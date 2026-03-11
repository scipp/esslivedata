# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Thread-safe registry of active job numbers.

Mediates between the background ingestion thread (Orchestrator.update)
and the UI thread (JobOrchestrator.commit_workflow / stop_workflow).
All lock acquisition is internal — callers never see the lock.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

import structlog

from ess.livedata.config.workflow_spec import JobNumber, ResultKey

if TYPE_CHECKING:
    from .data_service import DataService
    from .job_service import JobService

logger = structlog.get_logger(__name__)


class ActiveJobRegistry:
    """Thread-safe registry tracking which job numbers are currently active.

    Owns the synchronization between the background ingestion thread and the
    UI thread. The ingestion thread holds ``ingestion_guard()`` while
    processing messages; the UI thread calls ``activate`` / ``deactivate``
    when starting or stopping jobs.
    """

    def __init__(self, *, data_service: DataService, job_service: JobService) -> None:
        self._lock = threading.RLock()
        self._active: set[JobNumber] = set()
        self._data_service = data_service
        self._job_service = job_service

    def is_active(self, job_number: JobNumber) -> bool:
        """Check if a job number belongs to an active job."""
        return job_number in self._active

    @contextmanager
    def ingestion_guard(self):
        """Hold while processing a batch of messages.

        Prevents concurrent ``deactivate`` / ``cleanup`` from modifying the
        active set or deleting DataService buffers mid-iteration.
        """
        with self._lock:
            yield

    def activate(self, job_number: JobNumber) -> None:
        """Add a job number to the active set."""
        with self._lock:
            self._active.add(job_number)

    def deactivate(self, job_number: JobNumber) -> None:
        """Remove a job number from the active set and clean up its data.

        Removes DataService entries and JobService status entries keyed
        by the given job number. Serialized against ``ingestion_guard``
        to prevent dict-iteration crashes or orphaned buffers.
        """
        with self._lock:
            self._active.discard(job_number)
            self._cleanup(job_number)

    def restore(self, job_number: JobNumber) -> None:
        """Add a job number without acquiring the lock.

        Used during initialization when no other threads are running.
        """
        self._active.add(job_number)

    def _cleanup(self, job_number: JobNumber) -> None:
        """Remove buffered data and status tracking for a job number."""
        keys_to_remove = [
            key
            for key in self._data_service
            if isinstance(key, ResultKey) and key.job_id.job_number == job_number
        ]
        for key in keys_to_remove:
            del self._data_service[key]
        self._job_service.remove_jobs_by_number(job_number)
        logger.info(
            "Cleaned up job data",
            job_number=str(job_number),
            data_keys_removed=len(keys_to_remove),
        )
