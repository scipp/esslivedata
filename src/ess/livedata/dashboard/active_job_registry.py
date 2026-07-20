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

    Data lifecycle
    --------------
    Active-set membership and buffered-data eviction are intentionally
    decoupled into two methods:

    - :py:meth:`deactivate` removes a job from the active set, causing
      ``Orchestrator.forward`` to filter out any further results published
      for that job. This happens as soon as a job stops, so late messages
      are dropped immediately.
    - :py:meth:`cleanup` deletes the buffered data for a job. Callers defer
      this so a just-stopped job's data stays available to plots that may
      be created or reconfigured while the workflow is stopped (see
      :py:meth:`LayerSubscription.start`'s previous-job fallback).

    :py:class:`JobOrchestrator` invokes ``cleanup`` only when a job leaves
    ``state.previous`` — i.e. when a newer stop or commit replaces it.
    Stopping a workflow does **not** by itself call ``cleanup``.
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
        """Remove a job number from the active set.

        Does not delete buffered data — call :py:meth:`cleanup` separately
        when the job's data is no longer needed. Serialized against
        ``ingestion_guard`` so the ingest filter sees a consistent view.
        """
        with self._lock:
            self._active.discard(job_number)

    def restore(self, job_number: JobNumber) -> None:
        """Add a job number without acquiring the lock.

        Used during initialization when no other threads are running.
        """
        self._active.add(job_number)

    def cleanup(self, job_number: JobNumber) -> None:
        """Remove buffered data and status tracking for a job number.

        Independent of active-set membership: callers may retain buffered
        data after deactivation so that newly created plots can bind to a
        recently stopped job's results. See the class docstring for the
        data lifecycle that governs when callers should run this.

        Deletes are batched inside a DataService transaction, which holds
        the DataService lock for its duration: a concurrent pull observes
        either all keys or none, and subscribers get a single coalesced
        notification. Without this, a multi-source plot pulling mid-eviction
        could rebuild with only a subset of its sources still present.

        With today's orchestration there is no live subscriber bound to a
        job's keys at the moment its data is cleaned up (subscribers
        rebind to the new job before the old data is dropped), so the
        transaction is currently defensive. Keeping the guarantee here
        prevents the bug from resurfacing if a future caller — e.g. a UI
        action that explicitly evicts a job — runs cleanup while plots
        are still subscribed.
        """
        with self._lock:
            keys_to_remove = [
                key
                for key in self._data_service
                if isinstance(key, ResultKey) and key.job_id.job_number == job_number
            ]
            with self._data_service.transaction():
                for key in keys_to_remove:
                    del self._data_service[key]
            self._job_service.remove_jobs_by_number(job_number)
            logger.info(
                "Cleaned up job data",
                job_number=str(job_number),
                data_keys_removed=len(keys_to_remove),
            )
