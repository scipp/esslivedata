# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Thread-safe registry of the current generation per workflow.

Mediates between the background ingestion thread (Orchestrator.update)
and the UI thread (JobOrchestrator.commit_workflow / stop_workflow).
All lock acquisition is internal — callers never see the lock.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from ess.livedata.config.workflow_spec import JobNumber, WorkflowId

if TYPE_CHECKING:
    from .data_service import DataService
    from .job_service import JobService

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Generation:
    """One commit of a workflow: its wire-level job_number and its config.

    The config is retained so data stamped with this generation can be
    resolved back to the parameters it was computed with (provenance).
    """

    job_number: JobNumber
    config: Mapping[str, Any]


@dataclass
class _GenerationRecord:
    """Current and last generation of one workflow.

    ``last`` exists for provenance of a stopped workflow's retained data and
    for recognizing status heartbeats of a just-replaced job; only ``current``
    admits data.
    """

    current: Generation | None = None
    last: Generation | None = None
    stale_count: int = 0
    logged_stale: set[JobNumber] = field(default_factory=set)


class ActiveJobRegistry:
    """Thread-safe registry tracking the current generation per workflow.

    Owns the synchronization between the background ingestion thread and the
    UI thread. The ingestion thread holds ``ingestion_guard()`` while
    processing messages; the UI thread calls ``begin_generation`` /
    ``deactivate`` when committing or stopping workflows.

    Data lifecycle
    --------------
    The dashboard's data plane is keyed by stable ``DataKey``s, so buffered
    data is not evicted when a job stops — a stopped workflow's last data
    stays displayed under its keys. The one eviction point is
    :py:meth:`begin_generation` (called only from ``commit_workflow``): the
    workflow's buffers are cleared so the new generation starts blank and a
    windowed extractor can never aggregate across a parameter change.
    """

    def __init__(self, *, data_service: DataService, job_service: JobService) -> None:
        self._lock = threading.RLock()
        self._generations: dict[WorkflowId, _GenerationRecord] = {}
        self._data_service = data_service
        self._job_service = job_service

    @contextmanager
    def ingestion_guard(self):
        """Hold while processing a batch of messages.

        Prevents a concurrent generation flip from clearing DataService
        buffers mid-batch: post-flip, buffers hold only current-generation
        data (a batch admitted before the flip is cleared by it; a batch
        after it fails the ``is_current`` filter).
        """
        with self._lock:
            yield

    def is_current(self, workflow_id: WorkflowId, job_number: JobNumber) -> bool:
        """Check whether a wire job_number is the workflow's current generation."""
        with self._lock:
            record = self._generations.get(workflow_id)
            return (
                record is not None
                and record.current is not None
                and record.current.job_number == job_number
            )

    def is_known_job(self, job_number: JobNumber) -> bool:
        """Check whether a job_number is any workflow's current or last generation.

        Used to filter ``JobStatus`` heartbeats, which keep job-number
        semantics: a just-stopped or just-replaced job may still report its
        final states.
        """
        with self._lock:
            return any(
                gen is not None and gen.job_number == job_number
                for record in self._generations.values()
                for gen in (record.current, record.last)
            )

    def begin_generation(
        self, workflow_id: WorkflowId, job_number: JobNumber, config: Mapping[str, Any]
    ) -> None:
        """Flip a workflow to a new generation and clear its buffered data.

        This is the commit flip: the previous current generation rotates into
        ``last`` (its retained config still resolvable for provenance), the
        generation that drops off the two-entry window has its job-status
        entries pruned, and the workflow's DataService buffers are cleared —
        atomically with respect to ingest (callers hold ``ingestion_guard``)
        and to pulls (the clear holds the DataService lock across all keys).
        This is the only place buffers are cleared.
        """
        with self._lock:
            record = self._generations.setdefault(workflow_id, _GenerationRecord())
            dropped = record.last
            record.last = record.current
            record.current = Generation(job_number=job_number, config=config)
            keys = [k for k in self._data_service if k.workflow_id == workflow_id]
            self._data_service.clear_keys(keys)
            if dropped is not None:
                self._job_service.remove_jobs_by_number(dropped.job_number)
            logger.info(
                "Began new generation",
                workflow_id=str(workflow_id),
                job_number=str(job_number),
                data_keys_cleared=len(keys),
                stale_messages_dropped=record.stale_count,
            )
            record.stale_count = 0
            record.logged_stale.clear()

    def deactivate(self, workflow_id: WorkflowId) -> None:
        """Clear the workflow's current generation (rotating it into ``last``).

        Stops admitting data for the workflow without touching its buffers:
        the stopped generation's data stays displayed under its stable keys
        until the next :py:meth:`begin_generation` clears it. Serialized
        against ``ingestion_guard`` so the ingest filter sees a consistent
        view.
        """
        with self._lock:
            record = self._generations.get(workflow_id)
            if record is None or record.current is None:
                return
            dropped = record.last
            record.last = record.current
            record.current = None
            if dropped is not None:
                self._job_service.remove_jobs_by_number(dropped.job_number)

    def restore(self, workflow_id: WorkflowId, *, current: Generation) -> None:
        """Restore the persisted current generation without acquiring the lock.

        Used during initialization when no other threads are running. Only
        ``current`` survives a dashboard restart: a stopped backend job no
        longer heartbeats, so there is nothing a restored ``last`` generation
        would admit.
        """
        self._generations[workflow_id] = _GenerationRecord(current=current)

    def resolve_config(
        self, workflow_id: WorkflowId, job_number: JobNumber
    ) -> Mapping[str, Any] | None:
        """Resolve a generation stamp to the config it was committed with.

        Covers the current and last generation; older stamps (no longer
        resolvable) return None.
        """
        with self._lock:
            record = self._generations.get(workflow_id)
            if record is None:
                return None
            for gen in (record.current, record.last):
                if gen is not None and gen.job_number == job_number:
                    return gen.config
            return None

    def record_stale(self, workflow_id: WorkflowId, job_number: JobNumber) -> None:
        """Count a dropped data message from a non-current generation.

        Logs once per distinct stale job_number; the accumulated count is
        emitted with the next generation flip.
        """
        with self._lock:
            record = self._generations.setdefault(workflow_id, _GenerationRecord())
            record.stale_count += 1
            if job_number not in record.logged_stale:
                record.logged_stale.add(job_number)
                logger.info(
                    "Dropping data from non-current generation",
                    workflow_id=str(workflow_id),
                    job_number=str(job_number),
                )
