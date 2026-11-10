# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Mapping

import structlog

from ..config.workflow_spec import JobId, JobNumber, WorkflowConfig, WorkflowId
from .desired_state import DesiredWorkflow
from .job import JobState, JobStatus
from .job_manager import JobFactory, JobManager

logger = structlog.get_logger(__name__)

__all__ = ['JobReconciler']


class JobReconciler:
    """Starts and stops the jobs of a :class:`JobManager` to match the desired table.

    The worker wants ``JobId(source_name, job_number)`` for every source of every
    table entry whose workflow it owns (:meth:`JobFactory.owns`). Every commit
    mints a new job number, so comparing job ids covers a tombstone, a new
    generation, and a source dropped from a workflow. Reconciling twice against
    the same table changes nothing, so replayed or duplicate records need no
    deduplication.

    A job that fails to build is *rejected*. It is not retried until its record
    changes (which mints a new job number) or the process restarts, and its
    error is reported in :attr:`rejected` for as long as the table wants it.
    """

    def __init__(self, *, job_manager: JobManager, job_factory: JobFactory) -> None:
        self._job_manager = job_manager
        self._job_factory = job_factory
        self._rejected: dict[JobId, JobStatus] = {}
        self._unregistered: set[tuple[WorkflowId, JobNumber]] = set()

    @property
    def rejected(self) -> list[JobStatus]:
        """Error statuses of the wanted jobs that failed to build."""
        return list(self._rejected.values())

    def reconcile(self, table: Mapping[WorkflowId, DesiredWorkflow]) -> bool:
        """Stop the jobs the table does not want and build at most one missing job.

        Building a job can take seconds, so a call builds at most one. Heartbeats
        and running jobs thus continue while a restarted worker rebuilds.

        Returns
        -------
        :
            True if wanted jobs remain to be built by later calls.
        """
        self._log_unregistered(table)
        wanted = {
            JobId(source_name=source_name, job_number=want.job_number): want
            for want in table.values()
            if self._job_factory.owns(want.workflow_id)
            for source_name in want.jobs
        }
        for job in self._job_manager.all_jobs:
            if job.job_id not in wanted:
                self._job_manager.stop_job(job.job_id)
        self._rejected = {
            job_id: status
            for job_id, status in self._rejected.items()
            if job_id in wanted
        }
        existing = {job.job_id for job in self._job_manager.all_jobs}
        missing = [
            job_id
            for job_id in wanted
            if job_id not in existing and job_id not in self._rejected
        ]
        if missing:
            self._build(missing[0], wanted[missing[0]])
        return len(missing) > 1

    def _build(self, job_id: JobId, want: DesiredWorkflow) -> None:
        # Logged before building, so the journal names a job that kills the process.
        logger.info(
            'job_building', job_id=str(job_id), workflow_id=str(want.workflow_id)
        )
        desired = want.jobs[job_id.source_name]
        config = WorkflowConfig(
            identifier=want.workflow_id,
            job_id=job_id,
            params=desired.params,
            aux_source_names=desired.aux_source_names,
        )
        try:
            self._job_manager.schedule_job(config)
        except Exception as error:
            logger.exception(
                'job_rejected', job_id=str(job_id), workflow_id=str(want.workflow_id)
            )
            self._rejected[job_id] = JobStatus(
                job_id=job_id,
                workflow_id=want.workflow_id,
                state=JobState.error,
                error_message=str(error),
            )

    def _log_unregistered(self, table: Mapping[WorkflowId, DesiredWorkflow]) -> None:
        """Log each record of this instrument for an unregistered workflow once.

        No worker owns such a record, so it would otherwise go unreported.
        """
        unregistered = {
            (want.workflow_id, want.job_number)
            for want in table.values()
            if self._job_factory.is_unregistered(want.workflow_id)
        }
        for workflow_id, job_number in unregistered - self._unregistered:
            logger.warning(
                'desired_workflow_unregistered',
                workflow_id=str(workflow_id),
                job_number=str(job_number),
            )
        self._unregistered = unregistered
