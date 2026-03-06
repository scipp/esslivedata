# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import time

import structlog

from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.job import JobStatus

logger = structlog.get_logger(__name__)

# Timeout threshold for considering status stale (60 seconds in nanoseconds)
STATUS_HEARTBEAT_TIMEOUT_NS = 60_000_000_000


class JobService:
    def __init__(
        self,
        *,
        heartbeat_timeout_ns: int = STATUS_HEARTBEAT_TIMEOUT_NS,
    ) -> None:
        self._job_statuses: dict[JobId, JobStatus] = {}
        self._job_status_timestamps: dict[JobId, int] = {}
        self._removed_jobs: set[JobId] = set()
        self._heartbeat_timeout_ns = heartbeat_timeout_ns

    @property
    def job_statuses(self) -> dict[JobId, JobStatus]:
        """Access to all stored job statuses."""
        return self._job_statuses

    def status_updated(self, job_status: JobStatus) -> None:
        """Update the stored job status and record timestamp."""
        # Ignore in-flight heartbeats for jobs that have been eagerly removed.
        if job_status.job_id in self._removed_jobs:
            return

        logger.debug("Job status updated: %s", job_status)
        self._job_statuses[job_status.job_id] = job_status
        self._job_status_timestamps[job_status.job_id] = time.time_ns()

    def remove_job(self, job_id: JobId) -> None:
        """Eagerly remove a job from tracking.

        Marks the job so that in-flight heartbeats arriving after removal
        do not re-add it.
        """
        self._removed_jobs.add(job_id)
        self._job_statuses.pop(job_id, None)
        self._job_status_timestamps.pop(job_id, None)

    def is_status_stale(self, job_id: JobId) -> bool:
        """Check if a job's status is stale (no recent heartbeat).

        Parameters
        ----------
        job_id
            The job to check.

        Returns
        -------
        :
            True if status is older than heartbeat timeout or doesn't exist.
        """
        if job_id not in self._job_status_timestamps:
            return True

        last_update = self._job_status_timestamps[job_id]
        current_time = time.time_ns()
        age_ns = current_time - last_update

        return age_ns > self._heartbeat_timeout_ns

    def remove_stale_jobs(self) -> None:
        """Remove jobs whose heartbeats have timed out.

        Called periodically to clean up jobs that are no longer broadcasting,
        e.g., after a stop command was processed by the backend.
        """
        stale = [jid for jid in self._job_statuses if self.is_status_stale(jid)]
        for jid in stale:
            logger.debug("Removing stale job %s", jid)
            self._job_statuses.pop(jid, None)
            self._job_status_timestamps.pop(jid, None)
