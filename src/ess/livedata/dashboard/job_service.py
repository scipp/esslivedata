# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import time
from collections.abc import Callable
from uuid import UUID

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
        self._heartbeat_timeout_ns = heartbeat_timeout_ns
        self._on_status_updated: Callable[[JobStatus], None] | None = None

    @property
    def on_status_updated(self) -> Callable[[JobStatus], None] | None:
        return self._on_status_updated

    @on_status_updated.setter
    def on_status_updated(self, callback: Callable[[JobStatus], None]) -> None:
        self._on_status_updated = callback

    @property
    def job_statuses(self) -> dict[JobId, JobStatus]:
        """Access to all stored job statuses."""
        return self._job_statuses

    def status_updated(self, job_status: JobStatus) -> None:
        """Update the stored job status and record timestamp."""
        logger.debug("Job status updated: %s", job_status)
        self._job_statuses[job_status.job_id] = job_status
        self._job_status_timestamps[job_status.job_id] = time.time_ns()
        if self._on_status_updated is not None:
            self._on_status_updated(job_status)

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

    def remove_jobs_by_number(self, job_number: UUID) -> None:
        """Remove all jobs matching a given job number.

        Used to clean up local tracking when a workflow is restarted
        and the previous job set is no longer needed.
        """
        to_remove = [jid for jid in self._job_statuses if jid.job_number == job_number]
        for jid in to_remove:
            logger.debug("Removing job %s (job_number=%s)", jid, job_number)
            self._job_statuses.pop(jid, None)
            self._job_status_timestamps.pop(jid, None)
