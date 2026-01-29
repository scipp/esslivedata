# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import time
from collections.abc import Callable

import structlog

from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.job import JobState, JobStatus

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
        self._job_status_update_subscribers: list[Callable[[], None]] = []
        self._heartbeat_timeout_ns = heartbeat_timeout_ns

    @property
    def job_statuses(self) -> dict[JobId, JobStatus]:
        """Access to all stored job statuses."""
        return self._job_statuses

    def register_job_status_update_subscriber(
        self, callback: Callable[[], None]
    ) -> None:
        """Register a callback to be called when job status is updated."""
        self._job_status_update_subscribers.append(callback)
        # Immediately notify the new subscriber of current state
        try:
            callback()
        except Exception as e:
            logger.error("Error in job status update callback: %s", e)

    def status_updated(self, job_status: JobStatus) -> None:
        """Update the stored job status and record timestamp."""
        # Avoid re-adding removed jobs by in-flight status messages from backend.
        if (
            job_status.state == JobState.stopped
            and job_status.job_id in self._removed_jobs
        ):
            logger.debug("Ignoring status update for removed job %s", job_status.job_id)
            return

        logger.debug("Job status updated: %s", job_status)
        self._job_statuses[job_status.job_id] = job_status
        self._job_status_timestamps[job_status.job_id] = time.time_ns()
        self._notify_job_status_update()

    def _notify_job_status_update(self) -> None:
        """Notify listeners about job status updates."""
        logger.debug(
            "Job statuses updated for jobs: %s", list(self._job_statuses.keys())
        )

        # Notify all subscribers
        for callback in self._job_status_update_subscribers:
            try:
                callback()
            except Exception as e:
                logger.error("Error in job status update callback: %s", e)

    def remove_job(self, job_id: JobId) -> None:
        """Remove a job from tracking."""
        # Mark this job as removed to filter future status updates
        self._removed_jobs.add(job_id)

        # Remove from job statuses and timestamps
        if job_id in self._job_statuses:
            del self._job_statuses[job_id]
        if job_id in self._job_status_timestamps:
            del self._job_status_timestamps[job_id]

        # Notify subscribers of the status update (removal)
        self._notify_job_status_update()

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
