# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
from collections.abc import Callable

from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.job import JobState, JobStatus


class JobService:
    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._job_statuses: dict[JobId, JobStatus] = {}
        self._removed_jobs: set[JobId] = set()
        self._job_status_update_subscribers: list[Callable[[], None]] = []

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
            self._logger.error("Error in job status update callback: %s", e)

    def status_updated(self, job_status: JobStatus) -> None:
        """Update the stored job status."""
        # Avoid re-adding removed jobs by in-flight status messages from backend.
        if (
            job_status.state == JobState.stopped
            and job_status.job_id in self._removed_jobs
        ):
            self._logger.debug(
                "Ignoring status update for removed job %s", job_status.job_id
            )
            return

        self._logger.debug("Job status updated: %s", job_status)
        self._job_statuses[job_status.job_id] = job_status
        self._notify_job_status_update()

    def _notify_job_status_update(self) -> None:
        """Notify listeners about job status updates."""
        self._logger.debug(
            "Job statuses updated for jobs: %s", list(self._job_statuses.keys())
        )

        # Notify all subscribers
        for callback in self._job_status_update_subscribers:
            try:
                callback()
            except Exception as e:
                self._logger.error("Error in job status update callback: %s", e)

    def remove_job(self, job_id: JobId) -> None:
        """Remove a job from tracking."""
        # Mark this job as removed to filter future status updates
        self._removed_jobs.add(job_id)

        # Remove from job statuses
        if job_id in self._job_statuses:
            del self._job_statuses[job_id]

        # Notify subscribers of the status update (removal)
        self._notify_job_status_update()
