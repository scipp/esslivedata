# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Hashable
from typing import TypeVar

import scipp as sc

from ess.livedata.config.workflow_spec import JobId, JobNumber, ResultKey, WorkflowId
from ess.livedata.core.job import JobState, JobStatus

from .data_service import DataService

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')

SourceName = str
SourceData = dict[str | None, sc.DataArray]


class JobService:
    def __init__(
        self,
        *,
        data_service: DataService[ResultKey, sc.DataArray],
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._data_service = data_service
        self._job_data: dict[JobNumber, dict[SourceName, SourceData]] = {}
        self._job_info: dict[JobNumber, WorkflowId] = {}
        self._job_statuses: dict[JobId, JobStatus] = {}
        self._last_status_update: dict[JobId, float] = {}
        self._removed_jobs: set[JobId] = set()
        self._job_data_update_subscribers: list[Callable[[], None]] = []
        self._job_status_update_subscribers: list[Callable[[], None]] = []
        self._data_service.register_update_callback(self.data_updated)

    @property
    def job_data(self) -> dict[JobNumber, dict[SourceName, SourceData]]:
        return self._job_data

    @property
    def job_info(self) -> dict[JobNumber, WorkflowId]:
        return self._job_info

    @property
    def job_statuses(self) -> dict[JobId, JobStatus]:
        """Access to all stored job statuses."""
        return self._job_statuses

    def get_job_status(self, job_id: JobId) -> JobStatus | None:
        """Get the current status of a specific job."""
        return self._job_statuses.get(job_id)

    def get_last_status_update_time(self, job_id: JobId) -> float | None:
        """Get the timestamp of the last status update for a job."""
        return self._last_status_update.get(job_id)

    def is_job_status_stale(
        self, job_id: JobId, stale_threshold_seconds: float = 300
    ) -> bool:
        """
        Check if a job's status is stale (no updates received for a while).

        Args:
            job_id: The job to check
            stale_threshold_seconds: Consider stale if no updates for this many seconds

        Returns:
            True if the job status is stale, False otherwise
        """
        last_update = self._last_status_update.get(job_id)
        if last_update is None:
            return False  # No updates yet, not considered stale

        return time.time() - last_update > stale_threshold_seconds

    def get_stale_jobs(self, stale_threshold_seconds: float = 300) -> list[JobId]:
        """
        Get all jobs whose status updates are stale.

        Args:
            stale_threshold_seconds: Consider stale if no updates for this many seconds

        Returns:
            List of job IDs with stale status updates
        """
        stale_jobs = []
        current_time = time.time()

        for job_id, last_update in self._last_status_update.items():
            if current_time - last_update > stale_threshold_seconds:
                stale_jobs.append(job_id)

        return stale_jobs

    def register_job_update_subscriber(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when job data is updated."""
        self._job_data_update_subscribers.append(callback)
        # Immediately notify the new subscriber of current state
        try:
            callback()
        except Exception as e:
            self._logger.error("Error in job update callback: %s", e)

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
        """Update the stored job status and track when the update was received."""
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
        self._last_status_update[job_status.job_id] = time.time()
        self._notify_job_status_update()

    def data_updated(self, updated_keys: set[ResultKey]) -> None:
        notify_job_data_update = False
        for key in updated_keys:
            value = self._data_service[key]
            job_number = key.job_id.job_number
            source_name = key.job_id.source_name
            if (workflow_id := self._job_info.get(job_number)) is None:
                self._job_info[job_number] = key.workflow_id
            elif workflow_id != key.workflow_id:
                self._logger.warning(
                    "Workflow ID mismatch for job %s: existing %s, new %s. ",
                    key.job_id,
                    workflow_id,
                    key.workflow_id,
                )
            all_source_data = self._job_data.setdefault(job_number, {})
            # Store in DataService for access by plots etc.
            # Outputs are stored in a dict keyed by output_name.
            if source_name not in all_source_data:
                notify_job_data_update = True
            source_data = all_source_data.setdefault(source_name, {})
            source_data[key.output_name] = value
        if notify_job_data_update:
            self._notify_job_data_update()

    def _notify_job_data_update(self) -> None:
        """Notify listeners about job updates."""
        # For now we just log the job updates. In future we may want to have a more
        # sophisticated notification mechanism.
        self._logger.debug("Job data updated for jobs: %s", list(self._job_info.keys()))

        # Notify all subscribers
        for callback in self._job_data_update_subscribers:
            try:
                callback()
            except Exception as e:
                self._logger.error("Error in job update callback: %s", e)

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

        # Remove from last status update tracking
        if job_id in self._last_status_update:
            del self._last_status_update[job_id]

        # Check if we need to remove job data and info
        job_number = job_id.job_number
        source_name = job_id.source_name

        if job_number in self._job_data and source_name in self._job_data[job_number]:
            del self._job_data[job_number][source_name]

            # If this was the last source for this job number, remove the job entirely
            if not self._job_data[job_number]:
                del self._job_data[job_number]
                if job_number in self._job_info:
                    del self._job_info[job_number]

        # Notify subscribers of the status update (removal)
        self._notify_job_status_update()

    # To move from WorkflowController
    def stop_job(self, job: JobId) -> None: ...
