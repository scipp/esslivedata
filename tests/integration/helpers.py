# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Helper utilities for integration tests."""

import time
from collections.abc import Callable
from typing import Any

from ess.livedata.config.workflow_spec import JobId, JobNumber, WorkflowId
from ess.livedata.dashboard.job_service import JobService


class WaitTimeout(Exception):
    """Raised when waiting for a condition times out."""

    pass


def wait_for_condition(
    condition: Callable[[], bool], timeout: float = 5.0, poll_interval: float = 0.1
) -> None:
    """
    Wait for a condition to become true.

    Parameters
    ----------
    condition:
        Callable that returns True when the condition is met
    timeout:
        Maximum time to wait in seconds
    poll_interval:
        Time between condition checks in seconds

    Raises
    ------
    WaitTimeout:
        If the condition is not met within the timeout period
    """
    start_time = time.time()
    while not condition():
        if time.time() - start_time > timeout:
            raise WaitTimeout(
                f"Condition not met within {timeout} seconds: {condition}"
            )
        time.sleep(poll_interval)


# Workflow-specific helpers


def get_workflow_jobs(
    job_service: JobService, workflow_id: WorkflowId
) -> list[JobNumber]:
    """
    Get all job numbers associated with a specific workflow.

    Parameters
    ----------
    job_service:
        The JobService to query
    workflow_id:
        The workflow ID to filter by

    Returns
    -------
    :
        List of job numbers for the given workflow
    """
    return [
        job_number
        for job_number, wf_id in job_service.job_info.items()
        if wf_id == workflow_id
    ]


def wait_for_workflow_job_data(
    backend: Any,
    workflow_id: WorkflowId,
    source_names: list[str],
    timeout: float = 10.0,
    poll_interval: float = 0.5,
) -> None:
    """
    Wait for job data to arrive for a specific workflow and sources.

    This helper processes messages via backend.update() and waits until job data
    arrives for at least one job matching the workflow ID and containing data
    for at least one of the specified sources.

    Parameters
    ----------
    backend:
        The DashboardBackend instance (must have .update() and .job_service)
    workflow_id:
        The workflow ID to wait for
    source_names:
        List of source names to expect data for
    timeout:
        Maximum time to wait in seconds
    poll_interval:
        Time between checks in seconds

    Raises
    ------
    WaitTimeout:
        If job data does not arrive within the timeout period
    """

    def check_for_job_data():
        backend.update()
        # Check if any job has data for our expected sources
        for job_number, source_data in backend.job_service.job_data.items():
            # Verify this job belongs to our workflow
            if backend.job_service.job_info.get(job_number) == workflow_id:
                # Check if we have data for any of our sources
                if any(source in source_data for source in source_names):
                    return True
        return False

    wait_for_condition(check_for_job_data, timeout=timeout, poll_interval=poll_interval)


def wait_for_workflow_job_statuses(
    backend: Any,
    workflow_id: WorkflowId,
    source_names: list[str] | None = None,
    timeout: float = 10.0,
    poll_interval: float = 0.5,
) -> None:
    """
    Wait for job status updates for a specific workflow.

    This helper processes messages via backend.update() and waits until status
    updates arrive for jobs belonging to the specified workflow.

    Parameters
    ----------
    backend:
        The DashboardBackend instance (must have .update() and .job_service)
    workflow_id:
        The workflow ID to wait for
    source_names:
        Optional list of source names to filter by. If None, waits for any source.
    timeout:
        Maximum time to wait in seconds
    poll_interval:
        Time between checks in seconds

    Raises
    ------
    WaitTimeout:
        If status updates do not arrive within the timeout period
    """

    def check_for_status_updates():
        backend.update()
        statuses = get_workflow_job_statuses(
            backend.job_service, workflow_id, source_names
        )
        return len(statuses) > 0

    wait_for_condition(
        check_for_status_updates, timeout=timeout, poll_interval=poll_interval
    )


def get_workflow_job_data(
    job_service: JobService,
    workflow_id: WorkflowId,
    source_names: list[str] | None = None,
) -> dict[JobNumber, dict[str, dict]]:
    """
    Get job data for a specific workflow, optionally filtered by source names.

    Parameters
    ----------
    job_service:
        The JobService to query
    workflow_id:
        The workflow ID to filter by
    source_names:
        Optional list of source names to filter by. If None, returns all sources.

    Returns
    -------
    :
        Dictionary mapping job numbers to their source data.
        Structure: {job_number: {source_name: {output_name: data}}}
    """
    workflow_jobs = get_workflow_jobs(job_service, workflow_id)
    result = {}

    for job_number in workflow_jobs:
        all_source_data = job_service.job_data.get(job_number, {})

        if source_names is None:
            # Return all sources for this job
            result[job_number] = all_source_data
        else:
            # Filter by requested source names
            filtered_data = {
                source: data
                for source, data in all_source_data.items()
                if source in source_names
            }
            if filtered_data:
                result[job_number] = filtered_data

    return result


def get_workflow_job_statuses(
    job_service: JobService,
    workflow_id: WorkflowId,
    source_names: list[str] | None = None,
) -> dict[JobId, Any]:
    """
    Get job statuses for a specific workflow, optionally filtered by source names.

    Parameters
    ----------
    job_service:
        The JobService to query
    workflow_id:
        The workflow ID to filter by
    source_names:
        Optional list of source names to filter by. If None, returns all sources.

    Returns
    -------
    :
        Dictionary mapping job IDs to their statuses for the given workflow.
    """
    workflow_job_numbers = set(get_workflow_jobs(job_service, workflow_id))
    result = {}

    for job_id, status in job_service.job_statuses.items():
        # Only include jobs that belong to our workflow
        if job_id.job_number not in workflow_job_numbers:
            continue

        # Optionally filter by source names
        if source_names is None or job_id.source_name in source_names:
            result[job_id] = status

    return result
