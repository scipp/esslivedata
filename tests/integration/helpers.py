# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Helper utilities for integration tests."""

import time
from collections.abc import Callable
from typing import Any

from ess.livedata.config.workflow_spec import JobId


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


# Job-specific helpers


def wait_for_job_data(
    backend: Any,
    job_ids: list[JobId],
    timeout: float = 10.0,
    poll_interval: float = 0.5,
) -> dict[JobId, dict[str, dict]]:
    """
    Wait for job data to arrive for specific jobs.

    This helper processes messages via backend.update() and waits until all
    specified jobs have received data, then returns the data.

    Parameters
    ----------
    backend:
        The DashboardBackend instance (must have .update() and .job_service)
    job_ids:
        List of JobIds to wait for
    timeout:
        Maximum time to wait in seconds
    poll_interval:
        Time between checks in seconds

    Returns
    -------
    :
        Dictionary mapping JobId to source data for each job

    Raises
    ------
    WaitTimeout:
        If job data does not arrive for all jobs within the timeout period
    """

    def check_for_job_data():
        backend.update()
        for job_id in job_ids:
            job_data = backend.job_service.job_data.get(job_id.job_number, {})
            if job_id.source_name not in job_data:
                return False
        return True

    wait_for_condition(check_for_job_data, timeout=timeout, poll_interval=poll_interval)

    return {
        job_id: backend.job_service.job_data[job_id.job_number] for job_id in job_ids
    }


def wait_for_job_statuses(
    backend: Any,
    job_ids: list[JobId],
    timeout: float = 10.0,
    poll_interval: float = 0.5,
) -> dict[JobId, Any]:
    """
    Wait for job status updates for specific jobs.

    This helper processes messages via backend.update() and waits until all
    specified jobs have received status updates, then returns the statuses.

    Parameters
    ----------
    backend:
        The DashboardBackend instance (must have .update() and .job_service)
    job_ids:
        List of JobIds to wait for
    timeout:
        Maximum time to wait in seconds
    poll_interval:
        Time between checks in seconds

    Returns
    -------
    :
        Dictionary mapping JobId to status for each job

    Raises
    ------
    WaitTimeout:
        If status updates do not arrive for all jobs within the timeout period
    """

    def check_for_status_updates():
        backend.update()
        return all(job_id in backend.job_service.job_statuses for job_id in job_ids)

    wait_for_condition(
        check_for_status_updates, timeout=timeout, poll_interval=poll_interval
    )

    return {job_id: backend.job_service.job_statuses[job_id] for job_id in job_ids}
