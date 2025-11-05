# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Helper utilities for integration tests."""

import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

from ess.livedata.config.workflow_spec import JobId
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService

T = TypeVar('T')


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


def wait_for_data(data_service: DataService, key: Any, timeout: float = 5.0) -> Any:
    """
    Wait for data to appear in the DataService.

    Parameters
    ----------
    data_service:
        The DataService to monitor
    key:
        The key to wait for
    timeout:
        Maximum time to wait in seconds

    Returns
    -------
    :
        The data value once it appears

    Raises
    ------
    WaitTimeout:
        If the data does not appear within the timeout period
    """
    event = threading.Event()
    data_value = None

    def subscriber(keys: set) -> None:
        nonlocal data_value
        if key in keys:
            data_value = data_service[key]
            event.set()

    # Subscribe to data updates
    data_service.register_subscriber(subscriber)

    try:
        # Check if data already exists
        if key in data_service:
            return data_service[key]

        # Wait for data to arrive
        if not event.wait(timeout):
            raise WaitTimeout(
                f"Data for key {key} did not appear within {timeout} seconds"
            )
        return data_value
    finally:
        # Clean up: DataService doesn't have unsubscribe, but the subscriber
        # will just become inactive once this function returns
        pass


def wait_for_job_status(
    job_service: JobService,
    job_id: JobId,
    expected_status: str | None = None,
    timeout: float = 5.0,
) -> str:
    """
    Wait for a job to have a specific status (or any status if None).

    Parameters
    ----------
    job_service:
        The JobService to monitor
    job_id:
        The job ID to wait for
    expected_status:
        Expected status string (e.g., 'running', 'stopped').
        If None, waits for any status update.
    timeout:
        Maximum time to wait in seconds

    Returns
    -------
    :
        The job status once it matches the expected status

    Raises
    ------
    WaitTimeout:
        If the expected status is not reached within the timeout period
    """
    event = threading.Event()
    received_status = None

    def status_subscriber(updated_job_id: JobId, status: str) -> None:
        nonlocal received_status
        if updated_job_id == job_id:
            if expected_status is None or status == expected_status:
                received_status = status
                event.set()

    # Subscribe to job status updates
    job_service.register_job_status_update_subscriber(status_subscriber)

    try:
        # Check if job already has the expected status
        if job_id in job_service.job_statuses:
            current_status = job_service.job_statuses[job_id]
            if expected_status is None or current_status == expected_status:
                return current_status

        # Wait for status update
        if not event.wait(timeout):
            current = (
                job_service.job_statuses.get(job_id, 'unknown')
                if job_id in job_service.job_statuses
                else 'not found'
            )
            raise WaitTimeout(
                f"Job {job_id} did not reach status '{expected_status}' "
                f"within {timeout} seconds (current: {current})"
            )
        return received_status
    finally:
        pass


def wait_for_job_data(
    job_service: JobService, job_id: JobId, timeout: float = 5.0
) -> dict:
    """
    Wait for job data to arrive.

    Parameters
    ----------
    job_service:
        The JobService to monitor
    job_id:
        The job ID to wait for
    timeout:
        Maximum time to wait in seconds

    Returns
    -------
    :
        The job data once it arrives

    Raises
    ------
    WaitTimeout:
        If job data does not arrive within the timeout period
    """
    event = threading.Event()
    job_data = None

    def data_subscriber(updated_job_id: JobId) -> None:
        nonlocal job_data
        if updated_job_id == job_id:
            job_data = job_service.job_data.get(job_id)
            event.set()

    # Subscribe to job data updates
    job_service.register_job_update_subscriber(data_subscriber)

    try:
        # Check if data already exists
        if job_id in job_service.job_data:
            return job_service.job_data[job_id]

        # Wait for data to arrive
        if not event.wait(timeout):
            raise WaitTimeout(
                f"Job data for {job_id} did not arrive within {timeout} seconds"
            )
        return job_data
    finally:
        pass


@contextmanager
def collect_updates(data_service: DataService, max_updates: int | None = None):
    """
    Context manager to collect all data updates during a block.

    Parameters
    ----------
    data_service:
        The DataService to monitor
    max_updates:
        Maximum number of updates to collect (None for unlimited)

    Yields
    ------
    :
        A list that will be populated with tuples of (keys_added, keys_removed)
    """
    updates = []

    def subscriber(keys: set) -> None:
        updates.append(keys)
        if max_updates is not None and len(updates) >= max_updates:
            # Stop collecting
            pass

    data_service.register_subscriber(subscriber)
    try:
        yield updates
    finally:
        pass


def wait_for_multiple_data_updates(
    data_service: DataService, count: int, timeout: float = 5.0
) -> list[set]:
    """
    Wait for a specific number of data updates.

    Parameters
    ----------
    data_service:
        The DataService to monitor
    count:
        Number of updates to wait for
    timeout:
        Maximum time to wait in seconds

    Returns
    -------
    :
        List of sets containing the keys updated in each batch

    Raises
    ------
    WaitTimeout:
        If the expected number of updates is not received within the timeout period
    """
    with collect_updates(data_service) as updates:
        wait_for_condition(lambda: len(updates) >= count, timeout=timeout)
        return updates[:count]
