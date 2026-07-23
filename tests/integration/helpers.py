# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Helper utilities for integration tests."""

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config
from ess.livedata.config.workflow_spec import JobId, ResultKey

logger = logging.getLogger(__name__)


class WaitTimeout(Exception):
    """Raised when waiting for a condition times out."""

    pass


def dump_diagnostics(backend: Any, instrument: str = 'dummy') -> None:
    """Print pipeline state for debugging a wait timeout.

    Watermark offsets show per-topic message counts without consuming,
    distinguishing "backend never published" from "dashboard never consumed".
    Lines are prefixed DIAG so CI failure annotations can grep them.
    """
    from confluent_kafka import Consumer, TopicPartition

    logger.warning("DIAG data_service keys: %s", list(backend.data_service))
    logger.warning("DIAG job_statuses: %s", backend.job_service.job_statuses)
    config = load_config(namespace=config_names.kafka, env='dev')
    consumer = Consumer({**config, 'group.id': f'diag-{uuid.uuid4()}'})
    try:
        for suffix in (
            'beam_monitor',
            'livedata_commands',
            'livedata_responses',
            'livedata_heartbeat',
            'livedata_data',
        ):
            topic = f'{instrument}_{suffix}'
            try:
                low, high = consumer.get_watermark_offsets(
                    TopicPartition(topic, 0), timeout=5.0
                )
                logger.warning("DIAG topic %s: low=%s high=%s", topic, low, high)
            except Exception as e:
                logger.warning("DIAG topic %s: watermark fetch failed: %s", topic, e)
    finally:
        consumer.close()


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


def _get_result_keys_for_job_id(data_service: Any, job_id: JobId) -> list[ResultKey]:
    """Get all ResultKeys in DataService that match the given JobId."""
    return [key for key in data_service if key.job_id == job_id]


# Job-specific helpers


def wait_for_job_data(
    backend: Any,
    job_ids: list[JobId],
    timeout: float = 10.0,
    poll_interval: float = 0.5,
) -> dict[JobId, dict[ResultKey, Any]]:
    """
    Wait for job data to arrive for specific jobs.

    This helper processes messages via backend.update() and waits until all
    specified jobs have received data, then returns the data.

    Parameters
    ----------
    backend:
        The DashboardBackend instance (must have .update() and .data_service)
    job_ids:
        List of JobIds to wait for
    timeout:
        Maximum time to wait in seconds
    poll_interval:
        Time between checks in seconds

    Returns
    -------
    :
        Dictionary mapping JobId to {ResultKey: data} for each job

    Raises
    ------
    WaitTimeout:
        If job data does not arrive for all jobs within the timeout period
    """

    def check_for_job_data():
        backend.update()
        for job_id in job_ids:
            keys = _get_result_keys_for_job_id(backend.data_service, job_id)
            if not keys:
                return False
        return True

    try:
        wait_for_condition(
            check_for_job_data, timeout=timeout, poll_interval=poll_interval
        )
    except WaitTimeout:
        dump_diagnostics(backend)
        raise

    result = {}
    for job_id in job_ids:
        keys = _get_result_keys_for_job_id(backend.data_service, job_id)
        result[job_id] = {key: backend.data_service[key] for key in keys}
    return result


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

    try:
        wait_for_condition(
            check_for_status_updates, timeout=timeout, poll_interval=poll_interval
        )
    except WaitTimeout:
        dump_diagnostics(backend)
        raise

    return {job_id: backend.job_service.job_statuses[job_id] for job_id in job_ids}
