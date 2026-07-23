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
from ess.livedata.config.workflow_spec import DataKey, JobId, WorkflowId

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
            'detector',
            'livedata_commands',
            'livedata_responses',
            'livedata_heartbeat',
            'livedata_data',
            'livedata_roi',
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


def topic_high_watermark(topic: str) -> int:
    """Return the high watermark offset of a topic's single partition.

    Reads broker metadata without consuming, so it can prove that no new
    message was written to a topic (e.g. no spurious command was sent).
    """
    from confluent_kafka import Consumer, TopicPartition

    config = load_config(namespace=config_names.kafka, env='dev')
    consumer = Consumer({**config, 'group.id': f'watermark-{uuid.uuid4()}'})
    try:
        _low, high = consumer.get_watermark_offsets(
            TopicPartition(topic, 0), timeout=10.0
        )
        return high
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


def _get_job_data(
    data_service: Any, workflow_id: WorkflowId, source_name: str
) -> dict[DataKey, Any]:
    """Get all readable data in DataService for one workflow source.

    The data plane is keyed by the stable ``DataKey`` (workflow, source,
    output); the per-commit job_number is provenance, not identity, so jobs
    are matched via their source_name.

    A generation flip (recommit) clears buffers in place: the key stays
    iterable but reading it raises KeyError until new-generation data
    arrives. Such cleared keys are treated as absent.
    """
    result: dict[DataKey, Any] = {}
    for key in data_service:
        if key.workflow_id == workflow_id and key.source_name == source_name:
            try:
                result[key] = data_service[key]
            except KeyError:
                continue
    return result


def get_output_data(
    backend: Any,
    workflow_id: WorkflowId,
    source_name: str,
    output_name: str,
) -> Any | None:
    """Return the latest data for one output of a workflow source, or None."""
    for key, value in _get_job_data(
        backend.data_service, workflow_id, source_name
    ).items():
        if key.output_name == output_name:
            return value
    return None


def wait_for_backend_condition(
    backend: Any,
    condition: Callable[[], bool],
    timeout: float = 10.0,
    poll_interval: float = 0.5,
    instrument: str = 'dummy',
) -> None:
    """Pump the backend until ``condition()`` holds.

    Calls ``backend.update()`` before each check and dumps DIAG diagnostics
    on timeout, like the job-specific helpers.

    Parameters
    ----------
    backend:
        The DashboardBackend instance (must have .update())
    condition:
        Callable returning True when the awaited state is reached
    timeout:
        Maximum time to wait in seconds
    poll_interval:
        Time between checks in seconds
    instrument:
        Instrument name, used for topic names in timeout diagnostics

    Raises
    ------
    WaitTimeout:
        If the condition is not met within the timeout period
    """

    def check() -> bool:
        backend.update()
        return condition()

    try:
        wait_for_condition(check, timeout=timeout, poll_interval=poll_interval)
    except WaitTimeout:
        dump_diagnostics(backend, instrument=instrument)
        raise


# Job-specific helpers


def wait_for_job_data(
    backend: Any,
    workflow_id: WorkflowId,
    job_ids: list[JobId],
    timeout: float = 10.0,
    poll_interval: float = 0.5,
) -> dict[JobId, dict[DataKey, Any]]:
    """
    Wait for data to arrive for every source of the given jobs.

    This helper processes messages via backend.update() and waits until all
    specified jobs have received data, then returns the data.

    Parameters
    ----------
    backend:
        The DashboardBackend instance (must have .update() and .data_service)
    workflow_id:
        Workflow the jobs belong to
    job_ids:
        List of JobIds to wait for
    timeout:
        Maximum time to wait in seconds
    poll_interval:
        Time between checks in seconds

    Returns
    -------
    :
        Dictionary mapping JobId to {DataKey: data} for each job

    Raises
    ------
    WaitTimeout:
        If job data does not arrive for all jobs within the timeout period
    """

    result: dict[JobId, dict[DataKey, Any]] = {}

    def check_for_job_data():
        backend.update()
        result.clear()
        for job_id in job_ids:
            job_data = _get_job_data(
                backend.data_service, workflow_id, job_id.source_name
            )
            if not job_data:
                return False
            result[job_id] = job_data
        return True

    try:
        wait_for_condition(
            check_for_job_data, timeout=timeout, poll_interval=poll_interval
        )
    except WaitTimeout:
        dump_diagnostics(backend)
        raise

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
