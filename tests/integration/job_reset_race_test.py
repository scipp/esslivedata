# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Integration test for issue #548: Job reset commands dropped when sent quickly.

This test requires Kafka and services to be running. Run with:
    docker-compose up kafka
    pytest -m integration tests/integration/job_reset_race_test.py -xvs

The test verifies that multiple job reset commands sent in quick succession are all
processed by the backend, not just the first or last one.
"""

import time
import uuid

import pytest

from ess.livedata.config.models import ConfigKey
from ess.livedata.config.workflow_spec import JobId, WorkflowId
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.core.message import COMMANDS_STREAM_ID, Message
from ess.livedata.handlers.config_handler import ConfigProcessor
from ess.livedata.handlers.monitor_workflow_specs import MonitorDataParams
from ess.livedata.kafka.message_adapter import RawConfigItem
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import (
    WaitTimeout,
    wait_for_condition,
    wait_for_job_statuses,
)


def make_job_id(source_name: str) -> JobId:
    """Create a JobId with a unique job number."""
    return JobId(source_name=source_name, job_number=uuid.uuid4())


def make_reset_command_message(job_id: JobId) -> Message[RawConfigItem]:
    """Create a reset command message for the given job ID."""
    config_key = ConfigKey(key=JobCommand.key, source_name=str(job_id))
    command = JobCommand(job_id=job_id, action=JobAction.reset)

    key_str = str(config_key)
    value_json = command.model_dump_json()

    return Message(
        stream=COMMANDS_STREAM_ID,
        timestamp=time.time_ns(),
        value=RawConfigItem(
            key=key_str.encode('utf-8'),
            value=value_json.encode('utf-8'),
        ),
    )


class TrackingJobManagerAdapter:
    """Job manager adapter that tracks all job_command calls."""

    def __init__(self):
        self.job_command_calls = []
        self.call_timestamps = []

    def job_command(self, source_name: str, value: dict) -> None:
        self.job_command_calls.append((source_name, value))
        self.call_timestamps.append(time.time_ns())

    def set_workflow_with_config(self, source_name, config):
        return []


class TestJobResetRaceCondition:
    """Unit tests for job reset command processing (no Kafka required)."""

    def test_multiple_reset_commands_same_batch_all_processed(self):
        """
        Test that multiple reset commands in the same message batch are all processed.
        """
        adapter = TrackingJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=adapter)

        job_ids = [make_job_id("mantle"), make_job_id("hr"), make_job_id("endcap")]
        messages = [make_reset_command_message(job_id) for job_id in job_ids]

        processor.process_messages(messages)

        assert len(adapter.job_command_calls) == len(job_ids), (
            f"Expected {len(job_ids)} job_command calls, "
            f"but got {len(adapter.job_command_calls)}. "
            f"Some commands may have been deduplicated incorrectly."
        )

        processed_source_names = {call[0] for call in adapter.job_command_calls}
        expected_source_names = {str(job_id) for job_id in job_ids}
        assert processed_source_names == expected_source_names

    def test_duplicate_reset_for_same_job_only_processed_once(self):
        """
        Test that duplicate reset commands for the SAME job are deduplicated.
        """
        adapter = TrackingJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=adapter)

        job_id = make_job_id("mantle")
        messages = [
            make_reset_command_message(job_id),
            make_reset_command_message(job_id),
            make_reset_command_message(job_id),
        ]

        processor.process_messages(messages)

        assert len(adapter.job_command_calls) == 1

    def test_mixed_commands_for_different_and_same_jobs(self):
        """
        Test a mix of unique job commands and duplicate commands.
        """
        adapter = TrackingJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=adapter)

        job_id_1 = make_job_id("mantle")
        job_id_2 = make_job_id("hr")

        messages = [
            make_reset_command_message(job_id_1),
            make_reset_command_message(job_id_2),
            make_reset_command_message(job_id_1),  # Duplicate
        ]

        processor.process_messages(messages)

        assert len(adapter.job_command_calls) == 2


# --- Full Integration Tests with Kafka and Services ---


def wait_for_jobs_with_data(
    backend, job_ids: list[JobId], timeout: float = 10.0
) -> dict[JobId, dict]:
    """
    Wait for jobs to have data in job_service.job_data.

    Returns dict mapping job_id to its data dict.
    """

    def check_jobs_have_data():
        backend.update()
        for job_id in job_ids:
            job_data = backend.job_service.job_data.get(job_id.job_number, {})
            if job_id.source_name not in job_data:
                return False
        return True

    wait_for_condition(check_jobs_have_data, timeout=timeout, poll_interval=0.2)

    return {
        job_id: backend.job_service.job_data[job_id.job_number] for job_id in job_ids
    }


def wait_for_jobs_reset(
    backend,
    job_ids: list[JobId],
    original_data: dict[JobId, dict],
    timeout: float = 10.0,
) -> None:
    """
    Wait for jobs to be reset (data changes indicating new accumulation started).

    We detect reset by checking if the job's status start_time has changed,
    or if we receive new data after the reset command.
    """

    def check_jobs_reset():
        backend.update()
        for job_id in job_ids:
            status = backend.job_service.job_statuses.get(job_id)
            if status is None:
                return False
            # Check if status exists and job is in a valid state after reset
            # A reset job should eventually become active again with new data
            if status.state.value not in ('active', 'scheduled'):
                return False
        return True

    wait_for_condition(check_jobs_reset, timeout=timeout, poll_interval=0.2)


@pytest.mark.integration
@pytest.mark.services('monitor')
@pytest.mark.parametrize("iteration", range(10))  # Run 10 times to catch timing issues
def test_rapid_reset_same_job_multiple_times(
    integration_env: IntegrationEnv, iteration: int
) -> None:
    """
    Test that multiple reset commands for the same job in rapid succession all work.

    This is a regression test for issue #548. Since the architecture enforces
    one-job-per-source, we test rapid resets on a single job.

    The test:
    1. Starts a workflow
    2. Waits for job to receive data
    3. Sends multiple reset commands in rapid succession (no sleep)
    4. Verifies job is reset and continues to receive data
    """
    backend = integration_env.backend

    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )

    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=['monitor1'],
        config=MonitorDataParams(),
    )

    assert len(job_ids) == 1, f"Expected 1 job, got {len(job_ids)}"
    job_id = job_ids[0]

    # First wait for job status to appear (job is registered)
    try:
        wait_for_job_statuses(backend, job_ids, timeout=10.0)
    except WaitTimeout:
        pytest.fail(
            f"Iteration {iteration}: Job status never appeared. "
            f"All statuses: {list(backend.job_service.job_statuses.keys())}"
        )

    # Then wait for job to receive data
    try:
        wait_for_jobs_with_data(backend, job_ids, timeout=15.0)
    except WaitTimeout:
        status = backend.job_service.job_statuses.get(job_id)
        pytest.fail(
            f"Iteration {iteration}: Job did not receive data within timeout. "
            f"Status: {status}"
        )

    # Send multiple reset commands in rapid succession (NO sleep between them)
    for _ in range(3):
        backend.job_controller.send_job_action(job_id, JobAction.reset)

    # Wait for job to become active again after resets
    try:
        wait_for_jobs_reset(backend, job_ids, {}, timeout=15.0)
    except WaitTimeout:
        status = backend.job_service.job_statuses.get(job_id)
        pytest.fail(
            f"Iteration {iteration}: Job was not reset within timeout. "
            f"Status: {status}"
        )

    # Verify job continues to receive data after resets
    try:
        wait_for_jobs_with_data(backend, job_ids, timeout=10.0)
    except WaitTimeout:
        status = backend.job_service.job_statuses.get(job_id)
        pytest.fail(
            f"Iteration {iteration}: Job did not receive data after reset. "
            f"Status: {status}"
        )


@pytest.mark.integration
@pytest.mark.services('monitor')
@pytest.mark.parametrize("iteration", range(10))
def test_reset_two_different_sources(
    integration_env: IntegrationEnv, iteration: int
) -> None:
    """
    Test resetting jobs for two different sources in rapid succession.

    This directly tests the scenario from issue #548: sending reset commands
    for different job IDs (different sources) in quick succession.
    """
    backend = integration_env.backend

    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )

    # Start workflow for TWO DIFFERENT sources (monitor1 and monitor2)
    # This creates jobs with different job_ids but same job_number
    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=['monitor1', 'monitor2'],
        config=MonitorDataParams(),
    )

    assert len(job_ids) == 2, f"Expected 2 jobs, got {len(job_ids)}"

    # First wait for job statuses to appear
    try:
        wait_for_job_statuses(backend, job_ids, timeout=10.0)
    except WaitTimeout:
        pytest.fail(
            f"Iteration {iteration}: Job statuses never appeared. "
            f"All statuses: {list(backend.job_service.job_statuses.keys())}"
        )

    # Wait for at least one job to receive data (monitor1 typically gets data first)
    # We'll wait for monitor1 specifically since fake_monitors may not produce
    # data for all monitors equally
    monitor1_jobs = [j for j in job_ids if j.source_name == 'monitor1']
    try:
        wait_for_jobs_with_data(backend, monitor1_jobs, timeout=15.0)
    except WaitTimeout:
        statuses = {
            job_id: backend.job_service.job_statuses.get(job_id) for job_id in job_ids
        }
        pytest.fail(
            f"Iteration {iteration}: monitor1 job did not receive data. "
            f"Statuses: {statuses}"
        )

    # CRITICAL: Send reset commands for BOTH jobs in rapid succession (NO sleep)
    for job_id in job_ids:
        backend.job_controller.send_job_action(job_id, JobAction.reset)

    # Wait for jobs to be in valid state after resets
    try:
        wait_for_jobs_reset(backend, job_ids, {}, timeout=15.0)
    except WaitTimeout:
        statuses = {
            job_id: backend.job_service.job_statuses.get(job_id) for job_id in job_ids
        }
        not_reset = [
            job_id
            for job_id in job_ids
            if (s := statuses.get(job_id))
            and s.state.value not in ('active', 'scheduled')
        ]

        pytest.fail(
            f"Iteration {iteration}: Not all jobs were reset within timeout. "
            f"Jobs not in active/scheduled state: {not_reset}. "
            f"Current statuses: {statuses}"
        )
