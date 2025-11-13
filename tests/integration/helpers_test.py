# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for integration test helper functions.

These tests verify the critical filtering and waiting logic in helpers.py.
If these helpers have bugs, integration tests could give false positives
(passing when they should fail).
"""

import uuid
from typing import Any

import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import JobId, JobNumber, ResultKey, WorkflowId
from ess.livedata.core.job_manager import JobState, JobStatus
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from tests.integration.helpers import (
    WaitTimeout,
    wait_for_condition,
    wait_for_job_data,
    wait_for_job_statuses,
)


def make_test_result(value: str) -> sc.DataArray:
    """Create a scipp DataArray representing a result value.

    The returned DataArray has a time dimension so that LatestValueExtractor
    will extract the scalar value from it.
    """
    return sc.DataArray(
        sc.array(dims=['time'], values=[value], unit='dimensionless'),
        coords={'time': sc.array(dims=['time'], values=[0.0], unit='s')},
    )


def make_workflow_id(name: str, version: int = 1) -> WorkflowId:
    """Helper to create WorkflowId for tests."""
    return WorkflowId(
        instrument='test_instrument',
        namespace='test_namespace',
        name=name,
        version=version,
    )


def make_job_number(seed: int) -> JobNumber:
    """Helper to create deterministic JobNumber (UUID) for tests."""
    return uuid.UUID(int=seed)


def make_job_id(job_number: JobNumber, source_name: str) -> JobId:
    """Helper to create JobId for tests."""
    return JobId(job_number=job_number, source_name=source_name)


class IntegrationTestBackend:
    """Backend stub for testing wait helpers with real services."""

    def __init__(self, job_service: JobService):
        self.job_service = job_service
        self.update_count = 0
        # Callbacks to execute on each update (for simulating data arrival)
        self.on_update_callbacks: list = []

    def update(self):
        """Simulate processing messages."""
        self.update_count += 1
        for callback in self.on_update_callbacks:
            callback()


# Pytest fixtures


@pytest.fixture
def data_service() -> DataService[ResultKey, Any]:
    """Create a DataService for testing."""
    return DataService()


@pytest.fixture
def job_service(data_service: DataService[ResultKey, Any]) -> JobService:
    """Create a JobService with a DataService for testing."""
    return JobService(data_service=data_service)


class TestWaitForCondition:
    """Tests for wait_for_condition helper."""

    def test_succeeds_when_condition_becomes_true(self):
        """wait_for_condition should return when condition becomes true."""
        counter = [0]

        def condition():
            counter[0] += 1
            return counter[0] >= 3

        wait_for_condition(condition, timeout=2.0, poll_interval=0.05)

        assert counter[0] >= 3

    def test_raises_timeout_when_condition_never_true(self):
        """Should raise WaitTimeout if condition never becomes true."""

        def always_false():
            return False

        with pytest.raises(WaitTimeout, match="Condition not met within"):
            wait_for_condition(always_false, timeout=0.2, poll_interval=0.05)


class TestWaitForJobData:
    """Tests for wait_for_job_data helper."""

    def test_succeeds_when_data_arrives_for_single_job(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """wait_for_job_data should return when data arrives for single job."""
        backend = IntegrationTestBackend(job_service)
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)
        job_id = make_job_id(job_1, 'source1')

        # Simulate data arriving after a few updates
        def simulate_data_arrival():
            if backend.update_count == 2:
                data_service[
                    ResultKey(
                        workflow_id=workflow_a,
                        job_id=job_id,
                        output_name='output1',
                    )
                ] = make_test_result('data')

        backend.on_update_callbacks.append(simulate_data_arrival)

        result = wait_for_job_data(backend, [job_id], timeout=2.0, poll_interval=0.05)

        assert backend.update_count >= 2
        # Should return dict mapping JobId to job_data
        assert job_id in result
        assert 'source1' in result[job_id]
        assert result[job_id]['source1']['output1'].value == 'data'

    def test_succeeds_when_data_arrives_for_all_jobs(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """wait_for_job_data should return when data arrives for all specified jobs."""
        backend = IntegrationTestBackend(job_service)
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)
        job_ids = [
            make_job_id(job_1, 'source1'),
            make_job_id(job_1, 'source2'),
        ]

        # Simulate data arriving after a few updates
        def simulate_data_arrival():
            if backend.update_count == 2:
                for job_id in job_ids:
                    data_service[
                        ResultKey(
                            workflow_id=workflow_a,
                            job_id=job_id,
                            output_name='output1',
                        )
                    ] = make_test_result('data')

        backend.on_update_callbacks.append(simulate_data_arrival)

        wait_for_job_data(backend, job_ids, timeout=2.0, poll_interval=0.05)

        assert backend.update_count >= 2

    def test_must_not_succeed_if_only_some_jobs_have_data(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """CRITICAL: must NOT succeed if only some jobs have data (require all)."""
        backend = IntegrationTestBackend(job_service)
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)
        job_ids = [
            make_job_id(job_1, 'source1'),
            make_job_id(job_1, 'source2'),
        ]

        # Add data for only the first job
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=job_ids[0],
                output_name='output1',
            )
        ] = make_test_result('data')

        with pytest.raises(WaitTimeout):
            wait_for_job_data(backend, job_ids, timeout=0.3, poll_interval=0.05)

    def test_must_not_succeed_for_wrong_job(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """CRITICAL: must NOT succeed when data arrives for different job."""
        backend = IntegrationTestBackend(job_service)
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)
        job_id_expected = make_job_id(job_1, 'source1')
        job_id_other = make_job_id(job_1, 'other_source')

        # Add data for different job
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=job_id_other,
                output_name='output1',
            )
        ] = make_test_result('data')

        with pytest.raises(WaitTimeout):
            wait_for_job_data(
                backend, [job_id_expected], timeout=0.3, poll_interval=0.05
            )


class TestWaitForJobStatuses:
    """Tests for wait_for_job_statuses helper."""

    def test_succeeds_when_status_arrives_for_single_job(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """wait_for_job_statuses should return when status arrives for single job."""
        backend = IntegrationTestBackend(job_service)
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)
        job_id = make_job_id(job_1, 'source1')

        # Simulate status arriving after a few updates
        def simulate_status_arrival():
            if backend.update_count == 2:
                job_service.status_updated(
                    JobStatus(
                        job_id=job_id,
                        workflow_id=workflow_a,
                        state=JobState.active,
                    )
                )

        backend.on_update_callbacks.append(simulate_status_arrival)

        result = wait_for_job_statuses(
            backend, [job_id], timeout=2.0, poll_interval=0.05
        )

        assert backend.update_count >= 2
        # Should return dict mapping JobId to status
        assert job_id in result
        assert result[job_id].state == JobState.active

    def test_succeeds_when_status_arrives_for_all_jobs(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """wait_for_job_statuses should return when status arrives for all jobs."""
        backend = IntegrationTestBackend(job_service)
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)
        job_ids = [
            make_job_id(job_1, 'source1'),
            make_job_id(job_1, 'source2'),
        ]

        # Simulate status arriving after a few updates
        def simulate_status_arrival():
            if backend.update_count == 2:
                for job_id in job_ids:
                    job_service.status_updated(
                        JobStatus(
                            job_id=job_id,
                            workflow_id=workflow_a,
                            state=JobState.active,
                        )
                    )

        backend.on_update_callbacks.append(simulate_status_arrival)

        wait_for_job_statuses(backend, job_ids, timeout=2.0, poll_interval=0.05)

        assert backend.update_count >= 2

    def test_must_not_succeed_if_only_some_jobs_have_status(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """CRITICAL: must NOT succeed if only some jobs have status (require all)."""
        backend = IntegrationTestBackend(job_service)
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)
        job_ids = [
            make_job_id(job_1, 'source1'),
            make_job_id(job_1, 'source2'),
        ]

        # Add status for only the first job
        job_service.status_updated(
            JobStatus(
                job_id=job_ids[0],
                workflow_id=workflow_a,
                state=JobState.active,
            )
        )

        with pytest.raises(WaitTimeout):
            wait_for_job_statuses(backend, job_ids, timeout=0.3, poll_interval=0.05)

    def test_must_not_succeed_for_wrong_job(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """CRITICAL: must NOT succeed when status arrives for different job."""
        backend = IntegrationTestBackend(job_service)
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)
        job_id_expected = make_job_id(job_1, 'source1')
        job_id_other = make_job_id(job_1, 'other_source')

        # Add status for different job
        job_service.status_updated(
            JobStatus(
                job_id=job_id_other,
                workflow_id=workflow_a,
                state=JobState.active,
            )
        )

        with pytest.raises(WaitTimeout):
            wait_for_job_statuses(
                backend, [job_id_expected], timeout=0.3, poll_interval=0.05
            )
