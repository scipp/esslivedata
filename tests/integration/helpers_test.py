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

from ess.livedata.config.workflow_spec import JobId, JobNumber, ResultKey, WorkflowId
from ess.livedata.core.job_manager import JobState, JobStatus
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from tests.integration.helpers import (
    WaitTimeout,
    get_workflow_job_data,
    get_workflow_job_statuses,
    get_workflow_jobs,
    wait_for_condition,
    wait_for_workflow_job_data,
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


class TestGetWorkflowJobs:
    """Tests for get_workflow_jobs helper."""

    def test_returns_only_matching_workflow(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """get_workflow_jobs should only return jobs for specified workflow."""
        workflow_a = make_workflow_id('workflow_a')
        workflow_b = make_workflow_id('workflow_b')

        # Add data to populate job_info - JobService needs data to map jobs to workflows
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(make_job_number(1), 'source1'),
                output_name=None,
            )
        ] = 'data1'
        data_service[
            ResultKey(
                workflow_id=workflow_b,
                job_id=make_job_id(make_job_number(2), 'source1'),
                output_name=None,
            )
        ] = 'data2'
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(make_job_number(3), 'source1'),
                output_name=None,
            )
        ] = 'data3'

        result = get_workflow_jobs(job_service, workflow_a)

        assert result == [make_job_number(1), make_job_number(3)]

    def test_returns_empty_for_unknown_workflow(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """get_workflow_jobs should return empty list for non-existent workflow."""
        workflow_a = make_workflow_id('workflow_a')

        # Add data to populate job_info
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(make_job_number(1), 'source1'),
                output_name=None,
            )
        ] = 'data1'

        result = get_workflow_jobs(job_service, make_workflow_id('nonexistent'))

        assert result == []


class TestGetWorkflowJobData:
    """Tests for get_workflow_job_data helper."""

    def test_filters_by_workflow(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """get_workflow_job_data must only return data from specified workflow."""
        workflow_a = make_workflow_id('workflow_a')
        workflow_b = make_workflow_id('workflow_b')
        job_1 = make_job_number(1)
        job_2 = make_job_number(2)

        # Add job status to establish workflow_id mapping
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_1, 'source1'),
                workflow_id=workflow_a,
                state=JobState.active,
            )
        )
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_2, 'source1'),
                workflow_id=workflow_b,
                state=JobState.active,
            )
        )

        # Add data via DataService - JobService.data_updated() called automatically
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(job_1, 'source1'),
                output_name='output1',
            )
        ] = 'data1'
        data_service[
            ResultKey(
                workflow_id=workflow_b,
                job_id=make_job_id(job_2, 'source1'),
                output_name='output1',
            )
        ] = 'data2'

        result = get_workflow_job_data(job_service, workflow_a)

        # CRITICAL: Must not return data from workflow_b!
        assert job_1 in result
        assert job_2 not in result
        assert result[job_1] == {'source1': {'output1': 'data1'}}

    def test_filters_by_source_names(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """get_workflow_job_data must filter by source names when provided."""
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)

        # Add job status for all sources
        for source_name in ['source1', 'source2', 'source3']:
            job_service.status_updated(
                JobStatus(
                    job_id=make_job_id(job_1, source_name),
                    workflow_id=workflow_a,
                    state=JobState.active,
                )
            )

        # Add data via DataService for all sources
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(job_1, 'source1'),
                output_name='output1',
            )
        ] = 'data1'
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(job_1, 'source2'),
                output_name='output2',
            )
        ] = 'data2'
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(job_1, 'source3'),
                output_name='output3',
            )
        ] = 'data3'

        result = get_workflow_job_data(
            job_service,
            workflow_a,
            source_names=['source1', 'source3'],
        )

        # CRITICAL: Must only return requested sources!
        assert job_1 in result
        assert 'source1' in result[job_1]
        assert 'source3' in result[job_1]
        assert 'source2' not in result[job_1]

    def test_excludes_jobs_with_no_matching_sources(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """Jobs with no matching sources should not appear in result."""
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)
        job_2 = make_job_number(2)

        # Add job statuses
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_1, 'source1'),
                workflow_id=workflow_a,
                state=JobState.active,
            )
        )
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_2, 'other_source'),
                workflow_id=workflow_a,
                state=JobState.active,
            )
        )

        # Add data
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(job_1, 'source1'),
                output_name='output1',
            )
        ] = 'data1'
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(job_2, 'other_source'),
                output_name='output2',
            )
        ] = 'data2'

        result = get_workflow_job_data(
            job_service, workflow_a, source_names=['source1']
        )

        # Job 2 has no matching sources, should not appear
        assert job_1 in result
        assert job_2 not in result

    def test_returns_all_sources_when_none_specified(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """When source_names is None, should return all sources."""
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)

        # Add job statuses for both sources
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_1, 'source1'),
                workflow_id=workflow_a,
                state=JobState.active,
            )
        )
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_1, 'source2'),
                workflow_id=workflow_a,
                state=JobState.active,
            )
        )

        # Add data
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(job_1, 'source1'),
                output_name='output1',
            )
        ] = 'data1'
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(job_1, 'source2'),
                output_name='output2',
            )
        ] = 'data2'

        result = get_workflow_job_data(job_service, workflow_a, source_names=None)

        assert job_1 in result
        assert 'source1' in result[job_1]
        assert 'source2' in result[job_1]


class TestGetWorkflowJobStatuses:
    """Tests for get_workflow_job_statuses helper."""

    def test_filters_by_workflow(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """Must only return statuses from specified workflow."""
        workflow_a = make_workflow_id('workflow_a')
        workflow_b = make_workflow_id('workflow_b')

        # Add data to populate job_info
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(make_job_number(1), 'source1'),
                output_name=None,
            )
        ] = 'data1'
        data_service[
            ResultKey(
                workflow_id=workflow_b,
                job_id=make_job_id(make_job_number(2), 'source1'),
                output_name=None,
            )
        ] = 'data2'

        # Add job statuses
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(make_job_number(1), 'source1'),
                workflow_id=workflow_a,
                state=JobState.active,
            )
        )
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(make_job_number(2), 'source1'),
                workflow_id=workflow_b,
                state=JobState.finishing,
            )
        )

        result = get_workflow_job_statuses(job_service, workflow_a)

        # CRITICAL: Must not return statuses from workflow_b!
        assert make_job_id(make_job_number(1), 'source1') in result
        assert make_job_id(make_job_number(2), 'source1') not in result

    def test_filters_by_source_names(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """Must filter by source names when provided."""
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)

        # Add data to populate job_info
        for source_name in ['source1', 'source2', 'source3']:
            data_service[
                ResultKey(
                    workflow_id=workflow_a,
                    job_id=make_job_id(job_1, source_name),
                    output_name=None,
                )
            ] = f'data_{source_name}'

        # Add job statuses for multiple sources
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_1, 'source1'),
                workflow_id=workflow_a,
                state=JobState.active,
            )
        )
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_1, 'source2'),
                workflow_id=workflow_a,
                state=JobState.finishing,
            )
        )
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_1, 'source3'),
                workflow_id=workflow_a,
                state=JobState.error,
            )
        )

        result = get_workflow_job_statuses(
            job_service,
            workflow_a,
            source_names=['source1', 'source3'],
        )

        # CRITICAL: Must only return requested sources!
        assert make_job_id(job_1, 'source1') in result
        assert make_job_id(job_1, 'source3') in result
        assert make_job_id(job_1, 'source2') not in result

    def test_returns_all_sources_when_none_specified(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """When source_names is None, should return all sources."""
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)

        # Add data to populate job_info
        for source_name in ['source1', 'source2']:
            data_service[
                ResultKey(
                    workflow_id=workflow_a,
                    job_id=make_job_id(job_1, source_name),
                    output_name=None,
                )
            ] = f'data_{source_name}'

        # Add job statuses for multiple sources
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_1, 'source1'),
                workflow_id=workflow_a,
                state=JobState.active,
            )
        )
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_1, 'source2'),
                workflow_id=workflow_a,
                state=JobState.finishing,
            )
        )

        result = get_workflow_job_statuses(job_service, workflow_a, source_names=None)

        assert make_job_id(job_1, 'source1') in result
        assert make_job_id(job_1, 'source2') in result


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


class TestWaitForWorkflowJobData:
    """Tests for wait_for_workflow_job_data helper."""

    def test_succeeds_when_data_arrives(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """wait_for_workflow_job_data should return when correct data arrives."""
        backend = IntegrationTestBackend(job_service)
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)

        # Simulate data arriving after a few updates
        def simulate_data_arrival():
            if backend.update_count == 2:
                # Add job status
                job_service.status_updated(
                    JobStatus(
                        job_id=make_job_id(job_1, 'source1'),
                        workflow_id=workflow_a,
                        state=JobState.active,
                    )
                )
                # Add data
                data_service[
                    ResultKey(
                        workflow_id=workflow_a,
                        job_id=make_job_id(job_1, 'source1'),
                        output_name='output1',
                    )
                ] = 'data'

        backend.on_update_callbacks.append(simulate_data_arrival)

        wait_for_workflow_job_data(
            backend,
            workflow_a,
            ['source1'],
            timeout=2.0,
            poll_interval=0.05,
        )

        # Should have succeeded
        assert backend.update_count >= 2

    def test_must_not_succeed_for_wrong_workflow(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """CRITICAL: must NOT succeed for wrong workflow data."""
        backend = IntegrationTestBackend(job_service)
        workflow_a = make_workflow_id('workflow_a')
        workflow_b = make_workflow_id('workflow_b')
        job_1 = make_job_number(1)

        # Add data for different workflow
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_1, 'source1'),
                workflow_id=workflow_b,
                state=JobState.active,
            )
        )
        data_service[
            ResultKey(
                workflow_id=workflow_b,
                job_id=make_job_id(job_1, 'source1'),
                output_name='output1',
            )
        ] = 'data'

        with pytest.raises(WaitTimeout):
            wait_for_workflow_job_data(
                backend,
                workflow_a,  # Looking for workflow_a
                ['source1'],
                timeout=0.3,
                poll_interval=0.05,
            )

    def test_must_not_succeed_for_wrong_source(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """CRITICAL: wait_for_workflow_job_data must NOT succeed for wrong source."""
        backend = IntegrationTestBackend(job_service)
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)

        # Add data for correct workflow but wrong source
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_1, 'other_source'),
                workflow_id=workflow_a,
                state=JobState.active,
            )
        )
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(job_1, 'other_source'),
                output_name='output1',
            )
        ] = 'data'

        with pytest.raises(WaitTimeout):
            wait_for_workflow_job_data(
                backend,
                workflow_a,
                ['source1'],  # Looking for source1
                timeout=0.3,
                poll_interval=0.05,
            )

    def test_succeeds_with_any_requested_source(
        self, job_service: JobService, data_service: DataService[ResultKey, Any]
    ):
        """Should succeed if ANY requested source has data."""
        backend = IntegrationTestBackend(job_service)
        workflow_a = make_workflow_id('workflow_a')
        job_1 = make_job_number(1)

        # Add data for one of multiple requested sources
        job_service.status_updated(
            JobStatus(
                job_id=make_job_id(job_1, 'source2'),
                workflow_id=workflow_a,
                state=JobState.active,
            )
        )
        data_service[
            ResultKey(
                workflow_id=workflow_a,
                job_id=make_job_id(job_1, 'source2'),
                output_name='output1',
            )
        ] = 'data'

        wait_for_workflow_job_data(
            backend,
            workflow_a,
            ['source1', 'source2', 'source3'],  # source2 has data
            timeout=1.0,
            poll_interval=0.05,
        )

        # Should succeed because source2 has data
        assert job_1 in job_service.job_data
