# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for integration test helper functions.

These tests verify the critical filtering and waiting logic in helpers.py.
If these helpers have bugs, integration tests could give false positives
(passing when they should fail).
"""

import uuid

import pytest

from ess.livedata.config.workflow_spec import JobId, JobNumber, WorkflowId
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


class FakeJobService:
    """Fake JobService for testing helpers without real backend."""

    def __init__(self):
        # Map job_number -> workflow_id
        self.job_info: dict[JobNumber, WorkflowId] = {}
        # Map job_number -> {source_name: {output_name: data}}
        self.job_data: dict[JobNumber, dict[str, dict]] = {}
        # Map JobId -> status
        self.job_statuses: dict[JobId, str] = {}


class FakeBackend:
    """Fake backend for testing wait helpers."""

    def __init__(self, job_service: FakeJobService):
        self.job_service = job_service
        self.update_count = 0
        # Callbacks to execute on each update (for simulating data arrival)
        self.on_update_callbacks: list = []

    def update(self):
        """Simulate processing messages."""
        self.update_count += 1
        for callback in self.on_update_callbacks:
            callback()


# Tests for get_workflow_jobs


def test_get_workflow_jobs_returns_only_matching_workflow():
    """get_workflow_jobs should only return jobs for specified workflow."""
    job_service = FakeJobService()
    workflow_a = make_workflow_id('workflow_a')
    workflow_b = make_workflow_id('workflow_b')
    job_service.job_info = {
        make_job_number(1): workflow_a,
        make_job_number(2): workflow_b,
        make_job_number(3): workflow_a,
    }

    result = get_workflow_jobs(job_service, workflow_a)

    assert result == [make_job_number(1), make_job_number(3)]


def test_get_workflow_jobs_returns_empty_for_unknown_workflow():
    """get_workflow_jobs should return empty list for non-existent workflow."""
    job_service = FakeJobService()
    job_service.job_info = {
        make_job_number(1): make_workflow_id('workflow_a'),
    }

    result = get_workflow_jobs(job_service, make_workflow_id('nonexistent'))

    assert result == []


# Tests for get_workflow_job_data


def test_get_workflow_job_data_filters_by_workflow():
    """get_workflow_job_data must only return data from specified workflow."""
    job_service = FakeJobService()
    workflow_a = make_workflow_id('workflow_a')
    workflow_b = make_workflow_id('workflow_b')
    job_1 = make_job_number(1)
    job_2 = make_job_number(2)

    job_service.job_info = {
        job_1: workflow_a,
        job_2: workflow_b,
    }
    job_service.job_data = {
        job_1: {'source1': {'output1': 'data1'}},
        job_2: {'source1': {'output1': 'data2'}},
    }

    result = get_workflow_job_data(job_service, workflow_a)

    # CRITICAL: Must not return data from workflow_b!
    assert job_1 in result
    assert job_2 not in result
    assert result[job_1] == {'source1': {'output1': 'data1'}}


def test_get_workflow_job_data_filters_by_source_names():
    """get_workflow_job_data must filter by source names when provided."""
    job_service = FakeJobService()
    job_service.job_info = {
        make_job_number(1): make_workflow_id('workflow_a'),
    }
    job_service.job_data = {
        make_job_number(1): {
            'source1': {'output1': 'data1'},
            'source2': {'output2': 'data2'},
            'source3': {'output3': 'data3'},
        }
    }

    result = get_workflow_job_data(
        job_service, make_workflow_id('workflow_a'), source_names=['source1', 'source3']
    )

    # CRITICAL: Must only return requested sources!
    assert make_job_number(1) in result
    assert 'source1' in result[make_job_number(1)]
    assert 'source3' in result[make_job_number(1)]
    assert 'source2' not in result[make_job_number(1)]


def test_get_workflow_job_data_excludes_jobs_with_no_matching_sources():
    """Jobs with no matching sources should not appear in result."""
    job_service = FakeJobService()
    job_service.job_info = {
        make_job_number(1): make_workflow_id('workflow_a'),
        make_job_number(2): make_workflow_id('workflow_a'),
    }
    job_service.job_data = {
        make_job_number(1): {'source1': {'output1': 'data1'}},
        make_job_number(2): {'other_source': {'output2': 'data2'}},
    }

    result = get_workflow_job_data(
        job_service, make_workflow_id('workflow_a'), source_names=['source1']
    )

    # Job 2 has no matching sources, should not appear
    assert make_job_number(1) in result
    assert make_job_number(2) not in result


def test_get_workflow_job_data_returns_all_sources_when_none_specified():
    """When source_names is None, should return all sources."""
    job_service = FakeJobService()
    job_service.job_info = {
        make_job_number(1): make_workflow_id('workflow_a'),
    }
    job_service.job_data = {
        make_job_number(1): {
            'source1': {'output1': 'data1'},
            'source2': {'output2': 'data2'},
        }
    }

    result = get_workflow_job_data(
        job_service, make_workflow_id('workflow_a'), source_names=None
    )

    assert make_job_number(1) in result
    assert 'source1' in result[make_job_number(1)]
    assert 'source2' in result[make_job_number(1)]


# Tests for get_workflow_job_statuses


def test_get_workflow_job_statuses_filters_by_workflow():
    """get_workflow_job_statuses must only return statuses from specified workflow."""
    job_service = FakeJobService()
    job_service.job_info = {
        make_job_number(1): make_workflow_id('workflow_a'),
        make_job_number(2): make_workflow_id('workflow_b'),
    }
    job_service.job_statuses = {
        JobId(job_number=make_job_number(1), source_name='source1'): 'running',
        JobId(job_number=make_job_number(2), source_name='source1'): 'completed',
    }

    result = get_workflow_job_statuses(job_service, make_workflow_id('workflow_a'))

    # CRITICAL: Must not return statuses from workflow_b!
    assert JobId(job_number=make_job_number(1), source_name='source1') in result
    assert JobId(job_number=make_job_number(2), source_name='source1') not in result


def test_get_workflow_job_statuses_filters_by_source_names():
    """get_workflow_job_statuses must filter by source names when provided."""
    job_service = FakeJobService()
    job_service.job_info = {
        make_job_number(1): make_workflow_id('workflow_a'),
    }
    job_service.job_statuses = {
        JobId(job_number=make_job_number(1), source_name='source1'): 'running',
        JobId(job_number=make_job_number(1), source_name='source2'): 'completed',
        JobId(job_number=make_job_number(1), source_name='source3'): 'failed',
    }

    result = get_workflow_job_statuses(
        job_service, make_workflow_id('workflow_a'), source_names=['source1', 'source3']
    )

    # CRITICAL: Must only return requested sources!
    assert JobId(job_number=make_job_number(1), source_name='source1') in result
    assert JobId(job_number=make_job_number(1), source_name='source3') in result
    assert JobId(job_number=make_job_number(1), source_name='source2') not in result


def test_get_workflow_job_statuses_returns_all_sources_when_none_specified():
    """When source_names is None, should return all sources."""
    job_service = FakeJobService()
    job_service.job_info = {
        make_job_number(1): make_workflow_id('workflow_a'),
    }
    job_service.job_statuses = {
        JobId(job_number=make_job_number(1), source_name='source1'): 'running',
        JobId(job_number=make_job_number(1), source_name='source2'): 'completed',
    }

    result = get_workflow_job_statuses(
        job_service, make_workflow_id('workflow_a'), source_names=None
    )

    assert JobId(job_number=make_job_number(1), source_name='source1') in result
    assert JobId(job_number=make_job_number(1), source_name='source2') in result


# Tests for wait_for_condition


def test_wait_for_condition_succeeds_when_condition_becomes_true():
    """wait_for_condition should return when condition becomes true."""
    counter = [0]

    def condition():
        counter[0] += 1
        return counter[0] >= 3

    wait_for_condition(condition, timeout=2.0, poll_interval=0.05)

    assert counter[0] >= 3


def test_wait_for_condition_raises_timeout_when_condition_never_true():
    """wait_for_condition should raise WaitTimeout if condition doesn't become true."""

    def always_false():
        return False

    with pytest.raises(WaitTimeout, match="Condition not met within"):
        wait_for_condition(always_false, timeout=0.2, poll_interval=0.05)


# Tests for wait_for_workflow_job_data


def test_wait_for_workflow_job_data_succeeds_when_data_arrives():
    """wait_for_workflow_job_data should return when correct data arrives."""
    job_service = FakeJobService()
    backend = FakeBackend(job_service)

    # Simulate data arriving after a few updates
    def simulate_data_arrival():
        if backend.update_count == 2:
            job_service.job_info[make_job_number(1)] = make_workflow_id('workflow_a')
            job_service.job_data[make_job_number(1)] = {'source1': {'output1': 'data'}}

    backend.on_update_callbacks.append(simulate_data_arrival)

    wait_for_workflow_job_data(
        backend,
        make_workflow_id('workflow_a'),
        ['source1'],
        timeout=2.0,
        poll_interval=0.05,
    )

    # Should have succeeded
    assert backend.update_count >= 2


def test_wait_for_workflow_job_data_must_not_succeed_for_wrong_workflow():
    """CRITICAL: wait_for_workflow_job_data must NOT succeed for wrong workflow data."""
    job_service = FakeJobService()
    backend = FakeBackend(job_service)

    # Add data for different workflow
    job_service.job_info[make_job_number(1)] = make_workflow_id('workflow_b')
    job_service.job_data[make_job_number(1)] = {'source1': {'output1': 'data'}}

    with pytest.raises(WaitTimeout):
        wait_for_workflow_job_data(
            backend,
            make_workflow_id('workflow_a'),  # Looking for workflow_a
            ['source1'],
            timeout=0.3,
            poll_interval=0.05,
        )


def test_wait_for_workflow_job_data_must_not_succeed_for_wrong_source():
    """CRITICAL: wait_for_workflow_job_data must NOT succeed for wrong source."""
    job_service = FakeJobService()
    backend = FakeBackend(job_service)

    # Add data for correct workflow but wrong source
    job_service.job_info[make_job_number(1)] = make_workflow_id('workflow_a')
    job_service.job_data[make_job_number(1)] = {'other_source': {'output1': 'data'}}

    with pytest.raises(WaitTimeout):
        wait_for_workflow_job_data(
            backend,
            make_workflow_id('workflow_a'),
            ['source1'],  # Looking for source1
            timeout=0.3,
            poll_interval=0.05,
        )


def test_wait_for_workflow_job_data_succeeds_with_any_requested_source():
    """wait_for_workflow_job_data should succeed if ANY requested source has data."""
    job_service = FakeJobService()
    backend = FakeBackend(job_service)

    # Add data for one of multiple requested sources
    job_service.job_info[make_job_number(1)] = make_workflow_id('workflow_a')
    job_service.job_data[make_job_number(1)] = {'source2': {'output1': 'data'}}

    wait_for_workflow_job_data(
        backend,
        make_workflow_id('workflow_a'),
        ['source1', 'source2', 'source3'],  # source2 has data
        timeout=1.0,
        poll_interval=0.05,
    )

    # Should succeed because source2 has data
    assert make_job_number(1) in job_service.job_data
