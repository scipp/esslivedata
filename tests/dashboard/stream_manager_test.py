# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for StreamManager with role-based data streams."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.stream_manager import StreamManager


@pytest.fixture
def data_service() -> DataService:
    """Real DataService instance for testing."""
    return DataService()


@pytest.fixture
def sample_data() -> sc.DataArray:
    """Sample data array for testing."""
    return sc.DataArray(
        data=sc.array(dims=['x'], values=[1, 2, 3]),
        coords={'x': sc.array(dims=['x'], values=[10, 20, 30])},
    )


@pytest.fixture
def workflow_id() -> WorkflowId:
    """Sample workflow ID."""
    return WorkflowId(instrument="test", namespace="ns", name="wf", version=1)


def make_key(workflow_id: WorkflowId, source_name: str = "source") -> ResultKey:
    """Helper to create ResultKeys."""
    return ResultKey(
        workflow_id=workflow_id,
        job_id=JobId(source_name=source_name, job_number=uuid.uuid4()),
    )


def assert_dict_equal_with_scipp(actual: dict, expected: dict) -> None:
    """Assert two dictionaries are equal, using scipp.testing for scipp objects."""
    assert actual.keys() == expected.keys(), (
        f"Keys differ: {actual.keys()} != {expected.keys()}"
    )
    for key in expected:
        actual_value = actual[key]
        expected_value = expected[key]
        if type(expected_value).__module__.startswith('scipp'):
            assert_identical(actual_value, expected_value)
        else:
            assert actual_value == expected_value


class TestStreamManager:
    """Test cases for StreamManager class."""

    def test_make_stream_registers_subscriber(self, data_service, workflow_id):
        """Test that make_stream registers a subscriber with data service."""
        manager = StreamManager(data_service=data_service)

        key = make_key(workflow_id)
        keys_by_role = {'primary': [key]}

        manager.make_stream(keys_by_role, on_data=lambda d: None)

        assert len(data_service._subscribers) == 1

    def test_single_source_data_flow(self, data_service, sample_data, workflow_id):
        """Test data flow with a single source."""
        manager = StreamManager(data_service=data_service)
        received_data: list[Any] = []

        key = make_key(workflow_id)
        keys_by_role = {'primary': [key]}

        manager.make_stream(keys_by_role, on_data=lambda d: received_data.append(d))

        # Publish data
        data_service[key] = sample_data

        # Verify data received
        assert len(received_data) == 1
        assert_dict_equal_with_scipp(received_data[0], {key: sample_data})

    def test_multiple_sources_data_flow(self, data_service, sample_data, workflow_id):
        """Test data flow with multiple sources in single role."""
        manager = StreamManager(data_service=data_service)
        received_data: list[Any] = []

        key1 = make_key(workflow_id, "source1")
        key2 = make_key(workflow_id, "source2")
        keys_by_role = {'primary': [key1, key2]}

        manager.make_stream(keys_by_role, on_data=lambda d: received_data.append(d))

        # Publish data for both keys
        sample_data2 = sc.DataArray(
            data=sc.array(dims=['y'], values=[4, 5, 6]),
            coords={'y': sc.array(dims=['y'], values=[40, 50, 60])},
        )

        data_service[key1] = sample_data
        data_service[key2] = sample_data2

        # Should receive data for both keys
        assert len(received_data) == 2
        # First call has only key1
        assert_dict_equal_with_scipp(received_data[0], {key1: sample_data})
        # Second call has both keys
        assert_dict_equal_with_scipp(
            received_data[1], {key1: sample_data, key2: sample_data2}
        )

    def test_partial_data_updates(self, data_service, sample_data, workflow_id):
        """Test handling of partial data updates when only some keys have data."""
        manager = StreamManager(data_service=data_service)
        received_data: list[Any] = []

        key1 = make_key(workflow_id, "source1")
        key2 = make_key(workflow_id, "source2")
        keys_by_role = {'primary': [key1, key2]}

        manager.make_stream(keys_by_role, on_data=lambda d: received_data.append(d))

        # Send data for only one key
        data_service[key1] = sample_data

        # Should receive partial data
        assert len(received_data) == 1
        assert key1 in received_data[0]
        assert key2 not in received_data[0]
        assert_identical(received_data[0][key1], sample_data)

    def test_stream_independence(self, data_service, sample_data, workflow_id):
        """Test that multiple streams operate independently."""
        manager = StreamManager(data_service=data_service)
        received_data1: list[Any] = []
        received_data2: list[Any] = []

        # Create two independent streams with different keys
        key1 = make_key(workflow_id, "source1")
        key2 = make_key(workflow_id, "source2")

        manager.make_stream(
            {'primary': [key1]}, on_data=lambda d: received_data1.append(d)
        )
        manager.make_stream(
            {'primary': [key2]}, on_data=lambda d: received_data2.append(d)
        )

        # Send data for key1
        data_service[key1] = sample_data

        # Only stream1 should receive data
        assert len(received_data1) == 1
        assert len(received_data2) == 0

        # Send data for key2
        data_service[key2] = sample_data

        # Now stream2 should also have data, stream1 unchanged
        assert len(received_data1) == 1
        assert len(received_data2) == 1

    def test_incremental_updates(self, data_service, sample_data, workflow_id):
        """Test that incremental updates flow through correctly."""
        manager = StreamManager(data_service=data_service)
        received_data: list[Any] = []

        key = make_key(workflow_id)
        manager.make_stream(
            {'primary': [key]}, on_data=lambda d: received_data.append(d)
        )

        # Send initial data
        data_service[key] = sample_data

        # Send updated data
        updated_data = sc.DataArray(
            data=sc.array(dims=['x'], values=[7, 8, 9]),
            coords={'x': sc.array(dims=['x'], values=[70, 80, 90])},
        )
        data_service[key] = updated_data

        # Should receive both updates
        assert len(received_data) == 2
        assert_dict_equal_with_scipp(received_data[0], {key: sample_data})
        assert_dict_equal_with_scipp(received_data[1], {key: updated_data})

    def test_unrelated_key_filtering(self, data_service, sample_data, workflow_id):
        """Test that unrelated keys are filtered out."""
        manager = StreamManager(data_service=data_service)
        received_data: list[Any] = []

        target_key = make_key(workflow_id, "target")
        unrelated_key = make_key(workflow_id, "unrelated")

        manager.make_stream(
            {'primary': [target_key]}, on_data=lambda d: received_data.append(d)
        )

        # Publish data for unrelated key
        data_service[unrelated_key] = sample_data

        # Should not receive any data
        assert len(received_data) == 0

        # Publish data for target key
        data_service[target_key] = sample_data

        # Should receive data now
        assert len(received_data) == 1
        assert_dict_equal_with_scipp(received_data[0], {target_key: sample_data})


class TestStreamManagerMultiRole:
    """Test StreamManager with multiple roles (correlation plots)."""

    def test_multi_role_assembles_grouped_output(
        self, data_service, sample_data, workflow_id
    ):
        """Multiple roles output grouped dict[str, dict[ResultKey, data]]."""
        manager = StreamManager(data_service=data_service)
        received_data: list[Any] = []

        primary_key = make_key(workflow_id, "detector")
        x_axis_key = make_key(workflow_id, "position")
        keys_by_role = {'primary': [primary_key], 'x_axis': [x_axis_key]}

        manager.make_stream(keys_by_role, on_data=lambda d: received_data.append(d))

        position_data = sc.DataArray(
            data=sc.array(dims=['t'], values=[1.0, 2.0, 3.0]),
        )

        data_service[primary_key] = sample_data
        data_service[x_axis_key] = position_data

        # Last call should have grouped structure
        last_data = received_data[-1]
        assert 'primary' in last_data
        assert 'x_axis' in last_data
        assert_identical(last_data['primary'][primary_key], sample_data)
        assert_identical(last_data['x_axis'][x_axis_key], position_data)

    def test_multi_role_waits_for_all_roles(
        self, data_service, sample_data, workflow_id
    ):
        """Multi-role stream waits for data from each role before firing callback."""
        manager = StreamManager(data_service=data_service)
        received_data: list[Any] = []

        primary_key = make_key(workflow_id, "detector")
        x_axis_key = make_key(workflow_id, "position")
        keys_by_role = {'primary': [primary_key], 'x_axis': [x_axis_key]}

        manager.make_stream(keys_by_role, on_data=lambda d: received_data.append(d))

        # Only primary data - should NOT fire (multi-role requires all roles)
        data_service[primary_key] = sample_data
        assert len(received_data) == 0

        # Add x_axis data - should fire
        position_data = sc.DataArray(data=sc.array(dims=['t'], values=[1.0, 2.0]))
        data_service[x_axis_key] = position_data
        assert len(received_data) == 1
