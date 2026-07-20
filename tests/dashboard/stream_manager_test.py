# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for StreamManager with role-based data streams."""

from __future__ import annotations

import uuid

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
    return WorkflowId(instrument="test", name="wf", version=1)


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

        manager.make_stream(keys_by_role, on_update=lambda: None)

        assert len(data_service._subscribers) == 1

    def test_single_source_data_flow(self, data_service, sample_data, workflow_id):
        """Test data flow with a single source."""
        manager = StreamManager(data_service=data_service)
        updates: list[None] = []

        key = make_key(workflow_id)
        keys_by_role = {'primary': [key]}

        subscriber = manager.make_stream(
            keys_by_role, on_update=lambda: updates.append(None)
        )

        # Publish data
        data_service[key] = sample_data

        # Verify on_update fired and data can be assembled (grouped by role)
        assert len(updates) == 1
        snapshot = data_service.snapshot(subscriber)
        result = subscriber.assemble(snapshot)
        assert_dict_equal_with_scipp(result['primary'], {key: sample_data})

    def test_multiple_sources_data_flow(self, data_service, sample_data, workflow_id):
        """Test data flow with multiple sources in single role."""
        manager = StreamManager(data_service=data_service)
        updates: list[None] = []

        key1 = make_key(workflow_id, "source1")
        key2 = make_key(workflow_id, "source2")
        keys_by_role = {'primary': [key1, key2]}

        subscriber = manager.make_stream(
            keys_by_role, on_update=lambda: updates.append(None)
        )

        # Publish data for both keys
        sample_data2 = sc.DataArray(
            data=sc.array(dims=['y'], values=[4, 5, 6]),
            coords={'y': sc.array(dims=['y'], values=[40, 50, 60])},
        )

        data_service[key1] = sample_data
        # First update - has only key1
        assert len(updates) == 1
        snapshot = data_service.snapshot(subscriber)
        result = subscriber.assemble(snapshot)
        assert_dict_equal_with_scipp(result['primary'], {key1: sample_data})

        data_service[key2] = sample_data2
        # Second update - has both keys
        assert len(updates) == 2
        snapshot = data_service.snapshot(subscriber)
        result = subscriber.assemble(snapshot)
        assert_dict_equal_with_scipp(
            result['primary'],
            {key1: sample_data, key2: sample_data2},
        )

    def test_partial_data_updates(self, data_service, sample_data, workflow_id):
        """Test handling of partial data updates when only some keys have data."""
        manager = StreamManager(data_service=data_service)
        updates: list[None] = []

        key1 = make_key(workflow_id, "source1")
        key2 = make_key(workflow_id, "source2")
        keys_by_role = {'primary': [key1, key2]}

        subscriber = manager.make_stream(
            keys_by_role, on_update=lambda: updates.append(None)
        )

        # Send data for only one key
        data_service[key1] = sample_data

        # Should trigger on_update, assemble returns partial data (grouped by role)
        assert len(updates) == 1
        snapshot = data_service.snapshot(subscriber)
        result = subscriber.assemble(snapshot)
        assert result is not None
        primary = result['primary']
        assert key1 in primary
        assert key2 not in primary
        assert_identical(primary[key1], sample_data)

    def test_stream_independence(self, data_service, sample_data, workflow_id):
        """Test that multiple streams operate independently."""
        manager = StreamManager(data_service=data_service)
        updates1: list[None] = []
        updates2: list[None] = []

        # Create two independent streams with different keys
        key1 = make_key(workflow_id, "source1")
        key2 = make_key(workflow_id, "source2")

        manager.make_stream(
            {'primary': [key1]}, on_update=lambda: updates1.append(None)
        )
        manager.make_stream(
            {'primary': [key2]}, on_update=lambda: updates2.append(None)
        )

        # Send data for key1
        data_service[key1] = sample_data

        # Only stream1 should receive update
        assert len(updates1) == 1
        assert len(updates2) == 0

        # Send data for key2
        data_service[key2] = sample_data

        # Now stream2 should also have update, stream1 unchanged
        assert len(updates1) == 1
        assert len(updates2) == 1

    def test_incremental_updates(self, data_service, sample_data, workflow_id):
        """Test incremental updates trigger on_update and assemble reflects changes."""
        manager = StreamManager(data_service=data_service)
        updates: list[None] = []

        key = make_key(workflow_id)
        subscriber = manager.make_stream(
            {'primary': [key]}, on_update=lambda: updates.append(None)
        )

        # Send initial data
        data_service[key] = sample_data
        assert len(updates) == 1
        snapshot = data_service.snapshot(subscriber)
        result = subscriber.assemble(snapshot)
        assert_dict_equal_with_scipp(result['primary'], {key: sample_data})

        # Send updated data
        updated_data = sc.DataArray(
            data=sc.array(dims=['x'], values=[7, 8, 9]),
            coords={'x': sc.array(dims=['x'], values=[70, 80, 90])},
        )
        data_service[key] = updated_data
        assert len(updates) == 2
        snapshot = data_service.snapshot(subscriber)
        result = subscriber.assemble(snapshot)
        assert_dict_equal_with_scipp(result['primary'], {key: updated_data})

    def test_unrelated_key_filtering(self, data_service, sample_data, workflow_id):
        """Test that unrelated keys are filtered out."""
        manager = StreamManager(data_service=data_service)
        updates: list[None] = []

        target_key = make_key(workflow_id, "target")
        unrelated_key = make_key(workflow_id, "unrelated")

        subscriber = manager.make_stream(
            {'primary': [target_key]}, on_update=lambda: updates.append(None)
        )

        # Publish data for unrelated key
        data_service[unrelated_key] = sample_data

        # Should not trigger on_update (unrelated key)
        assert len(updates) == 0

        # Publish data for target key
        data_service[target_key] = sample_data

        # Should trigger on_update and assemble should return data
        assert len(updates) == 1
        snapshot = data_service.snapshot(subscriber)
        result = subscriber.assemble(snapshot)
        assert_dict_equal_with_scipp(result['primary'], {target_key: sample_data})


class TestStreamManagerMultiRole:
    """Test StreamManager with multiple roles (correlation plots)."""

    def test_multi_role_assembles_grouped_output(
        self, data_service, sample_data, workflow_id
    ):
        """Multiple roles output grouped dict[str, dict[ResultKey, data]]."""
        manager = StreamManager(data_service=data_service)
        updates: list[None] = []

        primary_key = make_key(workflow_id, "detector")
        x_axis_key = make_key(workflow_id, "position")
        keys_by_role = {'primary': [primary_key], 'x_axis': [x_axis_key]}

        subscriber = manager.make_stream(
            keys_by_role, on_update=lambda: updates.append(None)
        )

        position_data = sc.DataArray(
            data=sc.array(dims=['t'], values=[1.0, 2.0, 3.0]),
        )

        data_service[primary_key] = sample_data
        data_service[x_axis_key] = position_data

        # After both updates, assemble should return grouped structure
        snapshot = data_service.snapshot(subscriber)
        result = subscriber.assemble(snapshot)
        assert result is not None
        assert 'primary' in result
        assert 'x_axis' in result
        assert_identical(result['primary'][primary_key], sample_data)
        assert_identical(result['x_axis'][x_axis_key], position_data)

    def test_multi_role_waits_for_all_roles(
        self, data_service, sample_data, workflow_id
    ):
        """Multi-role stream waits for all role data before assemble returns."""
        manager = StreamManager(data_service=data_service)
        updates: list[None] = []

        primary_key = make_key(workflow_id, "detector")
        x_axis_key = make_key(workflow_id, "position")
        keys_by_role = {'primary': [primary_key], 'x_axis': [x_axis_key]}

        subscriber = manager.make_stream(
            keys_by_role, on_update=lambda: updates.append(None)
        )

        # Only primary data - on_update should fire but assemble returns None
        data_service[primary_key] = sample_data
        assert len(updates) == 1
        snapshot = data_service.snapshot(subscriber)
        assert subscriber.assemble(snapshot) is None

        # Add x_axis data - assemble should now return data
        position_data = sc.DataArray(data=sc.array(dims=['t'], values=[1.0, 2.0]))
        data_service[x_axis_key] = position_data
        assert len(updates) == 2
        snapshot = data_service.snapshot(subscriber)
        result = subscriber.assemble(snapshot)
        assert result is not None
        assert 'primary' in result
        assert 'x_axis' in result
