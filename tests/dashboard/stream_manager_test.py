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
from ess.livedata.dashboard.data_subscriber import Pipe
from ess.livedata.dashboard.stream_manager import StreamManager


class FakePipe(Pipe):
    """Fake implementation of Pipe for testing."""

    def __init__(self, data: Any) -> None:
        self.send_calls: list[Any] = []
        self.data: Any = data

    def send(self, data: Any) -> None:
        self.send_calls.append(data)
        self.data = data


class FakePipeFactory:
    """Fake pipe factory for testing."""

    def __init__(self) -> None:
        self.call_count = 0
        self.created_pipes: list[FakePipe] = []

    def __call__(self, data: Any) -> FakePipe:
        self.call_count += 1
        pipe = FakePipe(data)
        self.created_pipes.append(pipe)
        return pipe


@pytest.fixture
def data_service() -> DataService:
    """Real DataService instance for testing."""
    return DataService()


@pytest.fixture
def fake_pipe_factory() -> FakePipeFactory:
    """Fake pipe factory that creates FakePipe instances."""
    return FakePipeFactory()


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
    assert (
        actual.keys() == expected.keys()
    ), f"Keys differ: {actual.keys()} != {expected.keys()}"
    for key in expected:
        actual_value = actual[key]
        expected_value = expected[key]
        if type(expected_value).__module__.startswith('scipp'):
            assert_identical(actual_value, expected_value)
        else:
            assert actual_value == expected_value


class TestStreamManager:
    """Test cases for StreamManager class."""

    def test_make_stream_creates_pipe_and_registers_subscriber(
        self, data_service, fake_pipe_factory, workflow_id
    ):
        """Test that make_stream creates a pipe and registers a subscriber."""
        manager = StreamManager(
            data_service=data_service, pipe_factory=fake_pipe_factory
        )

        key = make_key(workflow_id)
        keys_by_role = {'primary': [key]}

        pipe = manager.make_stream(keys_by_role)

        assert isinstance(pipe, FakePipe)
        assert fake_pipe_factory.call_count == 1
        assert len(data_service._subscribers) == 1

    def test_single_source_data_flow(
        self, data_service, fake_pipe_factory, sample_data, workflow_id
    ):
        """Test data flow with a single source."""
        manager = StreamManager(
            data_service=data_service, pipe_factory=fake_pipe_factory
        )

        key = make_key(workflow_id)
        keys_by_role = {'primary': [key]}

        pipe = manager.make_stream(keys_by_role)

        # Publish data
        data_service[key] = sample_data

        # Verify data received
        assert len(pipe.send_calls) == 1
        assert_dict_equal_with_scipp(pipe.send_calls[0], {key: sample_data})

    def test_multiple_sources_data_flow(
        self, data_service, fake_pipe_factory, sample_data, workflow_id
    ):
        """Test data flow with multiple sources in single role."""
        manager = StreamManager(
            data_service=data_service, pipe_factory=fake_pipe_factory
        )

        key1 = make_key(workflow_id, "source1")
        key2 = make_key(workflow_id, "source2")
        keys_by_role = {'primary': [key1, key2]}

        pipe = manager.make_stream(keys_by_role)

        # Publish data for both keys
        sample_data2 = sc.DataArray(
            data=sc.array(dims=['y'], values=[4, 5, 6]),
            coords={'y': sc.array(dims=['y'], values=[40, 50, 60])},
        )

        data_service[key1] = sample_data
        data_service[key2] = sample_data2

        # Should receive data for both keys
        assert len(pipe.send_calls) == 2
        # First call has only key1
        assert_dict_equal_with_scipp(pipe.send_calls[0], {key1: sample_data})
        # Second call has both keys
        assert_dict_equal_with_scipp(
            pipe.send_calls[1], {key1: sample_data, key2: sample_data2}
        )

    def test_partial_data_updates(
        self, data_service, fake_pipe_factory, sample_data, workflow_id
    ):
        """Test handling of partial data updates when only some keys have data."""
        manager = StreamManager(
            data_service=data_service, pipe_factory=fake_pipe_factory
        )

        key1 = make_key(workflow_id, "source1")
        key2 = make_key(workflow_id, "source2")
        keys_by_role = {'primary': [key1, key2]}

        pipe = manager.make_stream(keys_by_role)

        # Send data for only one key
        data_service[key1] = sample_data

        # Should receive partial data
        assert len(pipe.send_calls) == 1
        assert key1 in pipe.send_calls[0]
        assert key2 not in pipe.send_calls[0]
        assert_identical(pipe.send_calls[0][key1], sample_data)

    def test_stream_independence(
        self, data_service, fake_pipe_factory, sample_data, workflow_id
    ):
        """Test that multiple streams operate independently."""
        manager = StreamManager(
            data_service=data_service, pipe_factory=fake_pipe_factory
        )

        # Create two independent streams with different keys
        key1 = make_key(workflow_id, "source1")
        key2 = make_key(workflow_id, "source2")

        pipe1 = manager.make_stream({'primary': [key1]})
        pipe2 = manager.make_stream({'primary': [key2]})

        # Send data for key1
        data_service[key1] = sample_data

        # Only pipe1 should receive data
        assert len(pipe1.send_calls) == 1
        assert len(pipe2.send_calls) == 0

        # Send data for key2
        data_service[key2] = sample_data

        # Now pipe2 should also have data, pipe1 unchanged
        assert len(pipe1.send_calls) == 1
        assert len(pipe2.send_calls) == 1

    def test_incremental_updates(
        self, data_service, fake_pipe_factory, sample_data, workflow_id
    ):
        """Test that incremental updates flow through correctly."""
        manager = StreamManager(
            data_service=data_service, pipe_factory=fake_pipe_factory
        )

        key = make_key(workflow_id)
        pipe = manager.make_stream({'primary': [key]})

        # Send initial data
        data_service[key] = sample_data

        # Send updated data
        updated_data = sc.DataArray(
            data=sc.array(dims=['x'], values=[7, 8, 9]),
            coords={'x': sc.array(dims=['x'], values=[70, 80, 90])},
        )
        data_service[key] = updated_data

        # Should receive both updates
        assert len(pipe.send_calls) == 2
        assert_dict_equal_with_scipp(pipe.send_calls[0], {key: sample_data})
        assert_dict_equal_with_scipp(pipe.send_calls[1], {key: updated_data})

    def test_unrelated_key_filtering(
        self, data_service, fake_pipe_factory, sample_data, workflow_id
    ):
        """Test that unrelated keys are filtered out."""
        manager = StreamManager(
            data_service=data_service, pipe_factory=fake_pipe_factory
        )

        target_key = make_key(workflow_id, "target")
        unrelated_key = make_key(workflow_id, "unrelated")

        pipe = manager.make_stream({'primary': [target_key]})

        # Publish data for unrelated key
        data_service[unrelated_key] = sample_data

        # Should not receive any data
        assert len(pipe.send_calls) == 0

        # Publish data for target key
        data_service[target_key] = sample_data

        # Should receive data now
        assert len(pipe.send_calls) == 1
        assert_dict_equal_with_scipp(pipe.send_calls[0], {target_key: sample_data})


class TestStreamManagerMultiRole:
    """Test StreamManager with multiple roles (correlation plots)."""

    def test_multi_role_assembles_grouped_output(
        self, data_service, fake_pipe_factory, sample_data, workflow_id
    ):
        """Multiple roles output grouped dict[str, dict[ResultKey, data]]."""
        manager = StreamManager(
            data_service=data_service, pipe_factory=fake_pipe_factory
        )

        primary_key = make_key(workflow_id, "detector")
        x_axis_key = make_key(workflow_id, "position")
        keys_by_role = {'primary': [primary_key], 'x_axis': [x_axis_key]}

        pipe = manager.make_stream(keys_by_role)

        position_data = sc.DataArray(
            data=sc.array(dims=['t'], values=[1.0, 2.0, 3.0]),
        )

        data_service[primary_key] = sample_data
        data_service[x_axis_key] = position_data

        # Last send should have grouped structure
        last_data = pipe.send_calls[-1]
        assert 'primary' in last_data
        assert 'x_axis' in last_data
        assert_identical(last_data['primary'][primary_key], sample_data)
        assert_identical(last_data['x_axis'][x_axis_key], position_data)


class TestStreamManagerOnFirstData:
    """Test on_first_data callback behavior."""

    def test_on_first_data_fires_when_data_available(
        self, data_service, fake_pipe_factory, sample_data, workflow_id
    ):
        """on_first_data fires when at least one key from each role has data."""
        manager = StreamManager(
            data_service=data_service, pipe_factory=fake_pipe_factory
        )

        callback_invoked = []

        def on_first_data(pipe):
            callback_invoked.append(pipe)

        key = make_key(workflow_id)
        pipe = manager.make_stream({'primary': [key]}, on_first_data=on_first_data)

        data_service[key] = sample_data

        assert len(callback_invoked) == 1
        assert callback_invoked[0] is pipe

    def test_multi_role_waits_for_all_roles(
        self, data_service, fake_pipe_factory, sample_data, workflow_id
    ):
        """Multi-role stream waits for data from each role before firing callback."""
        manager = StreamManager(
            data_service=data_service, pipe_factory=fake_pipe_factory
        )

        callback_invoked = []

        def on_first_data(pipe):
            callback_invoked.append(pipe)

        primary_key = make_key(workflow_id, "detector")
        x_axis_key = make_key(workflow_id, "position")
        keys_by_role = {'primary': [primary_key], 'x_axis': [x_axis_key]}

        manager.make_stream(keys_by_role, on_first_data=on_first_data)

        # Only primary data - should NOT fire
        data_service[primary_key] = sample_data
        assert len(callback_invoked) == 0

        # Add x_axis data - should fire
        position_data = sc.DataArray(data=sc.array(dims=['t'], values=[1.0, 2.0]))
        data_service[x_axis_key] = position_data
        assert len(callback_invoked) == 1


class TestStreamManagerInitialization:
    """Test stream initialization behavior."""

    def test_pipe_initialized_with_empty_dict(
        self, data_service, fake_pipe_factory, workflow_id
    ):
        """Pipe is initialized with empty dict when no data available."""
        manager = StreamManager(
            data_service=data_service, pipe_factory=fake_pipe_factory
        )

        key = make_key(workflow_id)
        pipe = manager.make_stream({'primary': [key]})

        assert isinstance(pipe, FakePipe)
        assert pipe.data == {}

    def test_pipe_receives_data_when_available(
        self, data_service, fake_pipe_factory, sample_data, workflow_id
    ):
        """Pipe receives data when it becomes available after creation."""
        manager = StreamManager(
            data_service=data_service, pipe_factory=fake_pipe_factory
        )

        key = make_key(workflow_id)
        pipe = manager.make_stream({'primary': [key]})

        # Initially empty
        assert pipe.data == {}

        # Publish data later
        data_service[key] = sample_data

        # Should receive data
        assert len(pipe.send_calls) == 1
        assert_dict_equal_with_scipp(pipe.send_calls[0], {key: sample_data})
