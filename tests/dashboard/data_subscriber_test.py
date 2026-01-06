# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for DataSubscriber with role-based assembly."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.data_subscriber import DataSubscriber, Pipe
from ess.livedata.dashboard.extractors import LatestValueExtractor


class FakePipe(Pipe):
    """Fake implementation of Pipe for testing."""

    def __init__(self, data: Any = None) -> None:
        self.init_data = data
        self.data = data
        self.send_calls: list[Any] = []

    def send(self, data: Any) -> None:
        self.send_calls.append(data)
        self.data = data


@pytest.fixture
def workflow_id() -> WorkflowId:
    """Sample workflow ID."""
    return WorkflowId(
        instrument='test_instrument',
        namespace='test_namespace',
        name='test_workflow',
        version=1,
    )


@pytest.fixture
def make_result_key(workflow_id):
    """Factory for creating ResultKeys."""

    def _make(source_name: str, output_name: str = 'result') -> ResultKey:
        return ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name=source_name, job_number=uuid.uuid4()),
            output_name=output_name,
        )

    return _make


@pytest.fixture
def fake_pipe_factory():
    """Fake pipe factory that tracks created pipes."""
    created_pipes: list[FakePipe] = []

    def factory(data: Any) -> FakePipe:
        pipe = FakePipe(data)
        created_pipes.append(pipe)
        return pipe

    factory.created_pipes = created_pipes
    return factory


class TestDataSubscriberSingleRole:
    """Test DataSubscriber with a single role (standard plots)."""

    def test_keys_returns_all_keys(self, make_result_key, fake_pipe_factory):
        """Test that keys property returns all keys from all roles."""
        key1 = make_result_key('detector1')
        key2 = make_result_key('detector2')
        keys_by_role = {'primary': [key1, key2]}
        extractors = {k: LatestValueExtractor() for k in [key1, key2]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            pipe_factory=fake_pipe_factory,
            extractors=extractors,
        )

        assert subscriber.keys == {key1, key2}

    def test_pipe_created_on_first_trigger(self, make_result_key, fake_pipe_factory):
        """Test that pipe is created on first trigger."""
        key = make_result_key('detector')
        keys_by_role = {'primary': [key]}
        extractors = {key: LatestValueExtractor()}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            pipe_factory=fake_pipe_factory,
            extractors=extractors,
        )

        # Before trigger, accessing pipe raises error
        with pytest.raises(RuntimeError, match="not yet initialized"):
            _ = subscriber.pipe

        # Trigger subscriber
        subscriber.trigger({key: 'value1'})

        # After trigger, pipe is accessible
        pipe = subscriber.pipe
        assert isinstance(pipe, FakePipe)

    def test_single_role_assembles_flat_dict(self, make_result_key, fake_pipe_factory):
        """Single role outputs flat dict[ResultKey, data] for standard plotters."""
        key1 = make_result_key('detector1')
        key2 = make_result_key('detector2')
        keys_by_role = {'primary': [key1, key2]}
        extractors = {k: LatestValueExtractor() for k in [key1, key2]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            pipe_factory=fake_pipe_factory,
            extractors=extractors,
        )

        subscriber.trigger({key1: 'value1', key2: 'value2'})

        # Should receive flat dict (not grouped by role)
        pipe = subscriber.pipe
        assert key1 in pipe.init_data
        assert key2 in pipe.init_data
        assert pipe.init_data[key1] == 'value1'
        assert pipe.init_data[key2] == 'value2'

    def test_multiple_triggers_send_to_pipe(self, make_result_key, fake_pipe_factory):
        """Test that subsequent triggers send data to existing pipe."""
        key = make_result_key('detector')
        keys_by_role = {'primary': [key]}
        extractors = {key: LatestValueExtractor()}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            pipe_factory=fake_pipe_factory,
            extractors=extractors,
        )

        # First trigger creates pipe
        subscriber.trigger({key: 'value1'})
        pipe = subscriber.pipe
        assert len(pipe.send_calls) == 0  # First trigger creates, doesn't send

        # Second trigger sends to pipe
        subscriber.trigger({key: 'value2'})
        assert len(pipe.send_calls) == 1
        assert key in pipe.send_calls[0]

    def test_partial_data_included(self, make_result_key, fake_pipe_factory):
        """Test that partial data is included in assembly."""
        key1 = make_result_key('detector1')
        key2 = make_result_key('detector2')
        keys_by_role = {'primary': [key1, key2]}
        extractors = {k: LatestValueExtractor() for k in [key1, key2]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            pipe_factory=fake_pipe_factory,
            extractors=extractors,
        )

        # Only provide data for key1
        subscriber.trigger({key1: 'value1'})

        pipe = subscriber.pipe
        assert key1 in pipe.init_data
        assert key2 not in pipe.init_data

    def test_keys_sorted_deterministically(self, workflow_id, fake_pipe_factory):
        """Test that keys are sorted deterministically in output."""
        # Create keys with specific ordering
        job_a = JobId(source_name='a_detector', job_number=uuid.uuid4())
        job_b = JobId(source_name='b_detector', job_number=uuid.uuid4())

        key_a = ResultKey(workflow_id=workflow_id, job_id=job_a, output_name='result')
        key_b = ResultKey(workflow_id=workflow_id, job_id=job_b, output_name='result')

        keys_by_role = {'primary': [key_b, key_a]}  # Intentionally unordered
        extractors = {k: LatestValueExtractor() for k in [key_a, key_b]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            pipe_factory=fake_pipe_factory,
            extractors=extractors,
        )

        subscriber.trigger({key_a: 'value_a', key_b: 'value_b'})

        result_keys = list(subscriber.pipe.init_data.keys())
        # Should be sorted alphabetically by source_name
        assert result_keys[0].job_id.source_name == 'a_detector'
        assert result_keys[1].job_id.source_name == 'b_detector'


class TestDataSubscriberMultiRole:
    """Test DataSubscriber with multiple roles (correlation plots)."""

    def test_multi_role_assembles_grouped_dict(
        self, make_result_key, fake_pipe_factory
    ):
        """Multiple roles output dict[str, dict[ResultKey, data]]."""
        primary_key = make_result_key('detector')
        x_axis_key = make_result_key('position')
        keys_by_role = {'primary': [primary_key], 'x_axis': [x_axis_key]}
        extractors = {k: LatestValueExtractor() for k in [primary_key, x_axis_key]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            pipe_factory=fake_pipe_factory,
            extractors=extractors,
        )

        subscriber.trigger({primary_key: 'detector_data', x_axis_key: 'position_data'})

        # Should receive grouped dict
        pipe = subscriber.pipe
        assert 'primary' in pipe.init_data
        assert 'x_axis' in pipe.init_data
        assert pipe.init_data['primary'][primary_key] == 'detector_data'
        assert pipe.init_data['x_axis'][x_axis_key] == 'position_data'

    def test_keys_property_includes_all_roles(self, make_result_key, fake_pipe_factory):
        """Keys property returns union of keys from all roles."""
        primary_key = make_result_key('detector')
        x_axis_key = make_result_key('position')
        y_axis_key = make_result_key('temperature')
        keys_by_role = {
            'primary': [primary_key],
            'x_axis': [x_axis_key],
            'y_axis': [y_axis_key],
        }
        extractors = {
            k: LatestValueExtractor() for k in [primary_key, x_axis_key, y_axis_key]
        }

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            pipe_factory=fake_pipe_factory,
            extractors=extractors,
        )

        assert subscriber.keys == {primary_key, x_axis_key, y_axis_key}


class TestDataSubscriberOnFirstData:
    """Test on_first_data callback behavior with role-based ready condition."""

    def test_single_role_fires_when_any_data_available(
        self, make_result_key, fake_pipe_factory
    ):
        """Single role fires on_first_data when any key has data."""
        callback_invoked = []

        def on_first_data(pipe):
            callback_invoked.append(pipe)

        key1 = make_result_key('detector1')
        key2 = make_result_key('detector2')
        keys_by_role = {'primary': [key1, key2]}
        extractors = {k: LatestValueExtractor() for k in [key1, key2]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            pipe_factory=fake_pipe_factory,
            extractors=extractors,
            on_first_data=on_first_data,
        )

        # Trigger with only one key
        subscriber.trigger({key1: 'value1'})

        # Should fire (at least one key from the one role)
        assert len(callback_invoked) == 1

    def test_multi_role_requires_data_from_each_role(
        self, make_result_key, fake_pipe_factory
    ):
        """Multi-role requires at least one key from EACH role to fire."""
        callback_invoked = []

        def on_first_data(pipe):
            callback_invoked.append(pipe)

        primary_key = make_result_key('detector')
        x_axis_key = make_result_key('position')
        keys_by_role = {'primary': [primary_key], 'x_axis': [x_axis_key]}
        extractors = {k: LatestValueExtractor() for k in [primary_key, x_axis_key]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            pipe_factory=fake_pipe_factory,
            extractors=extractors,
            on_first_data=on_first_data,
        )

        # Trigger with only primary - should NOT fire
        subscriber.trigger({primary_key: 'detector_data'})
        assert len(callback_invoked) == 0

        # Trigger with both roles - should fire
        subscriber.trigger({primary_key: 'detector_data', x_axis_key: 'position_data'})
        assert len(callback_invoked) == 1

    def test_on_first_data_fires_only_once(self, make_result_key, fake_pipe_factory):
        """on_first_data fires only once, even with multiple triggers."""
        callback_invoked = []

        def on_first_data(pipe):
            callback_invoked.append(pipe)

        key = make_result_key('detector')
        keys_by_role = {'primary': [key]}
        extractors = {key: LatestValueExtractor()}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            pipe_factory=fake_pipe_factory,
            extractors=extractors,
            on_first_data=on_first_data,
        )

        # Multiple triggers
        subscriber.trigger({key: 'value1'})
        subscriber.trigger({key: 'value2'})
        subscriber.trigger({key: 'value3'})

        # Should fire only once
        assert len(callback_invoked) == 1

    def test_no_callback_with_empty_data(self, make_result_key, fake_pipe_factory):
        """on_first_data does not fire when there's no data."""
        callback_invoked = []

        def on_first_data(pipe):
            callback_invoked.append(pipe)

        key = make_result_key('detector')
        keys_by_role = {'primary': [key]}
        extractors = {key: LatestValueExtractor()}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            pipe_factory=fake_pipe_factory,
            extractors=extractors,
            on_first_data=on_first_data,
        )

        # Trigger with empty store
        subscriber.trigger({})

        # Should not fire
        assert len(callback_invoked) == 0
