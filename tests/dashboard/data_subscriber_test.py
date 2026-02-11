# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for DataSubscriber with role-based assembly."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.data_subscriber import DataSubscriber
from ess.livedata.dashboard.extractors import LatestValueExtractor


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


class TestDataSubscriberSingleRole:
    """Test DataSubscriber with a single role (standard plots)."""

    def test_keys_returns_all_keys(self, make_result_key):
        """Test that keys property returns all keys from all roles."""
        key1 = make_result_key('detector1')
        key2 = make_result_key('detector2')
        keys_by_role = {'primary': [key1, key2]}
        extractors = {k: LatestValueExtractor() for k in [key1, key2]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_data=lambda d: None,
        )

        assert subscriber.keys == {key1, key2}

    def test_on_data_called_with_assembled_data(self, make_result_key):
        """Test that on_data callback receives assembled data."""
        received_data: list[Any] = []

        key1 = make_result_key('detector1')
        key2 = make_result_key('detector2')
        keys_by_role = {'primary': [key1, key2]}
        extractors = {k: LatestValueExtractor() for k in [key1, key2]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_data=lambda d: received_data.append(d),
        )

        subscriber.trigger({key1: 'value1', key2: 'value2'})

        assert len(received_data) == 1
        assert key1 in received_data[0]
        assert key2 in received_data[0]
        assert received_data[0][key1] == 'value1'
        assert received_data[0][key2] == 'value2'

    def test_single_role_assembles_flat_dict(self, make_result_key):
        """Single role outputs flat dict[ResultKey, data] for standard plotters."""
        received_data: list[Any] = []

        key1 = make_result_key('detector1')
        key2 = make_result_key('detector2')
        keys_by_role = {'primary': [key1, key2]}
        extractors = {k: LatestValueExtractor() for k in [key1, key2]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_data=lambda d: received_data.append(d),
        )

        subscriber.trigger({key1: 'value1', key2: 'value2'})

        # Should receive flat dict (not grouped by role)
        data = received_data[0]
        assert key1 in data
        assert key2 in data
        assert data[key1] == 'value1'
        assert data[key2] == 'value2'

    def test_multiple_triggers_call_on_data_each_time(self, make_result_key):
        """Test that each trigger calls on_data."""
        received_data: list[Any] = []

        key = make_result_key('detector')
        keys_by_role = {'primary': [key]}
        extractors = {key: LatestValueExtractor()}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_data=lambda d: received_data.append(d),
        )

        subscriber.trigger({key: 'value1'})
        subscriber.trigger({key: 'value2'})
        subscriber.trigger({key: 'value3'})

        assert len(received_data) == 3
        assert received_data[0][key] == 'value1'
        assert received_data[1][key] == 'value2'
        assert received_data[2][key] == 'value3'

    def test_partial_data_included(self, make_result_key):
        """Test that partial data is included in assembly."""
        received_data: list[Any] = []

        key1 = make_result_key('detector1')
        key2 = make_result_key('detector2')
        keys_by_role = {'primary': [key1, key2]}
        extractors = {k: LatestValueExtractor() for k in [key1, key2]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_data=lambda d: received_data.append(d),
        )

        # Only provide data for key1
        subscriber.trigger({key1: 'value1'})

        data = received_data[0]
        assert key1 in data
        assert key2 not in data

    def test_keys_sorted_deterministically(self, workflow_id):
        """Test that keys are sorted deterministically in output."""
        received_data: list[Any] = []

        # Create keys with specific ordering
        job_a = JobId(source_name='a_detector', job_number=uuid.uuid4())
        job_b = JobId(source_name='b_detector', job_number=uuid.uuid4())

        key_a = ResultKey(workflow_id=workflow_id, job_id=job_a, output_name='result')
        key_b = ResultKey(workflow_id=workflow_id, job_id=job_b, output_name='result')

        keys_by_role = {'primary': [key_b, key_a]}  # Intentionally unordered
        extractors = {k: LatestValueExtractor() for k in [key_a, key_b]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_data=lambda d: received_data.append(d),
        )

        subscriber.trigger({key_a: 'value_a', key_b: 'value_b'})

        result_keys = list(received_data[0].keys())
        # Should be sorted alphabetically by source_name
        assert result_keys[0].job_id.source_name == 'a_detector'
        assert result_keys[1].job_id.source_name == 'b_detector'


class TestDataSubscriberMultiRole:
    """Test DataSubscriber with multiple roles (correlation plots)."""

    def test_multi_role_assembles_grouped_dict(self, make_result_key):
        """Multiple roles output dict[str, dict[ResultKey, data]]."""
        received_data: list[Any] = []

        primary_key = make_result_key('detector')
        x_axis_key = make_result_key('position')
        keys_by_role = {'primary': [primary_key], 'x_axis': [x_axis_key]}
        extractors = {k: LatestValueExtractor() for k in [primary_key, x_axis_key]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_data=lambda d: received_data.append(d),
        )

        subscriber.trigger({primary_key: 'detector_data', x_axis_key: 'position_data'})

        # Should receive grouped dict
        data = received_data[0]
        assert 'primary' in data
        assert 'x_axis' in data
        assert data['primary'][primary_key] == 'detector_data'
        assert data['x_axis'][x_axis_key] == 'position_data'

    def test_keys_property_includes_all_roles(self, make_result_key):
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
            extractors=extractors,
            on_data=lambda d: None,
        )

        assert subscriber.keys == {primary_key, x_axis_key, y_axis_key}


class TestDataSubscriberReadyCondition:
    """Test on_data callback behavior with role-based ready condition."""

    def test_single_role_fires_when_any_data_available(self, make_result_key):
        """Single role fires on_data when any key has data."""
        callback_invoked: list[Any] = []

        key1 = make_result_key('detector1')
        key2 = make_result_key('detector2')
        keys_by_role = {'primary': [key1, key2]}
        extractors = {k: LatestValueExtractor() for k in [key1, key2]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_data=lambda d: callback_invoked.append(d),
        )

        # Trigger with only one key
        subscriber.trigger({key1: 'value1'})

        # Should fire (at least one key from the one role)
        assert len(callback_invoked) == 1

    def test_multi_role_requires_data_from_each_role(self, make_result_key):
        """Multi-role requires at least one key from EACH role to fire."""
        callback_invoked: list[Any] = []

        primary_key = make_result_key('detector')
        x_axis_key = make_result_key('position')
        keys_by_role = {'primary': [primary_key], 'x_axis': [x_axis_key]}
        extractors = {k: LatestValueExtractor() for k in [primary_key, x_axis_key]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_data=lambda d: callback_invoked.append(d),
        )

        # Trigger with only primary - should NOT fire
        subscriber.trigger({primary_key: 'detector_data'})
        assert len(callback_invoked) == 0

        # Trigger with both roles - should fire
        subscriber.trigger({primary_key: 'detector_data', x_axis_key: 'position_data'})
        assert len(callback_invoked) == 1

    def test_no_callback_with_empty_data(self, make_result_key):
        """on_data does not fire when there's no data."""
        callback_invoked: list[Any] = []

        key = make_result_key('detector')
        keys_by_role = {'primary': [key]}
        extractors = {key: LatestValueExtractor()}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_data=lambda d: callback_invoked.append(d),
        )

        # Trigger with empty store
        subscriber.trigger({})

        # Should not fire
        assert len(callback_invoked) == 0
