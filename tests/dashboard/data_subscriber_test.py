# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for DataSubscriber with role-based assembly."""

from __future__ import annotations

import uuid

import pytest

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.data_subscriber import DataSubscriber
from ess.livedata.dashboard.extractors import LatestValueExtractor


@pytest.fixture
def workflow_id() -> WorkflowId:
    """Sample workflow ID."""
    return WorkflowId(
        instrument='test_instrument',
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
            on_update=lambda: None,
        )

        assert subscriber.keys == {key1, key2}

    def test_assemble_returns_grouped_data(self, make_result_key):
        """assemble returns role-grouped data even for a single role."""
        key1 = make_result_key('detector1')
        key2 = make_result_key('detector2')
        keys_by_role = {'primary': [key1, key2]}
        extractors = {k: LatestValueExtractor() for k in [key1, key2]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_update=lambda: None,
        )

        data = subscriber.assemble({key1: 'value1', key2: 'value2'})

        assert data is not None
        primary = data['primary']
        assert primary[key1] == 'value1'
        assert primary[key2] == 'value2'

    def test_assemble_multiple_stores_with_different_data(self, make_result_key):
        """Test assemble with different store values."""
        key = make_result_key('detector')
        keys_by_role = {'primary': [key]}
        extractors = {key: LatestValueExtractor()}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_update=lambda: None,
        )

        data1 = subscriber.assemble({key: 'value1'})
        data2 = subscriber.assemble({key: 'value2'})
        data3 = subscriber.assemble({key: 'value3'})

        assert data1['primary'][key] == 'value1'
        assert data2['primary'][key] == 'value2'
        assert data3['primary'][key] == 'value3'

    def test_partial_data_included(self, make_result_key):
        """Test that partial data is included in assembly."""
        key1 = make_result_key('detector1')
        key2 = make_result_key('detector2')
        keys_by_role = {'primary': [key1, key2]}
        extractors = {k: LatestValueExtractor() for k in [key1, key2]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_update=lambda: None,
        )

        # Only provide data for key1
        data = subscriber.assemble({key1: 'value1'})

        primary = data['primary']
        assert key1 in primary
        assert key2 not in primary

    def test_keys_sorted_deterministically(self, workflow_id):
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
            extractors=extractors,
            on_update=lambda: None,
        )

        data = subscriber.assemble({key_a: 'value_a', key_b: 'value_b'})

        result_keys = list(data['primary'].keys())
        # Should be sorted alphabetically by source_name
        assert result_keys[0].job_id.source_name == 'a_detector'
        assert result_keys[1].job_id.source_name == 'b_detector'


class TestDataSubscriberMultiRole:
    """Test DataSubscriber with multiple roles (correlation plots)."""

    def test_multi_role_assembles_grouped_dict(self, make_result_key):
        """Multiple roles output dict[str, dict[ResultKey, data]]."""
        primary_key = make_result_key('detector')
        x_axis_key = make_result_key('position')
        keys_by_role = {'primary': [primary_key], 'x_axis': [x_axis_key]}
        extractors = {k: LatestValueExtractor() for k in [primary_key, x_axis_key]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_update=lambda: None,
        )

        data = subscriber.assemble(
            {primary_key: "detector_data", x_axis_key: "position_data"}
        )

        # Should return grouped dict
        assert data is not None
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
            on_update=lambda: None,
        )

        assert subscriber.keys == {primary_key, x_axis_key, y_axis_key}


class TestDataSubscriberReadyCondition:
    """Test on_data callback behavior with role-based ready condition."""

    def test_single_role_assemble_with_partial_data(self, make_result_key):
        """Single role assemble succeeds when any key has data."""
        key1 = make_result_key('detector1')
        key2 = make_result_key('detector2')
        keys_by_role = {'primary': [key1, key2]}
        extractors = {k: LatestValueExtractor() for k in [key1, key2]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_update=lambda: None,
        )

        # assemble with only one key
        result = subscriber.assemble({key1: 'value1'})

        # Should succeed (at least one key from the one role)
        assert result is not None
        assert 'primary' in result
        assert key1 in result['primary']
        assert key2 not in result['primary']

    def test_multi_role_requires_data_from_each_role(self, make_result_key):
        """Multi-role assemble returns None until all roles have data."""
        primary_key = make_result_key('detector')
        x_axis_key = make_result_key('position')
        keys_by_role = {'primary': [primary_key], 'x_axis': [x_axis_key]}
        extractors = {k: LatestValueExtractor() for k in [primary_key, x_axis_key]}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_update=lambda: None,
        )

        # assemble with only primary - should return None
        result = subscriber.assemble({primary_key: 'detector_data'})
        assert result is None

        # assemble with both roles - should return grouped data
        result = subscriber.assemble(
            {primary_key: "detector_data", x_axis_key: "position_data"}
        )
        assert result is not None
        assert 'primary' in result
        assert 'x_axis' in result

    def test_assemble_returns_none_with_empty_data(self, make_result_key):
        """assemble returns None when there's no data."""
        key = make_result_key('detector')
        keys_by_role = {'primary': [key]}
        extractors = {key: LatestValueExtractor()}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_update=lambda: None,
        )

        # assemble with empty store
        result = subscriber.assemble({})

        # Should return None
        assert result is None
