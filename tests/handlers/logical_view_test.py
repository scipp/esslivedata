# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for instrument.add_logical_view()."""

import scipp as sc

from ess.livedata.config import Instrument
from ess.livedata.config.instrument import LogicalViewConfig


def _identity_transform(da: sc.DataArray) -> sc.DataArray:
    """Simple identity transform for testing."""
    return da


def _sum_transform(da: sc.DataArray) -> sc.DataArray:
    """Simple sum transform for testing."""
    return da.sum()


class TestLogicalViewConfig:
    def test_config_creation(self):
        config = LogicalViewConfig(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        assert config.name == 'test_view'
        assert config.title == 'Test View'
        assert config.description == 'A test view.'
        assert config.source_names == ['detector1']
        assert config.transform is _identity_transform
        assert config.roi_support is True
        assert config.output_ndim is None

    def test_config_with_optional_params(self):
        config = LogicalViewConfig(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
            roi_support=False,
            output_ndim=3,
        )
        assert config.roi_support is False
        assert config.output_ndim == 3


class TestAddLogicalView:
    def test_add_logical_view_returns_handle(self):
        instrument = Instrument(name='test', detector_names=['detector1'])
        handle = instrument.add_logical_view(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        assert handle.workflow_id.name == 'test_view'

    def test_add_logical_view_registers_spec(self):
        instrument = Instrument(name='test', detector_names=['detector1'])
        instrument.add_logical_view(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        # Check that the spec was registered
        workflow_id = next(iter(instrument.workflow_factory.keys()))
        spec = instrument.workflow_factory[workflow_id]
        assert spec.name == 'test_view'
        assert spec.title == 'Test View'

    def test_add_multiple_logical_views(self):
        instrument = Instrument(name='test', detector_names=['detector1'])
        instrument.add_logical_view(
            name='view1',
            title='View 1',
            description='First view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        instrument.add_logical_view(
            name='view2',
            title='View 2',
            description='Second view.',
            source_names=['detector1'],
            transform=_sum_transform,
        )
        # Both views should be stored
        assert len(instrument._logical_views) == 2
        names = [config.name for config in instrument._logical_views]
        assert names == ['view1', 'view2']

    def test_add_logical_view_with_optional_params(self):
        instrument = Instrument(name='test', detector_names=['detector1'])
        instrument.add_logical_view(
            name='test_view',
            title='Test View',
            description='A test view with custom settings.',
            source_names=['detector1'],
            transform=_identity_transform,
            roi_support=False,
            output_ndim=3,
        )
        config = instrument._logical_views[0]
        assert config.roi_support is False
        assert config.output_ndim == 3

    def test_transform_function_preserved(self):
        """Verify that the transform function reference is preserved."""
        instrument = Instrument(name='test', detector_names=['detector1'])
        instrument.add_logical_view(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        config = instrument._logical_views[0]
        assert config.transform is _identity_transform
