# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for logical view registry."""

import pytest
import scipp as sc

from ess.livedata.config import Instrument
from ess.livedata.handlers.logical_view_registry import (
    LogicalViewConfig,
    LogicalViewRegistry,
)


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


class TestLogicalViewRegistry:
    def test_add_view_config(self):
        registry = LogicalViewRegistry()
        registry.add(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        assert len(registry) == 1

    def test_add_multiple_views(self):
        registry = LogicalViewRegistry()
        registry.add(
            name='view1',
            title='View 1',
            description='First view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        registry.add(
            name='view2',
            title='View 2',
            description='Second view.',
            source_names=['detector1'],
            transform=_sum_transform,
        )
        assert len(registry) == 2

    def test_iterate_over_configs(self):
        registry = LogicalViewRegistry()
        registry.add(
            name='view1',
            title='View 1',
            description='First view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        registry.add(
            name='view2',
            title='View 2',
            description='Second view.',
            source_names=['detector1'],
            transform=_sum_transform,
        )
        names = [config.name for config in registry]
        assert names == ['view1', 'view2']

    def test_register_specs_returns_handles(self):
        instrument = Instrument(name='test', detector_names=['detector1'])
        registry = LogicalViewRegistry()
        registry.add(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        handles = registry.register_specs(instrument)
        assert 'test_view' in handles
        assert handles['test_view'].workflow_id.name == 'test_view'

    def test_register_specs_creates_workflow_specs(self):
        instrument = Instrument(name='test', detector_names=['detector1'])
        registry = LogicalViewRegistry()
        registry.add(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        registry.register_specs(instrument)
        # Check that the spec was registered
        workflow_id = next(iter(instrument.workflow_factory.keys()))
        spec = instrument.workflow_factory[workflow_id]
        assert spec.name == 'test_view'
        assert spec.title == 'Test View'

    def test_attach_factories_requires_register_specs_first(self):
        instrument = Instrument(name='test', detector_names=['detector1'])
        registry = LogicalViewRegistry()
        registry.add(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        with pytest.raises(RuntimeError, match='Call register_specs'):
            registry.attach_factories(instrument)

    def test_full_registration_flow(self):
        """Test the complete two-phase registration pattern."""
        instrument = Instrument(name='test', detector_names=['detector1'])
        registry = LogicalViewRegistry()
        registry.add(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        # Phase 1: register specs (lightweight)
        handles = registry.register_specs(instrument)
        assert 'test_view' in handles
        # Phase 2: attach factories (heavy imports)
        registry.attach_factories(instrument)
        # Verify factory was attached
        workflow_id = handles['test_view'].workflow_id
        assert workflow_id in instrument.workflow_factory._factories

    def test_register_with_optional_params(self):
        instrument = Instrument(name='test', detector_names=['detector1'])
        registry = LogicalViewRegistry()
        registry.add(
            name='test_view',
            title='Test View',
            description='A test view with custom settings.',
            source_names=['detector1'],
            transform=_identity_transform,
            roi_support=False,
            output_ndim=3,
        )
        registry.register_specs(instrument)
        workflow_id = next(iter(instrument.workflow_factory.keys()))
        spec = instrument.workflow_factory[workflow_id]
        # ROI support affects aux_sources
        assert spec.aux_sources is None

    def test_transform_function_preserved(self):
        """Verify that the transform function reference is preserved."""
        instrument = Instrument(name='test', detector_names=['detector1'])
        registry = LogicalViewRegistry()
        registry.add(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        registry.register_specs(instrument)
        registry.attach_factories(instrument)
        # The config should still have the transform reference
        config = next(iter(registry))
        assert config.transform is _identity_transform
