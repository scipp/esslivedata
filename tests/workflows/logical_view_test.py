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

    def test_config_with_reduction_dim_string(self):
        config = LogicalViewConfig(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
            reduction_dim='dim_0',
        )
        assert config.reduction_dim == 'dim_0'

    def test_config_with_reduction_dim_list(self):
        config = LogicalViewConfig(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
            reduction_dim=['x_bin', 'y_bin'],
        )
        assert config.reduction_dim == ['x_bin', 'y_bin']

    def test_config_reduction_dim_defaults_to_none(self):
        config = LogicalViewConfig(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        assert config.reduction_dim is None


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

    def test_add_logical_view_with_reduction_dim_string(self):
        """Verify that reduction_dim as string is preserved."""
        instrument = Instrument(name='test', detector_names=['detector1'])
        instrument.add_logical_view(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
            reduction_dim='dim_0',
        )
        config = instrument._logical_views[0]
        assert config.reduction_dim == 'dim_0'

    def test_add_logical_view_with_reduction_dim_list(self):
        """Verify that reduction_dim as list is preserved."""
        instrument = Instrument(name='test', detector_names=['detector1'])
        instrument.add_logical_view(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
            reduction_dim=['x_bin', 'y_bin'],
        )
        config = instrument._logical_views[0]
        assert config.reduction_dim == ['x_bin', 'y_bin']

    def test_add_logical_view_reduction_dim_defaults_to_none(self):
        """Verify that reduction_dim defaults to None."""
        instrument = Instrument(name='test', detector_names=['detector1'])
        instrument.add_logical_view(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
        )
        config = instrument._logical_views[0]
        assert config.reduction_dim is None


class TestAddLogicalViewSpecOutputs:
    """Tests for workflow spec outputs based on roi_support."""

    def test_roi_support_true_includes_roi_outputs_in_spec(self):
        """Verify spec includes ROI outputs when roi_support=True (default)."""
        instrument = Instrument(name='test', detector_names=['detector1'])
        handle = instrument.add_logical_view(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
            roi_support=True,
        )
        spec = instrument.workflow_factory[handle.workflow_id]
        output_fields = set(spec.outputs.model_fields.keys())

        # ROI outputs should be present
        assert 'roi_rectangle' in output_fields
        assert 'roi_polygon' in output_fields
        assert 'roi_spectra_current' in output_fields
        assert 'roi_spectra_cumulative' in output_fields

    def test_roi_support_false_excludes_roi_outputs_from_spec(self):
        """Verify spec excludes ROI outputs when roi_support=False.

        This is important because the frontend uses the spec's outputs to determine
        what outputs are available for plotting. If ROI outputs are present in the
        spec but not actually produced by the workflow, the frontend will show
        misleading options.
        """
        instrument = Instrument(name='test', detector_names=['detector1'])
        handle = instrument.add_logical_view(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
            roi_support=False,
        )
        spec = instrument.workflow_factory[handle.workflow_id]
        output_fields = set(spec.outputs.model_fields.keys())

        # ROI outputs should NOT be present
        assert 'roi_rectangle' not in output_fields
        assert 'roi_polygon' not in output_fields
        assert 'roi_spectra_current' not in output_fields
        assert 'roi_spectra_cumulative' not in output_fields

        # Basic outputs should still be present
        assert 'cumulative' in output_fields
        assert 'current' in output_fields
        assert 'counts_total' in output_fields

    def test_roi_support_false_with_output_ndim_excludes_roi_outputs(self):
        """Verify custom ndim outputs also exclude ROI when roi_support=False."""
        instrument = Instrument(name='test', detector_names=['detector1'])
        handle = instrument.add_logical_view(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
            roi_support=False,
            output_ndim=1,
        )
        spec = instrument.workflow_factory[handle.workflow_id]
        output_fields = set(spec.outputs.model_fields.keys())

        # ROI outputs should NOT be present
        assert 'roi_rectangle' not in output_fields
        assert 'roi_polygon' not in output_fields

        # Basic outputs should still be present
        assert 'cumulative' in output_fields
        assert 'current' in output_fields

    def test_roi_support_false_sets_aux_sources_to_none(self):
        """Verify aux_sources is None when roi_support=False."""
        instrument = Instrument(name='test', detector_names=['detector1'])
        handle = instrument.add_logical_view(
            name='test_view',
            title='Test View',
            description='A test view.',
            source_names=['detector1'],
            transform=_identity_transform,
            roi_support=False,
        )
        spec = instrument.workflow_factory[handle.workflow_id]

        # aux_sources should be None for no ROI support
        assert spec.aux_sources is None
