# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for spec-based plotter selection functionality."""

import holoviews as hv
import pydantic
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import WorkflowOutputsBase, WorkflowSpec
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plotting import plotter_registry
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.stream_manager import StreamManager

hv.extension('bokeh')


class TestOutputMetadataExtraction:
    """Tests for WorkflowSpec.get_output_metadata()."""

    def test_extracts_metadata_from_field(self):
        """Test that metadata is correctly extracted from output field."""

        class TestOutputs(WorkflowOutputsBase):
            i_of_q: sc.DataArray = pydantic.Field(
                title='I(Q)',
                description='Test output',
                json_schema_extra={
                    'dims': ('Q',),
                    'coords': ['Q'],
                },
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test Workflow',
            description='Test',
            outputs=TestOutputs,
            params=None,
        )

        metadata = spec.get_output_metadata('i_of_q')

        assert metadata is not None
        dims, coords = metadata
        assert dims == ('Q',)
        assert coords == ['Q']

    def test_returns_none_for_missing_output(self):
        """Test that None is returned for non-existent output."""

        class TestOutputs(WorkflowOutputsBase):
            i_of_q: sc.DataArray = pydantic.Field(title='I(Q)')

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test Workflow',
            description='Test',
            outputs=TestOutputs,
            params=None,
        )

        metadata = spec.get_output_metadata('nonexistent')
        assert metadata is None

    def test_returns_none_for_field_without_metadata(self):
        """Test that None is returned when field has no metadata."""

        class TestOutputs(WorkflowOutputsBase):
            i_of_q: sc.DataArray = pydantic.Field(title='I(Q)')

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test Workflow',
            description='Test',
            outputs=TestOutputs,
            params=None,
        )

        metadata = spec.get_output_metadata('i_of_q')
        assert metadata is None

    def test_handles_2d_output_metadata(self):
        """Test extraction of 2D output metadata."""

        class TestOutputs(WorkflowOutputsBase):
            i_of_dspacing_two_theta: sc.DataArray = pydantic.Field(
                title='I(d, 2θ)',
                json_schema_extra={
                    'dims': ('dspacing', 'two_theta'),
                    'coords': ['dspacing', 'two_theta'],
                },
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test Workflow',
            description='Test',
            outputs=TestOutputs,
            params=None,
        )

        metadata = spec.get_output_metadata('i_of_dspacing_two_theta')

        assert metadata is not None
        dims, coords = metadata
        assert dims == ('dspacing', 'two_theta')
        assert coords == ['dspacing', 'two_theta']


class TestPlotterRegistryMetadataMatching:
    """Tests for PlotterRegistry.get_compatible_plotters_from_metadata()."""

    def test_matches_1d_data_to_lines_plotter(self):
        """Test that 1D metadata matches the lines plotter."""
        compatible = plotter_registry.get_compatible_plotters_from_metadata(
            dims=('Q',), coords=['Q']
        )

        assert 'lines' in compatible
        assert compatible['lines'].title == 'Lines'

    def test_matches_2d_data_to_image_plotter(self):
        """Test that 2D metadata matches the image plotter."""
        compatible = plotter_registry.get_compatible_plotters_from_metadata(
            dims=('x', 'y'), coords=['x', 'y']
        )

        assert 'image' in compatible
        assert compatible['image'].title == 'Image'

    def test_matches_3d_data_to_slicer_plotter(self):
        """Test that 3D metadata matches the slicer plotter (lenient)."""
        # Note: This is a false positive case - slicer requires evenly spaced coords
        # but we can't check that from metadata alone
        compatible = plotter_registry.get_compatible_plotters_from_metadata(
            dims=('x', 'y', 'z'), coords=['x', 'y', 'z']
        )

        assert 'slicer' in compatible
        assert compatible['slicer'].title == '3D Slicer'

    def test_matches_0d_data_to_timeseries_plotter(self):
        """Test that 0D metadata matches the timeseries plotter."""
        compatible = plotter_registry.get_compatible_plotters_from_metadata(
            dims=(), coords=[]
        )

        assert 'timeseries' in compatible
        assert compatible['timeseries'].title == 'Timeseries'

    def test_excludes_incompatible_plotters_by_dims(self):
        """Test that plotters are excluded based on dimension count."""
        compatible = plotter_registry.get_compatible_plotters_from_metadata(
            dims=('Q',), coords=['Q']
        )

        # 1D data should not match image (requires 2D) or slicer (requires 3D)
        assert 'image' not in compatible
        assert 'slicer' not in compatible


class TestPlottingControllerSpecBasedSelection:
    """Tests for PlottingController.get_available_plotters_from_spec()."""

    @pytest.fixture
    def data_service(self):
        """Create a DataService for testing."""
        return DataService()

    @pytest.fixture
    def job_service(self, data_service):
        """Create a JobService for testing."""
        return JobService(data_service=data_service)

    @pytest.fixture
    def stream_manager(self, data_service):
        """Create a StreamManager for testing."""
        return StreamManager(data_service=data_service, pipe_factory=hv.streams.Pipe)

    @pytest.fixture
    def plotting_controller(self, job_service, stream_manager):
        """Create a PlottingController for testing."""
        return PlottingController(
            job_service=job_service,
            stream_manager=stream_manager,
        )

    def test_returns_compatible_plotters_for_1d_output(self, plotting_controller):
        """Test that 1D output spec returns compatible plotters."""

        class TestOutputs(WorkflowOutputsBase):
            i_of_q: sc.DataArray = pydantic.Field(
                title='I(Q)',
                json_schema_extra={
                    'dims': ('Q',),
                    'coords': ['Q'],
                },
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test Workflow',
            description='Test',
            outputs=TestOutputs,
            params=None,
        )

        plotters, has_metadata = plotting_controller.get_available_plotters_from_spec(
            workflow_spec=spec, output_name='i_of_q'
        )

        assert has_metadata is True
        assert 'lines' in plotters
        assert 'image' not in plotters

    def test_returns_compatible_plotters_for_2d_output(self, plotting_controller):
        """Test that 2D output spec returns compatible plotters."""

        class TestOutputs(WorkflowOutputsBase):
            detector_data: sc.DataArray = pydantic.Field(
                title='Detector Data',
                json_schema_extra={
                    'dims': ('x', 'y'),
                    'coords': ['x', 'y'],
                },
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test Workflow',
            description='Test',
            outputs=TestOutputs,
            params=None,
        )

        plotters, has_metadata = plotting_controller.get_available_plotters_from_spec(
            workflow_spec=spec, output_name='detector_data'
        )

        assert has_metadata is True
        assert 'image' in plotters
        assert 'roi_detector' in plotters
        assert 'lines' not in plotters

    def test_returns_all_plotters_for_missing_metadata(self, plotting_controller):
        """Test that all plotters are returned when output has no metadata."""

        class TestOutputs(WorkflowOutputsBase):
            i_of_q: sc.DataArray = pydantic.Field(title='I(Q)')

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test Workflow',
            description='Test',
            outputs=TestOutputs,
            params=None,
        )

        plotters, has_metadata = plotting_controller.get_available_plotters_from_spec(
            workflow_spec=spec, output_name='i_of_q'
        )

        # Should return all plotters as fallback
        assert has_metadata is False
        assert len(plotters) > 0  # Should have multiple plotters from registry
        # Check that common plotters are present
        assert 'lines' in plotters
        assert 'image' in plotters

    def test_returns_all_plotters_for_nonexistent_output(self, plotting_controller):
        """Test that all plotters are returned for non-existent output."""

        class TestOutputs(WorkflowOutputsBase):
            i_of_q: sc.DataArray = pydantic.Field(
                title='I(Q)',
                json_schema_extra={
                    'dims': ('Q',),
                    'coords': ['Q'],
                },
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test Workflow',
            description='Test',
            outputs=TestOutputs,
            params=None,
        )

        plotters, has_metadata = plotting_controller.get_available_plotters_from_spec(
            workflow_spec=spec, output_name='nonexistent'
        )

        # Should return all plotters as fallback
        assert has_metadata is False
        assert len(plotters) > 0
        assert 'lines' in plotters
        assert 'image' in plotters
