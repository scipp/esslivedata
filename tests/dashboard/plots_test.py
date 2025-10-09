# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import uuid
import warnings

import holoviews as hv
import pytest
import scipp as sc
from holoviews.plotting.bokeh import BokehRenderer

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard import plots
from ess.livedata.dashboard.plot_params import (
    PlotParams2d,
    PlotParams3d,
    PlotScale,
    PlotScaleParams2d,
)

hv.extension('bokeh')


@pytest.fixture
def test_coordinates():
    """Create test coordinates for 2D data."""
    x = sc.arange('x', 10, dtype='float64')
    y = sc.arange('y', 8, dtype='float64')
    return {'x': x, 'y': y}


@pytest.fixture
def test_data_key():
    """Create a test ResultKey."""
    workflow_id = WorkflowId(
        instrument='test_instrument',
        namespace='test_namespace',
        name='test_workflow',
        version=1,
    )
    job_id = JobId(source_name='test_source', job_number=uuid.uuid4())
    return ResultKey(workflow_id=workflow_id, job_id=job_id, output_name='test_result')


@pytest.fixture(params=['linear', 'log'])
def color_scale(request):
    """Parametrize color scale for testing."""
    return PlotScale(request.param)


@pytest.fixture
def image_plotter(color_scale):
    """Create an ImagePlotter instance with parametrized color scale."""
    params = PlotParams2d()
    params.plot_scale.color_scale = color_scale
    return plots.ImagePlotter.from_params(params)


@pytest.fixture
def zero_data(test_coordinates):
    """Create test data with all zeros."""
    return sc.DataArray(
        sc.zeros(dims=['y', 'x'], shape=[8, 10], unit='counts'), coords=test_coordinates
    )


@pytest.fixture
def constant_nonzero_data(test_coordinates):
    """Create test data with constant non-zero values."""
    return sc.DataArray(
        sc.full(dims=['y', 'x'], shape=[8, 10], value=42.0, unit='counts'),
        coords=test_coordinates,
    )


@pytest.fixture
def negative_data(test_coordinates):
    """Create test data with negative values."""
    return sc.DataArray(
        sc.full(dims=['y', 'x'], shape=[8, 10], value=-5.0, unit='counts'),
        coords=test_coordinates,
    )


def render_to_bokeh(hv_element):
    """Helper function to render HoloViews element to Bokeh plot."""
    renderer = BokehRenderer.instance()
    bokeh_plot = renderer.get_plot(hv_element)
    assert bokeh_plot is not None
    assert hasattr(bokeh_plot, 'state')
    return bokeh_plot


class TestImagePlotter:
    def test_plot_with_all_zeros_does_not_raise(
        self, image_plotter, zero_data, test_data_key
    ):
        """Test that plotting image data with all zeros does not raise an exception."""
        result = image_plotter.plot(zero_data, test_data_key)
        assert result is not None

    def test_plot_with_all_zeros_renders_to_bokeh(
        self, image_plotter, zero_data, test_data_key, color_scale
    ):
        """Test that image data with all zeros can be rendered to a Bokeh plot."""
        hv_element = image_plotter.plot(zero_data, test_data_key)
        if color_scale == PlotScale.log:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "All-NaN slice encountered", RuntimeWarning
                )
                # Holoviews tries to compare None to int or float. Not sure what we can
                # do about that.
                with pytest.raises(hv.core.options.AbbreviatedException):
                    render_to_bokeh(hv_element)
        else:
            render_to_bokeh(hv_element)

    def test_plot_with_constant_nonzero_values_does_not_raise(
        self, image_plotter, constant_nonzero_data, test_data_key
    ):
        """Test image data with constant non-zero values does not raise an exception."""
        result = image_plotter.plot(constant_nonzero_data, test_data_key)
        assert result is not None

    def test_plot_with_constant_nonzero_values_renders_to_bokeh(
        self, image_plotter, constant_nonzero_data, test_data_key
    ):
        """Test image with constant non-zero values can be rendered to a Bokeh plot."""
        hv_element = image_plotter.plot(constant_nonzero_data, test_data_key)
        render_to_bokeh(hv_element)

    def test_plot_with_negative_values_does_not_raise(
        self, image_plotter, negative_data, test_data_key
    ):
        """Test plotting image data with negative values does not raise an exception."""
        result = image_plotter.plot(negative_data, test_data_key)
        assert result is not None

    def test_plot_with_negative_values_renders_to_bokeh(
        self, image_plotter, negative_data, test_data_key, color_scale
    ):
        """Test that image data with negative values can be rendered to a Bokeh plot."""
        hv_element = image_plotter.plot(negative_data, test_data_key)
        if color_scale == PlotScale.log:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "All-NaN slice encountered", RuntimeWarning
                )
                # Holoviews tries to compare None to int or float. Not sure what we can
                # do about that.
                with pytest.raises(hv.core.options.AbbreviatedException):
                    render_to_bokeh(hv_element)
        else:
            render_to_bokeh(hv_element)


class TestSlicerPlotter:
    @pytest.fixture
    def test_3d_coordinates(self):
        """Create test coordinates for 3D data."""
        x = sc.linspace('x', 0.0, 10.0, num=10, unit='m')
        y = sc.linspace('y', 0.0, 8.0, num=8, unit='m')
        z = sc.linspace('z', 0.0, 5.0, num=5, unit='s')
        return {'x': x, 'y': y, 'z': z}

    @pytest.fixture
    def test_3d_data(self, test_3d_coordinates):
        """Create 3D test data."""
        data = sc.DataArray(
            sc.arange('z', 0, 5 * 8 * 10, dtype='float64').fold(
                dim='z', sizes={'z': 5, 'y': 8, 'x': 10}
            ),
            coords=test_3d_coordinates,
        )
        data.data.unit = 'counts'
        return data

    @pytest.fixture
    def test_3d_data_no_coords(self):
        """Create 3D test data without coordinates."""
        return sc.DataArray(
            sc.arange('z', 0, 5 * 8 * 10, dtype='float64').fold(
                dim='z', sizes={'z': 5, 'y': 8, 'x': 10}
            ),
            coords={},
        )

    @pytest.fixture
    def slicer_plotter(self):
        """Create a SlicerPlotter instance."""
        params = PlotParams3d(
            slice_dimension='z',
            initial_slice_index=0,
            plot_scale=PlotScaleParams2d(),
        )
        return plots.SlicerPlotter.from_params(params)

    def test_initialization(self, slicer_plotter):
        """Test that SlicerPlotter initializes correctly."""
        assert hasattr(slicer_plotter, 'slice_stream')
        assert slicer_plotter.slice_stream is not None
        assert slicer_plotter.slice_stream.slice_index == 0

    def test_plot_slices_3d_data(self, slicer_plotter, test_3d_data, test_data_key):
        """Test that SlicerPlotter correctly slices 3D data."""
        result = slicer_plotter.plot(test_3d_data, test_data_key)
        assert isinstance(result, hv.Image)
        # The result should be a 2D image
        assert result is not None

    def test_plot_with_different_slice_index(self, test_3d_data, test_data_key):
        """Test that changing slice index affects the plot."""
        params = PlotParams3d(
            slice_dimension='z',
            initial_slice_index=2,
            plot_scale=PlotScaleParams2d(),
        )
        plotter = plots.SlicerPlotter.from_params(params)
        result = plotter.plot(test_3d_data, test_data_key)
        assert isinstance(result, hv.Image)

    def test_invalid_slice_dimension_raises(self, test_3d_data, test_data_key):
        """Test that invalid slice dimension raises ValueError."""
        params = PlotParams3d(
            slice_dimension='invalid_dim',
            initial_slice_index=0,
            plot_scale=PlotScaleParams2d(),
        )
        plotter = plots.SlicerPlotter.from_params(params)
        with pytest.raises(ValueError, match="Slice dimension 'invalid_dim' not found"):
            plotter({test_data_key: test_3d_data})

    def test_slice_label_with_coords(self, slicer_plotter, test_3d_data, test_data_key):
        """Test that slice label includes coordinate values when available."""
        label = slicer_plotter._format_slice_label(test_3d_data, 2)
        assert 'z=' in label
        assert 's' in label  # unit
        assert 'slice 2/4' in label

    def test_slice_label_without_coords(
        self, slicer_plotter, test_3d_data_no_coords, test_data_key
    ):
        """Test that slice label works without coordinates."""
        label = slicer_plotter._format_slice_label(test_3d_data_no_coords, 2)
        assert 'z[2/4]' in label

    def test_slice_index_clipping(self, test_3d_data, test_data_key):
        """Test that slice index is clipped to valid range."""
        # Create plotter with index beyond data range
        params = PlotParams3d(
            slice_dimension='z',
            initial_slice_index=100,  # Beyond valid range
            plot_scale=PlotScaleParams2d(),
        )
        plotter = plots.SlicerPlotter.from_params(params)
        # Call __call__ to trigger validation
        result = plotter({test_data_key: test_3d_data})
        # Should not raise, should clip to valid range
        assert result is not None

    def test_multiple_datasets(self, test_3d_data, test_data_key):
        """Test plotting multiple 3D datasets together."""
        params = PlotParams3d(
            slice_dimension='z',
            initial_slice_index=0,
            plot_scale=PlotScaleParams2d(),
        )
        plotter = plots.SlicerPlotter.from_params(params)

        # Create second dataset
        workflow_id2 = WorkflowId(
            instrument='test_instrument',
            namespace='test_namespace',
            name='test_workflow',
            version=1,
        )
        job_id2 = JobId(source_name='test_source2', job_number=uuid.uuid4())
        test_data_key2 = ResultKey(
            workflow_id=workflow_id2, job_id=job_id2, output_name='test_result'
        )

        data_dict = {test_data_key: test_3d_data, test_data_key2: test_3d_data}
        result = plotter(data_dict)
        assert result is not None

    def test_edge_coordinates(self, test_data_key):
        """Test handling of edge coordinates."""
        # Create data with edge coordinates
        x_edges = sc.linspace('x', 0.0, 10.0, num=11, unit='m')
        y_edges = sc.linspace('y', 0.0, 8.0, num=9, unit='m')
        z_edges = sc.linspace('z', 0.0, 5.0, num=6, unit='s')

        data = sc.DataArray(
            sc.ones(dims=['z', 'y', 'x'], shape=[5, 8, 10], unit='counts'),
            coords={'x': x_edges, 'y': y_edges, 'z': z_edges},
        )

        params = PlotParams3d(
            slice_dimension='z',
            initial_slice_index=0,
            plot_scale=PlotScaleParams2d(),
        )
        plotter = plots.SlicerPlotter.from_params(params)
        result = plotter.plot(data, test_data_key)
        assert isinstance(result, hv.Image)
