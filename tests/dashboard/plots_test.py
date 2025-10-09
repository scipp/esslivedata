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
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        return plots.SlicerPlotter.from_params(params)

    def test_initialization(self, slicer_plotter):
        """Test that SlicerPlotter initializes correctly."""
        assert slicer_plotter._dim_names is None
        assert slicer_plotter._dim_sizes is None
        assert len(slicer_plotter.autoscalers_by_slice_dim) == 3

    def test_plot_slices_3d_data(self, slicer_plotter, test_3d_data, test_data_key):
        """Test that SlicerPlotter correctly slices 3D data."""
        result = slicer_plotter.plot(
            test_3d_data, test_data_key, slice_dim='z', slice_idx=0
        )
        assert isinstance(result, hv.Image)
        # The result should be a 2D image
        assert result is not None

    def test_plot_with_different_slice_index(self, test_3d_data, test_data_key):
        """Test that changing slice index affects the plot."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)
        # Plot with different slice index
        result = plotter.plot(test_3d_data, test_data_key, slice_dim='z', slice_idx=2)
        assert isinstance(result, hv.Image)

    def test_can_slice_along_different_dimensions(
        self, slicer_plotter, test_3d_data, test_data_key
    ):
        """Test that we can slice along different dimensions."""
        # Can slice along z
        result_z = slicer_plotter.plot(
            test_3d_data, test_data_key, slice_dim='z', slice_idx=0
        )
        assert isinstance(result_z, hv.Image)

        # Can slice along y
        result_y = slicer_plotter.plot(
            test_3d_data, test_data_key, slice_dim='y', slice_idx=0
        )
        assert isinstance(result_y, hv.Image)

        # Can slice along x
        result_x = slicer_plotter.plot(
            test_3d_data, test_data_key, slice_dim='x', slice_idx=0
        )
        assert isinstance(result_x, hv.Image)

    def test_slice_label_with_coords(self, slicer_plotter, test_3d_data, test_data_key):
        """Test that slice label includes coordinate values when available."""
        label = slicer_plotter._format_slice_label(test_3d_data, 'z', 2)
        assert 'z=' in label
        assert 's' in label  # unit
        assert 'slice 2/4' in label

    def test_slice_label_without_coords(
        self, slicer_plotter, test_3d_data_no_coords, test_data_key
    ):
        """Test that slice label works without coordinates."""
        label = slicer_plotter._format_slice_label(test_3d_data_no_coords, 'z', 2)
        assert 'z[2/4]' in label

    def test_slice_index_clipping(self, test_3d_data, test_data_key):
        """Test that slice index is clipped to valid range."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)
        # Request index beyond data range
        # Should not raise, should clip to valid range
        result = plotter.plot(test_3d_data, test_data_key, slice_dim='z', slice_idx=100)
        assert result is not None

    def test_multiple_datasets(self, test_3d_data, test_data_key):
        """Test plotting multiple 3D datasets together."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
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

    def test_edge_coordinates(self, slicer_plotter, test_data_key):
        """Test handling of edge coordinates."""
        # Create data with edge coordinates
        x_edges = sc.linspace('x', 0.0, 10.0, num=11, unit='m')
        y_edges = sc.linspace('y', 0.0, 8.0, num=9, unit='m')
        z_edges = sc.linspace('z', 0.0, 5.0, num=6, unit='s')

        data = sc.DataArray(
            sc.ones(dims=['z', 'y', 'x'], shape=[5, 8, 10], unit='counts'),
            coords={'x': x_edges, 'y': y_edges, 'z': z_edges},
        )

        result = slicer_plotter.plot(data, test_data_key, slice_dim='z', slice_idx=0)
        assert isinstance(result, hv.Image)

    def test_inconsistent_dimensions_raises(self, test_data_key):
        """Test that data with inconsistent slice dimensions raises error."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)

        # First data has dims ['z', 'y', 'x']
        data1 = sc.DataArray(
            sc.ones(dims=['z', 'y', 'x'], shape=[5, 8, 10], unit='counts')
        )
        # Try slicing with invalid dimension
        with pytest.raises(ValueError, match="not found in data"):
            plotter.plot(data1, test_data_key, slice_dim='invalid_dim', slice_idx=0)

    def test_call_with_dimension_selector(self, test_3d_data, test_data_key):
        """Test that __call__ accepts slice_dim and index parameters."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)

        # Call with slice_dim and index as HoloViews would with multiple kdims
        result = plotter(
            {test_data_key: test_3d_data},
            slice_dim='z',
            z_index=2,
            y_index=0,
            x_index=0,
        )
        assert result is not None

    def test_call_slices_along_selected_dimension(self, test_3d_data, test_data_key):
        """Test that __call__ uses the selected dimension."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)

        # Call with different slice dimensions
        result_z = plotter(
            {test_data_key: test_3d_data},
            slice_dim='z',
            z_index=1,
            y_index=0,
            x_index=0,
        )
        assert result_z is not None

        result_y = plotter(
            {test_data_key: test_3d_data},
            slice_dim='y',
            z_index=0,
            y_index=3,
            x_index=0,
        )
        assert result_y is not None

    def test_kdims_before_initialization(self):
        """Test that kdims returns None before initialization."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)

        assert plotter.kdims is None

    def test_initialize_from_data_sets_kdims(self, test_3d_data, test_data_key):
        """Test that initialize_from_data enables kdims."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)

        # Initialize with data
        plotter.initialize_from_data({test_data_key: test_3d_data})

        # kdims should now be available (1 selector + 3 sliders = 4 kdims)
        kdims = plotter.kdims
        assert kdims is not None
        assert len(kdims) == 4

        # First kdim is the dimension selector
        assert kdims[0].name == 'slice_dim'
        assert kdims[0].values == ['z', 'y', 'x']
        assert kdims[0].default == 'z'

        # Next 3 kdims are the sliders for each dimension
        # Data has shape [5, 8, 10] for dims ['z', 'y', 'x']
        assert kdims[1].name == 'z_index'
        assert kdims[1].range == (0, 4)  # 5 slices (0-4)

        assert kdims[2].name == 'y_index'
        assert kdims[2].range == (0, 7)  # 8 slices (0-7)

        assert kdims[3].name == 'x_index'
        assert kdims[3].range == (0, 9)  # 10 slices (0-9)

    def test_initialize_from_data_with_empty_dict(self):
        """Test that initialize_from_data handles empty data gracefully."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)

        # Initialize with empty dict
        plotter.initialize_from_data({})

        # kdims should still be None
        assert plotter.kdims is None
