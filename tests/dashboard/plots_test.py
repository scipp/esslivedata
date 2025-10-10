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
        # kdims should be None before initialization
        assert slicer_plotter.kdims is None
        # Uses base class autoscalers dict (initialized lazily)
        assert slicer_plotter.autoscalers == {}

    def test_plot_slices_3d_data(self, slicer_plotter, test_3d_data, test_data_key):
        """Test that SlicerPlotter correctly slices 3D data."""
        slicer_plotter.initialize_from_data({test_data_key: test_3d_data})
        z_value = float(test_3d_data.coords['z'].values[0])
        result = slicer_plotter.plot(
            test_3d_data, test_data_key, slice_dim='z', z_value=z_value
        )
        assert isinstance(result, hv.Image)
        # The result should be a 2D image
        assert result is not None

    def test_plot_with_different_slice_index(self, test_3d_data, test_data_key):
        """Test that changing slice index affects the plot."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({test_data_key: test_3d_data})
        # Plot with different slice value
        z_value = float(test_3d_data.coords['z'].values[2])
        result = plotter.plot(
            test_3d_data, test_data_key, slice_dim='z', z_value=z_value
        )
        assert isinstance(result, hv.Image)

    def test_can_slice_along_different_dimensions(
        self, slicer_plotter, test_3d_data, test_data_key
    ):
        """Test that we can slice along different dimensions."""
        slicer_plotter.initialize_from_data({test_data_key: test_3d_data})

        # Can slice along z
        z_value = float(test_3d_data.coords['z'].values[0])
        result_z = slicer_plotter.plot(
            test_3d_data, test_data_key, slice_dim='z', z_value=z_value
        )
        assert isinstance(result_z, hv.Image)

        # Can slice along y
        y_value = float(test_3d_data.coords['y'].values[0])
        result_y = slicer_plotter.plot(
            test_3d_data, test_data_key, slice_dim='y', y_value=y_value
        )
        assert isinstance(result_y, hv.Image)

        # Can slice along x
        x_value = float(test_3d_data.coords['x'].values[0])
        result_x = slicer_plotter.plot(
            test_3d_data, test_data_key, slice_dim='x', x_value=x_value
        )
        assert isinstance(result_x, hv.Image)

    def test_kdims_with_coords(self, slicer_plotter, test_3d_data, test_data_key):
        """Test that kdims use coordinate values when available."""
        slicer_plotter.initialize_from_data({test_data_key: test_3d_data})
        kdims = slicer_plotter.kdims

        assert kdims is not None
        assert len(kdims) == 4  # selector + 3 sliders

        # Check dimension selector
        assert kdims[0].name == 'slice_dim'

        # Check that sliders use coord values for dimensions with coords
        z_dim = next(d for d in kdims if 'z' in d.name)
        assert z_dim.name == 'z_value'  # Uses value not index
        assert z_dim.unit == 's'
        assert hasattr(z_dim, 'values')  # Has discrete values

    def test_kdims_without_coords(
        self, slicer_plotter, test_3d_data_no_coords, test_data_key
    ):
        """Test that kdims fall back to indices without coordinates."""
        slicer_plotter.initialize_from_data({test_data_key: test_3d_data_no_coords})
        kdims = slicer_plotter.kdims

        assert kdims is not None
        # Check that sliders use integer indices for dimensions without coords
        z_dim = next(d for d in kdims if 'z' in d.name)
        assert z_dim.name == 'z_index'  # Uses index not value
        assert hasattr(z_dim, 'range')  # Has range not discrete values

    def test_plot_with_coord_value(self, test_3d_data, test_data_key):
        """Test plotting with coordinate value."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({test_data_key: test_3d_data})

        # Get a valid coordinate value from the data
        z_coord = test_3d_data.coords['z']
        z_value = float(z_coord.values[2])

        result = plotter.plot(
            test_3d_data, test_data_key, slice_dim='z', z_value=z_value
        )
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

        slicer_plotter.initialize_from_data({test_data_key: data})

        # For edge coords, slider uses midpoints
        z_midpoint = float(sc.midpoints(z_edges, dim='z').values[0])
        result = slicer_plotter.plot(
            data, test_data_key, slice_dim='z', z_value=z_midpoint
        )
        assert isinstance(result, hv.Image)

    def test_inconsistent_dimensions_raises(self, test_data_key):
        """Test that data with inconsistent slice dimensions raises error."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)

        # First data has dims ['z', 'y', 'x']
        data1 = sc.DataArray(
            sc.ones(dims=['z', 'y', 'x'], shape=[5, 8, 10], unit='counts')
        )
        # Try slicing with invalid dimension - scipp raises DimensionError
        with pytest.raises(sc.DimensionError, match="Expected dimension"):
            plotter.plot(
                data1, test_data_key, slice_dim='invalid_dim', invalid_dim_index=0
            )

    def test_call_with_dimension_selector(self, test_3d_data, test_data_key):
        """Test that __call__ accepts slice_dim and value parameters."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({test_data_key: test_3d_data})

        # Call with slice_dim and values as HoloViews would with multiple kdims
        z_value = float(test_3d_data.coords['z'].values[2])
        y_value = float(test_3d_data.coords['y'].values[0])
        x_value = float(test_3d_data.coords['x'].values[0])

        result = plotter(
            {test_data_key: test_3d_data},
            slice_dim='z',
            z_value=z_value,
            y_value=y_value,
            x_value=x_value,
        )
        assert result is not None

    def test_call_slices_along_selected_dimension(self, test_3d_data, test_data_key):
        """Test that __call__ uses the selected dimension."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({test_data_key: test_3d_data})

        # Call with different slice dimensions
        z_value = float(test_3d_data.coords['z'].values[1])
        y_value = float(test_3d_data.coords['y'].values[3])
        x_value = float(test_3d_data.coords['x'].values[0])

        result_z = plotter(
            {test_data_key: test_3d_data},
            slice_dim='z',
            z_value=z_value,
            y_value=y_value,
            x_value=x_value,
        )
        assert result_z is not None

        result_y = plotter(
            {test_data_key: test_3d_data},
            slice_dim='y',
            z_value=z_value,
            y_value=y_value,
            x_value=x_value,
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
        # Since test_3d_data has coords, they use coord values not indices
        assert kdims[1].name == 'z_value'
        assert kdims[1].unit == 's'
        assert hasattr(kdims[1], 'values')

        assert kdims[2].name == 'y_value'
        assert kdims[2].unit == 'm'
        assert hasattr(kdims[2], 'values')

        assert kdims[3].name == 'x_value'
        assert kdims[3].unit == 'm'
        assert hasattr(kdims[3], 'values')

    def test_initialize_from_data_with_empty_dict(self):
        """Test that initialize_from_data handles empty data gracefully."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)

        # Initialize with empty dict
        plotter.initialize_from_data({})

        # kdims should still be None
        assert plotter.kdims is None
