# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import uuid
import warnings

import holoviews as hv
import numpy as np
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
def coordinates_2d():
    """Create test coordinates for 2D data."""
    x = sc.arange('x', 10, dtype='float64')
    y = sc.arange('y', 8, dtype='float64')
    return {'x': x, 'y': y}


@pytest.fixture
def data_key():
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
def zero_data(coordinates_2d):
    """Create test data with all zeros."""
    return sc.DataArray(
        sc.zeros(dims=['y', 'x'], shape=[8, 10], unit='counts'), coords=coordinates_2d
    )


@pytest.fixture
def constant_nonzero_data(coordinates_2d):
    """Create test data with constant non-zero values."""
    return sc.DataArray(
        sc.full(dims=['y', 'x'], shape=[8, 10], value=42.0, unit='counts'),
        coords=coordinates_2d,
    )


@pytest.fixture
def negative_data(coordinates_2d):
    """Create test data with negative values."""
    return sc.DataArray(
        sc.full(dims=['y', 'x'], shape=[8, 10], value=-5.0, unit='counts'),
        coords=coordinates_2d,
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
        self, image_plotter, zero_data, data_key
    ):
        """Test that plotting image data with all zeros does not raise an exception."""
        result = image_plotter.plot(zero_data, data_key)
        assert result is not None

    def test_plot_with_all_zeros_renders_to_bokeh(
        self, image_plotter, zero_data, data_key, color_scale
    ):
        """Test that image data with all zeros can be rendered to a Bokeh plot."""
        hv_element = image_plotter.plot(zero_data, data_key)
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
        self, image_plotter, constant_nonzero_data, data_key
    ):
        """Test image data with constant non-zero values does not raise an exception."""
        result = image_plotter.plot(constant_nonzero_data, data_key)
        assert result is not None

    def test_plot_with_constant_nonzero_values_renders_to_bokeh(
        self, image_plotter, constant_nonzero_data, data_key
    ):
        """Test image with constant non-zero values can be rendered to a Bokeh plot."""
        hv_element = image_plotter.plot(constant_nonzero_data, data_key)
        render_to_bokeh(hv_element)

    def test_plot_with_negative_values_does_not_raise(
        self, image_plotter, negative_data, data_key
    ):
        """Test plotting image data with negative values does not raise an exception."""
        result = image_plotter.plot(negative_data, data_key)
        assert result is not None

    def test_plot_with_negative_values_renders_to_bokeh(
        self, image_plotter, negative_data, data_key, color_scale
    ):
        """Test that image data with negative values can be rendered to a Bokeh plot."""
        hv_element = image_plotter.plot(negative_data, data_key)
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
    def coordinates_3d(self):
        """Create test coordinates for 3D data."""
        x = sc.linspace('x', 0.0, 10.0, num=10, unit='m')
        y = sc.linspace('y', 0.0, 8.0, num=8, unit='m')
        z = sc.linspace('z', 0.0, 5.0, num=5, unit='s')
        return {'x': x, 'y': y, 'z': z}

    @pytest.fixture
    def data_3d(self, coordinates_3d):
        """Create 3D test data."""
        data = sc.DataArray(
            sc.arange('z', 0, 5 * 8 * 10, dtype='float64').fold(
                dim='z', sizes={'z': 5, 'y': 8, 'x': 10}
            ),
            coords=coordinates_3d,
        )
        data.data.unit = 'counts'
        return data

    @pytest.fixture
    def data_3d_no_coords(self):
        """Create 3D test data without coordinates."""
        return sc.DataArray(
            sc.arange('z', 0, 5 * 8 * 10, dtype='float64').fold(
                dim='z', sizes={'z': 5, 'y': 8, 'x': 10}
            ),
            coords={},
        )

    @pytest.fixture
    def slicer_plotter(self):
        """Create SlicerPlotter with linear color scale for easier testing."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        # Use linear scale to avoid NaN masking of zeros in tests
        params.plot_scale.color_scale = PlotScale.linear
        return plots.SlicerPlotter.from_params(params)

    def test_initialization(self, slicer_plotter):
        """Test that SlicerPlotter initializes correctly."""
        # kdims should be None before initialization
        assert slicer_plotter.kdims is None
        # Uses base class autoscalers dict (initialized lazily)
        assert slicer_plotter.autoscalers == {}

    def test_plot_slices_3d_data(self, slicer_plotter, data_3d, data_key):
        """Test that SlicerPlotter correctly slices 3D data."""
        slicer_plotter.initialize_from_data({data_key: data_3d})
        z_value = float(data_3d.coords['z'].values[0])
        result = slicer_plotter.plot(data_3d, data_key, slice_dim='z', z_value=z_value)
        assert isinstance(result, hv.Image)
        # Verify that the correct slice data is returned
        expected_slice = data_3d['z', 0]
        # HoloViews Image.data is a dictionary with keys 'x', 'y', 'values'
        data_dict = result.data
        # Compare with expected slice (HoloViews uses numpy arrays without units)
        np.testing.assert_allclose(
            data_dict['values'],
            expected_slice.values,
        )

    def test_plot_with_different_slice_index(self, data_3d, data_key):
        """Test that changing slice index affects the plot."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.linear
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({data_key: data_3d})
        # Plot with different slice value
        z_value = float(data_3d.coords['z'].values[2])
        result = plotter.plot(data_3d, data_key, slice_dim='z', z_value=z_value)
        assert isinstance(result, hv.Image)

        # Verify that the correct slice data is returned
        expected_slice = data_3d['z', 2]
        data_dict = result.data
        np.testing.assert_allclose(
            data_dict['values'],
            expected_slice.values,
        )

    @pytest.mark.parametrize('slice_dim', ['z', 'y', 'x'])
    def test_can_slice_along_different_dimensions(
        self, slicer_plotter, data_3d, data_key, slice_dim
    ):
        """Test that we can slice along different dimensions."""
        slicer_plotter.initialize_from_data({data_key: data_3d})

        # Get the coordinate value for the slice dimension
        slice_value = float(data_3d.coords[slice_dim].values[0])

        # Perform the slice
        result = slicer_plotter.plot(
            data_3d,
            data_key,
            slice_dim=slice_dim,
            **{f'{slice_dim}_value': slice_value},
        )

        assert isinstance(result, hv.Image)

        # Verify correct slice data
        expected = data_3d[slice_dim, 0]
        data_dict = result.data
        np.testing.assert_allclose(
            data_dict['values'],
            expected.values,
        )

    def test_kdims_with_coords(self, slicer_plotter, data_3d, data_key):
        """Test that kdims use coordinate values when available."""
        slicer_plotter.initialize_from_data({data_key: data_3d})
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

    def test_kdims_without_coords(self, slicer_plotter, data_3d_no_coords, data_key):
        """Test that kdims fall back to indices without coordinates."""
        slicer_plotter.initialize_from_data({data_key: data_3d_no_coords})
        kdims = slicer_plotter.kdims

        assert kdims is not None
        # Check that sliders use integer indices for dimensions without coords
        z_dim = next(d for d in kdims if 'z' in d.name)
        assert z_dim.name == 'z_index'  # Uses index not value
        assert hasattr(z_dim, 'range')  # Has range not discrete values

    def test_plot_with_coord_value(self, data_3d, data_key):
        """Test plotting with coordinate value."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({data_key: data_3d})

        # Get a valid coordinate value from the data
        z_coord = data_3d.coords['z']
        z_value = float(z_coord.values[2])

        result = plotter.plot(data_3d, data_key, slice_dim='z', z_value=z_value)
        assert result is not None

    def test_multiple_datasets_rejected_by_registry(self, data_3d, data_key):
        """Test slicer plotter is rejected for multiple datasets by the registry."""
        from ess.livedata.dashboard.plotting import plotter_registry

        # Create second dataset
        workflow_id2 = WorkflowId(
            instrument='test_instrument',
            namespace='test_namespace',
            name='test_workflow',
            version=1,
        )
        job_id2 = JobId(source_name='test_source2', job_number=uuid.uuid4())
        data_key2 = ResultKey(
            workflow_id=workflow_id2, job_id=job_id2, output_name='test_result'
        )

        # Single dataset should be compatible
        single_data = {data_key: data_3d}
        compatible = plotter_registry.get_compatible_plotters(single_data)
        assert 'slicer' in compatible

        # Multiple datasets should not be compatible
        multiple_data = {data_key: data_3d, data_key2: data_3d}
        compatible = plotter_registry.get_compatible_plotters(multiple_data)
        assert 'slicer' not in compatible

    def test_edge_coordinates(self, slicer_plotter, data_key):
        """Test handling of edge coordinates."""
        # Create data with edge coordinates
        x_edges = sc.linspace('x', 0.0, 10.0, num=11, unit='m')
        y_edges = sc.linspace('y', 0.0, 8.0, num=9, unit='m')
        z_edges = sc.linspace('z', 0.0, 5.0, num=6, unit='s')

        data = sc.DataArray(
            sc.ones(dims=['z', 'y', 'x'], shape=[5, 8, 10], unit='counts'),
            coords={'x': x_edges, 'y': y_edges, 'z': z_edges},
        )

        slicer_plotter.initialize_from_data({data_key: data})

        # For edge coords, slider uses midpoints
        z_midpoint = float(sc.midpoints(z_edges, dim='z').values[0])
        result = slicer_plotter.plot(data, data_key, slice_dim='z', z_value=z_midpoint)
        assert isinstance(result, hv.Image)

    def test_mixed_edge_and_bin_center_coordinates(self, slicer_plotter, data_key):
        """Test handling of mixed edge and bin-center coordinates."""
        # Create data with mixed coordinate types
        x_edges = sc.linspace('x', 0.0, 10.0, num=11, unit='m')  # edges
        y_centers = sc.linspace('y', 0.5, 7.5, num=8, unit='m')  # bin centers
        z_edges = sc.linspace('z', 0.0, 5.0, num=6, unit='s')  # edges

        data = sc.DataArray(
            sc.ones(dims=['z', 'y', 'x'], shape=[5, 8, 10], unit='counts'),
            coords={'x': x_edges, 'y': y_centers, 'z': z_edges},
        )

        slicer_plotter.initialize_from_data({data_key: data})
        kdims = slicer_plotter.kdims

        assert kdims is not None
        # Check that kdims are created correctly for mixed coords
        z_dim = next(d for d in kdims if 'z' in d.name)
        assert z_dim.name == 'z_value'
        assert z_dim.unit == 's'

        y_dim = next(d for d in kdims if 'y' in d.name)
        assert y_dim.name == 'y_value'
        assert y_dim.unit == 'm'

        # Slice along z using midpoint (since z has edges)
        z_midpoint = float(sc.midpoints(z_edges, dim='z').values[2])
        result = slicer_plotter.plot(data, data_key, slice_dim='z', z_value=z_midpoint)
        assert isinstance(result, hv.Image)

        # Verify the slice has correct data
        expected_slice = data['z', 2]
        data_dict = result.data
        np.testing.assert_allclose(
            data_dict['values'],
            expected_slice.values,
        )

    def test_inconsistent_dimensions_raises(self, data_key):
        """Test that data with inconsistent slice dimensions raises error."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)

        # First data has dims ['z', 'y', 'x']
        data1 = sc.DataArray(
            sc.ones(dims=['z', 'y', 'x'], shape=[5, 8, 10], unit='counts')
        )
        # Try slicing with invalid dimension - scipp raises DimensionError
        with pytest.raises(sc.DimensionError, match="Expected dimension"):
            plotter.plot(data1, data_key, slice_dim='invalid_dim', invalid_dim_index=0)

    def test_call_with_dimension_selector(self, data_3d, data_key):
        """Test that __call__ accepts slice_dim and value parameters."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({data_key: data_3d})

        # Call with slice_dim and values as HoloViews would with multiple kdims
        z_value = float(data_3d.coords['z'].values[2])
        y_value = float(data_3d.coords['y'].values[0])
        x_value = float(data_3d.coords['x'].values[0])

        result = plotter(
            {data_key: data_3d},
            slice_dim='z',
            z_value=z_value,
            y_value=y_value,
            x_value=x_value,
        )
        assert result is not None

    def test_call_slices_along_selected_dimension(self, data_3d, data_key):
        """Test that __call__ uses the selected dimension."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({data_key: data_3d})

        # Call with different slice dimensions
        z_value = float(data_3d.coords['z'].values[1])
        y_value = float(data_3d.coords['y'].values[3])
        x_value = float(data_3d.coords['x'].values[0])

        result_z = plotter(
            {data_key: data_3d},
            slice_dim='z',
            z_value=z_value,
            y_value=y_value,
            x_value=x_value,
        )
        assert result_z is not None

        result_y = plotter(
            {data_key: data_3d},
            slice_dim='y',
            z_value=z_value,
            y_value=y_value,
            x_value=x_value,
        )
        assert result_y is not None

    def test_initialize_from_data_sets_kdims(self, data_3d, data_key):
        """Test that initialize_from_data enables kdims."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)

        # Initialize with data
        plotter.initialize_from_data({data_key: data_3d})

        # kdims should now be available (1 selector + 3 sliders = 4 kdims)
        kdims = plotter.kdims
        assert kdims is not None
        assert len(kdims) == 4

        # First kdim is the dimension selector
        assert kdims[0].name == 'slice_dim'
        assert kdims[0].values == ['z', 'y', 'x']
        assert kdims[0].default == 'z'

        # Next 3 kdims are the sliders for each dimension
        # Since data_3d has coords, they use coord values not indices
        assert kdims[1].name == 'z_value'
        assert kdims[1].unit == 's'
        assert hasattr(kdims[1], 'values')

        assert kdims[2].name == 'y_value'
        assert kdims[2].unit == 'm'
        assert hasattr(kdims[2], 'values')

        assert kdims[3].name == 'x_value'
        assert kdims[3].unit == 'm'
        assert hasattr(kdims[3], 'values')

    def test_initialize_from_data_raises_if_no_data_given(self):
        """Test that initialize_from_data rejects empty data."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)

        # Initialize with empty dict
        with pytest.raises(ValueError, match='No data provided'):
            plotter.initialize_from_data({})

    def test_slice_returns_correct_coordinate_values(self, data_3d, data_key):
        """Test that the slice has correct coordinate values."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.linear
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({data_key: data_3d})

        # Slice along z at index 2
        z_value = float(data_3d.coords['z'].values[2])
        result = plotter.plot(data_3d, data_key, slice_dim='z', z_value=z_value)

        # Extract coordinate values from HoloViews Image
        data_dict = result.data

        # Get expected coordinates (directly from sliced data, not midpoints)
        sliced = data_3d['z', 2]
        expected_x = sliced.coords['x'].values
        expected_y = sliced.coords['y'].values

        # Verify coordinate values match
        np.testing.assert_allclose(data_dict['x'], expected_x)
        np.testing.assert_allclose(data_dict['y'], expected_y)

    def test_multiple_slices_have_different_values(self, data_3d, data_key):
        """Test that different slice indices produce different data values."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.linear
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({data_key: data_3d})

        # Get two different slices
        z_value_0 = float(data_3d.coords['z'].values[0])
        result_0 = plotter.plot(data_3d, data_key, slice_dim='z', z_value=z_value_0)

        z_value_2 = float(data_3d.coords['z'].values[2])
        result_2 = plotter.plot(data_3d, data_key, slice_dim='z', z_value=z_value_2)

        # Extract values
        data_dict_0 = result_0.data
        data_dict_2 = result_2.data

        # Values should be different (since data_3d uses arange)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                data_dict_0['values'],
                data_dict_2['values'],
            )

    def test_log_scale_masks_zeros_and_negatives(self, data_3d, data_key):
        """Test that log scale correctly masks zero and negative values as NaN."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        # Explicitly use log scale (which is the default)
        params.plot_scale.color_scale = PlotScale.log
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({data_key: data_3d})

        # Slice at z=0 which contains a zero value at position [0, 0]
        z_value = float(data_3d.coords['z'].values[0])
        result = plotter.plot(data_3d, data_key, slice_dim='z', z_value=z_value)

        data_dict = result.data
        expected_slice = data_3d['z', 0]

        # First value should be NaN (masked zero) in the log scale plot
        assert np.isnan(data_dict['values'][0, 0])
        # Original value was zero
        assert expected_slice.values[0, 0] == 0.0

        # Other positive values should remain unchanged
        np.testing.assert_allclose(
            data_dict['values'][0, 1:],  # Skip the NaN at [0, 0]
            expected_slice.values[0, 1:],
        )


class TestROIDetectorPlotter:
    @pytest.fixture
    def detector_data(self, coordinates_2d):
        """Create 2D detector data."""
        data = sc.arange('y', 0, 80, dtype='float64').fold(
            dim='y', sizes={'y': 8, 'x': 10}
        )
        return sc.DataArray(data, coords=coordinates_2d)

    @pytest.fixture
    def spectrum_data(self):
        """Create 1D spectrum data."""
        tof = sc.linspace('tof', 0.0, 100.0, num=50, unit='us')
        return sc.DataArray(
            sc.arange('tof', 50, dtype='float64', unit='counts'), coords={'tof': tof}
        )

    @pytest.fixture
    def roi_plotter(self):
        """Create ROIDetectorPlotter with linear color scale for testing."""
        params = PlotParams2d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.linear
        return plots.ROIDetectorPlotter.from_params(params)

    def test_initialization(self, roi_plotter):
        """Test that ROIDetectorPlotter initializes correctly."""
        assert roi_plotter is not None
        # Should have autoscaler support from base class
        assert hasattr(roi_plotter, 'autoscalers')

    def test_plot_returns_layout_with_detector_and_spectrum(
        self, roi_plotter, detector_data, spectrum_data, data_key
    ):
        """Test that plot returns a Layout with detector image and spectrum."""
        # Create data dict with detector_data and roi_spectrum
        roi_key = ResultKey(
            workflow_id=data_key.workflow_id,
            job_id=data_key.job_id,
            output_name='roi_spectrum',
        )
        data = {data_key: detector_data, roi_key: spectrum_data}

        result = roi_plotter(data)

        # Should return a Layout (not Overlay or single element)
        assert isinstance(result, hv.Layout)
        # Should have 2 elements: detector image + spectrum
        assert len(result) == 2

    def test_plot_with_missing_roi_spectrum_returns_single_element(
        self, roi_plotter, detector_data, data_key
    ):
        """Test that plot handles missing ROI spectrum gracefully."""
        # Only provide detector data, no ROI spectrum
        data = {data_key: detector_data}

        result = roi_plotter(data)

        # Should return an Overlay (detector image + BoxEdit rectangles)
        assert isinstance(result, hv.Overlay)
        # Should not raise an error

    def test_has_box_edit_stream_attribute(self, roi_plotter):
        """Test that ROIDetectorPlotter has a box_stream attribute for BoxEdit."""
        # After initialization, should have box_stream attribute
        assert hasattr(roi_plotter, 'box_stream')

    def test_detector_plot_includes_boxedit_overlay(
        self, roi_plotter, detector_data, spectrum_data, data_key
    ):
        """Test that detector plot includes BoxEdit rectangles overlay."""
        roi_key = ResultKey(
            workflow_id=data_key.workflow_id,
            job_id=data_key.job_id,
            output_name='roi_spectrum',
        )
        data = {data_key: detector_data, roi_key: spectrum_data}

        result = roi_plotter(data)

        # Layout should have 2 elements
        assert len(result) == 2

        # First element (detector) should be an Overlay (image + rectangles)
        detector_element = result[0]
        assert isinstance(detector_element, hv.Overlay)
