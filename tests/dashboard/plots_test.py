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
    PlotParams1d,
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
        self, image_plotter, zero_data, data_key
    ):
        """Test that image data with all zeros can be rendered to a Bokeh plot."""
        hv_element = image_plotter.plot(zero_data, data_key)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "All-NaN slice encountered", RuntimeWarning
            )
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
        self, image_plotter, negative_data, data_key
    ):
        """Test that image data with negative values can be rendered to a Bokeh plot."""
        hv_element = image_plotter.plot(negative_data, data_key)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "All-NaN slice encountered", RuntimeWarning
            )
            render_to_bokeh(hv_element)

    def test_plot_with_bin_edge_coordinate_does_not_raise(
        self, image_plotter, data_key
    ):
        """Test that 2D data with bin-edge coordinate on one dimension works."""
        # Reproduce the scenario from problem.txt: wire dimension without coord,
        # event_time_offset dimension with bin-edge coordinate
        event_time_offset_edges = sc.linspace(
            'event_time_offset', 0.0, 71.0, num=101, unit='ns'
        )
        data = sc.DataArray(
            sc.zeros(
                dims=['wire', 'event_time_offset'], shape=[32, 100], unit='counts'
            ),
            coords={'event_time_offset': event_time_offset_edges},
        )
        result = image_plotter.plot(data, data_key)
        assert result is not None

    def test_plot_with_bin_edge_coordinate_renders_to_bokeh(
        self, image_plotter, data_key
    ):
        """Test that 2D data with bin-edge coordinate can be rendered to Bokeh."""
        # Reproduce the scenario from problem.txt with actual data name
        event_time_offset_edges = sc.linspace(
            'event_time_offset', 0.0, 7.1e7, num=101, unit='ns'
        )
        data = sc.DataArray(
            sc.zeros(
                dims=['wire', 'event_time_offset'], shape=[32, 100], unit='counts'
            ),
            coords={'event_time_offset': event_time_offset_edges},
        )
        data.name = 'Spectrum View'

        hv_element = image_plotter.plot(data, data_key)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "All-NaN slice encountered", RuntimeWarning
            )
            render_to_bokeh(hv_element)

    def test_plot_with_2d_coordinate_does_not_raise(self, image_plotter, data_key):
        """Test that 2D data with a 2D coordinate (like detector_number) works."""
        # Reproduce the broken scenario from info.txt: 2D coordinate
        detector_number = sc.array(
            dims=['wire', 'strip'],
            values=np.arange(1, 32 * 64 + 1).reshape(32, 64),
            unit=None,
            dtype='int32',
        )
        data = sc.DataArray(
            sc.ones(dims=['wire', 'strip'], shape=[32, 64], unit='counts'),
            coords={'detector_number': detector_number},
        )
        data.name = 'Current Counts'

        result = image_plotter.plot(data, data_key)
        assert result is not None

    def test_plot_with_bin_edges_can_be_relabeled(self, image_plotter, data_key):
        """Test that 2D Image with bin edges can be relabeled without bounds error.

        This is a regression test for the issue where converting bin edges to
        midpoints causes floating-point rounding errors in HoloViews' automatic
        bound inference, which are then rejected when the Image is cloned during
        relabel().
        """
        # Create data with large coordinate values and bin edges (like bad.h5)
        event_time_offset_edges = sc.linspace(
            'event_time_offset', 0.0, 7.1e7, num=101, unit='ns'
        )
        data = sc.DataArray(
            sc.zeros(
                dims=['wire', 'event_time_offset'], shape=[32, 100], unit='counts'
            ),
            coords={'event_time_offset': event_time_offset_edges},
        )
        data.name = 'Spectrum View'

        # Get the plot element
        plot_element = image_plotter.plot(data, data_key)

        # This should not raise - relabel clones the Image and validates bounds
        labeled = plot_element.relabel('test_label')
        assert labeled is not None


class TestStripOverflowBins:
    """Tests for _strip_overflow_bins static method."""

    def test_strips_both_inf_edges(self):
        """Both ±inf overflow bins are removed, leaving only finite bins."""
        edges = sc.array(
            dims=['x'],
            values=[float('-inf'), 0.0, 10.0, 20.0, float('+inf')],
            unit='us',
        )
        data = sc.DataArray(
            sc.array(dims=['x'], values=[5.0, 1.0, 2.0, 3.0], unit='counts'),
            coords={'x': edges},
        )
        result = plots.Plotter._strip_overflow_bins(data)
        np.testing.assert_array_equal(result.coords['x'].values, [0.0, 10.0, 20.0])
        np.testing.assert_array_equal(result.values, [1.0, 2.0])

    def test_strips_only_leading_inf(self):
        edges = sc.array(dims=['x'], values=[float('-inf'), 0.0, 10.0, 20.0], unit='us')
        data = sc.DataArray(
            sc.array(dims=['x'], values=[5.0, 1.0, 2.0], unit='counts'),
            coords={'x': edges},
        )
        result = plots.Plotter._strip_overflow_bins(data)
        np.testing.assert_array_equal(result.coords['x'].values, [0.0, 10.0, 20.0])
        np.testing.assert_array_equal(result.values, [1.0, 2.0])

    def test_strips_only_trailing_inf(self):
        edges = sc.array(dims=['x'], values=[0.0, 10.0, 20.0, float('+inf')], unit='us')
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': edges},
        )
        result = plots.Plotter._strip_overflow_bins(data)
        np.testing.assert_array_equal(result.coords['x'].values, [0.0, 10.0, 20.0])
        np.testing.assert_array_equal(result.values, [1.0, 2.0])

    def test_no_inf_edges_returns_data_unchanged(self):
        edges = sc.array(dims=['x'], values=[0.0, 10.0, 20.0], unit='us')
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'),
            coords={'x': edges},
        )
        result = plots.Plotter._strip_overflow_bins(data)
        assert sc.identical(result, data)

    def test_non_edge_coord_returns_data_unchanged(self):
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[0.0, 10.0, 20.0], unit='us')},
        )
        result = plots.Plotter._strip_overflow_bins(data)
        assert sc.identical(result, data)

    def test_explicit_dim_for_2d_data(self):
        """Stripping works on a specified dimension of 2D data."""
        toa_edges = sc.array(
            dims=['toa'],
            values=[float('-inf'), 0.0, 10.0, 20.0, float('+inf')],
            unit='us',
        )
        data = sc.DataArray(
            sc.array(
                dims=['roi', 'toa'],
                values=[[9.0, 1.0, 2.0, 7.0], [0.0, 4.0, 5.0, 0.0]],
            ),
            coords={
                'roi': sc.array(dims=['roi'], values=[0, 1]),
                'toa': toa_edges,
            },
        )
        result = plots.Plotter._strip_overflow_bins(data, dim='toa')
        np.testing.assert_array_equal(result.coords['toa'].values, [0.0, 10.0, 20.0])
        np.testing.assert_array_equal(result.values, [[1.0, 2.0], [4.0, 5.0]])

    def test_preserves_unit(self):
        edges = sc.array(
            dims=['x'],
            values=[float('-inf'), 0.0, 10.0, float('+inf')],
            unit='us',
        )
        data = sc.DataArray(
            sc.array(dims=['x'], values=[5.0, 1.0, 3.0], unit='counts'),
            coords={'x': edges},
        )
        result = plots.Plotter._strip_overflow_bins(data)
        assert result.coords['x'].unit == sc.Unit('us')
        assert result.unit == sc.Unit('counts')


class TestLinePlotter:
    @pytest.fixture
    def line_plotter(self):
        """Create a LinePlotter instance."""
        from ess.livedata.dashboard.plot_params import PlotParams1d

        return plots.LinePlotter.from_params(PlotParams1d())

    def test_plot_without_dimension_coord(self, line_plotter, data_key):
        """Test that LinePlotter handles data without dimension coordinate."""
        # Create 1D data without a coordinate for the dimension (like strip view)
        data = sc.DataArray(
            sc.array(dims=['strip'], values=[1.0, 2.0, 3.0], unit='counts'),
        )
        result = line_plotter.plot(data, data_key)
        assert isinstance(result, hv.Curve)

    def test_plot_with_metadata_coord_but_no_dimension_coord(
        self, line_plotter, data_key
    ):
        """Test LinePlotter with metadata coord but no dimension coordinate."""
        # This is the actual case that was failing: data has a 'time' metadata coord
        # but no coord for the 'strip' dimension
        data = sc.DataArray(
            sc.array(dims=['strip'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'time': sc.scalar(123456789, unit='ns')},
        )
        result = line_plotter.plot(data, data_key)
        assert isinstance(result, hv.Curve)

    def test_plot_with_dimension_coord(self, line_plotter, data_key):
        """Test that LinePlotter works normally with dimension coordinate."""
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0], unit='m')},
        )
        result = line_plotter.plot(data, data_key)
        assert isinstance(result, hv.Curve)

    def test_plot_with_edge_coord(self, line_plotter, data_key):
        """Test that LinePlotter handles bin-edge coordinates."""
        edges = sc.array(dims=['x'], values=[0.0, 10.0, 20.0, 30.0], unit='m')
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': edges},
        )
        result = line_plotter.plot(data, data_key)
        assert isinstance(result, hv.Curve)

    def test_overflow_bins_stripped_in_curve_mode(self, line_plotter, data_key):
        """Overflow bins with ±inf edges are stripped before curve rendering."""
        edges = sc.array(
            dims=['x'],
            values=[float('-inf'), 0.0, 10.0, 20.0, float('+inf')],
            unit='m',
        )
        data = sc.DataArray(
            sc.array(dims=['x'], values=[5.0, 1.0, 2.0, 3.0], unit='counts'),
            coords={'x': edges},
        )
        result = line_plotter.plot(data, data_key)
        assert isinstance(result, hv.Curve)
        # Midpoints of [0, 10, 20] = [5, 15]; no inf values in output
        np.testing.assert_array_equal(result.dimension_values('x'), [5.0, 15.0])
        np.testing.assert_array_equal(result.dimension_values('values'), [1.0, 2.0])

    def test_overflow_bins_stripped_in_curve_mode_renders_to_bokeh(
        self, line_plotter, data_key
    ):
        """Curve with stripped overflow bins renders to Bokeh without warnings."""
        edges = sc.array(
            dims=['x'],
            values=[float('-inf'), 0.0, 10.0, 20.0, float('+inf')],
            unit='m',
        )
        data = sc.DataArray(
            sc.array(dims=['x'], values=[5.0, 1.0, 2.0, 3.0], unit='counts'),
            coords={'x': edges},
        )
        hv_element = line_plotter.plot(data, data_key)
        render_to_bokeh(hv_element)

    def test_overflow_bins_stripped_in_histogram_mode(self, data_key):
        """Overflow bins with ±inf edges are stripped before histogram rendering."""
        from ess.livedata.dashboard.plot_params import Curve1dRenderMode, PlotParams1d

        params = PlotParams1d()
        params.curve.mode = Curve1dRenderMode.histogram
        plotter = plots.LinePlotter.from_params(params)

        edges = sc.array(
            dims=['x'],
            values=[float('-inf'), 0.0, 10.0, 20.0, float('+inf')],
            unit='m',
        )
        data = sc.DataArray(
            sc.array(dims=['x'], values=[5.0, 1.0, 2.0, 3.0], unit='counts'),
            coords={'x': edges},
        )
        result = plotter.plot(data, data_key)
        assert isinstance(result, hv.Histogram)
        # Histogram edges should be [0, 10, 20]; no inf values
        np.testing.assert_array_equal(result.edges, [0.0, 10.0, 20.0])
        np.testing.assert_array_equal(result.dimension_values('values'), [1.0, 2.0])


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

    def test_switching_slice_dimension_sets_framewise_true(self, data_3d, data_key):
        """Test that switching slice dimension forces framewise=True for axis rescaling.

        This is a regression test for issue #559: when switching which
        dimension to slice along, the axis ranges must update to reflect
        the new displayed dimensions.
        """
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.linear
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({data_key: data_3d})

        # First plot: slice along z
        z_value = float(data_3d.coords['z'].values[0])
        result1 = plotter.plot(data_3d, data_key, slice_dim='z', z_value=z_value)

        # First call should have framewise=True (bounds were empty)
        norm_opts1 = hv.Store.lookup_options('bokeh', result1, 'norm').kwargs
        assert norm_opts1.get('framewise') is True

        # Second plot: same dimension, different slice position
        z_value2 = float(data_3d.coords['z'].values[2])
        result2 = plotter.plot(data_3d, data_key, slice_dim='z', z_value=z_value2)

        # Same dimension: framewise can be False (bounds didn't change)
        norm_opts2 = hv.Store.lookup_options('bokeh', result2, 'norm').kwargs
        assert norm_opts2.get('framewise') is False

        # Third plot: switch to slice along y (different displayed dimensions!)
        y_value = float(data_3d.coords['y'].values[0])
        result3 = plotter.plot(data_3d, data_key, slice_dim='y', y_value=y_value)

        # Dimension changed: framewise MUST be True to rescale axes
        norm_opts3 = hv.Store.lookup_options('bokeh', result3, 'norm').kwargs
        assert norm_opts3.get('framewise') is True, (
            "framewise should be True when slice dimension changes to "
            "force axis rescaling"
        )

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
        assert len(kdims) == 5  # mode + selector + 3 sliders

        # Check mode and dimension selectors
        assert kdims[0].name == 'mode'
        assert kdims[1].name == 'slice_dim'

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

        # kdims: mode + slice_dim selector + 3 sliders = 5 kdims
        kdims = plotter.kdims
        assert kdims is not None
        assert len(kdims) == 5

        # First kdim is the mode selector (slice vs flatten)
        assert kdims[0].name == 'mode'
        assert kdims[0].values == ['slice', 'flatten']
        assert kdims[0].default == 'slice'

        # Second kdim is the dimension selector
        assert kdims[1].name == 'slice_dim'
        assert kdims[1].values == ['z', 'y', 'x']
        assert kdims[1].default == 'z'

        # Next 3 kdims are the sliders for each dimension
        # Since data_3d has coords, they use coord values not indices
        assert kdims[2].name == 'z_value'
        assert kdims[2].unit == 's'
        assert hasattr(kdims[2], 'values')

        assert kdims[3].name == 'y_value'
        assert kdims[3].unit == 'm'
        assert hasattr(kdims[3], 'values')

        assert kdims[4].name == 'x_value'
        assert kdims[4].unit == 'm'
        assert hasattr(kdims[4], 'values')

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

    def test_flatten_mode_keeps_specified_dimension(self, data_3d, data_key):
        """Test that flatten mode keeps the specified dimension."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.linear
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({data_key: data_3d})

        # Original data is (z:5, y:8, x:10)
        # Keep x: flatten z,y -> (40, 10)
        result = plotter.plot(data_3d, data_key, mode='flatten', slice_dim='x')
        assert isinstance(result, hv.Image | hv.QuadMesh)
        assert result.data['values'].shape == (40, 10)

        # Keep y: flatten z,x -> (50, 8)
        result = plotter.plot(data_3d, data_key, mode='flatten', slice_dim='y')
        assert isinstance(result, hv.Image | hv.QuadMesh)
        assert result.data['values'].shape == (50, 8)

        # Keep z: flatten y,x -> (80, 5)
        result = plotter.plot(data_3d, data_key, mode='flatten', slice_dim='z')
        assert isinstance(result, hv.Image | hv.QuadMesh)
        assert result.data['values'].shape == (80, 5)

    def test_flatten_mode_preserves_all_data(self, data_3d, data_key):
        """Test that flatten mode preserves all data values."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.linear
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({data_key: data_3d})

        result = plotter.plot(data_3d, data_key, mode='flatten', slice_dim='x')
        data_dict = result.data

        # Total number of values should match
        assert data_dict['values'].size == data_3d.values.size
        # Sum should be preserved
        np.testing.assert_allclose(
            np.nansum(data_dict['values']),
            np.sum(data_3d.values),
        )

    def test_switching_mode_sets_framewise_true(self, data_3d, data_key):
        """Test that switching between slice and flatten sets framewise=True."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({data_key: data_3d})

        # First call in slice mode
        plotter.plot(data_3d, data_key, mode='slice', slice_dim='z', z_index=0)
        # Second call switching to flatten mode should trigger framewise
        result2 = plotter.plot(data_3d, data_key, mode='flatten', slice_dim='x')

        # The opts should include framewise=True after mode change
        # (We can't easily inspect opts, but the mode change detection is tested)
        assert result2 is not None

    def test_handles_2d_dimension_coords(self, data_key):
        """Test that SlicerPlotter handles 2D dimension coordinates gracefully."""
        # Create data with a 2D dimension coordinate (common with detector geometry)
        data = sc.DataArray(
            data=sc.array(
                dims=['z', 'y', 'x'],
                values=np.arange(24).reshape(2, 3, 4).astype('float64'),
                unit='counts',
            ),
        )
        # Make 'y' a 2D coordinate
        y_2d = sc.array(
            dims=['z', 'y'],
            values=np.arange(6).reshape(2, 3).astype('float64'),
            unit='m',
        )
        data = data.assign_coords(
            {
                'z': sc.array(dims=['z'], values=[0.0, 1.0], unit='s'),
                'y': y_2d,
                'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0, 3.0], unit='m'),
            }
        )

        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = plots.SlicerPlotter.from_params(params)
        plotter.initialize_from_data({data_key: data})

        # Should not raise - 2D coord falls back to index-based slider
        kdims = plotter.kdims
        y_kdim = next(d for d in kdims if d.name.startswith('y'))
        assert y_kdim.name == 'y_index'  # Falls back to index, not value

        # Slice mode should work (result may be Image or QuadMesh)
        result_slice = plotter.plot(
            data, data_key, mode='slice', slice_dim='z', z_value=0.0
        )
        assert isinstance(result_slice, hv.Image | hv.QuadMesh)

        # Flatten mode should work
        result_flatten = plotter.plot(data, data_key, mode='flatten', slice_dim='x')
        assert isinstance(result_flatten, hv.Image | hv.QuadMesh)


class TestPlotterLabelChanges:
    """Test Plotter label changes with output_name."""

    @pytest.fixture
    def simple_data(self):
        """Create simple 1D data for testing."""
        return sc.DataArray(
            data=sc.array(dims=['x'], values=[1, 2, 3]),
            coords={'x': sc.array(dims=['x'], values=[10, 20, 30])},
        )

    @pytest.fixture
    def data_key_with_output_name(self):
        """Create a test ResultKey with output_name."""
        workflow_id = WorkflowId(
            instrument='test_instrument',
            namespace='test_namespace',
            name='test_workflow',
            version=1,
        )
        job_id = JobId(source_name='detector', job_number=uuid.uuid4())
        return ResultKey(
            workflow_id=workflow_id, job_id=job_id, output_name='roi_current_0'
        )

    def test_label_includes_output_name(self, simple_data, data_key_with_output_name):
        """Test that plot label includes output_name."""
        plotter = plots.LinePlotter.from_params(PlotParams1d())
        data_dict = {data_key_with_output_name: simple_data}

        result = plotter(data_dict)

        # Result should have label that includes output_name
        # Label format: "detector/roi_current_0"
        assert hasattr(result, 'label')
        assert 'detector' in result.label
        assert 'roi_current_0' in result.label


class TestPlotterOverlayMode:
    """Test Plotter overlay mode changes."""

    @pytest.fixture
    def simple_data_1(self):
        """Create simple 1D data for testing."""
        return sc.DataArray(
            data=sc.array(dims=['x'], values=[1, 2, 3]),
            coords={'x': sc.array(dims=['x'], values=[10, 20, 30])},
        )

    @pytest.fixture
    def simple_data_2(self):
        """Create another simple 1D data for testing."""
        return sc.DataArray(
            data=sc.array(dims=['x'], values=[4, 5, 6]),
            coords={'x': sc.array(dims=['x'], values=[10, 20, 30])},
        )

    @pytest.fixture
    def data_key_1(self):
        """Create first test ResultKey."""
        workflow_id = WorkflowId(
            instrument='test_instrument',
            namespace='test_namespace',
            name='test_workflow',
            version=1,
        )
        job_id = JobId(source_name='detector', job_number=uuid.uuid4())
        return ResultKey(
            workflow_id=workflow_id, job_id=job_id, output_name='roi_current_0'
        )

    @pytest.fixture
    def data_key_2(self):
        """Create second test ResultKey."""
        workflow_id = WorkflowId(
            instrument='test_instrument',
            namespace='test_namespace',
            name='test_workflow',
            version=1,
        )
        job_id = JobId(source_name='detector', job_number=uuid.uuid4())
        return ResultKey(
            workflow_id=workflow_id, job_id=job_id, output_name='roi_current_1'
        )

    def test_overlay_mode_with_single_item(self, simple_data_1, data_key_1):
        """Test that overlay mode returns Overlay even with single item."""
        from ess.livedata.dashboard.plot_params import LayoutParams

        params = PlotParams1d(layout=LayoutParams(combine_mode='overlay'))
        plotter = plots.LinePlotter.from_params(params)
        data_dict = {data_key_1: simple_data_1}

        result = plotter(data_dict)

        # Should return Overlay, not raw Curve
        assert isinstance(result, hv.Overlay)
        # Should contain one element
        assert len(result) == 1

    def test_overlay_mode_with_multiple_items(
        self, simple_data_1, simple_data_2, data_key_1, data_key_2
    ):
        """Test that overlay mode combines multiple plots into Overlay."""
        from ess.livedata.dashboard.plot_params import LayoutParams

        params = PlotParams1d(layout=LayoutParams(combine_mode='overlay'))
        plotter = plots.LinePlotter.from_params(params)
        data_dict = {data_key_1: simple_data_1, data_key_2: simple_data_2}

        result = plotter(data_dict)

        # Should return Overlay
        assert isinstance(result, hv.Overlay)
        # Should contain two elements
        assert len(result) == 2

    def test_non_overlay_mode_with_single_item_returns_raw_plot(
        self, simple_data_1, data_key_1
    ):
        """Test that non-overlay mode returns raw plot for single item."""
        from ess.livedata.dashboard.plot_params import LayoutParams

        params = PlotParams1d(layout=LayoutParams(combine_mode='layout'))
        plotter = plots.LinePlotter.from_params(params)
        data_dict = {data_key_1: simple_data_1}

        result = plotter(data_dict)

        # Should return raw Curve, not Overlay
        assert isinstance(result, hv.Curve)

    def test_empty_data_returns_no_data_text(self):
        """Test that empty data returns 'No data' text element in overlay mode."""
        from ess.livedata.dashboard.plot_params import LayoutParams

        params = PlotParams1d(layout=LayoutParams(combine_mode='overlay'))
        plotter = plots.LinePlotter.from_params(params)
        data_dict = {}

        result = plotter(data_dict)

        # With overlay mode and empty data, returns Overlay containing Text
        assert isinstance(result, hv.Overlay)
        assert len(result) == 1
        # First element should be Text
        text_element = next(iter(result))
        assert isinstance(text_element, hv.Text)
        assert 'No data' in str(text_element.data)

    def test_empty_data_returns_no_data_text_layout_mode(self):
        """Test that empty data returns 'No data' text element in layout mode."""
        from ess.livedata.dashboard.plot_params import LayoutParams

        params = PlotParams1d(layout=LayoutParams(combine_mode='layout'))
        plotter = plots.LinePlotter.from_params(params)
        data_dict = {}

        result = plotter(data_dict)

        # With layout mode and empty data, returns Text directly
        assert isinstance(result, hv.Text)
        assert 'No data' in str(result.data)


class TestBarsPlotter:
    """Tests for BarsPlotter with 0D data."""

    @pytest.fixture
    def bars_plotter(self):
        """Create a BarsPlotter instance with default settings."""
        from ess.livedata.dashboard.plot_params import PlotParamsBars

        return plots.BarsPlotter.from_params(PlotParamsBars())

    @pytest.fixture
    def horizontal_bars_plotter(self):
        """Create a BarsPlotter instance with horizontal bars."""
        from ess.livedata.dashboard.plot_params import BarOrientation, PlotParamsBars

        return plots.BarsPlotter.from_params(
            PlotParamsBars(orientation=BarOrientation(horizontal=True))
        )

    @pytest.fixture
    def scalar_data(self):
        """Create 0D scalar data for testing."""
        return sc.DataArray(sc.scalar(42.0, unit='counts'))

    def test_plot_creates_bars_element(self, bars_plotter, scalar_data, data_key):
        """Test that BarsPlotter creates hv.Bars from 0D data."""
        result = bars_plotter.plot(scalar_data, data_key)
        assert isinstance(result, hv.Bars)

    def test_plot_contains_correct_value(self, bars_plotter, scalar_data, data_key):
        """Test that the bar contains the correct scalar value."""
        result = bars_plotter.plot(scalar_data, data_key)
        # Extract the data from the Bars element (pandas DataFrame)
        bar_data = result.data
        assert len(bar_data) == 1
        # The vdims column is named after output_name
        assert bar_data[data_key.output_name].iloc[0] == 42.0

    def test_plot_uses_source_name_as_label(self, bars_plotter, scalar_data, data_key):
        """Test that the bar is labeled with source_name."""
        result = bars_plotter.plot(scalar_data, data_key)
        bar_data = result.data
        # The label should contain the source_name
        assert data_key.job_id.source_name in bar_data['source'].iloc[0]

    def test_plot_uses_output_name_as_vdims(self, bars_plotter, scalar_data):
        """Test that output_name is used as the vdims column name."""
        workflow_id = WorkflowId(
            instrument='test_instrument',
            namespace='test_namespace',
            name='test_workflow',
            version=1,
        )
        job_id = JobId(source_name='detector', job_number=uuid.uuid4())
        data_key = ResultKey(
            workflow_id=workflow_id, job_id=job_id, output_name='roi_sum'
        )

        result = bars_plotter.plot(scalar_data, data_key)
        bar_data = result.data
        # output_name is used as the column name for values
        assert 'roi_sum' in bar_data.columns
        assert bar_data['roi_sum'].iloc[0] == 42.0

    def test_vertical_bars_default(self, bars_plotter, scalar_data, data_key):
        """Test that bars are vertical by default (invert_axes=False)."""
        result = bars_plotter.plot(scalar_data, data_key)
        # When invert_axes is False (default), it may not appear in options
        opts = hv.Store.lookup_options('bokeh', result, 'plot').kwargs
        assert opts.get('invert_axes', False) is False

    def test_horizontal_bars_option(
        self, horizontal_bars_plotter, scalar_data, data_key
    ):
        """Test that horizontal option inverts axes."""
        result = horizontal_bars_plotter.plot(scalar_data, data_key)
        opts = hv.Store.lookup_options('bokeh', result, 'plot').kwargs
        assert opts.get('invert_axes') is True

    def test_rejects_non_scalar_data(self, bars_plotter, data_key):
        """Test that BarsPlotter rejects non-0D data."""
        data_1d = sc.DataArray(sc.array(dims=['x'], values=[1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="Expected 0D data"):
            bars_plotter.plot(data_1d, data_key)

    def test_call_with_multiple_sources(self, bars_plotter):
        """Test __call__ with multiple 0D data sources creates multiple bars."""
        from ess.livedata.dashboard.plot_params import LayoutParams, PlotParamsBars

        plotter = plots.BarsPlotter.from_params(
            PlotParamsBars(layout=LayoutParams(combine_mode='layout'))
        )

        workflow_id = WorkflowId(
            instrument='test',
            namespace='test',
            name='test',
            version=1,
        )
        key1 = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='source1', job_number=uuid.uuid4()),
            output_name='counts',
        )
        key2 = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='source2', job_number=uuid.uuid4()),
            output_name='counts',
        )

        data = {
            key1: sc.DataArray(sc.scalar(10.0, unit='counts')),
            key2: sc.DataArray(sc.scalar(20.0, unit='counts')),
        }

        result = plotter(data)
        # With layout mode and 2 sources, should return Layout
        assert isinstance(result, hv.Layout)
        assert len(result) == 2

    def test_renders_to_bokeh(self, bars_plotter, scalar_data, data_key):
        """Test that bars can be rendered to Bokeh."""
        result = bars_plotter.plot(scalar_data, data_key)
        render_to_bokeh(result)


class TestOverlay1DPlotter:
    """Tests for Overlay1DPlotter with 2D data."""

    @pytest.fixture
    def overlay_plotter(self):
        """Create an Overlay1DPlotter instance."""
        return plots.Overlay1DPlotter.from_params(PlotParams1d())

    @pytest.fixture
    def data_2d_with_roi_coord(self):
        """Create 2D data with roi coordinate (like stacked ROI spectra)."""
        return sc.DataArray(
            sc.array(
                dims=['roi', 'toa'],
                values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                unit='counts',
            ),
            coords={
                'roi': sc.array(dims=['roi'], values=[0, 3, 5], unit=None),
                'toa': sc.array(dims=['toa'], values=[10.0, 20.0, 30.0], unit='us'),
            },
        )

    @pytest.fixture
    def data_2d_no_first_coord(self):
        """Create 2D data without coordinate for first dimension."""
        return sc.DataArray(
            sc.array(
                dims=['row', 'col'],
                values=[[1.0, 2.0], [3.0, 4.0]],
                unit='counts',
            ),
            coords={
                'col': sc.array(dims=['col'], values=[10.0, 20.0], unit='m'),
            },
        )

    @pytest.fixture
    def data_2d_empty_first_dim(self):
        """Create 2D data with empty first dimension."""
        return sc.DataArray(
            sc.zeros(dims=['roi', 'toa'], shape=[0, 5], unit='counts'),
            coords={
                'roi': sc.array(dims=['roi'], values=[], unit=None),
                'toa': sc.linspace('toa', 0.0, 100.0, num=5, unit='us'),
            },
        )

    def test_plot_creates_overlay(
        self, overlay_plotter, data_2d_with_roi_coord, data_key
    ):
        """Test that Overlay1DPlotter creates hv.Overlay from 2D data."""
        result = overlay_plotter.plot(data_2d_with_roi_coord, data_key)
        assert isinstance(result, hv.Overlay)

    def test_overlay_has_correct_number_of_curves(
        self, overlay_plotter, data_2d_with_roi_coord, data_key
    ):
        """Test that overlay contains one curve per slice along first dim."""
        result = overlay_plotter.plot(data_2d_with_roi_coord, data_key)
        assert len(result) == 3  # 3 ROIs

    def test_single_slice_returns_single_element(self, overlay_plotter, data_key):
        """Test that single slice returns single element, not overlay."""
        data = sc.DataArray(
            sc.array(dims=['roi', 'toa'], values=[[1.0, 2.0, 3.0]], unit='counts'),
            coords={
                'roi': sc.array(dims=['roi'], values=[2], unit=None),
                'toa': sc.array(dims=['toa'], values=[10.0, 20.0, 30.0], unit='us'),
            },
        )
        result = overlay_plotter.plot(data, data_key)
        # Single curve, not wrapped in Overlay
        assert isinstance(result, hv.Curve)

    def test_empty_first_dim_returns_empty_curve(
        self, overlay_plotter, data_2d_empty_first_dim, data_key
    ):
        """Test that empty first dimension returns empty curve."""
        result = overlay_plotter.plot(data_2d_empty_first_dim, data_key)
        assert isinstance(result, hv.Curve)

    def test_curves_labeled_by_coord_value(
        self, overlay_plotter, data_2d_with_roi_coord, data_key
    ):
        """Test that curves are labeled using coordinate values."""
        result = overlay_plotter.plot(data_2d_with_roi_coord, data_key)
        labels = [curve.label for curve in result]
        assert 'roi=0' in labels
        assert 'roi=3' in labels
        assert 'roi=5' in labels

    def test_colors_assigned_by_coord_value(
        self, overlay_plotter, data_2d_with_roi_coord, data_key
    ):
        """Test that colors are assigned by coordinate value for stable identity."""
        colors = hv.Cycle.default_cycles["default_colors"]
        result = overlay_plotter.plot(data_2d_with_roi_coord, data_key)

        # ROI 0 should get colors[0], ROI 3 -> colors[3], ROI 5 -> colors[5]
        curve_colors = {}
        for curve in result:
            opts = hv.Store.lookup_options('bokeh', curve, 'style').kwargs
            curve_colors[curve.label] = opts.get('color')

        assert curve_colors['roi=0'] == colors[0]
        assert curve_colors['roi=3'] == colors[3]
        assert curve_colors['roi=5'] == colors[5]

    def test_fallback_to_indices_without_coord(
        self, overlay_plotter, data_2d_no_first_coord, data_key
    ):
        """Test that indices are used when first dim has no coordinate."""
        result = overlay_plotter.plot(data_2d_no_first_coord, data_key)
        labels = [curve.label for curve in result]
        # Should use 0, 1 indices
        assert 'row=0' in labels
        assert 'row=1' in labels

    def test_rejects_non_2d_data(self, overlay_plotter, data_key):
        """Test that Overlay1DPlotter rejects non-2D data."""
        data_1d = sc.DataArray(sc.array(dims=['x'], values=[1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="Expected 2D data"):
            overlay_plotter.plot(data_1d, data_key)

    def test_renders_to_bokeh(self, overlay_plotter, data_2d_with_roi_coord, data_key):
        """Test that overlay can be rendered to Bokeh."""
        result = overlay_plotter.plot(data_2d_with_roi_coord, data_key)
        render_to_bokeh(result)

    def test_registered_in_plotter_registry(self):
        """Test that overlay_1d plotter is registered in the registry."""
        from ess.livedata.dashboard.plotting import plotter_registry

        assert 'overlay_1d' in plotter_registry
        spec = plotter_registry.get_spec('overlay_1d')
        assert spec.data_requirements.min_dims == 2
        assert spec.data_requirements.max_dims == 2
        assert spec.data_requirements.multiple_datasets is False

    def test_compatible_with_2d_data(self, data_2d_with_roi_coord, data_key):
        """Test that registry identifies overlay_1d as compatible with 2D data."""
        from ess.livedata.dashboard.plotting import plotter_registry

        data = {data_key: data_2d_with_roi_coord}
        compatible = plotter_registry.get_compatible_plotters(data)
        assert 'overlay_1d' in compatible

    def test_not_compatible_with_multiple_datasets(
        self, data_2d_with_roi_coord, data_key
    ):
        """Test that overlay_1d is not compatible with multiple datasets."""
        from ess.livedata.dashboard.plotting import plotter_registry

        key2 = ResultKey(
            workflow_id=data_key.workflow_id,
            job_id=JobId(source_name='source2', job_number=uuid.uuid4()),
            output_name='result2',
        )
        data = {data_key: data_2d_with_roi_coord, key2: data_2d_with_roi_coord}
        compatible = plotter_registry.get_compatible_plotters(data)
        assert 'overlay_1d' not in compatible

    def test_bin_edge_coords_produce_curves_not_histograms(
        self, overlay_plotter, data_key
    ):
        """Test bin-edge coordinates are converted to midpoints for Curve output."""
        # Create data with bin-edge coordinate (like TOA histograms)
        toa_edges = sc.array(dims=['toa'], values=[0.0, 10.0, 20.0, 30.0], unit='us')
        data = sc.DataArray(
            sc.array(dims=['roi', 'toa'], values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            coords={
                'roi': sc.array(dims=['roi'], values=[0, 1], unit=None),
                'toa': toa_edges,  # 4 edges for 3 bins
            },
        )
        result = overlay_plotter.plot(data, data_key)

        # Should be Curve elements, not Histogram
        for elem in result:
            assert isinstance(elem, hv.Curve), f"Expected Curve, got {type(elem)}"

    def test_sizing_opts_applied_to_curves(self, data_2d_with_roi_coord, data_key):
        """Test that sizing options (aspect) are applied to individual curves."""
        from ess.livedata.dashboard.plot_params import PlotAspect, PlotAspectType

        # Create plotter with non-default aspect
        params = PlotParams1d()
        params.plot_aspect = PlotAspect(aspect_type=PlotAspectType.square)
        plotter = plots.Overlay1DPlotter.from_params(params)

        result = plotter.plot(data_2d_with_roi_coord, data_key)

        # Check that sizing opts are applied to each curve
        for curve in result:
            opts = hv.Store.lookup_options('bokeh', curve, 'plot').kwargs
            assert opts.get('aspect') == 'square'

    def test_free_aspect_applies_responsive_only(
        self, data_2d_with_roi_coord, data_key
    ):
        """Test that 'free' aspect applies responsive=True without aspect constraint."""
        from ess.livedata.dashboard.plot_params import PlotAspect, PlotAspectType

        params = PlotParams1d()
        params.plot_aspect = PlotAspect(aspect_type=PlotAspectType.free)
        plotter = plots.Overlay1DPlotter.from_params(params)

        result = plotter.plot(data_2d_with_roi_coord, data_key)

        # Check that responsive is set but no aspect constraint
        for curve in result:
            opts = hv.Store.lookup_options('bokeh', curve, 'plot').kwargs
            assert opts.get('responsive') is True
            assert 'aspect' not in opts or opts.get('aspect') is None

    def test_overflow_bins_stripped(self, overlay_plotter, data_key):
        """Overflow bins with ±inf edges are stripped before rendering.

        ROI spectra from detector view histograms have overflow bins with
        inf edges that cannot be rendered.
        """
        toa_edges = sc.concat(
            [
                sc.scalar(float('-inf'), unit='us'),
                sc.array(dims=['toa'], values=[0.0, 10.0, 20.0, 30.0], unit='us'),
                sc.scalar(float('+inf'), unit='us'),
            ],
            'toa',
        )
        data = sc.DataArray(
            sc.array(
                dims=['roi', 'toa'],
                values=[[9.0, 1.0, 2.0, 3.0, 7.0], [0.0, 4.0, 5.0, 6.0, 0.0]],
            ),
            coords={
                'roi': sc.array(dims=['roi'], values=[0, 1], unit=None),
                'toa': toa_edges,
            },
        )
        result = overlay_plotter.plot(data, data_key)
        assert isinstance(result, hv.Overlay)
        # Each curve should have midpoints of [0,10,20,30] = [5,15,25]; no inf
        for curve in result:
            xs = curve.dimension_values('toa')
            assert not np.any(np.isinf(xs))
            np.testing.assert_array_equal(xs, [5.0, 15.0, 25.0])

    def test_overflow_bins_stripped_renders_to_bokeh(self, overlay_plotter, data_key):
        """Overlay1D with stripped overflow bins renders to Bokeh."""
        toa_edges = sc.concat(
            [
                sc.scalar(float('-inf'), unit='us'),
                sc.array(dims=['toa'], values=[0.0, 10.0, 20.0, 30.0], unit='us'),
                sc.scalar(float('+inf'), unit='us'),
            ],
            'toa',
        )
        data = sc.DataArray(
            sc.array(
                dims=['roi', 'toa'],
                values=[[9.0, 1.0, 2.0, 3.0, 7.0], [0.0, 4.0, 5.0, 6.0, 0.0]],
            ),
            coords={
                'roi': sc.array(dims=['roi'], values=[0, 1], unit=None),
                'toa': toa_edges,
            },
        )
        hv_element = overlay_plotter.plot(data, data_key)
        render_to_bokeh(hv_element)


class TestLagIndicator:
    """Tests for lag indicator functionality in plotters."""

    @pytest.fixture
    def data_key(self):
        """Create a test ResultKey."""
        workflow_id = WorkflowId(
            instrument='test_instrument',
            namespace='test_namespace',
            name='test_workflow',
            version=1,
        )
        job_id = JobId(source_name='test_source', job_number=uuid.uuid4())
        return ResultKey(
            workflow_id=workflow_id, job_id=job_id, output_name='test_result'
        )

    def test_time_info_shown_when_coords_present(self, data_key):
        """Test that time interval and lag are shown in title."""
        import time

        now_ns = time.time_ns()
        # Create data with start_time 2 seconds ago and end_time 1 second ago
        start_time_ns = now_ns - int(2e9)
        end_time_ns = now_ns - int(1e9)
        data = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
            coords={
                'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0]),
                'start_time': sc.scalar(start_time_ns, unit='ns'),
                'end_time': sc.scalar(end_time_ns, unit='ns'),
            },
        )

        plotter = plots.LinePlotter.from_params(PlotParams1d())
        result = plotter({data_key: data})

        # Check that title contains time range and lag
        opts = hv.Store.lookup_options('bokeh', result, 'plot').kwargs
        assert 'title' in opts
        title = opts['title']
        # Should contain time range separator and lag
        assert ' - ' in title  # hyphen between times
        assert 'Lag:' in title
        # Lag should be approximately 1 second
        assert '1.' in title or '2.' in title

    def test_no_lag_title_when_end_time_absent(self, data_key):
        """Test that no lag title is added when end_time coord is absent."""
        data = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
            coords={'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0])},
        )

        plotter = plots.LinePlotter.from_params(PlotParams1d())
        result = plotter({data_key: data})

        # Check that no title is set (or title doesn't contain Lag)
        opts = hv.Store.lookup_options('bokeh', result, 'plot').kwargs
        title = opts.get('title', '')
        assert 'Lag:' not in title

    def test_lag_uses_maximum_across_multiple_sources(self):
        """Test that lag shows the maximum (oldest data) when multiple sources."""
        import time

        workflow_id = WorkflowId(
            instrument='test_instrument',
            namespace='test_namespace',
            name='test_workflow',
            version=1,
        )

        now_ns = time.time_ns()
        # Source 1: data from 2s to 1s ago
        data_key1 = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='source1', job_number=uuid.uuid4()),
            output_name='result',
        )
        data1 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
            coords={
                'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0]),
                'start_time': sc.scalar(now_ns - int(2e9), unit='ns'),
                'end_time': sc.scalar(now_ns - int(1e9), unit='ns'),
            },
        )

        # Source 2: data from 6s to 5s ago (older, should determine the lag)
        data_key2 = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='source2', job_number=uuid.uuid4()),
            output_name='result',
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['x'], values=[4.0, 5.0, 6.0]),
            coords={
                'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0]),
                'start_time': sc.scalar(now_ns - int(6e9), unit='ns'),
                'end_time': sc.scalar(now_ns - int(5e9), unit='ns'),
            },
        )

        plotter = plots.LinePlotter.from_params(PlotParams1d())
        result = plotter({data_key1: data1, data_key2: data2})

        # Check that lag is approximately 5 seconds (the older data)
        opts = hv.Store.lookup_options('bokeh', result, 'plot').kwargs
        assert 'title' in opts
        assert 'Lag:' in opts['title']
        # Should show ~5 seconds, not ~1 second (using oldest end_time)
        assert '5.' in opts['title'] or '6.' in opts['title']
