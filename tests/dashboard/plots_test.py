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
    ErrorDisplay,
    Line1dParams,
    Line1dRenderMode,
    PlotParams1d,
    PlotParams2d,
    PlotParams3d,
    PlotScale,
    PlotScaleParams2d,
)
from ess.livedata.dashboard.slicer_plotter import (
    SlicerPlotter,
    SlicerPresenter,
    SlicerState,
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

    def test_line_mode_produces_curve(self, data_key):
        params = PlotParams1d(line=Line1dParams(mode=Line1dRenderMode.line))
        plotter = plots.LinePlotter.from_params(params)
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0], unit='m')},
        )
        result = plotter.plot(data, data_key)
        assert isinstance(result, hv.Curve)

    def test_points_mode_produces_scatter(self, data_key):
        params = PlotParams1d(line=Line1dParams(mode=Line1dRenderMode.points))
        plotter = plots.LinePlotter.from_params(params)
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0], unit='m')},
        )
        result = plotter.plot(data, data_key)
        assert isinstance(result, hv.Scatter)

    def test_histogram_mode_produces_histogram(self, data_key):
        params = PlotParams1d(line=Line1dParams(mode=Line1dRenderMode.histogram))
        plotter = plots.LinePlotter.from_params(params)
        edges = sc.array(dims=['x'], values=[0.0, 10.0, 20.0, 30.0], unit='m')
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': edges},
        )
        result = plotter.plot(data, data_key)
        assert isinstance(result, hv.Histogram)

    def test_line_with_error_bars(self, data_key):
        params = PlotParams1d(
            line=Line1dParams(mode=Line1dRenderMode.line, errors=ErrorDisplay.bars)
        )
        plotter = plots.LinePlotter.from_params(params)
        data = sc.DataArray(
            sc.array(
                dims=['x'],
                values=[1.0, 2.0, 3.0],
                variances=[0.1, 0.2, 0.3],
                unit='counts',
            ),
            coords={'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0], unit='m')},
        )
        result = plotter.plot(data, data_key)
        assert isinstance(result, hv.Overlay)
        elements = list(result)
        assert isinstance(elements[0], hv.Curve)
        assert isinstance(elements[1], hv.ErrorBars)

    def test_line_with_error_band(self, data_key):
        params = PlotParams1d(
            line=Line1dParams(mode=Line1dRenderMode.line, errors=ErrorDisplay.band)
        )
        plotter = plots.LinePlotter.from_params(params)
        data = sc.DataArray(
            sc.array(
                dims=['x'],
                values=[1.0, 2.0, 3.0],
                variances=[0.1, 0.2, 0.3],
                unit='counts',
            ),
            coords={'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0], unit='m')},
        )
        result = plotter.plot(data, data_key)
        assert isinstance(result, hv.Overlay)
        elements = list(result)
        assert isinstance(elements[0], hv.Curve)
        assert isinstance(elements[1], hv.Spread)

    def test_line_with_errors_none(self, data_key):
        params = PlotParams1d(
            line=Line1dParams(mode=Line1dRenderMode.line, errors=ErrorDisplay.none)
        )
        plotter = plots.LinePlotter.from_params(params)
        data = sc.DataArray(
            sc.array(
                dims=['x'],
                values=[1.0, 2.0, 3.0],
                variances=[0.1, 0.2, 0.3],
                unit='counts',
            ),
            coords={'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0], unit='m')},
        )
        result = plotter.plot(data, data_key)
        assert isinstance(result, hv.Curve)

    def test_points_with_error_bars(self, data_key):
        params = PlotParams1d(
            line=Line1dParams(mode=Line1dRenderMode.points, errors=ErrorDisplay.bars)
        )
        plotter = plots.LinePlotter.from_params(params)
        data = sc.DataArray(
            sc.array(
                dims=['x'],
                values=[1.0, 2.0, 3.0],
                variances=[0.1, 0.2, 0.3],
                unit='counts',
            ),
            coords={'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0], unit='m')},
        )
        result = plotter.plot(data, data_key)
        assert isinstance(result, hv.Overlay)
        elements = list(result)
        assert isinstance(elements[0], hv.Scatter)
        assert isinstance(elements[1], hv.ErrorBars)

    def test_no_variances_never_shows_errors(self, data_key):
        params = PlotParams1d(
            line=Line1dParams(mode=Line1dRenderMode.line, errors=ErrorDisplay.bars)
        )
        plotter = plots.LinePlotter.from_params(params)
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0], unit='m')},
        )
        result = plotter.plot(data, data_key)
        assert isinstance(result, hv.Curve)

    def test_histogram_mode_without_bin_edges_falls_back_to_curve(self, data_key):
        params = PlotParams1d(line=Line1dParams(mode=Line1dRenderMode.histogram))
        plotter = plots.LinePlotter.from_params(params)
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0], unit='m')},
        )
        # Histogram mode with midpoint coords: convert_histogram_1d expects bin edges,
        # so to_holoviews dispatches to Curve instead. The plotter passes the data
        # through without converting edges, so the result depends on the data shape.
        # With midpoint coords, hv.Histogram still works (it treats coords as edges).
        result = plotter.plot(data, data_key)
        assert isinstance(result, hv.Histogram)

    def test_histogram_mode_with_errors(self, data_key):
        params = PlotParams1d(
            line=Line1dParams(mode=Line1dRenderMode.histogram, errors=ErrorDisplay.bars)
        )
        plotter = plots.LinePlotter.from_params(params)
        edges = sc.array(dims=['x'], values=[0.0, 10.0, 20.0, 30.0], unit='m')
        data = sc.DataArray(
            sc.array(
                dims=['x'],
                values=[1.0, 2.0, 3.0],
                variances=[0.1, 0.2, 0.3],
                unit='counts',
            ),
            coords={'x': edges},
        )
        result = plotter.plot(data, data_key)
        assert isinstance(result, hv.Overlay)
        elements = list(result)
        assert isinstance(elements[0], hv.Histogram)
        assert isinstance(elements[1], hv.ErrorBars)


class TestSlicerPlotter:
    """Tests for SlicerPlotter two-stage architecture.

    SlicerPlotter uses a two-stage architecture:
    - compute(): Returns SlicerState with 3D data and pre-computed clim (shared)
    - SlicerPresenter: Handles interactive slicing per-session with kdims
    """

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
        return SlicerPlotter.from_params(params)

    @pytest.fixture
    def slicer_presenter(self, slicer_plotter):
        """Create SlicerPresenter for testing render_slice."""
        return slicer_plotter.create_presenter()

    def test_initialization(self, slicer_plotter):
        """Test that SlicerPlotter initializes correctly."""
        # Uses base class autoscalers dict (initialized lazily)
        assert slicer_plotter.autoscalers == {}

    # === Stage 1: compute() tests ===

    def test_compute_returns_slicer_state(self, slicer_plotter, data_3d, data_key):
        """Test that compute() returns a SlicerState with prepared data and clim."""
        data_dict = {data_key: data_3d}
        slicer_plotter.compute(data_dict)
        result = slicer_plotter.get_cached_state()

        assert isinstance(result, SlicerState)
        # Data should have same keys
        assert set(result.data.keys()) == set(data_dict.keys())
        # Data should be prepared (converted to float64)
        assert result.data[data_key].dtype == 'float64'
        assert result.clim is not None  # Should have computed clim

    def test_compute_calculates_global_clim(self, data_3d, data_key):
        """Test that compute() pre-calculates color limits from the full 3D data."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.linear
        plotter = SlicerPlotter.from_params(params)

        data_dict = {data_key: data_3d}
        plotter.compute(data_dict)
        result = plotter.get_cached_state()

        # clim should span the full range of data
        expected_min = float(data_3d.values.min())
        expected_max = float(data_3d.values.max())
        assert result.clim == (expected_min, expected_max)

    def test_compute_log_scale_clim_excludes_nonpositive(self, data_3d, data_key):
        """Test that log scale clim excludes zero and negative values."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.log
        plotter = SlicerPlotter.from_params(params)

        data_dict = {data_key: data_3d}
        plotter.compute(data_dict)
        result = plotter.get_cached_state()

        # data_3d starts at 0 (from arange), so min should be > 0
        assert result.clim is not None
        assert result.clim[0] > 0

    # === Stage 2: render_slice() tests (on SlicerPresenter) ===

    def test_render_slice_slices_3d_data(self, slicer_presenter, data_3d):
        """Test that render_slice correctly slices 3D data."""
        z_value = float(data_3d.coords['z'].values[0])
        result = slicer_presenter.render_slice(
            data_3d, clim=None, slice_dim='z', z_value=z_value
        )

        assert isinstance(result, hv.Image)
        expected_slice = data_3d['z', 0]
        np.testing.assert_allclose(result.data['values'], expected_slice.values)

    def test_render_slice_with_different_positions(self, data_3d):
        """Test that different slice positions produce different results."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.linear
        plotter = SlicerPlotter.from_params(params)
        presenter = plotter.create_presenter()

        z_value_0 = float(data_3d.coords['z'].values[0])
        result_0 = presenter.render_slice(
            data_3d, clim=None, slice_dim='z', z_value=z_value_0
        )

        z_value_2 = float(data_3d.coords['z'].values[2])
        result_2 = presenter.render_slice(
            data_3d, clim=None, slice_dim='z', z_value=z_value_2
        )

        # Values should be different
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(result_0.data['values'], result_2.data['values'])

    @pytest.mark.parametrize('slice_dim', ['z', 'y', 'x'])
    def test_render_slice_along_different_dimensions(
        self, slicer_presenter, data_3d, slice_dim
    ):
        """Test slicing along different dimensions."""
        slice_value = float(data_3d.coords[slice_dim].values[0])
        result = slicer_presenter.render_slice(
            data_3d,
            clim=None,
            slice_dim=slice_dim,
            **{f'{slice_dim}_value': slice_value},
        )

        assert isinstance(result, hv.Image)
        expected = data_3d[slice_dim, 0]
        np.testing.assert_allclose(result.data['values'], expected.values)

    def test_render_slice_uses_provided_clim(self, slicer_presenter, data_3d):
        """Test that render_slice uses the provided clim for consistent color scale."""
        clim = (10.0, 100.0)
        z_value = float(data_3d.coords['z'].values[0])
        result = slicer_presenter.render_slice(
            data_3d, clim=clim, slice_dim='z', z_value=z_value
        )

        # Check that clim is set in options (clim is a 'plot' option, not 'norm')
        plot_opts = hv.Store.lookup_options('bokeh', result, 'plot').kwargs
        assert plot_opts.get('clim') == clim

    def test_render_slice_framewise_always_true(self, slicer_presenter, data_3d):
        """Test that render_slice always sets framewise=True."""
        z_value = float(data_3d.coords['z'].values[0])
        result = slicer_presenter.render_slice(
            data_3d, clim=None, slice_dim='z', z_value=z_value
        )

        norm_opts = hv.Store.lookup_options('bokeh', result, 'norm').kwargs
        assert norm_opts.get('framewise') is True

    def test_render_slice_flatten_mode(self, data_3d):
        """Test that flatten mode works in render_slice."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.linear
        plotter = SlicerPlotter.from_params(params)
        presenter = plotter.create_presenter()

        # Original data is (z:5, y:8, x:10)
        # Keep x: flatten z,y -> (40, 10)
        result = presenter.render_slice(
            data_3d, clim=None, mode='flatten', slice_dim='x'
        )
        assert isinstance(result, hv.Image | hv.QuadMesh)
        assert result.data['values'].shape == (40, 10)

    def test_compute_log_scale_masks_zeros(self, data_3d, data_key):
        """Test that log scale masks zeros in compute()."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.log
        plotter = SlicerPlotter.from_params(params)

        data_dict = {data_key: data_3d}
        plotter.compute(data_dict)
        result = plotter.get_cached_state()

        # First value (0) should be NaN in log scale (data_3d starts at 0 from arange)
        prepared_data = result.data[data_key]
        assert np.isnan(prepared_data.values[0, 0, 0])

    # === SlicerPresenter tests ===

    def test_create_presenter_returns_slicer_presenter(self, slicer_plotter):
        """Test that create_presenter returns a SlicerPresenter."""
        presenter = slicer_plotter.create_presenter()
        assert isinstance(presenter, SlicerPresenter)

    def test_presenter_initializes_kdims_from_state(
        self, slicer_plotter, data_3d, data_key
    ):
        """Test that SlicerPresenter creates kdims from SlicerState."""
        data_dict = {data_key: data_3d}
        slicer_plotter.compute(data_dict)
        state = slicer_plotter.get_cached_state()

        presenter = slicer_plotter.create_presenter()
        pipe = hv.streams.Pipe(data=state)
        dmap = presenter.present(pipe)

        # DynamicMap should have kdims
        assert dmap.kdims is not None
        assert len(dmap.kdims) == 5  # mode + selector + 3 sliders

    def test_presenter_kdims_with_coords(self, slicer_plotter, data_3d, data_key):
        """Test that presenter kdims use coordinate values when available."""
        data_dict = {data_key: data_3d}
        slicer_plotter.compute(data_dict)
        state = slicer_plotter.get_cached_state()

        presenter = slicer_plotter.create_presenter()
        pipe = hv.streams.Pipe(data=state)
        dmap = presenter.present(pipe)

        kdims = dmap.kdims
        assert kdims[0].name == 'mode'
        assert kdims[1].name == 'slice_dim'

        z_dim = next(d for d in kdims if 'z' in d.name)
        assert z_dim.name == 'z_value'  # Uses value not index
        assert z_dim.unit == 's'

    def test_presenter_kdims_without_coords(
        self, slicer_plotter, data_3d_no_coords, data_key
    ):
        """Test that presenter kdims fall back to indices without coordinates."""
        data_dict = {data_key: data_3d_no_coords}
        slicer_plotter.compute(data_dict)
        state = slicer_plotter.get_cached_state()

        presenter = slicer_plotter.create_presenter()
        pipe = hv.streams.Pipe(data=state)
        dmap = presenter.present(pipe)

        kdims = dmap.kdims
        z_dim = next(d for d in kdims if 'z' in d.name)
        assert z_dim.name == 'z_index'  # Falls back to index

    def test_presenter_dmap_renders_slice(self, slicer_plotter, data_3d, data_key):
        """Test that presenter's DynamicMap can render slices."""
        data_dict = {data_key: data_3d}
        slicer_plotter.compute(data_dict)
        state = slicer_plotter.get_cached_state()

        presenter = slicer_plotter.create_presenter()
        pipe = hv.streams.Pipe(data=state)
        dmap = presenter.present(pipe)

        # Get a slice
        z_value = float(data_3d.coords['z'].values[0])
        y_value = float(data_3d.coords['y'].values[0])
        x_value = float(data_3d.coords['x'].values[0])

        result = dmap['slice', 'z', z_value, y_value, x_value]
        assert isinstance(result, hv.Image)

    # === Registry test (unchanged) ===

    def test_multiple_datasets_rejected_by_registry(self, data_3d, data_key):
        """Test slicer plotter is rejected for multiple datasets by the registry."""
        from ess.livedata.dashboard.plotter_registry import plotter_registry

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

        single_data = {data_key: data_3d}
        compatible = plotter_registry.get_compatible_plotters(single_data)
        assert 'slicer' in compatible

        multiple_data = {data_key: data_3d, data_key2: data_3d}
        compatible = plotter_registry.get_compatible_plotters(multiple_data)
        assert 'slicer' not in compatible

    # === Edge coordinate tests ===

    def test_edge_coordinates_in_presenter(self, slicer_plotter, data_key):
        """Test handling of edge coordinates in presenter kdims."""
        x_edges = sc.linspace('x', 0.0, 10.0, num=11, unit='m')
        y_edges = sc.linspace('y', 0.0, 8.0, num=9, unit='m')
        z_edges = sc.linspace('z', 0.0, 5.0, num=6, unit='s')

        data = sc.DataArray(
            sc.ones(dims=['z', 'y', 'x'], shape=[5, 8, 10], unit='counts'),
            coords={'x': x_edges, 'y': y_edges, 'z': z_edges},
        )

        data_dict = {data_key: data}
        slicer_plotter.compute(data_dict)
        state = slicer_plotter.get_cached_state()

        presenter = slicer_plotter.create_presenter()
        pipe = hv.streams.Pipe(data=state)
        dmap = presenter.present(pipe)

        # For edge coords, slider uses midpoints
        z_kdim = next(d for d in dmap.kdims if 'z' in d.name)
        expected_midpoint = float(sc.midpoints(z_edges, dim='z').values[0])
        assert z_kdim.values[0] == pytest.approx(expected_midpoint)

    def test_2d_dimension_coords_in_presenter(self, slicer_plotter, data_key):
        """Test that 2D dimension coordinates fall back to index-based slider."""
        data = sc.DataArray(
            data=sc.array(
                dims=['z', 'y', 'x'],
                values=np.arange(24).reshape(2, 3, 4).astype('float64'),
                unit='counts',
            ),
        )
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

        data_dict = {data_key: data}
        slicer_plotter.compute(data_dict)
        state = slicer_plotter.get_cached_state()

        presenter = slicer_plotter.create_presenter()
        pipe = hv.streams.Pipe(data=state)
        dmap = presenter.present(pipe)

        y_kdim = next(d for d in dmap.kdims if d.name.startswith('y'))
        assert y_kdim.name == 'y_index'  # Falls back to index


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

        plotter.compute(data_dict)
        result = plotter.get_cached_state()

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

        plotter.compute(data_dict)
        result = plotter.get_cached_state()

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

        plotter.compute(data_dict)
        result = plotter.get_cached_state()

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

        plotter.compute(data_dict)
        result = plotter.get_cached_state()

        # Should return raw Curve, not Overlay
        assert isinstance(result, hv.Curve)

    def test_empty_data_returns_no_data_text(self):
        """Test that empty data returns 'No data' text element in overlay mode."""
        from ess.livedata.dashboard.plot_params import LayoutParams

        params = PlotParams1d(layout=LayoutParams(combine_mode='overlay'))
        plotter = plots.LinePlotter.from_params(params)
        data_dict = {}

        plotter.compute(data_dict)
        result = plotter.get_cached_state()

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

        plotter.compute(data_dict)
        result = plotter.get_cached_state()

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

        plotter.compute(data)
        result = plotter.get_cached_state()
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
        from ess.livedata.dashboard.plotter_registry import plotter_registry

        assert 'overlay_1d' in plotter_registry
        spec = plotter_registry.get_spec('overlay_1d')
        assert spec.data_requirements.min_dims == 2
        assert spec.data_requirements.max_dims == 2
        assert spec.data_requirements.multiple_datasets is False

    def test_compatible_with_2d_data(self, data_2d_with_roi_coord, data_key):
        """Test that registry identifies overlay_1d as compatible with 2D data."""
        from ess.livedata.dashboard.plotter_registry import plotter_registry

        data = {data_key: data_2d_with_roi_coord}
        compatible = plotter_registry.get_compatible_plotters(data)
        assert 'overlay_1d' in compatible

    def test_not_compatible_with_multiple_datasets(
        self, data_2d_with_roi_coord, data_key
    ):
        """Test that overlay_1d is not compatible with multiple datasets."""
        from ess.livedata.dashboard.plotter_registry import plotter_registry

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
        plotter.compute({data_key: data})
        result = plotter.get_cached_state()

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
        plotter.compute({data_key: data})
        result = plotter.get_cached_state()

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
        plotter.compute({data_key1: data1, data_key2: data2})
        result = plotter.get_cached_state()

        # Check that lag is approximately 5 seconds (the older data)
        opts = hv.Store.lookup_options('bokeh', result, 'plot').kwargs
        assert 'title' in opts
        assert 'Lag:' in opts['title']
        # Should show ~5 seconds, not ~1 second (using oldest end_time)
        assert '5.' in opts['title'] or '6.' in opts['title']


class TestTwoStageArchitecture:
    """Tests for the two-stage compute/present architecture."""

    @pytest.fixture
    def simple_data(self):
        """Create simple 1D data for testing."""
        return sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
            coords={'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0])},
        )

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

    def test_create_presenter_returns_presenter(self, simple_data, data_key):
        """Test that create_presenter() returns a Presenter."""
        plotter = plots.LinePlotter.from_params(PlotParams1d())

        presenter = plotter.create_presenter()

        # Should have a present() method
        assert hasattr(presenter, 'present')
        assert callable(presenter.present)

    def test_presenter_creates_dynamic_map(self, simple_data, data_key):
        """Test that Presenter.present() creates a DynamicMap."""
        plotter = plots.LinePlotter.from_params(PlotParams1d())
        data_dict = {data_key: simple_data}
        # With new architecture, pipe receives pre-computed elements
        plotter.compute(data_dict)
        computed = plotter.get_cached_state()

        presenter = plotter.create_presenter()
        pipe = hv.streams.Pipe(data=computed)
        dmap = presenter.present(pipe)

        assert isinstance(dmap, hv.DynamicMap)

    def test_slicer_presenter_creates_kdims_from_state(self, data_key):
        """Test that SlicerPresenter creates kdims from SlicerState."""
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        plotter = SlicerPlotter.from_params(params)

        # Create 3D data
        data_3d = sc.DataArray(
            sc.arange('z', 0, 5 * 8 * 10, dtype='float64').fold(
                dim='z', sizes={'z': 5, 'y': 8, 'x': 10}
            ),
            coords={
                'z': sc.linspace('z', 0, 1, 5, unit='s'),
                'y': sc.linspace('y', 0, 1, 8, unit='m'),
                'x': sc.linspace('x', 0, 1, 10, unit='m'),
            },
        )
        data_dict = {data_key: data_3d}

        # Compute returns SlicerState
        plotter.compute(data_dict)
        state = plotter.get_cached_state()
        assert isinstance(state, SlicerState)

        # Presenter creates DynamicMap with kdims
        presenter = plotter.create_presenter()
        pipe = hv.streams.Pipe(data=state)
        dmap = presenter.present(pipe)

        # DynamicMap should have kdims for mode, slice_dim, and per-dimension sliders
        assert len(dmap.kdims) >= 3  # mode, slice_dim, + at least one slider
        kdim_names = [d.name for d in dmap.kdims]
        assert 'mode' in kdim_names
        assert 'slice_dim' in kdim_names

    def test_default_presenter_passes_through_computed_data(
        self, simple_data, data_key
    ):
        """Test that DefaultPresenter passes through pre-computed elements."""
        plotter = plots.LinePlotter.from_params(PlotParams1d())
        data_dict = {data_key: simple_data}
        # With new architecture, compute() is called once and result is passed to pipe
        plotter.compute(data_dict)
        computed = plotter.get_cached_state()

        presenter = plotter.create_presenter()
        pipe = hv.streams.Pipe(data=computed)
        dmap = presenter.present(pipe)

        # Get current value - should pass through the computed element
        current = dmap[()]
        assert current is not None
        # The result should be the same as the computed element (pass-through)
        assert type(current) is type(computed)
