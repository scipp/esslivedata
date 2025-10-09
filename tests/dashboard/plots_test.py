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
from ess.livedata.dashboard.plot_params import PlotParams2d, PlotScale

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
