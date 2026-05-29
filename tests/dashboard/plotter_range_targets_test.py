# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for per-plotter ``get_range_targets`` API.

Each plotter computes per-axis ``(lo, hi)`` targets from data on every
``compute()`` call. These tests assert that the targets match the data
extent framed with the same per-element padding HoloViews would apply
(images pad nothing, curves pad y only) and that ``AUTOSCALE_AXES``
matches the axes actually populated.
"""

from __future__ import annotations

import uuid

import holoviews as hv
import numpy as np
import pytest
import scipp as sc
from holoviews.core.util import range_pad

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.correlation_plotter import (
    PRIMARY,
    X_AXIS,
    Y_AXIS,
    Bin1dParams,
    Bin2dParams,
    CorrelationHistogram1dParams,
    CorrelationHistogram1dPlotter,
    CorrelationHistogram2dParams,
    CorrelationHistogram2dPlotter,
)
from ess.livedata.dashboard.plot_params import (
    PlotParams1d,
    PlotParams2d,
    PlotParams3d,
    PlotParamsBars,
    PlotScale,
    PlotScaleParams2d,
)
from ess.livedata.dashboard.plots import (
    BarsPlotter,
    ImagePlotter,
    LinePlotter,
    Overlay1DPlotter,
    _hv_axis_padding,
)
from ess.livedata.dashboard.slicer_plotter import SlicerPlotter

hv.extension('bokeh')


def _key(source: str = 'src', output: str = 'out') -> ResultKey:
    return ResultKey(
        workflow_id=WorkflowId(instrument='test', name='test', version=1),
        job_id=JobId(source_name=source, job_number=uuid.uuid4()),
        output_name=output,
    )


def _expected(
    element: type, axis: str, lo: float, hi: float, *, log: bool = False
) -> tuple[float, float]:
    """Range an axis should get: data extent framed with HoloViews' padding."""
    pad = dict(zip(('x', 'y', 'c'), _hv_axis_padding(element), strict=True))[axis]
    return range_pad(lo, hi, pad, log)


class TestLinePlotterRangeTargets:
    def test_autoscale_axes_declared(self):
        assert LinePlotter.AUTOSCALE_AXES == frozenset({'x', 'y'})

    def test_returns_none_before_compute(self):
        plotter = LinePlotter.from_params(PlotParams1d())
        assert plotter.get_range_targets(_key()) is None

    def test_targets_match_data_extent_linear(self):
        plotter = LinePlotter.from_params(PlotParams1d())
        key = _key()
        coord = sc.array(dims=['x'], values=[10.0, 20.0, 30.0], unit='m')
        data = sc.DataArray(
            sc.array(dims=['x'], values=[2.0, 5.0, 8.0], unit='counts'),
            coords={'x': coord},
        )
        plotter.compute({PRIMARY: {key: data}})

        targets = plotter.get_range_targets(key)
        assert set(targets) == {'x', 'y'}
        np.testing.assert_allclose(targets['x'], _expected(hv.Curve, 'x', 10.0, 30.0))
        np.testing.assert_allclose(targets['y'], _expected(hv.Curve, 'y', 2.0, 8.0))

    def test_targets_use_log_padding_when_logy(self):
        params = PlotParams1d()
        params.plot_scale.y_scale = PlotScale.log
        plotter = LinePlotter.from_params(params)
        key = _key()
        data = sc.DataArray(
            sc.array(dims=['x'], values=[2.0, 5.0, 8.0], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0], unit='m')},
        )
        plotter.compute({PRIMARY: {key: data}})

        targets = plotter.get_range_targets(key)
        np.testing.assert_allclose(targets['x'], _expected(hv.Curve, 'x', 10.0, 30.0))
        np.testing.assert_allclose(
            targets['y'], _expected(hv.Curve, 'y', 2.0, 8.0, log=True)
        )

    def test_log_padding_stays_positive_for_tiny_values(self):
        params = PlotParams1d()
        params.plot_scale.y_scale = PlotScale.log
        plotter = LinePlotter.from_params(params)
        key = _key()
        # Very small but positive lower bound; padding must stay positive.
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1e-300, 1e-200], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[1.0, 2.0], unit='m')},
        )
        plotter.compute({PRIMARY: {key: data}})

        targets = plotter.get_range_targets(key)
        assert targets['y'][0] > 0.0


class TestImagePlotterRangeTargets:
    def test_autoscale_axes_declared(self):
        assert ImagePlotter.AUTOSCALE_AXES == frozenset({'x', 'y', 'c'})

    def test_targets_for_2d_image(self):
        params = PlotParams2d()
        params.plot_scale.color_scale = PlotScale.linear
        plotter = ImagePlotter.from_params(params)
        key = _key()
        data = sc.DataArray(
            sc.array(
                dims=['y', 'x'],
                values=np.arange(12.0).reshape(3, 4),
                unit='counts',
            ),
            coords={
                'x': sc.arange('x', 4, dtype='float64'),
                'y': sc.arange('y', 3, dtype='float64'),
            },
        )
        plotter.compute({PRIMARY: {key: data}})

        targets = plotter.get_range_targets(key)
        assert set(targets) == {'x', 'y', 'c'}
        # Images pad nothing: x midpoints span [0, 3], extended by half-pixel
        # to [-0.5, 3.5], with no further padding on any axis.
        np.testing.assert_allclose(targets['x'], _expected(hv.Image, 'x', -0.5, 3.5))
        np.testing.assert_allclose(targets['y'], _expected(hv.Image, 'y', -0.5, 2.5))
        np.testing.assert_allclose(targets['c'], _expected(hv.Image, 'c', 0.0, 11.0))

    def test_targets_for_image_with_bin_edges(self):
        params = PlotParams2d()
        params.plot_scale.color_scale = PlotScale.linear
        plotter = ImagePlotter.from_params(params)
        key = _key()
        data = sc.DataArray(
            sc.array(
                dims=['y', 'x'],
                values=np.arange(6.0).reshape(2, 3),
                unit='counts',
            ),
            coords={
                'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0, 3.0], unit='m'),
                'y': sc.array(dims=['y'], values=[0.0, 1.0, 2.0], unit='m'),
            },
        )
        plotter.compute({PRIMARY: {key: data}})

        targets = plotter.get_range_targets(key)
        np.testing.assert_allclose(targets['x'], _expected(hv.Image, 'x', 0.0, 3.0))
        np.testing.assert_allclose(targets['y'], _expected(hv.Image, 'y', 0.0, 2.0))
        np.testing.assert_allclose(targets['c'], _expected(hv.Image, 'c', 0.0, 5.0))

    def test_targets_log_c_excludes_nonpositive(self):
        params = PlotParams2d()
        params.plot_scale.color_scale = PlotScale.log
        plotter = ImagePlotter.from_params(params)
        key = _key()
        data = sc.DataArray(
            sc.array(
                dims=['y', 'x'],
                values=np.array([[1.0, 2.0], [4.0, 8.0]]),
                unit='counts',
            ),
            coords={
                'x': sc.arange('x', 2, dtype='float64'),
                'y': sc.arange('y', 2, dtype='float64'),
            },
        )
        plotter.compute({PRIMARY: {key: data}})

        targets = plotter.get_range_targets(key)
        np.testing.assert_allclose(
            targets['c'], _expected(hv.Image, 'c', 1.0, 8.0, log=True)
        )


class TestOverlay1DPlotterRangeTargets:
    def test_autoscale_axes_declared(self):
        assert Overlay1DPlotter.AUTOSCALE_AXES == frozenset({'x', 'y'})

    def test_targets_union_across_slices(self):
        plotter = Overlay1DPlotter.from_params(PlotParams1d())
        key = _key()
        # 2D data: 3 slices along 'slice', 4 points along 'x'.
        values = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [-1.0, 0.0, 5.0, 7.0],
                [2.0, 2.0, 2.0, 6.0],
            ]
        )
        data = sc.DataArray(
            sc.array(dims=['slice', 'x'], values=values, unit='counts'),
            coords={
                'slice': sc.arange('slice', 3, dtype='float64'),
                'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0, 40.0], unit='m'),
            },
        )
        plotter.compute({PRIMARY: {key: data}})

        targets = plotter.get_range_targets(key)
        assert set(targets) == {'x', 'y'}
        np.testing.assert_allclose(targets['x'], _expected(hv.Curve, 'x', 10.0, 40.0))
        np.testing.assert_allclose(targets['y'], _expected(hv.Curve, 'y', -1.0, 7.0))


class TestBarsPlotterRangeTargets:
    def test_autoscale_axes_empty(self):
        assert BarsPlotter.AUTOSCALE_AXES == frozenset()

    def test_get_range_targets_returns_none(self):
        plotter = BarsPlotter.from_params(PlotParamsBars())
        key = _key()
        data = sc.DataArray(sc.scalar(42.0, unit='counts'))
        plotter.compute({PRIMARY: {key: data}})

        assert plotter.get_range_targets(key) is None


class TestSlicerPlotterRangeTargets:
    def test_autoscale_axes_c_only(self):
        assert SlicerPlotter.AUTOSCALE_AXES == frozenset({'c'})

    def test_targets_expose_c_only(self):
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.linear
        plotter = SlicerPlotter.from_params(params)
        key = _key()
        data = sc.DataArray(
            sc.arange('z', 0, 5 * 8 * 10, dtype='float64').fold(
                dim='z', sizes={'z': 5, 'y': 8, 'x': 10}
            ),
            coords={
                'z': sc.linspace('z', 0.0, 5.0, num=5, unit='s'),
                'y': sc.linspace('y', 0.0, 8.0, num=8, unit='m'),
                'x': sc.linspace('x', 0.0, 10.0, num=10, unit='m'),
            },
        )
        data.data.unit = 'counts'
        plotter.compute({PRIMARY: {key: data}})

        targets = plotter.get_range_targets(key)
        assert set(targets) == {'c'}
        np.testing.assert_allclose(targets['c'], _expected(hv.Image, 'c', 0.0, 399.0))

    def test_targets_log_c_excludes_zeros(self):
        params = PlotParams3d(plot_scale=PlotScaleParams2d())
        params.plot_scale.color_scale = PlotScale.log
        plotter = SlicerPlotter.from_params(params)
        key = _key()
        data = sc.DataArray(
            sc.arange('z', 0, 5 * 8 * 10, dtype='float64').fold(
                dim='z', sizes={'z': 5, 'y': 8, 'x': 10}
            ),
            coords={
                'z': sc.linspace('z', 0.0, 5.0, num=5, unit='s'),
                'y': sc.linspace('y', 0.0, 8.0, num=8, unit='m'),
                'x': sc.linspace('x', 0.0, 10.0, num=10, unit='m'),
            },
        )
        data.data.unit = 'counts'
        plotter.compute({PRIMARY: {key: data}})

        targets = plotter.get_range_targets(key)
        # Log scale excludes zero -> min positive value is 1.0; images pad nothing.
        assert targets['c'][0] > 0.0
        np.testing.assert_allclose(
            targets['c'], _expected(hv.Image, 'c', 1.0, 399.0, log=True)
        )


def _correlation_axis_data(times, values, unit='m'):
    return sc.DataArray(
        data=sc.array(dims=['time'], values=values, unit=unit),
        coords={'time': sc.array(dims=['time'], values=times, unit='ms')},
    )


def _correlation_source_data(times, values, unit='counts'):
    return sc.DataArray(
        data=sc.array(dims=['time'], values=values, unit=unit),
        coords={'time': sc.array(dims=['time'], values=times, unit='ms')},
    )


class TestCorrelationHistogramRangeTargets:
    def test_1d_autoscale_axes(self):
        assert CorrelationHistogram1dPlotter.AUTOSCALE_AXES == frozenset({'x', 'y'})

    def test_2d_autoscale_axes(self):
        assert CorrelationHistogram2dPlotter.AUTOSCALE_AXES == frozenset(
            {'x', 'y', 'c'}
        )

    def test_1d_targets_delegated_to_renderer(self):
        params = CorrelationHistogram1dParams(bins=Bin1dParams(x_bins=4))
        plotter = CorrelationHistogram1dPlotter(params=params)
        src_key = ResultKey(
            workflow_id=WorkflowId(instrument='test', name='test', version=1),
            job_id=JobId(source_name='detector', job_number=uuid.uuid4()),
            output_name='result',
        )
        axis_key = ResultKey(
            workflow_id=WorkflowId(instrument='test', name='test', version=1),
            job_id=JobId(source_name='position', job_number=uuid.uuid4()),
            output_name='result',
        )
        data = {
            PRIMARY: {
                src_key: _correlation_source_data(
                    [100, 200, 300, 400], [1.0, 2.0, 3.0, 4.0]
                )
            },
            X_AXIS: {
                axis_key: _correlation_axis_data(
                    [100, 200, 300, 400], [0.0, 1.0, 2.0, 3.0]
                )
            },
        }
        plotter.compute(data)

        assert plotter.get_range_targets(src_key) is not None
        # Delegation: same instance the renderer would return.
        renderer_targets = plotter._renderer.get_range_targets(src_key)
        assert plotter.get_range_targets(src_key) == renderer_targets

    def test_2d_targets_delegated_to_renderer(self):
        params = CorrelationHistogram2dParams(
            bins=Bin2dParams(x_bins=4, y_bins=4),
        )
        plotter = CorrelationHistogram2dPlotter(params=params)
        src_key = ResultKey(
            workflow_id=WorkflowId(instrument='test', name='test', version=1),
            job_id=JobId(source_name='detector', job_number=uuid.uuid4()),
            output_name='result',
        )
        x_axis_key = ResultKey(
            workflow_id=WorkflowId(instrument='test', name='test', version=1),
            job_id=JobId(source_name='position', job_number=uuid.uuid4()),
            output_name='result',
        )
        y_axis_key = ResultKey(
            workflow_id=WorkflowId(instrument='test', name='test', version=1),
            job_id=JobId(source_name='temperature', job_number=uuid.uuid4()),
            output_name='result',
        )
        data = {
            PRIMARY: {
                src_key: _correlation_source_data(
                    [100, 200, 300, 400], [1.0, 2.0, 3.0, 4.0]
                )
            },
            X_AXIS: {
                x_axis_key: _correlation_axis_data(
                    [100, 200, 300, 400], [0.0, 1.0, 2.0, 3.0]
                )
            },
            Y_AXIS: {
                y_axis_key: _correlation_axis_data(
                    [100, 200, 300, 400], [10.0, 20.0, 30.0, 40.0], unit='K'
                )
            },
        }
        plotter.compute(data)

        targets = plotter.get_range_targets(src_key)
        assert targets is not None
        assert set(targets) >= {'x', 'y', 'c'}
        assert targets == plotter._renderer.get_range_targets(src_key)

    def test_1d_iter_range_targets_delegated_to_renderer(self):
        params = CorrelationHistogram1dParams(bins=Bin1dParams(x_bins=4))
        plotter = CorrelationHistogram1dPlotter(params=params)
        src_key = ResultKey(
            workflow_id=WorkflowId(instrument='test', name='test', version=1),
            job_id=JobId(source_name='detector', job_number=uuid.uuid4()),
            output_name='result',
        )
        axis_key = ResultKey(
            workflow_id=WorkflowId(instrument='test', name='test', version=1),
            job_id=JobId(source_name='position', job_number=uuid.uuid4()),
            output_name='result',
        )
        data = {
            PRIMARY: {
                src_key: _correlation_source_data(
                    [100, 200, 300, 400], [1.0, 2.0, 3.0, 4.0]
                )
            },
            X_AXIS: {
                axis_key: _correlation_axis_data(
                    [100, 200, 300, 400], [0.0, 1.0, 2.0, 3.0]
                )
            },
        }
        plotter.compute(data)

        assert list(plotter.iter_range_targets()) == list(
            plotter._renderer.iter_range_targets()
        )
        # Sanity: the iteration is non-empty and includes the source key.
        keys = [k for k, _ in plotter.iter_range_targets()]
        assert src_key in keys

    def test_2d_iter_range_targets_delegated_to_renderer(self):
        params = CorrelationHistogram2dParams(
            bins=Bin2dParams(x_bins=4, y_bins=4),
        )
        plotter = CorrelationHistogram2dPlotter(params=params)
        src_key = ResultKey(
            workflow_id=WorkflowId(instrument='test', name='test', version=1),
            job_id=JobId(source_name='detector', job_number=uuid.uuid4()),
            output_name='result',
        )
        x_axis_key = ResultKey(
            workflow_id=WorkflowId(instrument='test', name='test', version=1),
            job_id=JobId(source_name='position', job_number=uuid.uuid4()),
            output_name='result',
        )
        y_axis_key = ResultKey(
            workflow_id=WorkflowId(instrument='test', name='test', version=1),
            job_id=JobId(source_name='temperature', job_number=uuid.uuid4()),
            output_name='result',
        )
        data = {
            PRIMARY: {
                src_key: _correlation_source_data(
                    [100, 200, 300, 400], [1.0, 2.0, 3.0, 4.0]
                )
            },
            X_AXIS: {
                x_axis_key: _correlation_axis_data(
                    [100, 200, 300, 400], [0.0, 1.0, 2.0, 3.0]
                )
            },
            Y_AXIS: {
                y_axis_key: _correlation_axis_data(
                    [100, 200, 300, 400], [10.0, 20.0, 30.0, 40.0], unit='K'
                )
            },
        }
        plotter.compute(data)

        items = list(plotter.iter_range_targets())
        assert items == list(plotter._renderer.iter_range_targets())
        # Sanity: c axis is included in the yielded targets.
        assert all('c' in targets for _, targets in items)
        assert src_key in [k for k, _ in items]


@pytest.mark.parametrize(
    ('plotter_cls', 'axes'),
    [
        (LinePlotter, frozenset({'x', 'y'})),
        (ImagePlotter, frozenset({'x', 'y', 'c'})),
        (Overlay1DPlotter, frozenset({'x', 'y'})),
        (BarsPlotter, frozenset()),
        (SlicerPlotter, frozenset({'c'})),
        (CorrelationHistogram1dPlotter, frozenset({'x', 'y'})),
        (CorrelationHistogram2dPlotter, frozenset({'x', 'y', 'c'})),
    ],
)
def test_autoscale_axes_table(plotter_cls, axes):
    """Per plan section 1.4, each plotter declares its autoscalable axes."""
    assert plotter_cls.AUTOSCALE_AXES == axes
