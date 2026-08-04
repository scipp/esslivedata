# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for correlation histogram plotters."""

import holoviews as hv
import numpy as np
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.dashboard.correlation_plotter import (
    PRIMARY,
    X_AXIS,
    Y_AXIS,
    AxisSpec,
    Bin1dParams,
    Bin2dParams,
    CorrelationHistogram1dParams,
    CorrelationHistogram1dPlotter,
    CorrelationHistogram2dParams,
    CorrelationHistogram2dPlotter,
    CorrelationHistogramPlotter,
)
from ess.livedata.dashboard.extractors import FullHistoryExtractor
from ess.livedata.dashboard.plot_params import PlotScaleParams, PlotScaleParams2d
from ess.livedata.dashboard.plots import ImagePlotter, LinePlotter, TitleResolver
from ess.livedata.dashboard.temporal_buffers import TemporalBuffer

hv.extension('bokeh')


def _make_line_renderer() -> LinePlotter:
    """Create a LinePlotter for testing."""
    return LinePlotter(scale_opts=PlotScaleParams(), mode='histogram')


def _make_result_key(source_name: str) -> DataKey:
    """Create a DataKey for testing with the given source name."""
    return DataKey(
        workflow_id=WorkflowId(instrument='test', name='test', version=1),
        source_name=source_name,
        output_name='result',
    )


def make_axis_data(
    times: list[int], values: list[float], time_unit: str = 'ms', value_unit: str = 'm'
) -> sc.DataArray:
    """Create axis data for correlation histogram."""
    return sc.DataArray(
        data=sc.array(dims=['time'], values=values, unit=value_unit),
        coords={'time': sc.array(dims=['time'], values=times, unit=time_unit)},
    )


def make_source_data(
    times: list[int],
    values: list[float],
    time_unit: str = 'ms',
    value_unit: str = 'counts',
) -> sc.DataArray:
    """Create source data for correlation histogram."""
    return sc.DataArray(
        data=sc.array(dims=['time'], values=values, unit=value_unit),
        coords={'time': sc.array(dims=['time'], values=times, unit=time_unit)},
    )


def histogram_of(plotter: CorrelationHistogramPlotter) -> dict[str, np.ndarray]:
    """Return the raw data of the single histogram in the plotter's cached state."""
    state = plotter.get_cached_state()
    assert state is not None
    return next(iter(state.values())).data


class TestWallClockJoinConvention:
    """Which instant of a window output the correlation join uses for x."""

    def test_window_output_joins_at_its_time_not_its_start_time(self):
        """A window is correlated against the axis as of when it closed.

        ``time`` is the interval's right edge, so a window straddling a step in
        the axis picks up the value after the step. Driven through the real
        buffer and extractor, since it is the buffered ``time`` coord — not the
        raw message — that the join reads.
        """
        axis_buffer = TemporalBuffer()
        for sample_time, value in [(100, 1.0), (250, 5.0)]:
            axis_buffer.add(
                sc.DataArray(
                    sc.scalar(value, unit='m'),
                    coords={'time': sc.scalar(sample_time, unit='ms')},
                )
            )

        source_buffer = TemporalBuffer()
        # The second window opens before the axis steps to 5.0 and closes after.
        for start_time, close_time, counts in [(100, 150, 10.0), (150, 300, 20.0)]:
            source_buffer.add(
                sc.DataArray(
                    sc.scalar(counts, unit='counts'),
                    coords={
                        'start_time': sc.scalar(start_time, unit='ms'),
                        'time': sc.scalar(close_time, unit='ms'),
                    },
                )
            )

        key = _make_result_key('detector')
        plotter = CorrelationHistogramPlotter(
            axes=[AxisSpec(role=X_AXIS, name='position', bins=2)],
            normalize=False,
            renderer=_make_line_renderer(),
        )
        plotter.compute(
            {
                PRIMARY: {key: FullHistoryExtractor().extract(source_buffer.get())},
                X_AXIS: {key: FullHistoryExtractor().extract(axis_buffer.get())},
            }
        )

        (histogram,) = plotter.get_cached_state()
        # Positions 1.0 and 5.0 over two bins: one window on each side. Joining
        # on start_time would put both at 1.0 and collapse this to a single bin.
        assert list(histogram.dimension_values('result')) == [10.0, 20.0]


class TestCorrelationHistogramPlotter:
    """Tests for the base CorrelationHistogramPlotter class."""

    def test_raises_when_primary_data_missing(self):
        """Should raise ValueError when no primary data is provided."""
        axes = [AxisSpec(role=X_AXIS, name='x', bins=10)]
        plotter = CorrelationHistogramPlotter(
            axes=axes, normalize=False, renderer=_make_line_renderer()
        )

        data = {
            PRIMARY: {},
            X_AXIS: {_make_result_key('position'): make_axis_data([100], [1.0])},
        }

        with pytest.raises(ValueError, match="at least one data source"):
            plotter.compute(data)

    def test_raises_when_axis_data_missing(self):
        """Should raise ValueError when required axis data is missing."""
        axes = [AxisSpec(role=X_AXIS, name='x', bins=10)]
        plotter = CorrelationHistogramPlotter(
            axes=axes, normalize=False, renderer=_make_line_renderer()
        )

        data = {
            PRIMARY: {_make_result_key('detector'): make_source_data([50], [10.0])},
            X_AXIS: {},  # Missing axis data
        }

        with pytest.raises(ValueError, match=f"role '{X_AXIS}'"):
            plotter.compute(data)

    def test_works_with_single_axis(self):
        """Should work with a single axis (1D histogram)."""
        axis_data = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])
        source_data = make_source_data(times=[150, 250], values=[10.0, 20.0])

        axes = [AxisSpec(role=X_AXIS, name='position', bins=10)]
        plotter = CorrelationHistogramPlotter(
            axes=axes, normalize=False, renderer=_make_line_renderer()
        )

        data = {
            PRIMARY: {_make_result_key('detector'): source_data},
            X_AXIS: {_make_result_key('position'): axis_data},
        }

        plotter.compute(data)
        result = plotter.get_cached_state()
        assert result is not None

    def test_works_with_multiple_axes(self):
        """Should work with multiple axes (2D histogram)."""
        x_axis = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])
        y_axis = make_axis_data(
            times=[100, 200, 300],
            values=[10.0, 20.0, 30.0],
            value_unit='K',
        )
        source_data = make_source_data(times=[150, 250], values=[10.0, 20.0])

        axes = [
            AxisSpec(role=X_AXIS, name='position', bins=5),
            AxisSpec(role=Y_AXIS, name='temperature', bins=5),
        ]
        plotter = CorrelationHistogramPlotter(
            axes=axes, normalize=False, renderer=_make_line_renderer()
        )

        data = {
            PRIMARY: {_make_result_key('detector'): source_data},
            X_AXIS: {_make_result_key('position'): x_axis},
            Y_AXIS: {_make_result_key('temperature'): y_axis},
        }

        plotter.compute(data)
        result = plotter.get_cached_state()
        assert result is not None

    def test_forwards_title_resolver_to_renderer(self):
        """title_resolver should be forwarded to the inner renderer."""
        from ess.livedata.dashboard.plots import TitleResolver

        axis_data = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])
        source_data = make_source_data(times=[150, 250], values=[10.0, 20.0])

        axes = [AxisSpec(role=X_AXIS, name='position', bins=10)]
        plotter = CorrelationHistogramPlotter(
            axes=axes, normalize=False, renderer=_make_line_renderer()
        )

        data = {
            PRIMARY: {_make_result_key('detector'): source_data},
            X_AXIS: {_make_result_key('position'): axis_data},
        }

        resolver = TitleResolver(source=lambda _: 'Detector', output=lambda _: 'I(d)')
        plotter.compute(data, title_resolver=resolver)
        result = plotter.get_cached_state()
        assert result is not None
        assert result.label == 'Detector/I(d)'

    def test_handles_axis_data_with_variances(self):
        """Axis values with variances are usable as correlation coordinates."""
        axis_data = sc.DataArray(
            data=sc.array(
                dims=['time'],
                values=[1.0, 2.0, 3.0],
                variances=[0.1, 0.1, 0.1],
                unit='m',
            ),
            coords={'time': sc.array(dims=['time'], values=[100, 200, 300], unit='ms')},
        )
        source_data = make_source_data(times=[150, 250], values=[10.0, 20.0])

        axes = [AxisSpec(role=X_AXIS, name='position', bins=2)]
        plotter = CorrelationHistogramPlotter(
            axes=axes, normalize=False, renderer=_make_line_renderer()
        )

        plotter.compute(
            {
                PRIMARY: {_make_result_key('detector'): source_data},
                X_AXIS: {_make_result_key('position'): axis_data},
            }
        )

        assert histogram_of(plotter)['values'].sum() == 30.0

    def test_handles_datetime64_time_coords(self):
        """Correlation works when timestamps are datetime64 rather than integers."""
        base_ns = 1_000_000_000_000_000_000
        axis_data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0], unit='m'),
            coords={
                'time': sc.datetimes(
                    dims=['time'],
                    values=[base_ns + i * 1_000_000_000 for i in range(3)],
                    unit='ns',
                )
            },
        )
        source_data = sc.DataArray(
            data=sc.array(dims=['time'], values=[10.0, 20.0], unit='counts'),
            coords={
                'time': sc.datetimes(
                    dims=['time'],
                    values=[base_ns + 500_000_000, base_ns + 1_500_000_000],
                    unit='ns',
                )
            },
        )

        axes = [AxisSpec(role=X_AXIS, name='position', bins=2)]
        plotter = CorrelationHistogramPlotter(
            axes=axes, normalize=False, renderer=_make_line_renderer()
        )

        plotter.compute(
            {
                PRIMARY: {_make_result_key('detector'): source_data},
                X_AXIS: {_make_result_key('position'): axis_data},
            }
        )

        assert histogram_of(plotter)['values'].sum() == 30.0


class TestHistogramValues:
    """Exact bin edges, bin contents, units and labels of the rendered histogram.

    The correlation join, the binning and the optional rate normalization all
    produce a plausible-looking histogram when they are wrong.
    """

    def _compute_1d(
        self, normalize: bool, values: list[float]
    ) -> CorrelationHistogramPlotter:
        """Histogram ``values`` sampled 50 ms after each of three axis readings."""
        plotter = CorrelationHistogramPlotter(
            axes=[AxisSpec(role=X_AXIS, name='position', bins=2)],
            normalize=normalize,
            renderer=_make_line_renderer(),
        )
        plotter.compute(
            {
                PRIMARY: {
                    _make_result_key('detector'): make_source_data(
                        times=[150, 250, 350], values=values
                    )
                },
                X_AXIS: {
                    _make_result_key('position'): make_axis_data(
                        times=[100, 200, 300], values=[1.0, 2.0, 3.0]
                    )
                },
            },
            title_resolver=TitleResolver(output=lambda _: 'Total counts'),
        )
        return plotter

    def test_bin_edges_and_contents_are_exact(self):
        """Two bins spanning 1 m to 3 m, filled by the joined axis value."""
        plotter = self._compute_1d(normalize=False, values=[10.0, 20.0, 30.0])

        data = histogram_of(plotter)
        np.testing.assert_allclose(data['position'], [1.0, 2.0, 3.0])
        # 10 counts join to 1 m; 20 and 30 join to 2 m and 3 m, one bin.
        np.testing.assert_array_equal(data['values'], [10.0, 50.0])

    def test_axis_dimension_carries_name_unit_and_output_title(self):
        plotter = self._compute_1d(normalize=False, values=[10.0, 20.0, 30.0])

        (histogram,) = plotter.get_cached_state()
        assert histogram.kdims[0].name == 'position'
        assert histogram.kdims[0].unit == 'm'
        assert histogram.vdims[0].label == 'Total counts'
        assert histogram.vdims[0].unit == 'counts'

    def test_per_second_normalization_divides_by_the_sample_interval(self):
        """Counts become a rate over the interval to the next sample.

        The trailing sample has no successor and takes the median interval.
        Bins then average the rates they hold rather than summing them.
        """
        plotter = self._compute_1d(normalize=True, values=[10.0, 20.0, 30.0])

        (histogram,) = plotter.get_cached_state()
        # Samples are 100 ms apart: 10, 20, 30 counts are 100, 200, 300 counts/s,
        # and the upper bin averages the latter two.
        np.testing.assert_allclose(histogram.dimension_values(1), [100.0, 250.0])
        assert histogram.vdims[0].unit == 'counts/s'

    def test_2d_histogram_maps_axes_to_rows_and_columns(self):
        """Y is the row axis and X the column axis; swapping them transposes.

        Each of four points sits in its own cell, so a transposition shows up in
        the values rather than only in the axis labels.
        """
        plotter = CorrelationHistogram2dPlotter(
            CorrelationHistogram2dParams(
                bins=Bin2dParams(
                    x_axis_source='position',
                    x_bins=2,
                    y_axis_source='temperature',
                    y_bins=2,
                )
            )
        )
        times = [100, 200, 300, 400]
        plotter.compute(
            {
                PRIMARY: {
                    _make_result_key('detector'): make_source_data(
                        times=[t + 50 for t in times], values=[1.0, 2.0, 3.0, 4.0]
                    )
                },
                X_AXIS: {
                    _make_result_key('motor'): make_axis_data(
                        times=times, values=[1.0, 2.0, 1.0, 2.0]
                    )
                },
                Y_AXIS: {
                    _make_result_key('sample'): make_axis_data(
                        times=times, values=[10.0, 10.0, 20.0, 20.0], value_unit='K'
                    )
                },
            }
        )

        (image,) = plotter.get_cached_state()
        np.testing.assert_array_equal(
            image.dimension_values(2, flat=False), [[1.0, 2.0], [3.0, 4.0]]
        )
        assert [(dim.label, dim.unit) for dim in image.kdims] == [
            ('position', 'm'),
            ('temperature', 'K'),
        ]


class TestDataPrecedingAxisHistory:
    """Points without a known axis value are excluded, consistently over time."""

    axis_data = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])

    def _make_plotter(self) -> CorrelationHistogramPlotter:
        return CorrelationHistogramPlotter(
            axes=[AxisSpec(role=X_AXIS, name='position', bins=4)],
            normalize=False,
            renderer=_make_line_renderer(),
        )

    def _compute(
        self, plotter: CorrelationHistogramPlotter, source_data: sc.DataArray
    ) -> None:
        plotter.compute(
            {
                PRIMARY: {_make_result_key('detector'): source_data},
                X_AXIS: {_make_result_key('position'): self.axis_data},
            }
        )

    def test_excludes_points_before_first_axis_reading(self):
        """Points predating the axis history contribute nothing to the histogram."""
        plotter = self._make_plotter()

        # t=50 precedes the first axis reading at t=100.
        self._compute(plotter, make_source_data([50, 150, 250], [10.0, 20.0, 30.0]))

        assert histogram_of(plotter)['values'].sum() == 50.0

    def test_renders_nothing_while_all_points_precede_axis(self):
        """Fully preceding data yields no plot rather than an error or a guess."""
        plotter = self._make_plotter()

        self._compute(plotter, make_source_data([50, 60, 70], [10.0, 20.0, 30.0]))

        assert not plotter.has_cached_state()

    def test_preceding_points_never_appear_and_then_vanish(self):
        """Growing history must not retroactively remove already-shown points.

        Correlating pre-axis points against a later reading would make them
        visible until the streams overlap, at which point they would silently
        disappear again.
        """
        plotter = self._make_plotter()

        self._compute(plotter, make_source_data([50, 60], [10.0, 20.0]))
        assert not plotter.has_cached_state()

        self._compute(
            plotter, make_source_data([50, 60, 150, 250], [10.0, 20.0, 30.0, 40.0])
        )

        assert histogram_of(plotter)['values'].sum() == 70.0

    def test_uses_latest_axis_reading_at_or_before_each_point(self):
        """Each point is correlated with the axis value in effect at its time."""
        plotter = self._make_plotter()

        self._compute(plotter, make_source_data([150, 250], [10.0, 20.0]))

        histogram = histogram_of(plotter)
        # Values 1.0 and 2.0 are in effect at t=150 and t=250, so the histogram
        # spans [1, 2] with the two counts landing in the outer bins.
        edges = [1.0, 1.25, 1.5, 1.75, 2.0]
        assert histogram['position'].tolist() == pytest.approx(edges)
        assert histogram['values'].tolist() == [10.0, 0.0, 0.0, 20.0]

    def test_excludes_points_preceding_any_axis_in_2d(self):
        """With multiple axes the latest axis start determines the cutoff."""
        plotter = CorrelationHistogramPlotter(
            axes=[
                AxisSpec(role=X_AXIS, name='position', bins=2),
                AxisSpec(role=Y_AXIS, name='temperature', bins=2),
            ],
            normalize=False,
            renderer=ImagePlotter(scale_opts=PlotScaleParams2d()),
        )

        plotter.compute(
            {
                PRIMARY: {
                    _make_result_key('detector'): make_source_data(
                        [150, 250, 350, 450], [10.0, 20.0, 30.0, 40.0]
                    )
                },
                X_AXIS: {_make_result_key('position'): self.axis_data},
                # Temperature only starts at t=300, so t=150 and t=250 are excluded.
                Y_AXIS: {
                    _make_result_key('temperature'): make_axis_data(
                        times=[300, 400], values=[10.0, 20.0], value_unit='K'
                    )
                },
            }
        )

        assert np.nansum(histogram_of(plotter)['values']) == 70.0


class TestConstantAxisValues:
    """A correlation axis that does not vary must still produce a usable plot."""

    def _histogram(self, axis_values: list[float]) -> dict[str, np.ndarray]:
        """Correlate three data points against one axis reading per point."""
        plotter = CorrelationHistogramPlotter(
            axes=[AxisSpec(role=X_AXIS, name='position', bins=4)],
            normalize=False,
            renderer=_make_line_renderer(),
        )
        times = [100 * (i + 1) for i in range(len(axis_values))]
        plotter.compute(
            {
                PRIMARY: {
                    _make_result_key('detector'): make_source_data(
                        [t + 50 for t in times], [10.0] * len(times)
                    )
                },
                X_AXIS: {
                    _make_result_key('position'): make_axis_data(times, axis_values)
                },
            }
        )
        state = plotter.get_cached_state()
        assert state is not None
        return next(iter(state.values())).data

    def test_constant_axis_spans_a_visible_range(self):
        """Identical values must not collapse the bins and the axis to a point.

        ``hist`` derives edges from the value range and leaves a degenerate one
        degenerate, giving zero-width bars on a zero-width axis.
        """
        edges = self._histogram([5.0, 5.0, 5.0])['position']

        assert edges[0] < edges[-1]
        assert np.all(np.diff(edges) > 0.0)

    def test_constant_axis_keeps_all_counts(self):
        """Widening the bins must not push data outside the histogram."""
        assert self._histogram([5.0, 5.0, 5.0])['values'].sum() == 30.0

    def test_constant_axis_at_zero_spans_a_visible_range(self):
        """A value of zero has no magnitude to widen relative to."""
        edges = self._histogram([0.0, 0.0, 0.0])['position']

        assert edges[0] < 0.0 < edges[-1]

    def test_varying_axis_bins_over_the_data_range(self):
        """Edges follow the values whenever they actually span a range."""
        edges = self._histogram([1.0, 2.0, 3.0])['position']

        assert edges[0] == pytest.approx(1.0)
        assert edges[-1] == pytest.approx(3.0)


class TestCorrelationHistogramPlotterOwnership:
    """Tests for presenter ownership in CorrelationHistogramPlotter."""

    def test_presenter_owned_by_outer_plotter_not_renderer(self):
        """The presenter should be owned by the CorrelationHistogramPlotter,
        not by the inner renderer (LinePlotter/ImagePlotter).

        This is critical for SessionPlotManager's plotter replacement detection
        to work correctly with correlation plotters.
        """
        axes = [AxisSpec(role=X_AXIS, name='x', bins=10)]
        renderer = _make_line_renderer()
        plotter = CorrelationHistogramPlotter(
            axes=axes, normalize=False, renderer=renderer
        )

        # Compute some data to populate cached state
        axis_data = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])
        source_data = make_source_data(times=[150, 250], values=[10.0, 20.0])
        data = {
            PRIMARY: {_make_result_key('detector'): source_data},
            X_AXIS: {_make_result_key('position'): axis_data},
        }
        plotter.compute(data)

        presenter = plotter.create_presenter()

        # Presenter should be owned by the outer plotter, not the renderer
        assert presenter.is_owned_by(plotter)
        assert not presenter.is_owned_by(renderer)


class TestCorrelationHistogram1dPlotter:
    """Tests for CorrelationHistogram1dPlotter wrapper."""

    def test_creates_correct_axis_spec(self):
        """Verifies 1D plotter creates correct axis configuration."""
        params = CorrelationHistogram1dParams(
            bins=Bin1dParams(x_axis_source='position', x_bins=50)
        )
        plotter = CorrelationHistogram1dPlotter(params)

        assert len(plotter._axes) == 1
        assert plotter._axes[0].role == X_AXIS
        assert plotter._axes[0].name == 'position'
        assert plotter._axes[0].bins == 50

    def test_axis_source_with_output_title_becomes_dimension_name(self):
        """When x_axis_source includes output title, it becomes the histogram
        dimension name and thus the X-axis label on the plot."""
        title = 'Cave Monitor (Delta)'
        params = CorrelationHistogram1dParams(
            bins=Bin1dParams(x_axis_source=title, x_bins=10)
        )
        plotter = CorrelationHistogram1dPlotter(params)

        axis_data = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])
        source_data = make_source_data(times=[150, 250], values=[10.0, 20.0])
        data = {
            PRIMARY: {_make_result_key('detector'): source_data},
            X_AXIS: {_make_result_key('monitor'): axis_data},
        }

        plotter.compute(data)
        result = plotter.get_cached_state()
        # Overlay wraps the histogram; get the first child element
        element = next(iter(result.values()))
        assert element.kdims[0].label == title

    def test_from_params_factory(self):
        """Verifies from_params factory method works."""
        params = CorrelationHistogram1dParams()
        plotter = CorrelationHistogram1dPlotter.from_params(params)
        assert isinstance(plotter, CorrelationHistogram1dPlotter)


class TestCorrelationHistogram2dPlotter:
    """Tests for CorrelationHistogram2dPlotter wrapper."""

    def test_creates_correct_axis_specs(self):
        """Verifies 2D plotter creates correct axis configuration."""
        params = CorrelationHistogram2dParams(
            bins=Bin2dParams(
                x_axis_source='position',
                x_bins=20,
                y_axis_source='temperature',
                y_bins=30,
            )
        )
        plotter = CorrelationHistogram2dPlotter(params)

        assert len(plotter._axes) == 2
        assert plotter._axes[0].role == Y_AXIS
        assert plotter._axes[0].name == 'temperature'
        assert plotter._axes[0].bins == 30
        assert plotter._axes[1].role == X_AXIS
        assert plotter._axes[1].name == 'position'
        assert plotter._axes[1].bins == 20

    def test_from_params_factory(self):
        """Verifies from_params factory method works."""
        params = CorrelationHistogram2dParams()
        plotter = CorrelationHistogram2dPlotter.from_params(params)
        assert isinstance(plotter, CorrelationHistogram2dPlotter)
