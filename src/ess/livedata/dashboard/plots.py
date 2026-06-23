# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""This file contains utilities for creating plots in the dashboard."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import holoviews as hv
import numpy as np
import scipp as sc
from bokeh.models import TeeHead
from holoviews.core.util import range_pad
from holoviews.plotting.util import get_axis_padding

from ess.livedata.config.workflow_spec import ResultKey
from ess.livedata.core.timestamp import Timestamp

from .data_roles import PRIMARY
from .plot_params import (
    LayoutParams,
    PlotAspect,
    PlotDisplayParams1d,
    PlotParams1d,
    PlotParams2d,
    PlotParamsBars,
    PlotParamsTimeseries,
    PlotScale,
    PlotScaleParams,
    PlotScaleParams2d,
    TickParams,
    TimeseriesDownsamplingParams,
)
from .range_hook import Axis, RangeTargets
from .scipp_to_holoviews import HvConverter1d, to_holoviews
from .time_utils import format_time_ns_local
from .timeseries_downsample import downsample_timeseries


def _latest_time_ns(primary: dict[ResultKey, sc.DataArray]) -> int | None:
    """Latest time-coord value across the primary dict, as int64 nanoseconds.

    Caller is the timeseries plotter, so each DataArray has a non-empty
    datetime64 ``time`` coord (guaranteed by FullHistoryExtractor).
    """
    if not primary:
        return None
    return max(
        int(np.datetime64(da.coords['time'].values[-1], 'ns').astype('int64'))
        for da in primary.values()
    )


# Used only to widen a zero-width range (see _ensure_span); the per-axis
# padding fraction itself comes from HoloViews (see _hv_axis_padding).
_DEGENERATE_PAD = 0.05
_DEGENERATE_PAD_MIN = 0.5
_DEGENERATE_LOG_FACTOR = 1.1


def _hv_axis_padding(element_type: type) -> tuple[float, float, float]:
    """Per-axis ``(xpad, ypad, zpad)`` padding HoloViews applies for an element.

    Sourced from the element's registered Bokeh plot class so autoscale ranges
    frame the data exactly as HoloViews would: images pad nothing, curves pad
    only y, histograms and scatter pad both axes. The values are fractions of
    the data span, matching how :func:`range_pad` interprets them.
    """
    plot_cls = hv.Store.registry['bokeh'][element_type]
    return get_axis_padding(plot_cls.param.padding.default)


def _ensure_span(lo: float, hi: float, *, log: bool) -> tuple[float, float]:
    """Widen a zero-width range so Bokeh has something to render.

    ``range_pad`` derives padding from the span, so it leaves a single-valued
    range (constant image, single point or bin) untouched; HoloViews handles
    this separately via ``default_span``. Bump multiplicatively on log axes to
    keep the lower bound positive, additively on linear axes.
    """
    if hi != lo:
        return lo, hi
    if log:
        return lo / _DEGENERATE_LOG_FACTOR, hi * _DEGENERATE_LOG_FACTOR
    offset = max(abs(lo) * _DEGENERATE_PAD, _DEGENERATE_PAD_MIN)
    return lo - offset, hi + offset


def _pad_range(lo: float, hi: float, *, pad: float, log: bool) -> tuple[float, float]:
    """Pad ``(lo, hi)`` by fraction ``pad`` using HoloViews' range padding.

    ``pad`` is the per-axis fraction HoloViews assigns to the element type (see
    :func:`_hv_axis_padding`); :func:`range_pad` applies it in data space, or in
    log space when ``log=True``. ``pad=0`` (e.g. every image axis) is a no-op
    beyond the zero-width guard.
    """
    lo, hi = _ensure_span(lo, hi, log=log)
    return range_pad(lo, hi, pad, log)


def _bounds_for_log(
    bounds: tuple[float, float], *, log: bool
) -> tuple[float, float] | None:
    """Validate explicit ``(lo, hi)`` bounds for an axis.

    Returns ``None`` if any bound is non-finite, or if ``log=True`` and the
    lower bound is non-positive — log axes require strictly positive bounds.
    """
    lo, hi = bounds
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    if log and lo <= 0.0:
        return None
    return lo, hi


def _finite_min_max(
    values: np.ndarray, *, log: bool = False
) -> tuple[float, float] | None:
    """Return finite (min, max) of ``values``, or ``None`` if none qualify.

    Parameters
    ----------
    values:
        Input array.
    log:
        If True, also drop non-positive values. ``LogColorMapper`` and
        Bokeh's log axes reject non-positive bounds, so any padding derived
        from a non-positive ``lo`` would break rendering.
    """
    if values.size == 0:
        return None
    mask = np.isfinite(values)
    if log:
        mask &= values > 0
    finite = values[mask]
    if finite.size == 0:
        return None
    return float(finite.min()), float(finite.max())


def _value_extent_with_errors(
    data: sc.DataArray, *, show_errors: bool, log: bool
) -> tuple[float, float] | None:
    """Finite (min, max) of the y values, widened to the error whiskers.

    When error bars / bands are shown they extend the rendered y-range to
    ``value ± stddev`` (see ``HvConverter1d.error_bars`` / ``spread``), so the
    extent must include those whiskers or autoscale would clip them. Mirrors
    the range HoloViews itself derives for an ``ErrorBars`` / ``Spread``
    element. Returns ``None`` when no value qualifies (see
    :func:`_finite_min_max`).
    """
    values = data.values
    if show_errors and data.variances is not None:
        std = np.sqrt(data.variances)
        values = np.concatenate(
            [values.ravel(), (values - std).ravel(), (values + std).ravel()]
        )
    return _finite_min_max(values, log=log)


def _normalize_to_rate(da: sc.DataArray) -> sc.DataArray:
    """Normalize data to rate (per second) using start_time/end_time coords.

    Only applies to data with unit 'counts'. Returns the input unchanged if
    the unit is not 'counts', the required time coordinates are missing, or
    the duration is not positive.
    """
    if da.unit != 'counts':
        return da
    if 'start_time' not in da.coords or 'end_time' not in da.coords:
        return da
    duration_s = (da.coords['end_time'] - da.coords['start_time']).to(
        unit='s', dtype='float64'
    )
    if duration_s.value <= 0:
        return da
    return da / duration_s


def _identity(x: str) -> str:
    return x


def _typed_opts(element_types: Iterable[type], **options: Any) -> list[hv.Options]:
    """Build per-element-type ``Options`` for declaring style once on a DynamicMap.

    ``responsive`` and most plot options are invalid on ``Layout``, so styling
    hoisted onto a DynamicMap must be keyed to concrete leaf element types rather
    than applied generically.
    """
    return [getattr(hv.opts, t.__name__)(**options) for t in element_types]


@dataclass(frozen=True)
class TitleResolver:
    """Resolves raw source and output names to human-readable display titles.

    Parameters
    ----------
    source:
        Maps raw source names to display titles.
    output:
        Maps raw output names to display titles.
    dim:
        Maps coord/dim names to display titles for plot axis labels.
    include_output_in_label:
        Whether to include the output name in legend labels. When all layers
        in a cell share the same output, the output is already on the Y-axis
        and repeating it in the legend is redundant.
    """

    source: Callable[[str], str] = _identity
    output: Callable[[str], str] = _identity
    dim: Callable[[str], str] = _identity
    include_output_in_label: bool = True

    def get_legend_label(self, source_name: str, output_name: str) -> str:
        """Build the legend label for a data layer."""
        source = self.source(source_name)
        if self.include_output_in_label:
            return f'{source}/{self.output(output_name)}'
        return source

    def get_axis_label(self, output_name: str) -> str:
        """Resolve an output name to a display title for axis labels."""
        return self.output(output_name)


class PresenterBase:
    """
    Base class for presenters with dirty flag tracking.

    Tracks whether new data has been computed since the last time this
    presenter consumed an update. This enables efficient polling-based
    update detection in multi-session scenarios.

    Parameters
    ----------
    plotter:
        The plotter that creates and manages this presenter's state.
    owner:
        Optional "logical owner" for identity checks. Used when a plotter
        delegates presenter creation to an inner renderer but wants to be
        recognized as the owner for lifecycle management (e.g., detecting
        plotter replacement). Defaults to plotter if not specified.
    """

    def __init__(self, plotter: Plotter, *, owner: Plotter | None = None) -> None:
        self._plotter = plotter
        self._owner = owner
        self._dirty: bool = False

    def is_owned_by(self, plotter: Plotter) -> bool:
        """Check if this presenter is owned by the given plotter."""
        owner = self._owner if self._owner is not None else self._plotter
        return owner is plotter

    def _mark_dirty(self) -> None:
        """Mark this presenter as having a pending update."""
        self._dirty = True

    def has_pending_update(self) -> bool:
        """Check if there is a pending update to present."""
        return self._dirty

    def consume_update(self) -> Any:
        """
        Consume the pending update and return the cached state.

        Clears the dirty flag and returns the plotter's cached state.
        """
        self._dirty = False
        return self._plotter.get_cached_state()

    def present(self, pipe: hv.streams.Pipe) -> hv.DynamicMap | hv.Element:
        """Create a DynamicMap or Element for this session from a data pipe."""
        raise NotImplementedError("Subclasses must implement present()")


class DefaultPresenter(PresenterBase):
    """
    Default presenter for standard plotters.

    Pipe receives pre-computed HoloViews elements from PlotDataService.
    DynamicMap just passes through the data - no computation per-session.

    Plotters requiring interactive controls (kdims) must override
    create_presenter() to return a custom presenter.
    """

    def present(self, pipe: hv.streams.Pipe) -> hv.DynamicMap:
        """Create a DynamicMap that passes through pre-computed elements.

        Static styling is applied once on the DynamicMap (see
        :meth:`Plotter.style_opts`) rather than per element on every tick.
        """

        def passthrough(data):
            return data

        dmap = hv.DynamicMap(passthrough, streams=[pipe], cache_size=1)
        return dmap.opts(*self._plotter.style_opts())


class StaticPresenter(PresenterBase):
    """
    Presenter for static plots. Returns the element directly.

    Static plots (rectangles, lines, etc.) don't need DynamicMaps since their
    content doesn't change. The pipe.data contains the pre-computed hv.Element
    which is returned as-is.
    """

    def present(self, pipe: hv.streams.Pipe) -> hv.Element:
        """Return the static element from the pipe data."""
        return pipe.data


@dataclass(frozen=True)
class TimeBounds:
    """Data-time bounds backing the freshness/lag indicator.

    ``min_end`` is the end time of the oldest data, giving worst-case staleness.
    ``min_start`` and ``max_end`` bound the full data range and are present only
    when start times exist on the data. ``created_at`` is the wall-clock instant
    these bounds were computed, anchoring the two distinct freshness measures:
    data age (``now`` minus the data's end time, the headline indicator) and
    pipeline lag (frozen at compute time, the per-layer diagnostic detail).
    """

    min_end: Timestamp
    created_at: Timestamp
    min_start: Timestamp | None = None
    max_end: Timestamp | None = None

    def age_seconds(self, now: Timestamp | None = None) -> float:
        """Wall-clock age of the oldest displayed data (``now`` minus its end).

        Recomputed against a shared ``now`` so ages are comparable across plots
        and grow while a stream is stalled -- the headline freshness indicator.
        """
        now = Timestamp.now() if now is None else now
        return (now - self.min_end).to_seconds()

    def lag_seconds(self) -> float:
        """Pipeline latency: oldest data's end time to when it was plotted.

        Frozen at compute time -- the reduction's processing delay, anchored to
        each plot's own render moment and so not comparable across plots.
        """
        return (self.created_at - self.min_end).to_seconds()


def _compute_time_bounds(data: dict[str, sc.DataArray]) -> TimeBounds | None:
    """
    Extract time bounds from start_time/end_time coordinates.

    Returns the earliest start_time, the earliest end_time (oldest data, for
    worst-case lag), and the latest end_time across all DataArrays. Returns None
    if no end_time coordinate is found.
    """
    min_start: Timestamp | None = None
    min_end: Timestamp | None = None
    max_end: Timestamp | None = None

    for da in data.values():
        if 'start_time' in da.coords:
            start = Timestamp.from_scipp(da.coords['start_time'])
            min_start = start if min_start is None else min(min_start, start)
        if 'end_time' in da.coords:
            end = Timestamp.from_scipp(da.coords['end_time'])
            min_end = end if min_end is None else min(min_end, end)
            max_end = end if max_end is None else max(max_end, end)

    if min_end is None:
        return None
    return TimeBounds(
        min_end=min_end,
        created_at=Timestamp.now(),
        min_start=min_start,
        max_end=max_end,
    )


def merge_time_bounds(bounds: Iterable[TimeBounds | None]) -> TimeBounds | None:
    """Combine bounds across layers: earliest start, oldest end, latest end.

    ``min_end`` is the oldest across layers, so the cell's age reflects its most
    stale layer (worst case). ``created_at`` takes the earliest to match.
    """
    present = [b for b in bounds if b is not None]
    if not present:
        return None
    starts = [b.min_start for b in present if b.min_start is not None]
    ends = [b.max_end for b in present if b.max_end is not None]
    return TimeBounds(
        min_end=min(b.min_end for b in present),
        created_at=min(b.created_at for b in present),
        min_start=min(starts) if starts else None,
        max_end=max(ends) if ends else None,
    )


def format_time_info(bounds: TimeBounds) -> str:
    """Format time bounds as a range + lag string, e.g. "12:34:56 (Lag: 2.3s)".

    ``Lag`` is the frozen pipeline latency, not wall-clock staleness.
    """
    lag_s = bounds.lag_seconds()
    if bounds.min_start is not None and bounds.max_end is not None:
        start_str = format_time_ns_local(bounds.min_start)
        end_str = format_time_ns_local(bounds.max_end)
        return f'{start_str} - {end_str} (Lag: {lag_s:.1f}s)'
    return f'{format_time_ns_local(bounds.min_end)} (Lag: {lag_s:.1f}s)'


class Plotter:
    """
    Base class for plots that support autoscaling.

    Tracks presenters via WeakSet and marks them dirty when state changes.
    This enables efficient polling-based update detection.
    """

    AUTOSCALE_AXES: ClassVar[frozenset[Axis]] = frozenset()
    """Per-axis autoscale support. Override per subclass."""

    def __init__(
        self,
        *,
        aspect_params: PlotAspect | None = None,
        layout_params: LayoutParams | None = None,
        normalize_to_rate: bool = False,
    ):
        """
        Initialize the plotter.

        Parameters
        ----------
        layout_params:
            Layout parameters for combining multiple datasets. If None, uses defaults.
        normalize_to_rate:
            If True, normalize counts data to rate (counts/s) using
            start_time/end_time coordinates before plotting.
        """
        self._normalize_to_rate = normalize_to_rate
        self._cached_state: Any | None = None
        self._time_bounds: TimeBounds | None = None
        self._range_targets: dict[ResultKey, RangeTargets] = {}
        self._presenters: weakref.WeakSet[PresenterBase] = weakref.WeakSet()
        self.layout_params = layout_params or LayoutParams()
        aspect_params = aspect_params or PlotAspect()

        # All non-free aspect types are enforced by a JS hook
        # (see frame_aspect.py) that adjusts the Bokeh figure dimensions.
        # HoloViews' own aspect/data_aspect opts are not set — they conflict
        # with responsive mode in Panel containers (upstream bug).
        self._sizing_opts: dict[str, Any] = {'responsive': True}

    @staticmethod
    def _make_tick_opts(tick_params: TickParams | None) -> dict[str, Any]:
        """
        Create tick options from TickParams.

        Parameters
        ----------
        tick_params:
            Tick configuration parameters.

        Returns
        -------
        :
            Dictionary of tick options for HoloViews plots.
        """
        if tick_params is None:
            return {}
        opts: dict[str, Any] = {}
        if tick_params.custom_xticks:
            opts['xticks'] = tick_params.xticks
        if tick_params.custom_yticks:
            opts['yticks'] = tick_params.yticks
        return opts

    @staticmethod
    def _make_2d_base_opts(
        scale_opts: PlotScaleParams2d, tick_params: TickParams | None = None
    ) -> dict[str, Any]:
        """
        Create base options for 2D image plots.

        Parameters
        ----------
        scale_opts:
            Scaling options for axes and color.
        tick_params:
            Tick configuration parameters.

        Returns
        -------
        :
            Dictionary of base options for HoloViews image plots.
        """
        opts: dict[str, Any] = {
            'colorbar': True,
            'cmap': 'viridis',
            'logx': scale_opts.x_scale == PlotScale.log,
            'logy': scale_opts.y_scale == PlotScale.log,
            'logz': scale_opts.color_scale == PlotScale.log,
        }
        if tick_params is not None:
            if tick_params.custom_xticks:
                opts['xticks'] = tick_params.xticks
            if tick_params.custom_yticks:
                opts['yticks'] = tick_params.yticks
        return opts

    @staticmethod
    def _convert_bin_edges_to_midpoints(
        data: sc.DataArray, dim: str | None = None
    ) -> sc.DataArray:
        """
        Convert bin-edge coordinates to midpoints for curve plotting.

        Histograms with many narrow bins don't display well - the black outlines
        dominate. Converting to midpoint coordinates yields a Curve instead.

        Parameters
        ----------
        data:
            DataArray that may have bin-edge coordinates.
        dim:
            Dimension to convert. If None, uses the single dimension of 1D data.

        Returns
        -------
        :
            DataArray with midpoint coordinates if edges were present.
        """
        if dim is None:
            dim = data.dim
        if dim in data.coords and data.coords.is_edges(dim):
            return data.assign_coords({dim: sc.midpoints(data.coords[dim])})
        return data

    @staticmethod
    def _prepare_2d_image_data(data: sc.DataArray, use_log_scale: bool) -> sc.DataArray:
        """
        Convert to float64 and mask non-positive values for log scale.

        With logz=True we need to exclude zero values: The value bounds
        calculation should properly adjust the color limits. Since zeros can never
        be included we want to adjust to the lowest positive value.

        Parameters
        ----------
        data:
            Input data array.
        use_log_scale:
            Whether to apply log scale masking.

        Returns
        -------
        :
            Prepared data array with appropriate dtype and masking.
        """
        plot_data = data.to(dtype='float64')
        if use_log_scale:
            plot_data = plot_data.assign(
                sc.where(
                    plot_data.data <= sc.scalar(0.0, unit=plot_data.unit),
                    sc.scalar(np.nan, unit=plot_data.unit, dtype=plot_data.dtype),
                    plot_data.data,
                )
            )
        return plot_data

    @staticmethod
    def _get_log_scale_clim(data: sc.DataArray) -> tuple[float, float] | None:
        """
        Return fallback clim for log scale if data is all NaN.

        HoloViews' LogColorMapper fails when color_mapper.low is None (which
        happens when all data is NaN). This provides explicit bounds to avoid
        the "TypeError: '>' not supported between instances of 'NoneType' and 'int'"
        error in _draw_colorbar. Limits are not returned if data is not 'None' as in
        this case we let Holoviews handle the bounds.

        Parameters
        ----------
        data:
            Data array (after log scale masking, may contain NaN).

        Returns
        -------
        :
            Tuple of (low, high) bounds, or None if data has valid positive values.
        """
        vmin = float(data.data.nanmin().value)
        vmax = float(data.data.nanmax().value)
        # If all NaN, nanmin/nanmax return nan
        if np.isnan(vmin) or np.isnan(vmax) or vmax <= vmin:
            # Return placeholder bounds for empty/invalid data
            return (1.0, 10.0)
        return None

    def compute(
        self,
        data: dict[str, dict[ResultKey, sc.DataArray]],
        *,
        title_resolver: TitleResolver | None = None,
        **kwargs,
    ) -> None:
        """
        Compute plot elements from input data and cache the result.

        This is Stage 1 of the two-stage plotter architecture. It transforms
        input data into HoloViews elements, caches the result, and marks all
        registered presenters as dirty.

        Parameters
        ----------
        data:
            Role-grouped data. Standard plotters consume the ``primary`` role.
        title_resolver:
            Resolves source/output names to display titles. If None, raw names
            are used.
        **kwargs:
            Additional keyword arguments passed to plot().
        """
        data = data.get(PRIMARY, {})
        if self._normalize_to_rate:
            data = {key: _normalize_to_rate(da) for key, da in data.items()}

        self._range_targets = {}
        resolver = title_resolver or TitleResolver()
        plots: list[hv.Element] = []
        try:
            for plot_index, (data_key, da) in enumerate(data.items()):
                label = resolver.get_legend_label(
                    data_key.job_id.source_name, data_key.output_name
                )
                output_display_name = resolver.get_axis_label(data_key.output_name)
                source_display_name = resolver.source(data_key.job_id.source_name)
                plot_element = self.plot(
                    da,
                    data_key,
                    label=label,
                    source_display_name=source_display_name,
                    output_display_name=output_display_name,
                    dim_label=resolver.dim,
                    plot_index=plot_index,
                    **kwargs,
                )
                plots.append(plot_element)
        except Exception as e:
            self._range_targets = {}
            plots = [
                hv.Text(0.5, 0.5, f"Error: {e}").opts(
                    text_align='center', text_baseline='middle', **self._sizing_opts
                )
            ]

        if len(plots) == 0:
            plots = [
                hv.Text(0.5, 0.5, 'No data').opts(
                    text_align='center', text_baseline='middle', **self._sizing_opts
                )
            ]

        if self.layout_params.combine_mode == 'overlay':
            result = hv.Overlay(plots)
        elif len(plots) == 1:
            result = plots[0]
        else:
            result = hv.Layout(plots).cols(self.layout_params.layout_columns)

        # Time bounds drive the cell titlebar's freshness indicator; they are
        # kept off the plot title to avoid minting an OptionTree entry per tick.
        self._time_bounds = _compute_time_bounds(data)
        self._set_cached_state(result)

    def create_presenter(self, *, owner: Plotter | None = None) -> PresenterBase:
        """
        Create a presenter for this plotter.

        Stage 2 of the two-stage architecture. Returns a fresh presenter
        instance that can be used to create session-bound DynamicMaps.
        The presenter is registered with this plotter and will be marked
        dirty when compute() produces new state.

        Override this method in subclasses that need custom presenters
        (e.g., ROI plotters with edit streams).

        Parameters
        ----------
        owner:
            Optional "logical owner" for identity checks. Used when a plotter
            delegates presenter creation to an inner renderer. Defaults to self.

        Returns
        -------
        :
            A presenter instance for this plotter.
        """
        presenter = DefaultPresenter(self, owner=owner)
        self._presenters.add(presenter)
        return presenter

    def _set_cached_state(self, state: Any) -> None:
        """Store computed state and mark all presenters dirty."""
        self._cached_state = state
        self.mark_presenters_dirty()

    def mark_presenters_dirty(self) -> None:
        """Mark all registered presenters as having pending updates."""
        # Convert to list to avoid RuntimeError if WeakSet is modified during iteration
        # (e.g., by garbage collector removing dead references)
        for presenter in list(self._presenters):
            presenter._mark_dirty()

    def get_cached_state(self) -> Any | None:
        """Get the last computed state, or None if not yet computed."""
        return self._cached_state

    @property
    def time_bounds(self) -> TimeBounds | None:
        """Time bounds of the most recently computed data, or None if absent."""
        return self._time_bounds

    def has_cached_state(self) -> bool:
        """Check if state has been computed."""
        return self._cached_state is not None

    def get_range_targets(self, data_key: ResultKey) -> RangeTargets | None:
        """Per-axis ``(lo, hi)`` targets computed at the last ``compute()``.

        Returns ``None`` when no targets have been computed for ``data_key``
        (e.g. for plotters whose ``AUTOSCALE_AXES`` is empty, or before the
        first ``compute()`` call).
        """
        return self._range_targets.get(data_key)

    def iter_range_targets(self) -> Iterator[tuple[ResultKey, RangeTargets]]:
        """Iterate ``(data_key, targets)`` pairs computed at the last ``compute()``.

        Empty when no ``compute()`` has happened yet or when the plotter's
        ``AUTOSCALE_AXES`` is empty.
        """
        return iter(self._range_targets.items())

    def style_opts(self) -> list[hv.Options]:
        """Static HoloViews opts applied once on the presenter's DynamicMap.

        Hoisting styling here keeps ``compute()`` a pure data->element transform:
        the per-tick build emits bare elements and styling is declared once at
        render time, rather than minting an entry in the process-global option
        store for every element on every tick. Subclasses extend this with opts
        for their leaf element types; the base provides the container-level opts.
        """
        # title='' stops Bokeh promoting a single overlaid element's label to the
        # plot title when there is no legend to carry it.
        return [
            hv.opts.Overlay(shared_axes=True, title=''),
            hv.opts.Layout(shared_axes=False),
        ]

    def plot(
        self, data: sc.DataArray, data_key: ResultKey, *, label: str = '', **kwargs
    ) -> Any:
        """Create a plot from the given data.

        Override this method for plotters that use the default compute() flow.
        Plotters that override compute() entirely don't need to implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement plot() or override compute()"
        )


_LINE1D_BASE_METHOD: dict[str, str] = {
    'line': 'curve',
    'points': 'scatter',
    'histogram': 'histogram',
}
_LINE1D_ERROR_METHOD: dict[str, str] = {
    'bars': 'error_bars',
    'band': 'spread',
}
# HoloViews element each mode renders as, for sourcing axis padding.
_LINE1D_ELEMENT: dict[str, type] = {
    'line': hv.Curve,
    'points': hv.Scatter,
    'histogram': hv.Histogram,
}
_LINE1D_HISTOGRAM_FALLBACK = 'line'
# Every leaf element type a 1-D line plot may render (base modes plus error
# displays), for declaring style opts once on the DynamicMap.
_LINE1D_LEAF_ELEMENTS: tuple[type, ...] = (
    hv.Curve,
    hv.Scatter,
    hv.Histogram,
    hv.ErrorBars,
    hv.Spread,
)


def _color_error_element(el: hv.Element, color: Any) -> hv.Element:
    """Apply ``color`` to an error element, including ``ErrorBars`` endcaps.

    HoloViews maps ``color`` to the Whisker body line only; its endcaps are
    separate ``TeeHead`` glyphs whose ``line_color`` otherwise stays black.
    ``Spread`` (a subclass of ``ErrorBars``) renders as a filled band with no
    endcaps, so it takes the plain ``color`` path.
    """
    if type(el) is hv.ErrorBars:
        return el.opts(
            color=color,
            upper_head=TeeHead(line_color=color),
            lower_head=TeeHead(line_color=color),
        )
    return el.opts(color=color)


def _resolve_line1d_mode(
    mode: str, data: sc.DataArray, dim: str | None = None
) -> tuple[str, sc.DataArray]:
    """Return (actual_mode, data) with bin edges converted to midpoints if needed.

    When mode is 'histogram' but the data has no bin-edge coordinate on ``dim``,
    falls back to 'line'. When mode is not 'histogram', bin edges are always
    converted to midpoints.
    """
    if dim is None:
        dim = data.dim
    has_edges = dim in data.coords and data.coords.is_edges(dim)
    if mode == 'histogram' and has_edges:
        return 'histogram', data
    actual_mode = mode if mode != 'histogram' else _LINE1D_HISTOGRAM_FALLBACK
    if has_edges:
        data = data.assign_coords({dim: sc.midpoints(data.coords[dim])})
    return actual_mode, data


class LinePlotter(Plotter):
    """Plotter for 1D plots from scipp DataArrays.

    Supports line, scatter, and histogram rendering modes with optional
    error display (bars, band, or none).
    """

    AUTOSCALE_AXES: ClassVar[frozenset[Axis]] = frozenset({'x', 'y'})

    def __init__(
        self,
        scale_opts: PlotScaleParams,
        tick_params: TickParams | None = None,
        *,
        mode: str = 'line',
        errors: str = 'bars',
        **kwargs,
    ):
        """
        Initialize the line plotter.

        Parameters
        ----------
        scale_opts:
            Scaling options for axes.
        tick_params:
            Tick configuration parameters.
        mode:
            Rendering mode: 'line', 'points', or 'histogram'.
        errors:
            Error display mode: 'bars', 'band', or 'none'.
        **kwargs:
            Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._mode = mode
        self._errors = errors
        self._logx = scale_opts.x_scale == PlotScale.log
        self._logy = scale_opts.y_scale == PlotScale.log
        self._colors = hv.Cycle.default_cycles["default_colors"]
        self._base_opts: dict[str, Any] = {
            'logx': self._logx,
            'logy': self._logy,
            **self._make_tick_opts(tick_params),
        }
        self._downsampling: TimeseriesDownsamplingParams | None = None
        self._last_compute_data_time_ns: int | None = None

    @classmethod
    def from_display_params(
        cls, params: PlotDisplayParams1d, *, normalize_to_rate: bool = False
    ):
        """Create LinePlotter from display parameters."""
        return cls(
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
            tick_params=params.ticks,
            mode=params.line.mode,
            errors=params.line.errors,
            normalize_to_rate=normalize_to_rate,
        )

    @classmethod
    def from_params(cls, params: PlotParams1d):
        """Create LinePlotter from PlotParams1d."""
        return cls.from_display_params(
            params, normalize_to_rate=params.rate.normalize_to_rate
        )

    @classmethod
    def from_timeseries_params(cls, params: PlotParamsTimeseries):
        """Create LinePlotter for the timeseries plotter, with downsampling on.

        Downsampling and update throttling live at the plotter rather than at
        the extractor: the subscription still pulls the full-history, and
        per-plot config can change without re-subscribing.
        """
        instance = cls.from_display_params(params)
        instance._downsampling = params.downsampling
        return instance

    def compute(
        self,
        data: dict[str, dict[ResultKey, sc.DataArray]],
        *,
        title_resolver: TitleResolver | None = None,
        **kwargs,
    ) -> None:
        """Compute plot state, with timeseries throttling and downsampling.

        For non-timeseries instances (``_downsampling is None``) this just
        delegates to ``Plotter.compute``.

        For timeseries instances the call short-circuits when the new data's
        latest timestamp is less than ``fine_period_seconds`` past the timestamp
        seen at the last compute that actually ran. Returning early skips
        ``_set_cached_state``, which leaves presenters non-dirty and prevents
        the downstream ``pipe.send`` / Bokeh patch / WebSocket flush /
        browser repaint. When the call does proceed, every primary DataArray
        is reduced via ``downsample_timeseries`` before delegating to the
        standard plot path.
        """
        if self._downsampling is None:
            super().compute(data, title_resolver=title_resolver, **kwargs)
            return
        primary = data.get(PRIMARY, {})
        latest_ns = _latest_time_ns(primary)
        if self._should_skip_for_throttle(latest_ns):
            return
        downsampled = {
            key: downsample_timeseries(
                da,
                fine_period_seconds=self._downsampling.fine_period_seconds,
                recent_seconds=self._downsampling.recent_seconds,
                coarse_period_seconds=self._downsampling.coarse_period_seconds,
            )
            for key, da in primary.items()
        }
        data_for_super = {**data, PRIMARY: downsampled}
        super().compute(data_for_super, title_resolver=title_resolver, **kwargs)
        if latest_ns is not None:
            self._last_compute_data_time_ns = latest_ns

    def _should_skip_for_throttle(self, latest_ns: int | None) -> bool:
        if (
            latest_ns is None
            or self._last_compute_data_time_ns is None
            or self._downsampling is None
        ):
            return False
        period_ns = int(self._downsampling.fine_period_seconds * 1e9)
        return (latest_ns - self._last_compute_data_time_ns) < period_ns

    def _compute_line_range_targets(
        self, data: sc.DataArray, mode: str
    ) -> RangeTargets:
        """Per-axis ``(lo, hi)`` targets for the given 1-D data."""
        xpad, ypad, _ = _hv_axis_padding(_LINE1D_ELEMENT[mode])
        targets: RangeTargets = {}
        dim = data.dim
        if dim in data.coords:
            coord_values = data.coords[dim].values
            coord_extent = _finite_min_max(coord_values, log=self._logx)
            if coord_extent is not None:
                targets['x'] = _pad_range(*coord_extent, pad=xpad, log=self._logx)
        value_extent = _value_extent_with_errors(
            data, show_errors=self._errors != 'none', log=self._logy
        )
        if value_extent is not None:
            targets['y'] = _pad_range(*value_extent, pad=ypad, log=self._logy)
        return targets

    def plot(
        self,
        data: sc.DataArray,
        data_key: ResultKey,
        *,
        label: str = '',
        output_display_name: str = '',
        dim_label: Callable[[str], str] | None = None,
        plot_index: int = 0,
        **kwargs,
    ) -> hv.Element | hv.Overlay:
        """Create a 1D plot from a scipp DataArray."""
        mode, da = _resolve_line1d_mode(self._mode, data)
        targets = self._compute_line_range_targets(data, mode)
        if targets:
            self._range_targets[data_key] = targets
        converter = HvConverter1d(
            da, value_label=output_display_name, dim_label=dim_label
        )
        base_method = getattr(converter, _LINE1D_BASE_METHOD[mode])
        base = base_method(label=label)

        if da.variances is not None and self._errors != 'none':
            # An error element must be colored explicitly to match its line and
            # to color its endcaps; this picks a distinct cycle color per source
            # but forfeits HoloViews' cross-overlay auto-cycling. Error-free lines
            # keep no explicit color so auto-cycling still distinguishes sources
            # across independently overlaid layers.
            color = self._colors[plot_index % len(self._colors)]
            base = base.opts(color=color)
            if mode == 'histogram':
                # Error elements need midpoint coords (N values, not N+1 edges)
                converter = HvConverter1d(
                    da.assign_coords({da.dim: sc.midpoints(da.coords[da.dim])}),
                    value_label=output_display_name,
                    dim_label=dim_label,
                )
            error_method = getattr(converter, _LINE1D_ERROR_METHOD[self._errors])
            error = _color_error_element(error_method(label=label), color)
            return base * error

        return base

    def style_opts(self) -> list[hv.Options]:
        return [
            *_typed_opts(_LINE1D_LEAF_ELEMENTS, **self._base_opts, **self._sizing_opts),
            *super().style_opts(),
        ]


class ImagePlotter(Plotter):
    """Plotter for 2D images from scipp DataArrays."""

    AUTOSCALE_AXES: ClassVar[frozenset[Axis]] = frozenset({'x', 'y', 'c'})

    def __init__(
        self,
        scale_opts: PlotScaleParams2d,
        tick_params: TickParams | None = None,
        **kwargs,
    ):
        """
        Initialize the image plotter.

        Parameters
        ----------
        scale_opts:
            Scaling options for axes and color.
        tick_params:
            Tick configuration parameters.
        **kwargs:
            Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._scale_opts = scale_opts
        self._base_opts = self._make_2d_base_opts(scale_opts, tick_params)

    @classmethod
    def from_params(cls, params: PlotParams2d):
        """Create ImagePlotter from PlotParams2d."""
        rate = getattr(params, 'rate', None)
        return cls(
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
            tick_params=params.ticks,
            normalize_to_rate=rate.normalize_to_rate if rate is not None else False,
        )

    def _compute_image_range_targets(
        self,
        element: hv.Element,
        plot_data: sc.DataArray,
        use_log_scale: bool,
    ) -> RangeTargets:
        """Per-axis ``(lo, hi)`` targets for the rendered 2-D image element.

        ``x``/``y`` extents are read from ``element.range`` so they match the
        bounds HoloViews actually renders -- half-pixel extension for
        midpoint coords, exact edge values for bin edges -- without
        duplicating that logic here. ``c`` extent is derived from the data
        because HoloViews does not expose a NaN-filtered or log-filtered
        value range.
        """
        xpad, ypad, cpad = _hv_axis_padding(hv.Image)
        targets: RangeTargets = {}
        if plot_data.ndim == 2:
            logx = self._scale_opts.x_scale == PlotScale.log
            logy = self._scale_opts.y_scale == PlotScale.log
            if x_extent := _bounds_for_log(element.range(0), log=logx):
                targets['x'] = _pad_range(*x_extent, pad=xpad, log=logx)
            if y_extent := _bounds_for_log(element.range(1), log=logy):
                targets['y'] = _pad_range(*y_extent, pad=ypad, log=logy)
        extent = _finite_min_max(plot_data.values, log=use_log_scale)
        if extent is not None:
            targets['c'] = _pad_range(*extent, pad=cpad, log=use_log_scale)
        return targets

    def plot(
        self,
        data: sc.DataArray,
        data_key: ResultKey,
        *,
        label: str = '',
        output_display_name: str = '',
        dim_label: Callable[[str], str] | None = None,
        **kwargs,
    ) -> hv.Image:
        """Create a 2D plot from a scipp DataArray."""
        # Prepare data with appropriate dtype and log scale masking
        use_log_scale = self._scale_opts.color_scale == PlotScale.log
        plot_data = self._prepare_2d_image_data(data, use_log_scale)
        if output_display_name:
            plot_data.name = output_display_name

        # We are using the masked data here since Holoviews (at least with the Bokeh
        # backend) show values below the color limits with the same color as the lowest
        # value in the colormap, which is not what we want for, e.g., zeros on a log
        # scale plot. The nan values will be shown as transparent.
        histogram = to_holoviews(plot_data, label=label, dim_label=dim_label)

        targets = self._compute_image_range_targets(histogram, plot_data, use_log_scale)
        if targets:
            self._range_targets[data_key] = targets

        # base_opts are declared once in style_opts(); only the data-dependent clim
        # guard (log scale with all-NaN data) must be set per element here.
        if use_log_scale and (clim := self._get_log_scale_clim(plot_data)) is not None:
            return histogram.opts(clim=clim)
        return histogram

    def style_opts(self) -> list[hv.Options]:
        return [
            *_typed_opts(
                (hv.Image, hv.QuadMesh), **self._base_opts, **self._sizing_opts
            ),
            *super().style_opts(),
        ]


class BarsPlotter(Plotter):
    """Plotter for bar charts of 0D scalar data."""

    AUTOSCALE_AXES: ClassVar[frozenset[Axis]] = frozenset()

    def __init__(
        self,
        *,
        horizontal: bool = False,
        **kwargs,
    ):
        """
        Initialize the bars plotter.

        Parameters
        ----------
        horizontal:
            If True, bars are horizontal; if False, bars are vertical.
        **kwargs:
            Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._horizontal = horizontal
        self._bars_opts: dict[str, Any] = {
            'invert_axes': horizontal,
            'show_legend': False,
            'toolbar': None,
            'yrotation' if horizontal else 'xrotation': 45 if horizontal else 25,
        }

    @classmethod
    def from_params(cls, params: PlotParamsBars):
        """Create BarsPlotter from PlotParamsBars."""
        return cls(
            horizontal=params.orientation.horizontal,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            normalize_to_rate=params.rate.normalize_to_rate,
        )

    def plot(
        self,
        data: sc.DataArray,
        data_key: ResultKey,
        *,
        label: str = '',
        source_display_name: str = '',
        output_display_name: str = '',
        **kwargs,
    ) -> hv.Bars:
        """Create a bar chart from a 0D scipp DataArray."""
        if data.ndim != 0:
            raise ValueError(f"Expected 0D data, got {data.ndim}D")

        bar_label = source_display_name or data_key.job_id.source_name
        value = float(data.value)
        unit = str(data.unit) if data.unit is not None else None
        vdim_label = output_display_name or data_key.output_name or 'values'
        vdim = hv.Dimension(
            data_key.output_name or 'values', label=vdim_label, unit=unit
        )
        return hv.Bars(
            [(bar_label, value)],
            kdims=['source'],
            vdims=[vdim],
            label=label,
        )

    def style_opts(self) -> list[hv.Options]:
        return [
            hv.opts.Bars(**self._bars_opts, **self._sizing_opts),
            *super().style_opts(),
        ]


class Overlay1DPlotter(Plotter):
    """
    Plotter that slices 2D data along the first dimension and overlays as 1D elements.

    Takes 2D data with dims [slice_dim, plot_dim] and creates an overlay of 1D elements,
    one for each position along the first dimension. Useful for visualizing multiple
    spectra (e.g., ROI spectra) from a single 2D array.

    Colors are assigned by coordinate value when coordinates are integer-like,
    providing stable color identity across updates. For non-integer coordinates
    colors are assigned by position, since rounding values to an integer color
    index would collapse closely-spaced coordinates onto the same color.

    Supports the same line style options (mode, errors) as LinePlotter.
    """

    AUTOSCALE_AXES: ClassVar[frozenset[Axis]] = frozenset({'x', 'y'})

    def __init__(
        self,
        scale_opts: PlotScaleParams,
        tick_params: TickParams | None = None,
        *,
        mode: str = 'line',
        errors: str = 'bars',
        **kwargs,
    ):
        """
        Initialize the overlay 1D plotter.

        Parameters
        ----------
        scale_opts:
            Scaling options for axes.
        tick_params:
            Tick configuration parameters.
        mode:
            Rendering mode: 'line', 'points', or 'histogram'.
        errors:
            Error display mode: 'bars', 'band', or 'none'.
        **kwargs:
            Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._mode = mode
        self._errors = errors
        self._logx = scale_opts.x_scale == PlotScale.log
        self._logy = scale_opts.y_scale == PlotScale.log
        self._base_opts: dict[str, Any] = {
            'logx': self._logx,
            'logy': self._logy,
            **self._make_tick_opts(tick_params),
        }
        self._colors = hv.Cycle.default_cycles["default_colors"]

    @classmethod
    def from_params(cls, params: PlotParams1d):
        """Create Overlay1DPlotter from PlotParams1d."""
        from .plot_params import CombineMode

        return cls(
            layout_params=LayoutParams(combine_mode=CombineMode.overlay),
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
            tick_params=params.ticks,
            normalize_to_rate=params.rate.normalize_to_rate,
            mode=params.line.mode,
            errors=params.line.errors,
        )

    def _compute_overlay_range_targets(
        self, data: sc.DataArray, mode: str
    ) -> RangeTargets:
        """Union x/y targets across all slices of the 2-D overlay data."""
        xpad, ypad, _ = _hv_axis_padding(_LINE1D_ELEMENT[mode])
        targets: RangeTargets = {}
        plot_dim = data.dims[1]
        if plot_dim in data.coords:
            coord_values = data.coords[plot_dim].values
            coord_extent = _finite_min_max(coord_values, log=self._logx)
            if coord_extent is not None:
                targets['x'] = _pad_range(*coord_extent, pad=xpad, log=self._logx)
        value_extent = _value_extent_with_errors(
            data, show_errors=self._errors != 'none', log=self._logy
        )
        if value_extent is not None:
            targets['y'] = _pad_range(*value_extent, pad=ypad, log=self._logy)
        return targets

    def plot(
        self,
        data: sc.DataArray,
        data_key: ResultKey,
        *,
        label: str = '',
        output_display_name: str = '',
        dim_label: Callable[[str], str] | None = None,
        **kwargs,
    ) -> hv.Overlay | hv.Element:
        """
        Create overlaid elements from a 2D DataArray.

        Slices along the first dimension and creates an element for each slice.
        """
        del kwargs, label  # Unused
        if data.ndim != 2:
            raise ValueError(f"Expected 2D data, got {data.ndim}D")

        slice_dim = data.dims[0]
        slice_size = data.sizes[slice_dim]

        if slice_size == 0:
            return hv.Curve([])

        actual_mode, plot_data = _resolve_line1d_mode(
            self._mode, data, dim=data.dims[1]
        )
        targets = self._compute_overlay_range_targets(data, actual_mode)
        if targets:
            self._range_targets[data_key] = targets

        # Get coordinate values for labels and colors
        if slice_dim in data.coords:
            coord_values = data.coords[slice_dim].values
        else:
            coord_values = np.arange(slice_size)

        # Integer coords (e.g. ROI indices) color by value, giving each slice a
        # stable color even when the set of slices changes between updates. For
        # non-integer coords (e.g. distances in metres) ``int()`` would collapse
        # nearby values onto the same color, so fall back to position instead.
        color_by_value = np.issubdtype(coord_values.dtype, np.integer)

        data = plot_data
        use_histogram = actual_mode == 'histogram'

        # A single slice needs no label: a lone labelled element renders its
        # label as the plot title, whereas labels only earn their keep as legend
        # entries distinguishing multiple overlaid slices.
        label_slices = slice_size > 1

        elements: list[hv.Element] = []
        for i in range(slice_size):
            slice_data = data[slice_dim, i]
            if output_display_name:
                slice_data.name = output_display_name
            coord_val = coord_values[i]

            color_idx = (int(coord_val) if color_by_value else i) % len(self._colors)
            color = self._colors[color_idx]

            curve_label = f"{slice_dim}={coord_val}" if label_slices else ''
            converter = HvConverter1d(
                slice_data, value_label=output_display_name, dim_label=dim_label
            )
            base_method = getattr(converter, _LINE1D_BASE_METHOD[actual_mode])
            base = base_method(label=curve_label).opts(color=color)

            if slice_data.variances is not None and self._errors != 'none':
                if use_histogram:
                    # Error elements need midpoint coords (N values, not N+1 edges)
                    mid = slice_data.assign_coords(
                        {
                            slice_data.dim: sc.midpoints(
                                slice_data.coords[slice_data.dim]
                            )
                        }
                    )
                    converter = HvConverter1d(
                        mid, value_label=output_display_name, dim_label=dim_label
                    )
                error_method = getattr(converter, _LINE1D_ERROR_METHOD[self._errors])
                error_el = _color_error_element(error_method(label=curve_label), color)
                elements.append(base)
                elements.append(error_el)
            else:
                elements.append(base)

        if len(elements) == 1:
            return elements[0]
        return hv.Overlay(elements)

    def style_opts(self) -> list[hv.Options]:
        # Per-slice ``color`` stays on the elements in plot(); the rest is static.
        return [
            *_typed_opts(_LINE1D_LEAF_ELEMENTS, **self._base_opts, **self._sizing_opts),
            *super().style_opts(),
        ]
