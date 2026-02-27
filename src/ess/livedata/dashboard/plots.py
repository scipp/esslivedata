# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""This file contains utilities for creating plots in the dashboard."""

from __future__ import annotations

import time
import weakref
from typing import Any, ClassVar, cast

import holoviews as hv
import numpy as np
import scipp as sc

from ess.livedata.config.workflow_spec import ResultKey

from .autoscaler import Autoscaler
from .plot_params import (
    LayoutParams,
    PlotAspect,
    PlotAspectType,
    PlotParams1d,
    PlotParams2d,
    PlotParamsBars,
    PlotScale,
    PlotScaleParams,
    PlotScaleParams2d,
    TickParams,
)
from .scipp_to_holoviews import HvConverter1d, to_holoviews
from .time_utils import format_time_ns_local


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
        """Create a DynamicMap that passes through pre-computed elements."""

        def passthrough(data):
            return data

        return hv.DynamicMap(passthrough, streams=[pipe], cache_size=1)


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


def _compute_time_info(data: dict[str, sc.DataArray]) -> str | None:
    """
    Compute time interval and lag from start_time/end_time coordinates.

    Returns a formatted string like "12:34:56 - 12:35:02 (Lag: 2.3s)" or None
    if no timing coordinates are found. Uses the earliest start_time and
    latest end_time across all DataArrays to show the full data range.
    Lag is computed from the earliest end_time (oldest data) to show worst-case
    staleness.
    """
    now_ns = time.time_ns()
    min_start: int | None = None
    min_end: int | None = None
    max_end: int | None = None

    for da in data.values():
        if 'start_time' in da.coords:
            start_ns = da.coords['start_time'].value
            if min_start is None or start_ns < min_start:
                min_start = start_ns
        if 'end_time' in da.coords:
            end_ns = da.coords['end_time'].value
            if min_end is None or end_ns < min_end:
                min_end = end_ns
            if max_end is None or end_ns > max_end:
                max_end = end_ns

    if min_end is None:
        return None

    # Use min_end for lag (oldest data = maximum lag)
    lag_s = (now_ns - min_end) / 1e9

    if min_start is not None and max_end is not None:
        start_str = format_time_ns_local(min_start)
        end_str = format_time_ns_local(max_end)
        return f'{start_str} - {end_str} (Lag: {lag_s:.1f}s)'
    else:
        end_str = format_time_ns_local(min_end)
        return f'{end_str} (Lag: {lag_s:.1f}s)'


class Plotter:
    """
    Base class for plots that support autoscaling.

    Tracks presenters via WeakSet and marks them dirty when state changes.
    This enables efficient polling-based update detection.
    """

    def __init__(
        self,
        *,
        aspect_params: PlotAspect | None = None,
        layout_params: LayoutParams | None = None,
        **kwargs,
    ):
        """
        Initialize the plotter.

        Parameters
        ----------
        layout_params:
            Layout parameters for combining multiple datasets. If None, uses defaults.
        **kwargs:
            Additional keyword arguments passed to the autoscaler if created.
        """
        self._cached_state: Any | None = None
        self._presenters: weakref.WeakSet[PresenterBase] = weakref.WeakSet()
        self.autoscaler_kwargs = kwargs
        self.autoscalers: dict[ResultKey, Autoscaler] = {}
        self.layout_params = layout_params or LayoutParams()
        aspect_params = aspect_params or PlotAspect()

        self._sizing_opts: dict[str, Any]
        match aspect_params.aspect_type:
            case PlotAspectType.free:
                self._sizing_opts = {}
            case PlotAspectType.equal:
                self._sizing_opts = {'aspect': 'equal'}
            case PlotAspectType.square:
                self._sizing_opts = {'aspect': 'square'}
            case PlotAspectType.aspect:
                self._sizing_opts = {'aspect': aspect_params.ratio}
            case PlotAspectType.data_aspect:
                self._sizing_opts = {'data_aspect': aspect_params.ratio}
        self._sizing_opts['responsive'] = True

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

    def compute(self, data: dict[ResultKey, sc.DataArray], **kwargs) -> None:
        """
        Compute plot elements from input data and cache the result.

        This is Stage 1 of the two-stage plotter architecture. It transforms
        input data into HoloViews elements, caches the result, and marks all
        registered presenters as dirty.

        Parameters
        ----------
        data:
            Dictionary mapping ResultKeys to DataArrays.
        **kwargs:
            Additional keyword arguments passed to plot().
        """
        plots: list[hv.Element] = []
        try:
            for data_key, da in data.items():
                label = f'{data_key.job_id.source_name}/{data_key.output_name}'
                plot_element = self.plot(da, data_key, label=label, **kwargs)
                plots.append(plot_element)
        except Exception as e:
            plots = [
                hv.Text(0.5, 0.5, f"Error: {e}").opts(
                    text_align='center', text_baseline='middle'
                )
            ]

        if len(plots) == 0:
            plots = [
                hv.Text(0.5, 0.5, 'No data').opts(
                    text_align='center', text_baseline='middle'
                )
            ]

        plots = [self._apply_generic_options(p) for p in plots]

        if self.layout_params.combine_mode == 'overlay':
            result = hv.Overlay(plots).opts(shared_axes=True)
        elif len(plots) == 1:
            result = plots[0]
        else:
            result = (
                hv.Layout(plots)
                .opts(shared_axes=False)
                .cols(self.layout_params.layout_columns)
            )

        # Add time interval and lag indicator as plot title
        time_info = _compute_time_info(data)
        if time_info is not None:
            result = result.opts(title=time_info, fontsize={'title': '10pt'})

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

    def has_cached_state(self) -> bool:
        """Check if state has been computed."""
        return self._cached_state is not None

    def _apply_generic_options(self, plot_element: hv.Element) -> hv.Element:
        """Apply generic options like aspect ratio to a plot element."""
        return plot_element.opts(**self._sizing_opts)

    def _update_autoscaler_and_get_framewise(
        self,
        data: sc.DataArray,
        data_key: ResultKey,
        *,
        coord_data: sc.DataArray | None = None,
    ) -> bool:
        """Update autoscaler with data and return whether bounds changed."""
        if data_key not in self.autoscalers:
            self.autoscalers[data_key] = Autoscaler(**self.autoscaler_kwargs)
        return self.autoscalers[data_key].update_bounds(data, coord_data=coord_data)

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


class LinePlotter(Plotter):
    """Plotter for 1D plots from scipp DataArrays.

    Supports line, scatter, and histogram rendering modes with optional
    error display (bars, band, or none).
    """

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
        self._base_opts: dict[str, Any] = {
            'logx': True if scale_opts.x_scale == PlotScale.log else False,
            'logy': True if scale_opts.y_scale == PlotScale.log else False,
            **self._make_tick_opts(tick_params),
        }

    @classmethod
    def from_params(cls, params: PlotParams1d):
        """Create LinePlotter from PlotParams1d."""
        return cls(
            grow_threshold=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
            tick_params=params.ticks,
            mode=params.line.mode,
            errors=params.line.errors,
        )

    _BASE_METHOD: ClassVar[dict[str, str]] = {
        'line': 'curve',
        'points': 'scatter',
        'histogram': 'histogram',
    }
    _ERROR_METHOD: ClassVar[dict[str, str]] = {
        'bars': 'error_bars',
        'band': 'spread',
    }

    _HISTOGRAM_FALLBACK: ClassVar[str] = 'line'

    def plot(
        self, data: sc.DataArray, data_key: ResultKey, *, label: str = '', **kwargs
    ) -> hv.Element | hv.Overlay:
        """Create a 1D plot from a scipp DataArray."""
        converter = HvConverter1d(data)
        if self._mode == 'histogram' and converter.has_edges:
            mode = 'histogram'
            da = data
        else:
            mode = self._mode if self._mode != 'histogram' else self._HISTOGRAM_FALLBACK
            da = self._convert_bin_edges_to_midpoints(data)
            converter = HvConverter1d(da)

        framewise = self._update_autoscaler_and_get_framewise(da, data_key)
        opts = dict(framewise=framewise, **self._base_opts)

        base_method = getattr(converter, self._BASE_METHOD[mode])
        base = base_method(label=label).opts(**opts)

        if da.variances is not None and self._errors != 'none':
            if mode == 'histogram':
                # Error elements need midpoint coords (N values, not N+1 edges)
                converter = HvConverter1d(self._convert_bin_edges_to_midpoints(da))
            error_method = getattr(converter, self._ERROR_METHOD[self._errors])
            error_element = error_method(label=label).opts(**opts, **self._sizing_opts)
            # Apply sizing opts to child elements individually. Bokeh needs
            # responsive/aspect on each element to size the figure correctly;
            # applying them only to the composite Overlay is not sufficient.
            return base.opts(**self._sizing_opts) * error_element

        return base


class ImagePlotter(Plotter):
    """Plotter for 2D images from scipp DataArrays."""

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
        return cls(
            grow_threshold=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
            tick_params=params.ticks,
        )

    def plot(
        self, data: sc.DataArray, data_key: ResultKey, *, label: str = '', **kwargs
    ) -> hv.Image:
        """Create a 2D plot from a scipp DataArray."""
        # Prepare data with appropriate dtype and log scale masking
        use_log_scale = self._scale_opts.color_scale == PlotScale.log
        plot_data = self._prepare_2d_image_data(data, use_log_scale)

        framewise = self._update_autoscaler_and_get_framewise(plot_data, data_key)
        # We are using the masked data here since Holoviews (at least with the Bokeh
        # backend) show values below the color limits with the same color as the lowest
        # value in the colormap, which is not what we want for, e.g., zeros on a log
        # scale plot. The nan values will be shown as transparent.
        histogram = to_holoviews(plot_data, label=label)
        opts = dict(self._base_opts)
        opts['framewise'] = framewise
        # Set explicit clim for log scale when data is all NaN to avoid HoloViews error
        if use_log_scale and (clim := self._get_log_scale_clim(plot_data)) is not None:
            opts['clim'] = clim
        return histogram.opts(**opts)


class BarsPlotter(Plotter):
    """Plotter for bar charts of 0D scalar data."""

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

    @classmethod
    def from_params(cls, params: PlotParamsBars):
        """Create BarsPlotter from PlotParamsBars."""
        return cls(
            horizontal=params.orientation.horizontal,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
        )

    def plot(
        self, data: sc.DataArray, data_key: ResultKey, *, label: str = '', **kwargs
    ) -> hv.Bars:
        """Create a bar chart from a 0D scipp DataArray."""
        if data.ndim != 0:
            raise ValueError(f"Expected 0D data, got {data.ndim}D")

        bar_label = data_key.job_id.source_name
        value = float(data.value)
        bars = hv.Bars(
            [(bar_label, value)],
            kdims=['source'],
            vdims=[data_key.output_name or ''],
            label=label,
        )
        opts = {'invert_axes': self._horizontal, 'show_legend': False, 'toolbar': None}
        if self._horizontal:
            opts['yrotation'] = 45
        else:
            opts['xrotation'] = 25
        return cast(hv.Bars, bars.opts(**opts))


class Overlay1DPlotter(Plotter):
    """
    Plotter that slices 2D data along the first dimension and overlays as 1D curves.

    Takes 2D data with dims [slice_dim, plot_dim] and creates an overlay of 1D curves,
    one for each position along the first dimension. Useful for visualizing multiple
    spectra (e.g., ROI spectra) from a single 2D array.

    Colors are assigned by coordinate value (not position) when coordinates are
    integer-like, providing stable color identity across updates.
    """

    def __init__(
        self,
        scale_opts: PlotScaleParams,
        tick_params: TickParams | None = None,
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
        **kwargs:
            Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._base_opts: dict[str, Any] = {
            'logx': scale_opts.x_scale == PlotScale.log,
            'logy': scale_opts.y_scale == PlotScale.log,
            **self._make_tick_opts(tick_params),
        }
        self._colors = hv.Cycle.default_cycles["default_colors"]

    @classmethod
    def from_params(cls, params: PlotParams1d):
        """Create Overlay1DPlotter from PlotParams1d."""
        from .plot_params import CombineMode

        return cls(
            grow_threshold=0.1,
            layout_params=LayoutParams(combine_mode=CombineMode.overlay),
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
            tick_params=params.ticks,
        )

    def plot(
        self, data: sc.DataArray, data_key: ResultKey, *, label: str = '', **kwargs
    ) -> hv.Overlay | hv.Element:
        """
        Create overlaid curves from a 2D DataArray.

        Slices along the first dimension and creates a curve for each slice.
        """
        del kwargs, label  # Unused
        if data.ndim != 2:
            raise ValueError(f"Expected 2D data, got {data.ndim}D")

        slice_dim = data.dims[0]
        slice_size = data.sizes[slice_dim]

        if slice_size == 0:
            return hv.Curve([]).opts(**self._base_opts)

        # Update autoscaler with full 2D data to establish global bounds
        framewise = self._update_autoscaler_and_get_framewise(data, data_key)

        # Get coordinate values for labels and colors
        if slice_dim in data.coords:
            coord_values = data.coords[slice_dim].values
        else:
            coord_values = np.arange(slice_size)

        # Pre-convert bin-edge coords to midpoints (shared across all slices)
        data = self._convert_bin_edges_to_midpoints(data, dim=data.dims[1])

        curves: list[hv.Element] = []
        for i in range(slice_size):
            slice_data = data[slice_dim, i]
            coord_val = coord_values[i]

            # Assign color by coordinate value for stable identity
            color_idx = int(coord_val) % len(self._colors)
            color = self._colors[color_idx]

            curve_label = f"{slice_dim}={coord_val}"
            curve = to_holoviews(slice_data, label=curve_label)
            curve = curve.opts(
                color=color, framewise=framewise, **self._base_opts, **self._sizing_opts
            )
            curves.append(curve)

        if len(curves) == 1:
            return curves[0]
        return hv.Overlay(curves).opts(shared_axes=True)
