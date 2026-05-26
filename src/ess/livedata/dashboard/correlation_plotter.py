# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Correlation histogram plotters and related data structures.

This module contains the plotter implementations for correlation histograms,
along with simplified parameter models used by the PlotConfigModal wizard.

Correlation histograms receive pre-structured data from DataSubscriber:
- "primary": dict[ResultKey, DataArray] - data to histogram (may have multiple sources)
- "x_axis": dict[ResultKey, DataArray] - x-axis correlation values
- "y_axis": dict[ResultKey, DataArray] - y-axis correlation values (2D only)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import pydantic
import scipp as sc
import structlog

from ess.livedata.config.workflow_spec import ResultKey

from .data_roles import PRIMARY, X_AXIS, Y_AXIS
from .plot_params import (
    Line1dParams,
    Line1dRenderMode,
    PlotDisplayParams1d,
    PlotDisplayParams2d,
)
from .plots import ImagePlotter, LinePlotter, PresenterBase, TitleResolver

_logger = structlog.get_logger()


class NormalizationParams(pydantic.BaseModel):
    per_second: bool = pydantic.Field(
        default=False,
        description="Divide data by time bin width to obtain a rate. When enabled, "
        "each histogram bin represents a rate (rather than counts), computed as a mean "
        "instead of a sum over all contributions.",
    )


class _CorrelationHistogramBase(pydantic.BaseModel):
    normalization: NormalizationParams = pydantic.Field(
        default_factory=NormalizationParams,
        title="Normalization",
        description="Options for normalizing the correlation histogram.",
    )


class Bin1dParams(pydantic.BaseModel):
    """Bin parameters for 1D correlation histograms."""

    x_axis_source: str | None = pydantic.Field(
        default=None,
        frozen=True,
        title="X Axis",
        description="Data source used for the X axis.",
    )
    x_bins: int = pydantic.Field(
        default=100,
        ge=1,
        le=5000,
        title="X Bins",
        description="Number of bins for X axis (range auto-determined from data).",
    )


class Bin2dParams(pydantic.BaseModel):
    """Bin parameters for 2D correlation histograms."""

    x_axis_source: str | None = pydantic.Field(
        default=None,
        frozen=True,
        title="X Axis",
        description="Data source used for the X axis.",
    )
    x_bins: int = pydantic.Field(
        default=20,
        ge=1,
        le=1000,
        title="X Bins",
        description="Number of bins for X axis (range auto-determined from data).",
    )
    y_axis_source: str | None = pydantic.Field(
        default=None,
        frozen=True,
        title="Y Axis",
        description="Data source used for the Y axis.",
    )
    y_bins: int = pydantic.Field(
        default=20,
        ge=1,
        le=1000,
        title="Y Bins",
        description="Number of bins for Y axis (range auto-determined from data).",
    )


class CorrelationHistogram1dParams(_CorrelationHistogramBase, PlotDisplayParams1d):
    """Params for 1D correlation histogram with auto-determined ranges.

    Used by PlotConfigModal wizard. The plotter auto-determines bin edges from data.
    Inherits display options (layout, plot_scale, ticks, plot_aspect) from
    PlotDisplayParams1d. Renders as histogram by default.
    """

    bins: Bin1dParams = pydantic.Field(
        default_factory=Bin1dParams,
        title="Histogram Bins",
        description="Bin configuration for the histogram.",
    )
    line: Line1dParams = pydantic.Field(
        default_factory=lambda: Line1dParams(mode=Line1dRenderMode.histogram),
        description="1D line rendering options (defaults to histogram mode).",
    )


class CorrelationHistogram2dParams(_CorrelationHistogramBase, PlotDisplayParams2d):
    """Params for 2D correlation histogram with auto-determined ranges.

    Used by PlotConfigModal wizard. The plotter auto-determines bin edges from data.
    Inherits display options (layout, plot_scale, ticks, plot_aspect) from
    PlotDisplayParams2d.
    """

    bins: Bin2dParams = pydantic.Field(
        default_factory=Bin2dParams,
        title="Histogram Bins",
        description="Bin configuration for the histogram.",
    )


# Plotter names that are correlation histogram types
CORRELATION_HISTOGRAM_PLOTTERS = frozenset(
    {'correlation_histogram_1d', 'correlation_histogram_2d'}
)


def _make_lookup(axis_data: sc.DataArray, data_max_time: sc.Variable) -> sc.bins.Lookup:
    """Create lookup table with appropriate mode based on time overlap.

    Uses 'previous' mode normally, but falls back to 'nearest' if all data
    timestamps are before the first axis timestamp (which would produce NaNs).
    """
    axis_values = sc.values(axis_data) if axis_data.variances is not None else axis_data
    axis_min_time = axis_data.coords['time'].min()
    mode = 'nearest' if data_max_time < axis_min_time else 'previous'
    if mode == 'nearest':  # scipp>=25.11.0 rejects int coords in this mode
        dim = axis_values.dim
        coord = axis_values.coords[dim]
        # Only convert to float64 if needed; datetime64 coords are accepted as-is
        if coord.dtype != sc.DType.datetime64:
            axis_values = axis_values.assign_coords({dim: coord.astype('float64')})
    return sc.lookup(axis_values, mode=mode)


@dataclass(frozen=True)
class AxisSpec:
    """Specification for a correlation axis."""

    role: str
    """Data role to use for this axis (e.g., X_AXIS, Y_AXIS)."""
    name: str
    """Coordinate name to assign in the data."""
    bins: int
    """Number of bins for this axis."""


class CorrelationHistogramPlotter:
    """Base plotter for correlation histograms with arbitrary number of axes.

    Receives role-grouped data from DataSubscriber:
    - "primary": dict[ResultKey, DataArray] - data to histogram
    - One or more axis roles (e.g., "x_axis", "y_axis") containing correlation values

    Not a ``Plotter`` subclass — composes a renderer (``LinePlotter`` or
    ``ImagePlotter``) rather than building elements directly. Mirrors the
    same lazy-compute gating as the base class: ``compute`` stashes input
    and only triggers the histogram + renderer build while a consumer holds
    interest. The renderer's own ``_active_tokens`` is the gate so we don't
    duplicate token state.
    """

    kdims: list[str] | None = None

    def __init__(
        self,
        axes: list[AxisSpec],
        normalize: bool,
        renderer: LinePlotter | ImagePlotter,
    ) -> None:
        self._axes = axes
        self._normalize = normalize
        self._renderer = renderer
        self._pending: (
            tuple[
                dict[ResultKey, sc.DataArray],
                dict[str, sc.DataArray],
                TitleResolver | None,
            ]
            | None
        ) = None
        self._dirty: bool = False
        self._build_lock = threading.Lock()

    def initialize_from_data(self, data: dict[str, Any]) -> None:
        """No-op: histogram edges are computed dynamically on each call."""

    def compute(
        self,
        data: dict[str, Any],
        *,
        title_resolver: TitleResolver | None = None,
    ) -> None:
        """Validate input eagerly; stash extracted data; build if a consumer is active.

        Validation (presence of primary + axis roles) runs on every call so
        bad input raises immediately at the orchestrator. Only the heavy
        histogram + renderer build is deferred.

        Parameters
        ----------
        data
            Role-grouped data with "primary" role and axis roles.
        title_resolver
            Resolves source/output names to display titles.
        """
        histogram_data, axis_data = self._extract(data)
        with self._build_lock:
            self._pending = (histogram_data, axis_data, title_resolver)
            self._dirty = True
            should_build = self._renderer.has_active_interest
        if should_build:
            self._build_pending()

    def _extract(
        self, data: dict[str, Any]
    ) -> tuple[dict[ResultKey, sc.DataArray], dict[str, sc.DataArray]]:
        histogram_data: dict[ResultKey, sc.DataArray] = data.get(PRIMARY, {})
        if not histogram_data:
            raise ValueError(
                "Correlation histogram requires at least one data source to histogram."
            )
        axis_data: dict[str, sc.DataArray] = {}
        for axis in self._axes:
            axis_dict = data.get(axis.role, {})
            ax = next(iter(axis_dict.values()), None) if axis_dict else None
            if ax is None:
                raise ValueError(
                    f"Correlation histogram requires data for role '{axis.role}', "
                    "but it was not found in the data."
                )
            axis_data[axis.name] = ax
        return histogram_data, axis_data

    def get_cached_state(self) -> Any | None:
        """Get the last computed state from the renderer."""
        return self._renderer.get_cached_state()

    def has_cached_state(self) -> bool:
        """Check if the renderer has computed state."""
        return self._renderer.has_cached_state()

    def create_presenter(self) -> PresenterBase:
        """Create a presenter owned by this plotter.

        Uses the renderer for presentation.
        """
        return self._renderer.create_presenter(owner=self)

    def mark_presenters_dirty(self) -> None:
        """Mark all presenters as dirty by delegating to the renderer."""
        self._renderer.mark_presenters_dirty()

    def set_active(self, token: object, active: bool) -> None:
        """Update consumer interest on the renderer and build if newly active.

        The renderer owns the active-token set — presenters live on it via
        ``create_presenter``'s delegation, so there's nothing to gain by
        tracking a second set here. The 0→1 transition triggers our own
        ``_build_pending`` so the histogram step runs only when something
        will actually consume the result.
        """
        was_active = self._renderer.has_active_interest
        self._renderer.set_active(token, active)
        if self._renderer.has_active_interest and not was_active and self._dirty:
            self._build_pending()

    def _build_pending(self) -> None:
        with self._build_lock:
            if self._pending is None or not self._dirty:
                return
            histogram_data, axis_data, title_resolver = self._pending
            self._dirty = False
        try:
            self._build_histograms(histogram_data, axis_data, title_resolver)
        except Exception:
            _logger.exception(
                "CorrelationHistogramPlotter._build_histograms failed",
                plotter_type=type(self).__name__,
            )

    def _build_histograms(
        self,
        histogram_data: dict[ResultKey, sc.DataArray],
        axis_data: dict[str, sc.DataArray],
        title_resolver: TitleResolver | None,
    ) -> None:
        bin_spec = {axis.name: axis.bins for axis in self._axes}

        histograms: dict[ResultKey, sc.DataArray] = {}
        for key, source_data in histogram_data.items():
            dependent = source_data.copy(deep=False)
            data_max_time = dependent.coords['time'].max()

            for axis in self._axes:
                lut = _make_lookup(axis_data[axis.name], data_max_time)
                dependent.coords[axis.name] = lut[dependent.coords['time']]

            if self._normalize:
                times = dependent.coords['time']
                widths = (times[1:] - times[:-1]).to(dtype='float64', unit='s')
                widths = sc.concat([widths, widths.median()], dim='time')
                dependent = dependent / widths
                histograms[key] = dependent.bin(bin_spec).bins.mean()
            else:
                histograms[key] = dependent.hist(bin_spec)

        self._renderer.compute({PRIMARY: histograms}, title_resolver=title_resolver)


class CorrelationHistogram1dPlotter(CorrelationHistogramPlotter):
    """Plotter for 1D correlation histograms."""

    def __init__(self, params: CorrelationHistogram1dParams) -> None:
        axes = [
            AxisSpec(
                role=X_AXIS,
                name=params.bins.x_axis_source or 'x',
                bins=params.bins.x_bins,
            )
        ]
        renderer = LinePlotter.from_display_params(params)
        super().__init__(
            axes=axes, normalize=params.normalization.per_second, renderer=renderer
        )

    @classmethod
    def from_params(cls, params: CorrelationHistogram1dParams):
        """Factory method for plotter registry."""
        return cls(params=params)


class CorrelationHistogram2dPlotter(CorrelationHistogramPlotter):
    """Plotter for 2D correlation histograms."""

    def __init__(self, params: CorrelationHistogram2dParams) -> None:
        # Y axis first: dims[0] maps to vertical, dims[1] to horizontal
        axes = [
            AxisSpec(
                role=Y_AXIS,
                name=params.bins.y_axis_source or 'y',
                bins=params.bins.y_bins,
            ),
            AxisSpec(
                role=X_AXIS,
                name=params.bins.x_axis_source or 'x',
                bins=params.bins.x_bins,
            ),
        ]
        renderer = ImagePlotter.from_params(params)
        super().__init__(
            axes=axes, normalize=params.normalization.per_second, renderer=renderer
        )

    @classmethod
    def from_params(cls, params: CorrelationHistogram2dParams):
        """Factory method for plotter registry."""
        return cls(params=params)
