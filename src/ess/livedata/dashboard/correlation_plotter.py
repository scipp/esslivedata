# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Correlation histogram plotters and related data structures.

This module contains the plotter implementations for correlation histograms,
along with simplified parameter models used by the PlotConfigModal wizard.

Correlation histograms receive pre-structured data from DataSubscriber:
- "primary": dict[DataKey, DataArray] - data to histogram (may have multiple sources)
- "x_axis": dict[DataKey, DataArray] - x-axis correlation values
- "y_axis": dict[DataKey, DataArray] - y-axis correlation values (2D only)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import pydantic
import scipp as sc

from ess.livedata.config.workflow_spec import DataKey

from .data_roles import PRIMARY, X_AXIS, Y_AXIS
from .plot_params import (
    LegendPosition,
    Line1dParams,
    Line1dRenderMode,
    PlotDisplayParams1d,
    PlotDisplayParams2d,
)
from .plots import (
    ImagePlotter,
    LinePlotter,
    PresenterBase,
    TimeBounds,
    TitleResolver,
    ensure_span,
    is_degenerate_span,
)
from .range_hook import Axis, RangeTargets


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


@dataclass(frozen=True)
class AxisSpec:
    """Specification for a correlation axis."""

    role: str
    """Data role to use for this axis (e.g., X_AXIS, Y_AXIS)."""
    name: str
    """Coordinate name to assign in the data."""
    bins: int
    """Number of bins for this axis."""


def _axis_bins(values: sc.Variable, axis: AxisSpec) -> sc.Variable | int:
    """Binning for ``values``: a bin count, or explicit edges if degenerate.

    ``hist`` derives edges from the value range and does not guard a degenerate
    one (scipp/scipp#3935). A stationary device, or a single axis reading
    correlated with every data point, yields bins of zero width: nothing is
    drawn, and the axis range derived from those edges collapses to a point.
    Bin over the interval such a range is widened to instead -- fixing the edges
    rather than the view, because zero-width bars stay invisible however wide
    the axis around them.
    """
    lo = sc.nanmin(values).value
    hi = sc.nanmax(values).value
    if not is_degenerate_span(lo, hi):
        return axis.bins
    return sc.linspace(
        axis.name, *ensure_span(lo, hi, log=False), axis.bins + 1, unit=values.unit
    )


class CorrelationHistogramPlotter:
    """Base plotter for correlation histograms with arbitrary number of axes.

    Receives role-grouped data from DataSubscriber:
    - "primary": dict[DataKey, DataArray] - data to histogram
    - One or more axis roles (e.g., "x_axis", "y_axis") containing correlation values

    Each point of the primary data is correlated with the axis value in effect at
    its timestamp, i.e. the most recent axis reading at or before it. Points
    predating the first reading of any axis have no such value and are excluded
    from the histogram; correlating them with a reading taken later would be
    fabricating the axis history. The plot therefore starts empty until the axes
    and the data overlap in time.
    """

    AUTOSCALE_AXES: ClassVar[frozenset[Axis]] = frozenset()
    """Subclasses delegate to the inner renderer's ``AUTOSCALE_AXES``."""

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

    def initialize_from_data(self, data: dict[str, Any]) -> None:
        """No-op: histogram edges are computed dynamically on each call."""

    def compute(
        self,
        data: dict[str, Any],
        *,
        title_resolver: TitleResolver | None = None,
    ) -> None:
        """Compute histograms for all data sources and render.

        Data points predating the first reading of any axis are excluded. If that
        leaves nothing to histogram, the render is skipped and the layer keeps
        showing its placeholder.

        Parameters
        ----------
        data
            Role-grouped data with "primary" role and axis roles.
        title_resolver
            Resolves source/output names to display titles.
        """
        histogram_data: dict[DataKey, sc.DataArray] = data.get(PRIMARY, {})
        if not histogram_data:
            raise ValueError(
                "Correlation histogram requires at least one data source to histogram."
            )

        # Extract and validate all axis data
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

        lookups = {
            name: sc.lookup(sc.values(ax), mode='previous')
            for name, ax in axis_data.items()
        }
        # Earliest time at which every axis has a reading. Before it, 'previous'
        # lookup yields NaN, which hist()/bin() would drop without a trace.
        start = max(ax.coords['time'].min() for ax in axis_data.values())

        histograms: dict[DataKey, sc.DataArray] = {}
        for key, source_data in histogram_data.items():
            dependent = source_data['time', start:].copy(deep=False)
            if dependent.sizes['time'] == 0:
                continue

            # Add all axis coordinates via lookup
            for name, lut in lookups.items():
                dependent.coords[name] = lut[dependent.coords['time']]

            bin_spec = {
                axis.name: _axis_bins(dependent.coords[axis.name], axis)
                for axis in self._axes
            }

            # Bin data with optional normalization by time width
            if self._normalize:
                times = dependent.coords['time']
                widths = (times[1:] - times[:-1]).to(dtype='float64', unit='s')
                widths = sc.concat([widths, widths.median()], dim='time')
                dependent = dependent / widths
                histograms[key] = dependent.bin(bin_spec).bins.mean()
            else:
                histograms[key] = dependent.hist(bin_spec)

        if not histograms:
            return

        self._renderer.compute({PRIMARY: histograms}, title_resolver=title_resolver)

    def get_cached_state(self) -> Any | None:
        """Get the last computed state from the renderer."""
        return self._renderer.get_cached_state()

    @property
    def time_bounds(self) -> TimeBounds | None:
        """Time bounds of the most recently computed data (from the renderer)."""
        return self._renderer.time_bounds

    def has_cached_state(self) -> bool:
        """Check if the renderer has computed state."""
        return self._renderer.has_cached_state()

    @property
    def is_overlayable(self) -> bool:
        """Delegate to the inner renderer, which knows its combine mode."""
        return self._renderer.is_overlayable

    @property
    def legend_position(self) -> LegendPosition | None:
        """Delegate the legend placement to the inner renderer.

        The renderer draws the legend, so it owns the setting; the cell asks the
        outer plotter for it when collating layers into a shared figure.
        """
        return self._renderer.legend_position

    @property
    def autoscale_axes(self) -> frozenset[Axis]:
        """Delegate effective autoscale axes to the inner renderer.

        The renderer narrows its axes for static config such as manual color
        limits, so the cell controller inherits that here without the outer
        plotter knowing about it.
        """
        return self._renderer.autoscale_axes

    def get_range_targets(self, data_key: DataKey) -> RangeTargets | None:
        """Delegate per-axis target lookup to the inner renderer."""
        return self._renderer.get_range_targets(data_key)

    def iter_range_targets(self) -> Iterator[tuple[DataKey, RangeTargets]]:
        """Delegate per-key target iteration to the inner renderer."""
        return self._renderer.iter_range_targets()

    def create_presenter(self) -> PresenterBase:
        """Create a presenter owned by this plotter.

        Uses the renderer for presentation.
        """
        return self._renderer.create_presenter(owner=self)

    def mark_presenters_dirty(self) -> None:
        """Mark all presenters as dirty by delegating to the renderer."""
        self._renderer.mark_presenters_dirty()


class CorrelationHistogram1dPlotter(CorrelationHistogramPlotter):
    """Plotter for 1D correlation histograms."""

    AUTOSCALE_AXES: ClassVar[frozenset[Axis]] = frozenset({'x', 'y'})

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

    AUTOSCALE_AXES: ClassVar[frozenset[Axis]] = frozenset({'x', 'y', 'c'})

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
