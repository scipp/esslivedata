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

from dataclasses import dataclass
from typing import Any

import pydantic
import scipp as sc

from ess.livedata.config.workflow_spec import ResultKey

from .data_roles import PRIMARY, X_AXIS, Y_AXIS
from .plot_params import (
    Curve1dParams,
    Curve1dRenderMode,
    PlotDisplayParams1d,
    PlotDisplayParams2d,
)
from .plots import ImagePlotter, LinePlotter, PresenterBase


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
    curve: Curve1dParams = pydantic.Field(
        default_factory=lambda: Curve1dParams(mode=Curve1dRenderMode.histogram),
        description="1D curve rendering options (defaults to histogram mode).",
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

    Receives pre-structured data from DataSubscriber (multi-role assembly):
    - "primary": dict[ResultKey, DataArray] - data to histogram
    - One or more axis roles (e.g., "x_axis", "y_axis") containing correlation values
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

    def initialize_from_data(self, data: dict[str, Any]) -> None:
        """No-op: histogram edges are computed dynamically on each call."""

    def compute(self, data: dict[str, Any]) -> None:
        """Compute histograms for all data sources and render.

        Parameters
        ----------
        data
            Structured data from DataSubscriber with "primary" role and axis roles.
        """
        histogram_data: dict[ResultKey, sc.DataArray] = data.get(PRIMARY, {})
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

        # Build bin spec
        bin_spec = {axis.name: axis.bins for axis in self._axes}

        histograms: dict[ResultKey, sc.DataArray] = {}
        for key, source_data in histogram_data.items():
            dependent = source_data.copy(deep=False)
            data_max_time = dependent.coords['time'].max()

            # Add all axis coordinates via lookup
            for axis in self._axes:
                lut = _make_lookup(axis_data[axis.name], data_max_time)
                dependent.coords[axis.name] = lut[dependent.coords['time']]

            # Bin data with optional normalization by time width
            if self._normalize:
                times = dependent.coords['time']
                widths = (times[1:] - times[:-1]).to(dtype='float64', unit='s')
                widths = sc.concat([widths, widths.median()], dim='time')
                dependent = dependent / widths
                histograms[key] = dependent.bin(bin_spec).bins.mean()
            else:
                histograms[key] = dependent.hist(bin_spec)

        self._renderer.compute(histograms)

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
        renderer = LinePlotter.from_params(params)
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
