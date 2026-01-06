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

from typing import Any

import pydantic
import scipp as sc

from ess.livedata.config.workflow_spec import ResultKey

from .data_roles import PRIMARY, X_AXIS, Y_AXIS
from .plot_params import PlotDisplayParams1d, PlotDisplayParams2d
from .plots import ImagePlotter, LinePlotter


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
    PlotDisplayParams1d.
    """

    bins: Bin1dParams = pydantic.Field(
        default_factory=Bin1dParams,
        title="Histogram Bins",
        description="Bin configuration for the histogram.",
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
    return sc.lookup(axis_values, mode=mode)


class CorrelationHistogram1dPlotter:
    """Plotter for 1D correlation histograms.

    Receives pre-structured data from DataSubscriber (multi-role assembly):
    - "primary": dict[ResultKey, DataArray] - data to histogram
    - "x_axis": dict[ResultKey, DataArray] - x-axis correlation values
    """

    kdims: list[str] | None = None

    def __init__(self, params: CorrelationHistogram1dParams) -> None:
        self._num_bins = params.bins.x_bins
        self._x_name = params.bins.x_axis_source or 'x'
        self._normalize = params.normalization.per_second
        self._renderer = LinePlotter(
            scale_opts=params.plot_scale,
            tick_params=params.ticks,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            as_histogram=True,
        )

    def initialize_from_data(self, data: dict[str, Any]) -> None:
        """No-op: histogram edges are computed dynamically on each call."""

    def __call__(self, data: dict[str, Any]) -> Any:
        """Compute histograms for all data sources and render.

        Parameters
        ----------
        data
            Structured data from DataSubscriber with "primary" and "x_axis" roles.
        """
        histogram_data: dict[ResultKey, sc.DataArray] = data.get(PRIMARY, {})
        x_axis_dict = data.get(X_AXIS, {})
        x_axis_data: sc.DataArray | None = (
            next(iter(x_axis_dict.values()), None) if x_axis_dict else None
        )

        if x_axis_data is None:
            raise ValueError(
                f"Correlation histogram requires x-axis data (role '{X_AXIS}'), "
                "but it was not found in the data."
            )
        if not histogram_data:
            raise ValueError(
                "Correlation histogram requires at least one data source to histogram."
            )

        x_name = self._x_name
        histograms: dict[ResultKey, sc.DataArray] = {}
        for key, source_data in histogram_data.items():
            dependent = source_data.copy(deep=False)
            x_lut = _make_lookup(x_axis_data, dependent.coords['time'].max())
            dependent.coords[x_name] = x_lut[dependent.coords['time']]

            if self._normalize:
                times = dependent.coords['time']
                widths = (times[1:] - times[:-1]).to(dtype='float64', unit='s')
                widths = sc.concat([widths, widths.median()], dim='time')
                dependent = dependent / widths
                histograms[key] = dependent.bin({x_name: self._num_bins}).bins.mean()
            else:
                histograms[key] = dependent.hist({x_name: self._num_bins})

        return self._renderer(histograms)

    @classmethod
    def from_params(cls, params: CorrelationHistogram1dParams):
        """Factory method for plotter registry."""
        return cls(params=params)


class CorrelationHistogram2dPlotter:
    """Plotter for 2D correlation histograms.

    Receives pre-structured data from DataSubscriber (multi-role assembly):
    - "primary": dict[ResultKey, DataArray] - data to histogram
    - "x_axis": dict[ResultKey, DataArray] - x-axis correlation values
    - "y_axis": dict[ResultKey, DataArray] - y-axis correlation values
    """

    kdims: list[str] | None = None

    def __init__(self, params: CorrelationHistogram2dParams) -> None:
        self._x_bins = params.bins.x_bins
        self._y_bins = params.bins.y_bins
        self._x_name = params.bins.x_axis_source or 'x'
        self._y_name = params.bins.y_axis_source or 'y'
        self._normalize = params.normalization.per_second
        self._renderer = ImagePlotter(
            scale_opts=params.plot_scale,
            tick_params=params.ticks,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
        )

    def initialize_from_data(self, data: dict[str, Any]) -> None:
        """No-op: histogram edges are computed dynamically on each call."""

    def __call__(self, data: dict[str, Any]) -> Any:
        """Compute 2D histograms for all data sources and render.

        Parameters
        ----------
        data
            Structured data from DataSubscriber with "primary", "x_axis",
            and "y_axis" roles.
        """
        histogram_data: dict[ResultKey, sc.DataArray] = data.get(PRIMARY, {})
        x_axis_dict = data.get(X_AXIS, {})
        y_axis_dict = data.get(Y_AXIS, {})
        x_axis_data: sc.DataArray | None = (
            next(iter(x_axis_dict.values()), None) if x_axis_dict else None
        )
        y_axis_data: sc.DataArray | None = (
            next(iter(y_axis_dict.values()), None) if y_axis_dict else None
        )

        if x_axis_data is None:
            raise ValueError(
                f"2D correlation histogram requires x-axis data (role '{X_AXIS}'), "
                "but it was not found in the data."
            )
        if y_axis_data is None:
            raise ValueError(
                f"2D correlation histogram requires y-axis data (role '{Y_AXIS}'), "
                "but it was not found in the data."
            )
        if not histogram_data:
            raise ValueError(
                "Correlation histogram requires at least one data source to histogram."
            )

        x_name, y_name = self._x_name, self._y_name
        histograms: dict[ResultKey, sc.DataArray] = {}
        for key, source_data in histogram_data.items():
            dependent = source_data.copy(deep=False)
            data_max_time = dependent.coords['time'].max()
            x_lut = _make_lookup(x_axis_data, data_max_time)
            y_lut = _make_lookup(y_axis_data, data_max_time)
            dependent.coords[x_name] = x_lut[dependent.coords['time']]
            dependent.coords[y_name] = y_lut[dependent.coords['time']]

            if self._normalize:
                times = dependent.coords['time']
                widths = (times[1:] - times[:-1]).to(dtype='float64', unit='s')
                widths = sc.concat([widths, widths.median()], dim='time')
                dependent = dependent / widths
                histograms[key] = dependent.bin(
                    {y_name: self._y_bins, x_name: self._x_bins}
                ).bins.mean()
            else:
                histograms[key] = dependent.hist(
                    {y_name: self._y_bins, x_name: self._x_bins}
                )

        return self._renderer(histograms)

    @classmethod
    def from_params(cls, params: CorrelationHistogram2dParams):
        """Factory method for plotter registry."""
        return cls(params=params)
