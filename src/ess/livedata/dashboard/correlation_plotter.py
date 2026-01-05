# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Correlation histogram plotters and related data structures.

This module contains the plotter implementations for correlation histograms,
along with simplified parameter models used by the PlotConfigModal wizard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pydantic
import scipp as sc

from ess.livedata.config.workflow_spec import ResultKey

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


# TODO This class should no longer be necessary with the multi-subscription mechanism?
@dataclass
class CorrelationHistogramData:
    """Structured data for correlation histogram plotters.

    Separates data sources (to be histogrammed) from axis sources (defining bins).
    """

    data_sources: dict[ResultKey, sc.DataArray]
    """Primary data sources to histogram. Multiple sources = multiple histograms."""

    axis_data: dict[str, sc.DataArray]
    """Axis data for correlation. Keys are 'x' and optionally 'y'."""


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

    Receives structured CorrelationHistogramData with data sources and axis data.
    Computes a histogram for EACH data source, creating overlay/layout as configured.
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
            value_margin_factor=0.1,
            as_histogram=True,
        )

    def initialize_from_data(self, data: CorrelationHistogramData) -> None:
        """No-op: histogram edges are computed dynamically on each call."""

    def __call__(self, data: CorrelationHistogramData) -> Any:
        """Compute histograms for all data sources and render."""
        if not isinstance(data, CorrelationHistogramData):
            raise TypeError(
                f"CorrelationHistogram1dPlotter expected CorrelationHistogramData, "
                f"got {type(data).__name__}. This usually indicates the data pipeline "
                f"was set up without axis_keys, causing MergingStreamAssembler to be "
                f"used instead of CorrelationHistogramAssembler."
            )
        if 'x' not in data.axis_data:
            raise ValueError("Correlation histogram requires x-axis data")

        x_name = self._x_name
        histograms: dict[ResultKey, sc.DataArray] = {}
        for key, source_data in data.data_sources.items():
            dependent = source_data.copy(deep=False)
            x_lut = _make_lookup(data.axis_data['x'], dependent.coords['time'].max())
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

    Receives structured CorrelationHistogramData with data sources and axis data.
    Computes a 2D histogram for EACH data source, creating overlay/layout as configured.
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
            value_margin_factor=0.1,
        )

    def initialize_from_data(self, data: CorrelationHistogramData) -> None:
        """No-op: histogram edges are computed dynamically on each call."""

    def __call__(self, data: CorrelationHistogramData) -> Any:
        """Compute 2D histograms for all data sources and render."""
        if 'x' not in data.axis_data or 'y' not in data.axis_data:
            raise ValueError("2D correlation histogram requires x-axis and y-axis data")

        x_name, y_name = self._x_name, self._y_name
        histograms: dict[ResultKey, sc.DataArray] = {}
        for key, source_data in data.data_sources.items():
            dependent = source_data.copy(deep=False)
            data_max_time = dependent.coords['time'].max()
            x_lut = _make_lookup(data.axis_data['x'], data_max_time)
            y_lut = _make_lookup(data.axis_data['y'], data_max_time)
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
