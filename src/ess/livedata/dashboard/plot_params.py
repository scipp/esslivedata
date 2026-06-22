# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Param models for configuring plotters via widgets."""

from __future__ import annotations

import enum
from enum import StrEnum

import pydantic


class WindowMode(enum.StrEnum):
    """Enumeration of extraction modes.

    - ``since_start``: latest cumulative value (subscribes to the
      ``since_start`` stream of the selected output view).
    - ``window``: the most recent per-update value, optionally with prior
      updates aggregated over an additional lookback window.
    """

    since_start = 'since_start'
    window = 'window'


class WindowAggregation(StrEnum):
    """Enumeration of aggregation methods for window mode."""

    auto = 'auto'
    nansum = 'nansum'
    nanmean = 'nanmean'
    sum = 'sum'
    mean = 'mean'


class PlotScale(enum.StrEnum):
    """Enumeration of plot scales."""

    linear = 'linear'
    log = 'log'


class Line1dRenderMode(StrEnum):
    """Enumeration of rendering modes for 1D plots."""

    line = 'line'
    points = 'points'
    histogram = 'histogram'


class ErrorDisplay(StrEnum):
    """Enumeration of error display modes for 1D plots."""

    bars = 'bars'
    band = 'band'
    none = 'none'


class Line1dParams(pydantic.BaseModel):
    """Parameters for 1D line rendering."""

    mode: Line1dRenderMode = pydantic.Field(
        default=Line1dRenderMode.line,
        description=(
            "Rendering mode: 'line' for smooth curves, 'points' for discrete "
            "markers, or 'histogram' for step plot."
        ),
        title="Line Mode",
    )
    errors: ErrorDisplay = pydantic.Field(
        default=ErrorDisplay.bars,
        description=(
            "Error display mode: 'bars' for error whiskers, 'band' for filled "
            "band, or 'none' to suppress errors."
        ),
        title="Error Display",
    )


class CombineMode(enum.StrEnum):
    """Enumeration of combine modes for multiple datasets."""

    overlay = 'overlay'
    layout = 'layout'


class PlotAspectType(enum.StrEnum):
    """Enumeration of aspect types."""

    square = 'Square'
    equal = 'Equal'
    aspect = 'Fixed plot aspect ratio'
    data_aspect = 'Fixed data aspect ratio'
    free = 'Free'


class StretchMode(enum.StrEnum):
    """Stretch mode for responsive plots with fixed aspect.

    When using a fixed aspect ratio, choose how the plot fills its container:
    - 'Fill width': Plot fills container width, height determined by aspect ratio.
      Use when container is more portrait-ish than the plot.
    - 'Fill height': Plot fills container height, width determined by aspect ratio.
      Use when container is more landscape-ish than the plot.
    """

    width = 'Fill width'
    height = 'Fill height'


class PlotAspect(pydantic.BaseModel):
    aspect_type: PlotAspectType = pydantic.Field(
        default=PlotAspectType.free,
        description="Aspect type for the plot.",
        title="Aspect Type",
    )
    ratio: float = pydantic.Field(
        default=1.0,
        description="Aspect ratio (width/height) for 'Fixed plot/data aspect ratio'.",
        title="Aspect Ratio",
        ge=0.1,
        le=10.0,
    )
    stretch_mode: StretchMode = pydantic.Field(
        default=StretchMode.width,
        description=(
            "How the plot fills its container when using fixed aspect. "
            "'Fill width' for tall containers, 'Fill height' for wide containers. "
            "Ignored when aspect type is 'Free'."
        ),
        title="Stretch Mode",
    )


class TickParams(pydantic.BaseModel):
    """Parameters for axis tick configuration."""

    custom_xticks: bool = pydantic.Field(
        default=False,
        description="Enable custom x-axis tick count instead of automatic.",
        title="Custom X Ticks",
    )
    xticks: int = pydantic.Field(
        default=5,
        description="Number of ticks on x-axis (when custom ticks enabled).",
        title="X Ticks",
        ge=2,
        le=20,
    )
    custom_yticks: bool = pydantic.Field(
        default=False,
        description="Enable custom y-axis tick count instead of automatic.",
        title="Custom Y Ticks",
    )
    yticks: int = pydantic.Field(
        default=5,
        description="Number of ticks on y-axis (when custom ticks enabled).",
        title="Y Ticks",
        ge=2,
        le=20,
    )


class PlotScaleParams(pydantic.BaseModel):
    x_scale: PlotScale = pydantic.Field(
        default=PlotScale.linear, description="Scale for x-axis", title="X Axis Scale"
    )
    y_scale: PlotScale = pydantic.Field(
        default=PlotScale.linear, description="Scale for y-axis", title="Y Axis Scale"
    )


class PlotScaleParams2d(PlotScaleParams):
    color_scale: PlotScale = pydantic.Field(
        default=PlotScale.log,
        description="Scale for color axis",
        title="Color Axis Scale",
    )


class LayoutParams(pydantic.BaseModel):
    """Parameters for layout configuration."""

    combine_mode: CombineMode = pydantic.Field(
        default=CombineMode.overlay,
        description="How to combine multiple datasets: overlay or layout.",
        title="Combine Mode",
    )
    layout_columns: int = pydantic.Field(
        default=2,
        description="Number of columns to use when combining plots in layout mode.",
        title="Layout Columns",
        ge=1,
        le=5,
    )


class WindowParams(pydantic.BaseModel):
    """Parameters for windowing and aggregation."""

    mode: WindowMode = pydantic.Field(
        default=WindowMode.window,
        description=(
            "Extraction mode: 'since_start' for the cumulative value since the "
            "run started, 'window' for the most recent update plus optional "
            "lookback over prior updates."
        ),
        title="Mode",
    )
    window_duration_seconds: float = pydantic.Field(
        default=0.0,
        description=(
            "Additional history to aggregate alongside the latest update "
            "(seconds). 0 means only the latest update."
        ),
        title="Window Duration (s)",
        ge=0.0,
        le=60.0,
    )
    aggregation: WindowAggregation = pydantic.Field(
        default=WindowAggregation.auto,
        description=(
            "Aggregation method for window mode. 'auto' uses 'nansum' for "
            "counts (unit='counts') and 'nanmean' otherwise."
        ),
        title="Aggregation",
    )


class PlotParamsBase(pydantic.BaseModel):
    """Base class for plot parameters."""

    layout: LayoutParams = pydantic.Field(
        default_factory=LayoutParams,
        description="Layout options for combining multiple datasets.",
    )
    plot_aspect: PlotAspect = pydantic.Field(
        default_factory=PlotAspect,
        description="Aspect ratio options for the plot.",
    )


class PlotDisplayParams1d(PlotParamsBase):
    """Display parameters for 1D plots without windowing.

    Used by correlation histograms and other derived 1D plots that don't
    need window configuration (they process full timeseries data).
    """

    plot_scale: PlotScaleParams = pydantic.Field(
        default_factory=PlotScaleParams,
        description="Scaling options for the plot axes.",
    )
    ticks: TickParams = pydantic.Field(
        default_factory=TickParams,
        description="Tick configuration for plot axes.",
    )
    line: Line1dParams = pydantic.Field(
        default_factory=Line1dParams,
        description="1D line rendering options.",
    )


class PlotDisplayParams2d(PlotParamsBase):
    """Display parameters for 2D plots without windowing.

    Used by 2D correlation histograms and other derived 2D plots that don't
    need window configuration.
    """

    plot_scale: PlotScaleParams2d = pydantic.Field(
        default_factory=PlotScaleParams2d,
        description="Scaling options for the plot and color axes.",
    )
    ticks: TickParams = pydantic.Field(
        default_factory=TickParams,
        description="Tick configuration for plot axes.",
    )


class RateNormalizationParams(pydantic.BaseModel):
    """Parameters for normalizing counts to rate (counts per second)."""

    normalize_to_rate: bool = pydantic.Field(
        default=False,
        description="Display as rate (counts per second) by dividing by the "
        "time duration between start_time and end_time.",
        title="Counts Per Second",
    )


_WINDOW_DESCRIPTION = (
    "The live reduction emits a new result roughly once per second "
    "(longer for heavy workflows or under heavy load). The window "
    "duration sets how much recent history the dashboard aggregates "
    "for display; it does not affect what or how often the reduction "
    "emits."
    "<br><br>"
    "The window duration is a target, not a guarantee: the aggregation "
    "time range always covers at least one cadence interval, and will "
    "differ from the requested duration when it does not align with "
    "the reduction's current cadence. To compare values across "
    "different cadences or window durations, enable 'Counts Per Second' "
    "in the 'Rate' tab."
)


_TIMESERIES_DOWNSAMPLING_DESCRIPTION = (
    "Controls how much detail this plot shows over the lifetime of the run. "
    "Performance depends on the total number of points displayed: too many "
    "and the plot - and the rest of the dashboard - becomes sluggish."
    "<br><br>"
    "The plot keeps a trailing <em>Recent</em> window at <em>Fine Period</em> "
    "resolution, plus a coarser band at <em>Coarse Period</em> that extends "
    "the rest of the run's history (set <em>Coarse Period</em> to 0 to drop "
    "older data instead)."
    "<br><br>"
    "Aim to keep the total under a few thousand points for smooth plots. "
    "Some examples for a 1 Hz source: the defaults (1 s / 1 h / 5 min) stay "
    "near 4 000 points regardless of run length; raising <em>Recent "
    "Window</em> to 12 h at 1 s <em>Fine Period</em> adds about 43 000 points "
    "- likely too many; for week-long runs, a <em>Coarse Period</em> of 1 h "
    "keeps the count below 4 000."
)


class TimeseriesDownsamplingParams(pydantic.BaseModel):
    """Parameters controlling timeseries downsampling and update throttling."""

    fine_period_seconds: float = pydantic.Field(
        default=1.0,
        description=(
            "Sample period for the Recent Window, and minimum interval between "
            "plot updates. Larger values reduce points and update frequency."
        ),
        title="Fine Period (s)",
        gt=0.0,
    )
    recent_seconds: float = pydantic.Field(
        default=3600.0,
        description=(
            "Length of the trailing window kept at <em>Fine Period</em> "
            "resolution. Older data falls into the <em>Coarse Period</em> band "
            "(or is dropped if Coarse Period is 0)."
        ),
        title="Recent Window (s)",
        ge=0.0,
    )
    coarse_period_seconds: float = pydantic.Field(
        default=300.0,
        description=(
            "Coarser sample period for data older than the Recent Window. "
            "Set to 0 to drop older data entirely."
        ),
        title="Coarse Period (s)",
        ge=0.0,
    )


class PlotParamsTimeseries(PlotDisplayParams1d):
    """Parameters for the timeseries plotter (downsampling + display)."""

    downsampling: TimeseriesDownsamplingParams = pydantic.Field(
        default_factory=TimeseriesDownsamplingParams,
        description=_TIMESERIES_DOWNSAMPLING_DESCRIPTION,
    )


class PlotParams1d(PlotDisplayParams1d):
    """Common parameters for 1D plots with windowing support."""

    window: WindowParams = pydantic.Field(
        default_factory=WindowParams,
        description=_WINDOW_DESCRIPTION,
    )
    rate: RateNormalizationParams = pydantic.Field(
        default_factory=RateNormalizationParams,
        description="Rate normalization options.",
    )


class PlotParams2d(PlotDisplayParams2d):
    """Common parameters for 2D plots with windowing support."""

    window: WindowParams = pydantic.Field(
        default_factory=WindowParams,
        description=_WINDOW_DESCRIPTION,
    )
    rate: RateNormalizationParams = pydantic.Field(
        default_factory=RateNormalizationParams,
        description="Rate normalization options.",
    )


class PlotParams3d(PlotParamsBase):
    """Parameters for 3D slicer plots."""

    window: WindowParams = pydantic.Field(
        default_factory=WindowParams,
        description=_WINDOW_DESCRIPTION,
    )
    plot_scale: PlotScaleParams2d = pydantic.Field(
        default_factory=PlotScaleParams2d,
        description="Scaling options for the plot axes and color.",
    )
    ticks: TickParams = pydantic.Field(
        default_factory=TickParams,
        description="Tick configuration for plot axes.",
    )
    rate: RateNormalizationParams = pydantic.Field(
        default_factory=RateNormalizationParams,
        description="Rate normalization options.",
    )


class BarOrientation(pydantic.BaseModel):
    """Orientation options for bar plots."""

    horizontal: bool = pydantic.Field(
        default=False,
        description="If True, bars are horizontal; if False, bars are vertical.",
        title="Horizontal Bars",
    )


class PlotParamsBars(PlotParamsBase):
    """Parameters for bar plots of 0D scalar data."""

    window: WindowParams = pydantic.Field(
        default_factory=WindowParams,
        description=_WINDOW_DESCRIPTION,
    )
    orientation: BarOrientation = pydantic.Field(
        default_factory=BarOrientation,
        description="Bar orientation options.",
    )
    rate: RateNormalizationParams = pydantic.Field(
        default_factory=RateNormalizationParams,
        description="Rate normalization options.",
    )


class PlotParamsTable(PlotParamsBase):
    """Parameters for tabular display of 0D scalar data."""

    window: WindowParams = pydantic.Field(
        default_factory=WindowParams,
        description=_WINDOW_DESCRIPTION,
    )
    rate: RateNormalizationParams = pydantic.Field(
        default_factory=RateNormalizationParams,
        description="Rate normalization options.",
    )
