# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Param models for configuring plotters via widgets."""

from __future__ import annotations

import enum
from enum import StrEnum

import pydantic


class WindowMode(enum.StrEnum):
    """Enumeration of extraction modes."""

    latest = 'latest'
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


class Curve1dRenderMode(StrEnum):
    """Enumeration of rendering modes for 1D curves."""

    curve = 'curve'
    histogram = 'histogram'


class Curve1dParams(pydantic.BaseModel):
    """Parameters for 1D curve rendering."""

    mode: Curve1dRenderMode = pydantic.Field(
        default=Curve1dRenderMode.curve,
        description=(
            "Rendering mode: 'curve' for smooth lines or 'histogram' for step plot."
        ),
        title="Curve Mode",
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
        default=WindowMode.latest,
        description="Extraction mode: 'latest' for single frame (typically accumulated "
        "for 1 second), 'window' for aggregation over multiple frames.",
        title="Mode",
    )
    window_duration_seconds: float = pydantic.Field(
        default=1.0,
        description="Time duration to aggregate in window mode (seconds).",
        title="Window Duration (s)",
        ge=0.1,
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
    curve: Curve1dParams = pydantic.Field(
        default_factory=Curve1dParams,
        description="1D curve rendering options.",
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


class PlotParams1d(PlotDisplayParams1d):
    """Common parameters for 1D plots with windowing support."""

    window: WindowParams = pydantic.Field(
        default_factory=WindowParams,
        description="Windowing and aggregation options.",
    )


class PlotParams2d(PlotDisplayParams2d):
    """Common parameters for 2D plots with windowing support."""

    window: WindowParams = pydantic.Field(
        default_factory=WindowParams,
        description="Windowing and aggregation options.",
    )


class PlotParams3d(PlotParamsBase):
    """Parameters for 3D slicer plots."""

    window: WindowParams = pydantic.Field(
        default_factory=WindowParams,
        description="Windowing and aggregation options.",
    )
    plot_scale: PlotScaleParams2d = pydantic.Field(
        default_factory=PlotScaleParams2d,
        description="Scaling options for the plot axes and color.",
    )
    ticks: TickParams = pydantic.Field(
        default_factory=TickParams,
        description="Tick configuration for plot axes.",
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
        description="Windowing and aggregation options.",
    )
    orientation: BarOrientation = pydantic.Field(
        default_factory=BarOrientation,
        description="Bar orientation options.",
    )
