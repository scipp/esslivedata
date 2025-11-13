# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Param models for configuring plotters via widgets."""

import enum
from enum import StrEnum

import pydantic

from ..config.roi_names import get_roi_mapper


def _get_default_max_roi_count() -> int:
    """Get the default maximum ROI count from the mapper configuration."""
    return get_roi_mapper().total_rois


class WindowMode(str, enum.Enum):
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
    last = 'last'
    max = 'max'


class PlotScale(str, enum.Enum):
    """Enumeration of plot scales."""

    linear = 'linear'
    log = 'log'


class CombineMode(str, enum.Enum):
    """Enumeration of combine modes for multiple datasets."""

    overlay = 'overlay'
    layout = 'layout'


class PlotAspectType(str, enum.Enum):
    """Enumeration of aspect types."""

    square = 'Square'
    equal = 'Equal'
    aspect = 'Fixed plot aspect ratio'
    data_aspect = 'Fixed data aspect ratio'
    free = 'Free'


class PlotAspect(pydantic.BaseModel):
    aspect_type: PlotAspectType = pydantic.Field(
        default=PlotAspectType.square,
        description="Aspect type to use if custom is disabled.",
        title="Aspect Type",
    )
    ratio: float = pydantic.Field(
        default=1.0,
        description="Aspect ratio (width/height) to use if custom is enabled.",
        title="Aspect Ratio",
        ge=0.1,
        le=10.0,
    )
    fix_width: bool = pydantic.Field(
        default=False,
        description="Whether to fix the width of the plot.",
        title="Fix Width",
    )
    width: int = pydantic.Field(
        default=400,
        description="Width of the plot in pixels.",
        title="Width",
        ge=100,
        le=2000,
    )
    fix_height: bool = pydantic.Field(
        default=False,
        description="Whether to fix the height of the plot.",
        title="Fix Height",
    )
    height: int = pydantic.Field(
        default=400,
        description="Height of the plot in pixels.",
        title="Height",
        ge=100,
        le=2000,
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
        default=CombineMode.layout,
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


class PlotParams1d(PlotParamsBase):
    """Common parameters for 1d plots."""

    window: WindowParams = pydantic.Field(
        default_factory=WindowParams,
        description="Windowing and aggregation options.",
    )
    plot_scale: PlotScaleParams = pydantic.Field(
        default_factory=PlotScaleParams,
        description="Scaling options for the plot axes.",
    )


class PlotParams2d(PlotParamsBase):
    """Common parameters for 2d plots."""

    window: WindowParams = pydantic.Field(
        default_factory=WindowParams,
        description="Windowing and aggregation options.",
    )
    plot_scale: PlotScaleParams2d = pydantic.Field(
        default_factory=PlotScaleParams2d,
        description="Scaling options for the plot and color axes.",
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


class ROIOptions(pydantic.BaseModel):
    """Options for ROI detector plots."""

    max_roi_count: int = pydantic.Field(
        default_factory=lambda: min(3, _get_default_max_roi_count()),
        description="Maximum number of regions of interest (ROIs) that can be defined.",
        title="Max ROI Count",
        ge=1,
        le=_get_default_max_roi_count(),
    )


class PlotParamsROIDetector(PlotParams2d):
    """Parameters for ROI detector plots."""

    roi_options: ROIOptions = pydantic.Field(
        default_factory=ROIOptions,
        description="Options for ROI selection and display.",
    )
