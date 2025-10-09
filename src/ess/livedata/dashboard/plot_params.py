# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Param models for configuring plotters via widgets."""

import enum

import pydantic


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

    plot_scale: PlotScaleParams = pydantic.Field(
        default_factory=PlotScaleParams,
        description="Scaling options for the plot axes.",
    )


class PlotParams2d(PlotParamsBase):
    """Common parameters for 2d plots."""

    plot_scale: PlotScaleParams2d = pydantic.Field(
        default_factory=PlotScaleParams2d,
        description="Scaling options for the plot and color axes.",
    )


class PlotParams3d(PlotParamsBase):
    """Parameters for 3D slicer plots."""

    slice_dimension: str = pydantic.Field(
        description="Dimension to slice along. The other two dimensions will be "
        "displayed as a 2D image.",
        title="Slice Dimension",
    )

    initial_slice_index: int = pydantic.Field(
        default=0,
        ge=0,
        description="Initial slice index to display.",
        title="Initial Slice Index",
    )

    plot_scale: PlotScaleParams2d = pydantic.Field(
        default_factory=PlotScaleParams2d,
        description="Scaling options for the plot axes and color.",
    )
