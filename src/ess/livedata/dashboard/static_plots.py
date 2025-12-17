# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Static plotters for geometric overlays without data sources.

These plotters create plots directly from their params, without subscribing
to any workflow data. They are used for overlay elements like rectangles,
vertical lines, and horizontal lines.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

import holoviews as hv
import pydantic


class StaticPlotter(ABC):
    """Base class for static plotters that create plots from params only."""

    @abstractmethod
    def create_static_plot(self) -> hv.Element:
        """Create a plot element from the stored params."""


# =============================================================================
# Rectangles Plotter
# =============================================================================


class RectanglesCoordinates(pydantic.BaseModel):
    """Wrapper for rectangle coordinate input to get full-width card."""

    coordinates: str = pydantic.Field(
        default="[]",
        title="Coordinates",
        description='Enter as [[x0,y0,x1,y1], ...], e.g., [[0,0,10,10],[20,20,30,30]]',
    )

    @pydantic.field_validator('coordinates')
    @classmethod
    def validate_coordinates(cls, v: str) -> str:
        """Validate rectangle coordinate structure."""
        try:
            coords = json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid format: {e}") from e

        if not isinstance(coords, list):
            raise ValueError("Must be a list of rectangles")

        for i, rect in enumerate(coords):
            if not isinstance(rect, list | tuple):
                raise ValueError(f"Rectangle {i + 1}: must be a list [x0, y0, x1, y1]")
            if len(rect) != 4:
                raise ValueError(
                    f"Rectangle {i + 1}: expected 4 coordinates [x0, y0, x1, y1], "
                    f"got {len(rect)}"
                )
            for j, val in enumerate(rect):
                if not isinstance(val, int | float):
                    raise ValueError(
                        f"Rectangle {i + 1}, coordinate {j + 1}: must be a number"
                    )
        return v

    def parse(self) -> list[tuple[float, float, float, float]]:
        """Parse validated coordinates into list of rectangle tuples."""
        coords = json.loads(self.coordinates)
        return [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in coords]


class RectanglesStyle(pydantic.BaseModel):
    """Style options for rectangles."""

    color: str = pydantic.Field(
        default="red",
        title="Color",
        description="Rectangle fill color (e.g., 'red', '#ff0000')",
    )
    alpha: float = pydantic.Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        title="Opacity",
        description="Fill transparency (0 = transparent, 1 = opaque)",
    )
    line_width: float = pydantic.Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        title="Line Width",
        description="Border line width in pixels",
    )


class RectanglesParams(pydantic.BaseModel):
    """Parameters for static rectangles overlay."""

    geometry: RectanglesCoordinates = pydantic.Field(
        default_factory=RectanglesCoordinates,
        title="Rectangle Coordinates",
        description="List of [x0, y0, x1, y1] corner coordinates for each rectangle.",
    )
    style: RectanglesStyle = pydantic.Field(
        default_factory=RectanglesStyle,
        title="Appearance",
        description="Visual styling options.",
    )


class RectanglesPlotter(StaticPlotter):
    """Plotter for static rectangles overlay."""

    def __init__(self, params: RectanglesParams) -> None:
        self.params = params

    @classmethod
    def from_params(cls, params: RectanglesParams) -> RectanglesPlotter:
        """Create plotter from params."""
        return cls(params)

    def create_static_plot(self) -> hv.Element:
        """Create rectangles element from params."""
        rects = self.params.geometry.parse()
        style = self.params.style

        if not rects:
            # Return empty rectangles element
            return hv.Rectangles([]).opts(
                fill_alpha=style.alpha,
                fill_color=style.color,
                line_width=style.line_width,
            )

        return hv.Rectangles(rects).opts(
            fill_alpha=style.alpha,
            fill_color=style.color,
            line_width=style.line_width,
        )


# =============================================================================
# Vertical Lines Plotter
# =============================================================================


class VLinesCoordinates(pydantic.BaseModel):
    """Wrapper for vertical line coordinate input."""

    positions: str = pydantic.Field(
        default="[]",
        title="X Positions",
        description='Enter as [x1, x2, ...], e.g., [10, 20, 30]',
    )

    @pydantic.field_validator('positions')
    @classmethod
    def validate_positions(cls, v: str) -> str:
        """Validate line position structure."""
        try:
            positions = json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid format: {e}") from e

        if not isinstance(positions, list):
            raise ValueError("Must be a list of x positions")

        for i, pos in enumerate(positions):
            if not isinstance(pos, int | float):
                raise ValueError(f"Position {i + 1}: must be a number")
        return v

    def parse(self) -> list[float]:
        """Parse validated positions into list of floats."""
        return [float(p) for p in json.loads(self.positions)]


class LinesStyle(pydantic.BaseModel):
    """Style options for lines."""

    color: str = pydantic.Field(
        default="red",
        title="Color",
        description="Line color (e.g., 'red', '#ff0000')",
    )
    alpha: float = pydantic.Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        title="Opacity",
        description="Line transparency (0 = transparent, 1 = opaque)",
    )
    line_width: float = pydantic.Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        title="Line Width",
        description="Line width in pixels",
    )
    line_dash: str = pydantic.Field(
        default="solid",
        title="Line Style",
        description="Line style: solid, dashed, dotted, dotdash",
    )


class VLinesParams(pydantic.BaseModel):
    """Parameters for static vertical lines overlay."""

    geometry: VLinesCoordinates = pydantic.Field(
        default_factory=VLinesCoordinates,
        title="Line Positions",
        description="List of x-coordinates for vertical lines.",
    )
    style: LinesStyle = pydantic.Field(
        default_factory=LinesStyle,
        title="Appearance",
        description="Visual styling options.",
    )


class VLinesPlotter(StaticPlotter):
    """Plotter for static vertical lines overlay."""

    def __init__(self, params: VLinesParams) -> None:
        self.params = params

    @classmethod
    def from_params(cls, params: VLinesParams) -> VLinesPlotter:
        """Create plotter from params."""
        return cls(params)

    def create_static_plot(self) -> hv.Element:
        """Create VLines element from params."""
        positions = self.params.geometry.parse()
        style = self.params.style

        if not positions:
            return hv.VLines([]).opts(
                alpha=style.alpha,
                color=style.color,
                line_width=style.line_width,
                line_dash=style.line_dash,
            )

        return hv.VLines(positions).opts(
            alpha=style.alpha,
            color=style.color,
            line_width=style.line_width,
            line_dash=style.line_dash,
        )


# =============================================================================
# Horizontal Lines Plotter
# =============================================================================


class HLinesCoordinates(pydantic.BaseModel):
    """Wrapper for horizontal line coordinate input."""

    positions: str = pydantic.Field(
        default="[]",
        title="Y Positions",
        description='Enter as [y1, y2, ...], e.g., [10, 20, 30]',
    )

    @pydantic.field_validator('positions')
    @classmethod
    def validate_positions(cls, v: str) -> str:
        """Validate line position structure."""
        try:
            positions = json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid format: {e}") from e

        if not isinstance(positions, list):
            raise ValueError("Must be a list of y positions")

        for i, pos in enumerate(positions):
            if not isinstance(pos, int | float):
                raise ValueError(f"Position {i + 1}: must be a number")
        return v

    def parse(self) -> list[float]:
        """Parse validated positions into list of floats."""
        return [float(p) for p in json.loads(self.positions)]


class HLinesParams(pydantic.BaseModel):
    """Parameters for static horizontal lines overlay."""

    geometry: HLinesCoordinates = pydantic.Field(
        default_factory=HLinesCoordinates,
        title="Line Positions",
        description="List of y-coordinates for horizontal lines.",
    )
    style: LinesStyle = pydantic.Field(
        default_factory=LinesStyle,
        title="Appearance",
        description="Visual styling options.",
    )


class HLinesPlotter(StaticPlotter):
    """Plotter for static horizontal lines overlay."""

    def __init__(self, params: HLinesParams) -> None:
        self.params = params

    @classmethod
    def from_params(cls, params: HLinesParams) -> HLinesPlotter:
        """Create plotter from params."""
        return cls(params)

    def create_static_plot(self) -> hv.Element:
        """Create HLines element from params."""
        positions = self.params.geometry.parse()
        style = self.params.style

        if not positions:
            return hv.HLines([]).opts(
                alpha=style.alpha,
                color=style.color,
                line_width=style.line_width,
                line_dash=style.line_dash,
            )

        return hv.HLines(positions).opts(
            alpha=style.alpha,
            color=style.color,
            line_width=style.line_width,
            line_dash=style.line_dash,
        )


# =============================================================================
# Registration
# =============================================================================


def _register_static_plotters() -> None:
    """Register static plotters with the plotter registry."""
    from .plotting import (
        DataRequirements,
        PlotterCategory,
        plotter_registry,
    )

    # Dummy data requirements for static plotters (never validated)
    static_data_requirements = DataRequirements(min_dims=0, max_dims=0)

    plotter_registry.register_plotter(
        name='rectangles',
        title='Rectangles',
        description='Draw static rectangles as an overlay. '
        'Define corners as [x0, y0, x1, y1] coordinates.',
        data_requirements=static_data_requirements,
        factory=RectanglesPlotter.from_params,
        category=PlotterCategory.STATIC,
    )

    plotter_registry.register_plotter(
        name='vlines',
        title='Vertical Lines',
        description='Draw static vertical lines as an overlay. '
        'Define x-positions for the lines.',
        data_requirements=static_data_requirements,
        factory=VLinesPlotter.from_params,
        category=PlotterCategory.STATIC,
    )

    plotter_registry.register_plotter(
        name='hlines',
        title='Horizontal Lines',
        description='Draw static horizontal lines as an overlay. '
        'Define y-positions for the lines.',
        data_requirements=static_data_requirements,
        factory=HLinesPlotter.from_params,
        category=PlotterCategory.STATIC,
    )
