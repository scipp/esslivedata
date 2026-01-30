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
from enum import StrEnum

import holoviews as hv
import pydantic
from pydantic_core import core_schema


class LineDash(StrEnum):
    """Line dash styles for HoloViews plots."""

    solid = "solid"
    dashed = "dashed"
    dotted = "dotted"
    dotdash = "dotdash"


class Color(str):
    """A CSS color value (hex, rgb, or named color).

    Detected by ParamWidget to render as a ColorPicker.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema()
        )


def _parse_number_list(v: str) -> list[int | float]:
    """Parse a comma-separated string into a list of numbers.

    Example: "10, 20, 30" -> [10, 20, 30]
    """
    v = v.strip()
    if not v:
        return []
    try:
        result = json.loads(f"[{v}]")
        if isinstance(result, list):
            return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid number: {e}") from e
    return []


def _parse_rectangle_list(v: str) -> list[list[int | float]]:
    """Parse comma-separated JSON arrays into a list of rectangle coordinates.

    Example: "[0,0,10,10], [20,20,30,30]" -> [[0,0,10,10], [20,20,30,30]]
    """
    v = v.strip()
    try:
        result = json.loads(f"[{v}]")
        if isinstance(result, list):
            return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid format: {e}") from e
    return []


class StaticPlotter(ABC):
    """Base class for static plotters that create plots from params only.

    Static plotters don't inherit from Plotter since they don't process data,
    but they still need to track presenters for the dirty flag mechanism.
    """

    def __init__(self) -> None:
        import weakref

        from .plots import PresenterBase

        self._presenters: weakref.WeakSet[PresenterBase] = weakref.WeakSet()
        self._cached_state: hv.Element | None = None

    @abstractmethod
    def create_static_plot(self) -> hv.Element:
        """Create a plot element from the stored params."""

    def create_presenter(self):
        """Create a StaticPresenter for this plotter."""
        from .plots import StaticPresenter

        presenter = StaticPresenter(self)
        self._presenters.add(presenter)
        return presenter

    def _set_cached_state(self, state: hv.Element) -> None:
        """Store computed state and mark all presenters dirty."""
        self._cached_state = state
        # Convert to list to avoid RuntimeError if WeakSet is modified during iteration
        for presenter in list(self._presenters):
            presenter._mark_dirty()

    def get_cached_state(self) -> hv.Element | None:
        """Get the last computed state, or None if not yet computed."""
        return self._cached_state

    def has_cached_state(self) -> bool:
        """Check if state has been computed."""
        return self._cached_state is not None

    def compute(self, data: dict) -> None:
        """Compute the static plot and cache the result.

        Parameters
        ----------
        data
            Ignored. Static plotters don't use data; they create plots purely
            from their params. This parameter exists for interface compatibility
            with dynamic plotters.
        """
        del data  # Unused - static plotters ignore data
        element = self.create_static_plot()
        self._set_cached_state(element)


class RectanglesCoordinates(pydantic.BaseModel):
    """Wrapper for rectangle coordinate input to get full-width card."""

    coordinates: str = pydantic.Field(
        default="",
        title="Coordinates",
        description='E.g., [0,0,10,10], [20,20,30,30]',
    )

    @pydantic.field_validator('coordinates')
    @classmethod
    def validate_coordinates(cls, v: str) -> str:
        """Validate rectangle coordinate structure."""
        coords = _parse_rectangle_list(v)
        if not coords:
            raise ValueError("At least one rectangle is required")
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
        coords = _parse_rectangle_list(self.coordinates)
        return [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in coords]


class BaseStyle(pydantic.BaseModel):
    """Common style options for static overlays."""

    color: Color = pydantic.Field(
        default="#ff0000",
        title="Color",
    )
    line_width: float = pydantic.Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        title="Line Width",
        description="Line width in pixels",
    )
    line_dash: LineDash = pydantic.Field(
        default=LineDash.solid,
        title="Line Style",
        description="Line style: solid, dashed, dotted, dotdash",
    )


class RectanglesStyle(BaseStyle):
    """Style options for rectangles."""

    alpha: float = pydantic.Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        title="Fill Opacity",
        description="Fill transparency (0 = transparent, 1 = opaque)",
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
        super().__init__()
        self.params = params

    @classmethod
    def from_params(cls, params: RectanglesParams) -> RectanglesPlotter:
        """Create plotter from params."""
        return cls(params)

    def create_static_plot(self) -> hv.Element:
        """Create rectangles element from params."""
        rects = self.params.geometry.parse()
        style = self.params.style
        return hv.Rectangles(rects).opts(
            fill_alpha=style.alpha,
            fill_color=style.color,
            line_color=style.color,
            line_width=style.line_width,
            line_dash=style.line_dash,
        )


class LinesCoordinates(pydantic.BaseModel):
    """Wrapper for line coordinate input."""

    positions: str = pydantic.Field(
        default="",
        title="Positions",
        description='Enter as comma-separated values, e.g., 10, 20, 30',
    )

    @pydantic.field_validator('positions')
    @classmethod
    def validate_positions(cls, v: str) -> str:
        """Validate line position structure."""
        positions = _parse_number_list(v)
        if not positions:
            raise ValueError("At least one position is required")
        for i, pos in enumerate(positions):
            if not isinstance(pos, int | float):
                raise ValueError(f"Position {i + 1}: must be a number")
        return v

    def parse(self) -> list[float]:
        """Parse validated positions into list of floats."""
        return [float(p) for p in _parse_number_list(self.positions)]


class LinesStyle(BaseStyle):
    """Style options for lines."""

    alpha: float = pydantic.Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        title="Opacity",
        description="Line transparency (0 = transparent, 1 = opaque)",
    )


class VLinesParams(pydantic.BaseModel):
    """Parameters for static vertical lines overlay."""

    geometry: LinesCoordinates = pydantic.Field(
        default_factory=LinesCoordinates,
        title="X Positions",
        description="X-coordinates where vertical lines will be drawn.",
    )
    style: LinesStyle = pydantic.Field(
        default_factory=LinesStyle,
        title="Appearance",
        description="Visual styling options.",
    )


class HLinesParams(pydantic.BaseModel):
    """Parameters for static horizontal lines overlay."""

    geometry: LinesCoordinates = pydantic.Field(
        default_factory=LinesCoordinates,
        title="Y Positions",
        description="Y-coordinates where horizontal lines will be drawn.",
    )
    style: LinesStyle = pydantic.Field(
        default_factory=LinesStyle,
        title="Appearance",
        description="Visual styling options.",
    )


class LinesPlotter(StaticPlotter):
    """Plotter for static lines overlay (vertical or horizontal)."""

    def __init__(
        self, params: VLinesParams | HLinesParams, element_class: type
    ) -> None:
        super().__init__()
        self.params = params
        self._element_class = element_class

    @classmethod
    def vlines(cls, params: VLinesParams) -> LinesPlotter:
        """Create vertical lines plotter from params."""
        return cls(params, hv.VLines)

    @classmethod
    def hlines(cls, params: HLinesParams) -> LinesPlotter:
        """Create horizontal lines plotter from params."""
        return cls(params, hv.HLines)

    def create_static_plot(self) -> hv.Element:
        """Create lines element from params."""
        positions = self.params.geometry.parse()
        style = self.params.style
        return self._element_class(positions).opts(
            alpha=style.alpha,
            color=style.color,
            line_width=style.line_width,
            line_dash=style.line_dash,
        )


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
        factory=LinesPlotter.vlines,
        category=PlotterCategory.STATIC,
    )

    plotter_registry.register_plotter(
        name='hlines',
        title='Horizontal Lines',
        description='Draw static horizontal lines as an overlay. '
        'Define y-positions for the lines.',
        data_requirements=static_data_requirements,
        factory=LinesPlotter.hlines,
        category=PlotterCategory.STATIC,
    )
