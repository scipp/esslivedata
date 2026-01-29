# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Interactive ROI request plotters for user-drawn ROI selection.

These plotters create interactive BoxEdit/PolyDraw elements that allow users
to draw ROIs visually. Edits are published to Kafka for backend processing.

These plotters subscribe to the ROI readback output stream for job_id context.
The readback data itself is not used for display - only the ResultKey is needed
to identify where to publish ROI updates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

import holoviews as hv
import pydantic
import scipp as sc
import structlog

from ess.livedata.config.models import Interval, PolygonROI, RectangleROI
from ess.livedata.config.roi_names import (
    ROIGeometryType,
    get_default_index_offset,
    get_default_num_rois,
    get_roi_mapper,
)

from .plots import Plotter
from .static_plots import Color, LineDash, RectanglesCoordinates

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from .roi_publisher import ROIPublisher


@runtime_checkable
class ROIPublisherAware(Protocol):
    """Protocol for plotters that can publish ROI updates."""

    def set_roi_publisher(self, publisher: ROIPublisher | None) -> None:
        """Set the ROI publisher for this plotter."""
        ...


def _get_max_rois_for_geometry(geometry_type: ROIGeometryType) -> int:
    """Get max ROI count for a geometry type from central config."""
    geom = get_roi_mapper().geometry_for_type(geometry_type)
    return geom.num_rois if geom else get_default_num_rois(geometry_type)


class RectangleConverter:
    """Converter for rectangle ROIs using BoxEdit stream."""

    def parse_stream_data(
        self,
        data: dict[str, Any],
        x_unit: str | None,
        y_unit: str | None,
        index_offset: int = 0,
    ) -> dict[int, RectangleROI]:
        """
        Convert BoxEdit data dictionary to RectangleROI instances.

        BoxEdit returns data as a dictionary with keys 'x0', 'x1', 'y0', 'y1',
        where each value is a list of coordinates for all boxes.

        Parameters
        ----------
        data:
            Dictionary from BoxEdit stream with keys x0, x1, y0, y1.
        x_unit:
            Unit for x coordinates (from the detector data coordinates).
        y_unit:
            Unit for y coordinates (from the detector data coordinates).
        index_offset:
            Not used for rectangles (always 0).

        Returns
        -------
        :
            Dictionary mapping box index to RectangleROI. Empty boxes are skipped.
        """
        if not data or not data.get("x0"):
            return {}

        x0_list = data.get("x0", [])
        x1_list = data.get("x1", [])
        y0_list = data.get("y0", [])
        y1_list = data.get("y1", [])

        rois = {}
        for i, (x0, x1, y0, y1) in enumerate(
            zip(x0_list, x1_list, y0_list, y1_list, strict=True)
        ):
            # Skip empty/invalid boxes (where corners are equal)
            if x0 == x1 or y0 == y1:
                continue

            # Ensure min < max
            x_min, x_max = (x0, x1) if x0 < x1 else (x1, x0)
            y_min, y_max = (y0, y1) if y0 < y1 else (y1, y0)

            rois[i] = RectangleROI(
                x=Interval(min=x_min, max=x_max, unit=x_unit),
                y=Interval(min=y_min, max=y_max, unit=y_unit),
            )

        return rois

    def to_hv_data(
        self, rois: dict[int, RectangleROI], index_to_color: dict[int, str] | None
    ) -> list[tuple[float, ...]]:
        """
        Convert RectangleROI instances to HoloViews Rectangles format.

        Parameters
        ----------
        rois:
            Dictionary mapping ROI index to RectangleROI.
        index_to_color:
            Optional mapping from ROI index to color string. If provided, each
            rectangle tuple will include the color as a fifth element.

        Returns
        -------
        :
            List of (x0, y0, x1, y1) or (x0, y0, x1, y1, color) tuples for HoloViews
            Rectangles. Returned in sorted order by ROI index.
            All coordinates are explicitly cast to float to ensure compatibility
            with BoxEdit drag operations.
        """
        rectangles = []
        for idx in sorted(rois.keys()):
            roi = rois[idx]
            rect_tuple = (
                float(roi.x.min),
                float(roi.y.min),
                float(roi.x.max),
                float(roi.y.max),
            )
            if index_to_color is not None and idx in index_to_color:
                rect_tuple = (*rect_tuple, index_to_color[idx])
            rectangles.append(rect_tuple)
        return rectangles

    def to_stream_data(self, rois: dict[int, RectangleROI]) -> dict[str, list[float]]:
        """
        Convert RectangleROI instances to BoxEdit data format.

        Parameters
        ----------
        rois:
            Dictionary mapping ROI index to RectangleROI.

        Returns
        -------
        :
            Dictionary with keys 'x0', 'x1', 'y0', 'y1' in BoxEdit format.
            Empty dict with empty lists if no ROIs.
        """
        if not rois:
            return {"x0": [], "y0": [], "x1": [], "y1": []}

        sorted_indices = sorted(rois.keys())
        return {
            "x0": [rois[i].x.min for i in sorted_indices],
            "y0": [rois[i].y.min for i in sorted_indices],
            "x1": [rois[i].x.max for i in sorted_indices],
            "y1": [rois[i].y.max for i in sorted_indices],
        }


class PolygonConverter:
    """Converter for polygon ROIs using PolyDraw stream."""

    def parse_stream_data(
        self,
        data: dict[str, Any],
        x_unit: str | None,
        y_unit: str | None,
        index_offset: int = 0,
    ) -> dict[int, PolygonROI]:
        """
        Convert PolyDraw data dictionary to PolygonROI instances.

        PolyDraw returns data as a dictionary with keys 'xs', 'ys',
        where each value is a list of lists of coordinates for all polygons.

        Parameters
        ----------
        data:
            Dictionary from PolyDraw stream with keys 'xs', 'ys'.
        x_unit:
            Unit for x coordinates (from the detector data coordinates).
        y_unit:
            Unit for y coordinates (from the detector data coordinates).
        index_offset:
            Starting index for polygon ROIs (e.g., 4 for indices 4-7).

        Returns
        -------
        :
            Dictionary mapping polygon index to PolygonROI. Empty polygons are skipped.
        """
        if not data or not data.get("xs"):
            return {}

        xs_list = data.get("xs", [])
        ys_list = data.get("ys", [])

        rois = {}
        for i, (xs, ys) in enumerate(zip(xs_list, ys_list, strict=True)):
            # Polygons are always closed by HoloViews, so 3 vertices define a triangle.
            if len(xs) < 3 or len(ys) < 3:
                continue

            rois[index_offset + i] = PolygonROI(
                x=list(xs), y=list(ys), x_unit=x_unit, y_unit=y_unit
            )

        return rois

    def to_hv_data(
        self, rois: dict[int, PolygonROI], index_to_color: dict[int, str] | None
    ) -> list[dict[str, Any]]:
        """
        Convert PolygonROI instances to HoloViews Polygons format.

        Parameters
        ----------
        rois:
            Dictionary mapping ROI index to PolygonROI.
        index_to_color:
            Optional mapping from ROI index to color string.

        Returns
        -------
        :
            List of dicts with 'x', 'y' (and optionally 'color') for HoloViews Polygons.
            Returned in sorted order by ROI index.
        """
        polygons = []
        for idx in sorted(rois.keys()):
            roi = rois[idx]
            # Explicit float() ensures Python floats for Bokeh JSON serialization.
            poly_dict: dict[str, Any] = {
                'x': [float(v) for v in roi.x],
                'y': [float(v) for v in roi.y],
            }
            if index_to_color is not None and idx in index_to_color:
                poly_dict['color'] = index_to_color[idx]
            polygons.append(poly_dict)
        return polygons

    def to_stream_data(
        self, rois: dict[int, PolygonROI]
    ) -> dict[str, list[list[float]]]:
        """
        Convert PolygonROI instances to PolyDraw data format.

        Parameters
        ----------
        rois:
            Dictionary mapping ROI index to PolygonROI.

        Returns
        -------
        :
            Dictionary with keys 'xs', 'ys' in PolyDraw format.
            Empty dict with empty lists if no ROIs.
        """
        if not rois:
            return {"xs": [], "ys": []}

        sorted_indices = sorted(rois.keys())
        return {
            "xs": [[float(v) for v in rois[i].x] for i in sorted_indices],
            "ys": [[float(v) for v in rois[i].y] for i in sorted_indices],
        }


if TYPE_CHECKING:
    from ess.livedata.config.workflow_spec import ResultKey

# TypeVars for generic base class
ROIType = TypeVar('ROIType', RectangleROI, PolygonROI)
ParamsType = TypeVar('ParamsType', bound=pydantic.BaseModel)
ConverterType = TypeVar('ConverterType', RectangleConverter, PolygonConverter)


class OptionalRectanglesCoordinates(RectanglesCoordinates):
    """Wrapper for optional rectangle coordinate input.

    Unlike RectanglesCoordinates, this allows empty coordinates
    for request plotters where no initial rectangles are configured.
    """

    coordinates: str = pydantic.Field(
        default="",
        title="Coordinates",
        description='E.g., [0,0,10,10], [20,20,30,30] (leave empty for none)',
    )

    @pydantic.field_validator('coordinates')
    @classmethod
    def validate_coordinates(cls, v: str) -> str:
        """Validate rectangle coordinate structure, allowing empty."""
        v = v.strip()
        if not v:
            return ""  # Allow empty instead of raising
        return super().validate_coordinates(v)


class RectanglesRequestStyle(pydantic.BaseModel):
    """Style options for ROI request rectangles."""

    color: Color = pydantic.Field(
        default=Color("#808080"),
        title="Color",
    )
    line_width: float = pydantic.Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        title="Line Width",
        description="Line width in pixels",
    )
    line_dash: LineDash = pydantic.Field(
        default=LineDash.dashed,
        title="Line Style",
        description="Line style: solid, dashed, dotted, dotdash",
    )


class RectanglesRequestOptions(pydantic.BaseModel):
    """Options for rectangles request plotter."""

    max_roi_count: int = pydantic.Field(
        default_factory=lambda: _get_max_rois_for_geometry("rectangle"),
        ge=1,
        le=_get_max_rois_for_geometry("rectangle"),
        title="Max ROIs",
        description="Maximum number of rectangles that can be drawn.",
    )


class RectanglesRequestParams(pydantic.BaseModel):
    """Parameters for interactive rectangles request plotter."""

    geometry: OptionalRectanglesCoordinates = pydantic.Field(
        default_factory=OptionalRectanglesCoordinates,
        title="Initial Coordinates",
        description="Initial rectangle coordinates. Leave empty for none.",
    )
    style: RectanglesRequestStyle = pydantic.Field(
        default_factory=RectanglesRequestStyle,
        title="Appearance",
        description="Visual styling options.",
    )
    options: RectanglesRequestOptions = pydantic.Field(
        default_factory=RectanglesRequestOptions,
        title="Options",
        description="Drawing options.",
    )


class BaseROIRequestPlotter(Plotter, ABC, Generic[ROIType, ParamsType, ConverterType]):
    """Base class for interactive ROI request plotters.

    Implements the Template Method pattern for creating interactive ROI editors.
    Subclasses provide hooks for ROI type-specific behavior.
    """

    def __init__(
        self,
        params: ParamsType,
        roi_publisher: ROIPublisher | None = None,
    ) -> None:
        super().__init__()
        self._params = params
        self._roi_publisher = roi_publisher
        self._converter = self._create_converter()
        self._roi_mapper = get_roi_mapper()

        # Runtime state set during plot creation
        self._result_key: ResultKey | None = None
        self._current_rois: dict[int, ROIType] = {}
        self._index_offset: int = 0
        # Units extracted from readback data coordinates
        self._x_unit: str | None = None
        self._y_unit: str | None = None
        # Store edit stream reference. While the DynamicMap keeps streams alive via
        # source linkage, we store an explicit reference for clarity and defensive
        # programming in case HoloViews internals change.
        self._edit_stream: hv.streams.Stream | None = None
        # Store DynamicMap for idempotent plot() calls
        self._dmap: hv.DynamicMap | None = None

    def set_roi_publisher(self, publisher: ROIPublisher | None) -> None:
        """Set the ROI publisher for this plotter."""
        self._roi_publisher = publisher

    @abstractmethod
    def _create_converter(self) -> ConverterType:
        """Create the converter for this ROI type."""

    @abstractmethod
    def _geometry_type(self) -> str:
        """Return geometry type name ('rectangle' or 'polygon')."""

    @abstractmethod
    def _get_index_offset(self) -> int:
        """Return index offset for ROI indices (0 for rectangles, 4 for polygons)."""

    @abstractmethod
    def _parse_initial_geometry(self) -> dict[int, ROIType]:
        """Parse initial geometry from params using self._index_offset."""

    @abstractmethod
    def _create_element(self, data: list) -> hv.Element:
        """Create HoloViews element (Rectangles or Polygons)."""

    @abstractmethod
    def _create_edit_stream(
        self, dmap: hv.DynamicMap, initial_data: dict
    ) -> hv.streams.Stream:
        """Create edit stream (BoxEdit or PolyDraw)."""

    @abstractmethod
    def _should_skip_edit(self, new_rois: dict[int, ROIType]) -> bool:
        """Return True if this edit event should be skipped."""

    @abstractmethod
    def _get_style(self) -> Any:
        """Return the style params object with color, line_width, line_dash."""

    def plot(
        self,
        data: sc.DataArray,
        data_key: ResultKey,
        **kwargs,
    ) -> hv.DynamicMap:
        """Create interactive ROI editor with edit stream.

        Parameters
        ----------
        data:
            ROI readback DataArray (used only for job_id extraction).
        data_key:
            Key identifying this data stream (provides job_id for publishing).
        **kwargs:
            Unused additional arguments.

        Returns
        -------
        :
            DynamicMap with edit interactivity.
        """
        del kwargs

        # Return existing DynamicMap if already initialized (idempotent)
        if self._dmap is not None:
            return self._dmap

        self._result_key = data_key
        self._index_offset = self._get_index_offset()

        # Extract units from readback data coordinates (canvas uses these units)
        if 'x' in data.coords:
            self._x_unit = str(data.coords['x'].unit) if data.coords['x'].unit else None
        if 'y' in data.coords:
            self._y_unit = str(data.coords['y'].unit) if data.coords['y'].unit else None
        initial_rois = self._parse_initial_geometry()

        # Create pipe for programmatic updates
        pipe = hv.streams.Pipe(data=[])
        dmap = hv.DynamicMap(self._create_element, streams=[pipe])

        # Create and store edit stream (see __init__ comment for why we store this)
        self._edit_stream = self._create_edit_stream(
            dmap, self._converter.to_stream_data(initial_rois)
        )

        # Initialize pipe with current data
        pipe.send(self._converter.to_hv_data(initial_rois, index_to_color=None))

        # Store initial state
        self._current_rois = initial_rois

        # Watch for edits
        self._edit_stream.param.watch(self._on_edit, 'data')

        # Publish initial state
        self._publish_rois(initial_rois)

        # Apply styling and store for idempotent returns
        self._dmap = self._apply_styling(dmap)
        return self._dmap

    def _on_edit(self, event) -> None:
        """Handle edit stream data changes from user interaction."""
        data = event.new if hasattr(event, 'new') else event
        if data is None:
            data = {}

        try:
            new_rois = self._converter.parse_stream_data(
                data,
                x_unit=self._x_unit,
                y_unit=self._y_unit,
                index_offset=self._index_offset,
            )

            # Skip if unchanged
            if new_rois == self._current_rois:
                return

            # Allow subclass-specific skip logic
            if self._should_skip_edit(new_rois):
                return

            self._current_rois = new_rois
            self._publish_rois(new_rois)

        except Exception as e:
            logger.error("Failed to process %s edit: %s", self._geometry_type(), e)

    def _apply_styling(self, dmap: hv.DynamicMap) -> hv.DynamicMap:
        """Apply common styling options to the DynamicMap."""
        style = self._get_style()
        return dmap.opts(
            color=style.color,
            # Transparent fill so users can see the underlying image while editing.
            fill_alpha=0,
            line_width=style.line_width,
            line_dash=style.line_dash,
            # Bokeh bug: line_dash='dashed' doesn't render with WebGL backend
            backend_opts={'plot.output_backend': 'canvas'},
        )

    def _publish_rois(self, rois: dict[int, ROIType]) -> None:
        """Publish ROIs to Kafka."""
        if not self._roi_publisher or not self._result_key:
            return

        geometry = self._roi_mapper.geometry_for_type(self._geometry_type())
        if geometry is None:
            logger.warning("%s geometry not configured", self._geometry_type())
            return

        self._roi_publisher.publish(self._result_key.job_id, rois, geometry)
        logger.info(
            "Published %d %s ROI(s) for job %s",
            len(rois),
            self._geometry_type(),
            self._result_key.job_id,
        )


class RectanglesRequestPlotter(
    BaseROIRequestPlotter[RectangleROI, RectanglesRequestParams, RectangleConverter]
):
    """Interactive plotter for ROI rectangle requests.

    Creates a BoxEdit-enabled DynamicMap that allows users to draw rectangles.
    Publishes ROI updates to Kafka when shapes are edited.
    """

    def _create_converter(self) -> RectangleConverter:
        return RectangleConverter()

    def _geometry_type(self) -> str:
        return "rectangle"

    def _get_index_offset(self) -> int:
        return 0

    def _parse_initial_geometry(self) -> dict[int, RectangleROI]:
        """Parse initial rectangles from params."""
        coords_str = self._params.geometry.coordinates
        if not coords_str or coords_str.strip() == '':
            return {}

        try:
            rects = self._params.geometry.parse()
        except Exception:
            logger.warning("Failed to parse initial rectangle coordinates")
            return {}

        rois: dict[int, RectangleROI] = {}
        for i, (x0, y0, x1, y1) in enumerate(rects):
            rois[i] = RectangleROI(
                x=Interval(min=min(x0, x1), max=max(x0, x1), unit=None),
                y=Interval(min=min(y0, y1), max=max(y0, y1), unit=None),
            )
        return rois

    def _create_element(self, data: list) -> hv.Rectangles:
        return hv.Rectangles(data)

    def _create_edit_stream(
        self, dmap: hv.DynamicMap, initial_data: dict
    ) -> hv.streams.BoxEdit:
        return hv.streams.BoxEdit(
            source=dmap,
            num_objects=self._params.options.max_roi_count,
            data=initial_data,
        )

    def _should_skip_edit(self, new_rois: dict[int, RectangleROI]) -> bool:
        del new_rois  # Rectangles never skip edits
        return False

    def _get_style(self) -> RectanglesRequestStyle:
        return self._params.style

    @classmethod
    def from_params(cls, params: RectanglesRequestParams) -> RectanglesRequestPlotter:
        """Create plotter from params (concrete type hint for registry)."""
        return cls(params)


class PolygonsRequestStyle(pydantic.BaseModel):
    """Style options for ROI request polygons."""

    color: Color = pydantic.Field(default=Color("#808080"), title="Color")
    line_width: float = pydantic.Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        title="Line Width",
        description="Line width in pixels",
    )
    line_dash: LineDash = pydantic.Field(
        default=LineDash.dashed,
        title="Line Style",
        description="Line style: solid, dashed, dotted, dotdash",
    )


class PolygonsCoordinates(pydantic.BaseModel):
    """Wrapper for polygon coordinate input."""

    coordinates: str = pydantic.Field(
        default="",
        title="Coordinates",
        description='E.g., [[0,0],[10,0],[5,10]], [[20,20],[30,20],[30,30],[20,30]]',
    )

    def parse(self) -> list[tuple[list[float], list[float]]]:
        """Parse validated coordinates into list of (xs, ys) tuples."""
        import json

        coords_str = self.coordinates.strip()
        if not coords_str:
            return []

        try:
            # Parse as JSON array of polygons
            result = json.loads(f"[{coords_str}]")
            polygons = []
            for poly in result:
                if not isinstance(poly, list) or len(poly) < 3:
                    continue
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                polygons.append((xs, ys))
            return polygons
        except (json.JSONDecodeError, IndexError, TypeError):
            return []


class PolygonsRequestOptions(pydantic.BaseModel):
    """Options for polygons request plotter."""

    max_roi_count: int = pydantic.Field(
        default_factory=lambda: _get_max_rois_for_geometry("polygon"),
        ge=1,
        le=_get_max_rois_for_geometry("polygon"),
        title="Max ROIs",
        description="Maximum number of polygons that can be drawn.",
    )


class PolygonsRequestParams(pydantic.BaseModel):
    """Parameters for interactive polygons request plotter."""

    geometry: PolygonsCoordinates = pydantic.Field(
        default_factory=PolygonsCoordinates,
        title="Initial Coordinates",
        description="Initial polygon coordinates. Leave empty for none.",
    )
    style: PolygonsRequestStyle = pydantic.Field(
        default_factory=PolygonsRequestStyle,
        title="Appearance",
        description="Visual styling options.",
    )
    options: PolygonsRequestOptions = pydantic.Field(
        default_factory=PolygonsRequestOptions,
        title="Options",
        description="Drawing options.",
    )


class PolygonsRequestPlotter(
    BaseROIRequestPlotter[PolygonROI, PolygonsRequestParams, PolygonConverter]
):
    """Interactive plotter for ROI polygon requests.

    Creates a PolyDraw-enabled DynamicMap that allows users to draw polygons.
    Publishes ROI updates to Kafka when shapes are edited.
    """

    def _create_converter(self) -> PolygonConverter:
        return PolygonConverter()

    def _geometry_type(self) -> str:
        return "polygon"

    def _get_index_offset(self) -> int:
        # Polygons start after rectangles so each geometry type gets distinct colors
        # from the color cycle when displayed together.
        poly_geom = self._roi_mapper.geometry_for_type("polygon")
        return (
            poly_geom.index_offset if poly_geom else get_default_index_offset("polygon")
        )

    def _parse_initial_geometry(self) -> dict[int, PolygonROI]:
        """Parse initial polygons from params."""
        coords_str = self._params.geometry.coordinates
        if not coords_str or coords_str.strip() == '':
            return {}

        try:
            polygons = self._params.geometry.parse()
        except Exception:
            logger.warning("Failed to parse initial polygon coordinates")
            return {}

        rois: dict[int, PolygonROI] = {}
        for i, (xs, ys) in enumerate(polygons):
            if len(xs) >= 3:
                rois[self._index_offset + i] = PolygonROI(
                    x=xs, y=ys, x_unit=None, y_unit=None
                )
        return rois

    def _create_element(self, data: list) -> hv.Polygons:
        return hv.Polygons(data)

    def _create_edit_stream(
        self, dmap: hv.DynamicMap, initial_data: dict
    ) -> hv.streams.PolyDraw:
        return hv.streams.PolyDraw(
            source=dmap,
            num_objects=self._params.options.max_roi_count,
            drag=True,
            show_vertices=True,
            data=initial_data,
        )

    def _should_skip_edit(self, new_rois: dict[int, PolygonROI]) -> bool:
        """Skip publishing while user is actively drawing a polygon.

        PolyDraw reports the cursor position as a trailing duplicate vertex
        (last vertex == second-to-last vertex). We only publish when the
        user clicks to confirm a vertex, which removes the duplicate.
        This avoids race conditions with backend updates during drawing.

        NOTE: This relies on undocumented Bokeh PolyDrawTool behavior.
        In poly_draw_tool.ts, "new" mode initializes with [x,x]/[y,y] and
        "add" mode captures-then-pushes the last vertex, creating a brief
        duplicate after each click until cursor movement updates it.
        This is fundamental to the rubber-band preview UX but not a
        documented API guarantee.
        """
        for roi in new_rois.values():
            if len(roi.x) >= 2:
                if roi.x[-1] == roi.x[-2] and roi.y[-1] == roi.y[-2]:
                    return True
        return False

    def _get_style(self) -> PolygonsRequestStyle:
        return self._params.style

    @classmethod
    def from_params(cls, params: PolygonsRequestParams) -> PolygonsRequestPlotter:
        """Create plotter from params (concrete type hint for registry)."""
        return cls(params)
