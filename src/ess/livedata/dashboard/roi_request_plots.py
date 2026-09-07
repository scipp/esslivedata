# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Interactive ROI request plotters for user-drawn ROI selection.

These plotters create interactive BoxEdit/PolyDraw elements that allow users
to draw ROIs visually. Edits are published to Kafka for backend processing and
written back into the plotter's params, so that they survive plotter rebuilds.

These plotters subscribe to the ROI readback output stream for source context.
The readback data itself is not used for display - only the DataKey is needed
to identify where to publish ROI updates.

Architecture
------------
ROI request plotters follow the two-stage compute/present pattern:

1. compute(): Extracts data-dependent info (DataKey, coordinate units) and
   forwards the raw data. Called once when data arrives.

2. create_presenter(): Creates a presenter with HoloViews config and an edit
   handler callback. The callback reads and updates the plotter's live ROI
   state, which is shared across all sessions.

3. Presenter.__init__(): Creates session-bound Pipe, DynamicMap, and edit
   streams. Each browser session gets its own presenter instance.

4. Presenter.present(): Returns the pre-created DynamicMap (ignores the
   passed pipe since ROI editors don't update from data changes).

The presenter handles only HoloViews mechanics. All domain logic (ROI parsing,
comparison, skip logic, publishing) stays in the plotter via the edit callback.

Coordinate units
----------------
Stored geometry is bare numbers, meaning coordinates as read off the plot's
axes -- the same numbers the static overlays draw. The unit is stamped on at
parse time from the data and never stored alongside them: it belongs to the
view a layer is bound to, so persisting it would create a value that goes
stale as soon as the layer is pointed at another view, plus rules to resolve
the disagreement. A re-derived unit cannot disagree.

Leaving the unit off is not an option either: ``Interval`` and ``PolygonROI``
read a missing unit as pixel indices, which would silently displace every ROI
on a view with physical coordinates.

Units therefore arrive with the data, which is why stored geometry is seeded
in compute() rather than in __init__.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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

from .data_roles import PRIMARY
from .plots import Plotter, PresenterBase
from .static_plots import Color, LineDash, RectanglesCoordinates

if TYPE_CHECKING:
    from .roi_publisher import ROIPublisher

logger = structlog.get_logger(__name__)

_COORD_PRECISION = 6
"""Significant digits kept in stored ROI coordinates.

A coordinate is a cursor pixel mapped onto the axis, so a 2000-pixel axis
carries barely more than three digits of information and everything past that
is float noise -- noise the user reads, since the stored string is also the
params field they edit. Significant digits rather than decimal places: axes
here run from pixel indices through Q in 1/angstrom to wavelength in m, and a
fixed number of decimals would erase the small-magnitude ones.
"""


def _format_coord(value: float) -> str:
    """Format a coordinate for storage, as JSON the coordinate parsers accept."""
    return f'{value:.{_COORD_PRECISION}g}'


@runtime_checkable
class ROIPublisherAware(Protocol):
    """Protocol for plotters that can publish ROI updates."""

    def set_roi_publisher(self, publisher: ROIPublisher | None) -> None:
        """Set the ROI publisher for this plotter."""
        ...


@runtime_checkable
class ParamsPersisterAware(Protocol):
    """Protocol for plotters that rewrite their own params from user interaction.

    ROI request plotters rewrite their geometry as the user draws. The persister
    stores the updated params with the layer, so that a plotter rebuilt later
    (dashboard restart, layer re-added, new job generation) seeds from the last
    edited state instead of clobbering it with the config-time geometry.
    """

    def set_params_persister(
        self, persister: Callable[[pydantic.BaseModel], None]
    ) -> None:
        """Set the callback that stores params updated by this plotter."""
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
        # Length, not truthiness: the browser syncs the columns back as numpy
        # arrays, whose truth value raises for every length but one.
        if len(data.get("x0", ())) == 0:
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
        if len(data.get("xs", ())) == 0:
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
    from ess.livedata.config.workflow_spec import DataKey


class OptionalRectanglesCoordinates(RectanglesCoordinates):
    """Wrapper for optional rectangle coordinate input.

    Unlike RectanglesCoordinates, this allows empty coordinates
    for request plotters where no initial rectangles are configured.
    """

    coordinates: str = pydantic.Field(
        default="",
        title="Coordinates",
        description='In plot axis units, e.g. [0,0,10,10], [20,20,30,30]',
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
        title="Coordinates",
        description="Rectangles to start from; rewritten as you draw.",
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


class BaseROIRequestPresenter(PresenterBase, ABC):
    """
    Base presenter for ROI request plotters.

    Handles only HoloViews mechanics: creates session-bound Pipe, DynamicMap,
    and edit streams in __init__. Edit events are forwarded to a callback
    provided by the plotter, which handles all domain logic (ROI parsing,
    comparison, skip logic, publishing).

    Parameters
    ----------
    plotter:
        The plotter that created this presenter.
    initial_hv_data:
        Initial data in HoloViews format for pipe initialization.
    initial_stream_data:
        Initial data in edit stream format.
    style:
        Style parameters (color, line_width, line_dash).
    max_roi_count:
        Maximum number of ROIs that can be drawn.
    on_edit:
        Callback for handling edit events. Receives raw edit stream data and
        returns the accepted ROIs in HoloViews form, or ``None`` if the edit
        changed nothing. The plotter provides this callback with
        closure-captured state.
    """

    def __init__(
        self,
        *,
        plotter: Plotter,
        initial_hv_data: list,
        initial_stream_data: dict,
        style: Any,
        max_roi_count: int,
        on_edit: Callable[[dict], list | None],
    ) -> None:
        super().__init__(plotter)
        self._style = style
        self._max_roi_count = max_roi_count
        self._on_edit_callback = on_edit

        # Create session-bound components
        self._pipe = hv.streams.Pipe(data=[])
        self._dmap = hv.DynamicMap(self._create_element, streams=[self._pipe])
        self._edit_stream = self._create_edit_stream(self._dmap, initial_stream_data)

        # Initialize pipe with data
        self._pipe.send(initial_hv_data)

        # Set up edit callback
        self._edit_stream.param.watch(self._handle_edit, 'data')

    @abstractmethod
    def _create_element(self, data: list) -> hv.Element:
        """Create HoloViews element (Rectangles or Polygons)."""

    @abstractmethod
    def _create_edit_stream(
        self, dmap: hv.DynamicMap, initial_data: dict
    ) -> hv.streams.Stream:
        """Create edit stream (BoxEdit or PolyDraw)."""

    def present(self, pipe: hv.streams.Pipe) -> hv.DynamicMap:
        """
        Return pre-created DynamicMap.

        The passed pipe is ignored - ROI request plotters create their own
        internal pipe and don't update from external data changes.

        Parameters
        ----------
        pipe:
            Pipe from the session layer (ignored).

        Returns
        -------
        :
            The session's styled DynamicMap with edit interactivity.
        """
        del pipe  # ROI request plotters don't use the shared pipe for updates
        return self._apply_styling(self._dmap)

    def _handle_edit(self, event) -> None:
        """Forward edit stream events to the plotter's callback.

        The accepted set is pushed back through the pipe, so that the element
        this session renders always matches the plotter's live ROIs. Without
        it the element stays at whatever this presenter was constructed with:
        drawn ROIs live only in the browser's edit tool, and the next
        re-render -- a tab switch, a cell rebuilt on job state change --
        rebuilds that tool from the stale element and syncs the difference
        back as an edit, silently dropping ROIs the user drew.
        """
        data = event.new if hasattr(event, 'new') else event
        try:
            accepted = self._on_edit_callback(data or {})
        except Exception as e:
            logger.error("Failed to process ROI edit: %s", e)
            return
        # None means the edit changed nothing, so the element already agrees.
        if accepted is not None:
            self._pipe.send(accepted)

    def _apply_styling(self, dmap: hv.DynamicMap) -> hv.DynamicMap:
        """Apply common styling options to the DynamicMap."""
        return dmap.opts(
            # Rectangles/Polygons are fillable glyphs, so a bare `color` would set
            # only the (invisible) fill and leave the outline at Bokeh's default.
            line_color=self._style.color,
            # Transparent fill so users can see the underlying image while editing.
            fill_alpha=0,
            line_width=self._style.line_width,
            line_dash=self._style.line_dash,
            # Bokeh bug: line_dash='dashed' doesn't render with WebGL backend
            backend_opts={'plot.output_backend': 'canvas'},
        )


class RectanglesRequestPresenter(BaseROIRequestPresenter):
    """Presenter for rectangle ROI requests using BoxEdit."""

    def _create_element(self, data: list) -> hv.Rectangles:
        return hv.Rectangles(data)

    def _create_edit_stream(
        self, dmap: hv.DynamicMap, initial_data: dict
    ) -> hv.streams.BoxEdit:
        return hv.streams.BoxEdit(
            source=dmap,
            num_objects=self._max_roi_count,
            data=initial_data,
        )


class PolygonsRequestPresenter(BaseROIRequestPresenter):
    """Presenter for polygon ROI requests using PolyDraw."""

    def _create_element(self, data: list) -> hv.Polygons:
        return hv.Polygons(data)

    def _create_edit_stream(
        self, dmap: hv.DynamicMap, initial_data: dict
    ) -> hv.streams.PolyDraw:
        return hv.streams.PolyDraw(
            source=dmap,
            num_objects=self._max_roi_count,
            drag=True,
            show_vertices=True,
            data=initial_data,
        )


class BaseROIRequestPlotter[
    ROIType: (RectangleROI, PolygonROI),
    ParamsType: pydantic.BaseModel,
    ConverterType: (RectangleConverter, PolygonConverter),
](Plotter, ABC):
    """Base class for interactive ROI request plotters.

    Implements compute() to extract data-dependent info and create_presenter()
    to create per-session presenters. Domain logic (ROI parsing, comparison,
    skip logic, publishing, persistence) is handled via a closure-based edit
    callback.
    """

    def __init__(
        self,
        params: ParamsType,
        roi_publisher: ROIPublisher | None = None,
    ) -> None:
        super().__init__()
        self._params = params
        self._roi_publisher = roi_publisher
        self._params_persister: Callable[[pydantic.BaseModel], None] | None = None
        self._converter = self._create_converter()
        self._roi_mapper = get_roi_mapper()

        # Initialize static config from params
        self._index_offset = self._get_index_offset()

        # Live ROI state shared across all sessions of this plotter. Seeded from
        # params on the first compute(), where the axis units become known.
        # Sessions' edit handlers read and update this under _roi_lock; new
        # presenters are seeded from it so a session opened after edits sees
        # the current ROIs.
        self._current_rois: dict[int, ROIType] = {}
        self._seeded_from_params = False
        self._published_initial = False
        self._roi_lock = threading.Lock()

        # Data-dependent state (set during compute())
        self._data_key: DataKey | None = None
        self._x_unit: str | None = None
        self._y_unit: str | None = None

    def set_roi_publisher(self, publisher: ROIPublisher | None) -> None:
        """Set the ROI publisher for this plotter."""
        self._roi_publisher = publisher

    def set_params_persister(
        self, persister: Callable[[pydantic.BaseModel], None]
    ) -> None:
        """Set the callback that stores params updated by this plotter."""
        self._params_persister = persister

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
        """Parse the geometry stored in params, in the plot's axis units."""

    @abstractmethod
    def _format_geometry(self, rois: dict[int, ROIType]) -> str:
        """Serialize ROIs into the coordinate string format params store."""

    @abstractmethod
    def _should_skip_edit(self, new_rois: dict[int, ROIType]) -> bool:
        """Return True if this edit event should be skipped."""

    @abstractmethod
    def _get_style(self) -> Any:
        """Return the style params object with color, line_width, line_dash."""

    @abstractmethod
    def _get_max_roi_count(self) -> int:
        """Return maximum number of ROIs that can be drawn."""

    @abstractmethod
    def create_presenter(self) -> PresenterBase:
        """Create a presenter for this plotter."""

    def compute(
        self, data: dict[str, dict[DataKey, sc.DataArray]], **kwargs
    ) -> dict[DataKey, sc.DataArray]:
        """
        Extract data-dependent info and forward data to presenter.

        Stores the DataKey and coordinate units from the ROI readback data.
        These are used by the edit handler callback created in create_presenter().

        Parameters
        ----------
        data:
            Role-grouped data; the ``primary`` role contains the ROI readback.
        **kwargs:
            Unused.

        Returns
        -------
        :
            The primary-role data, forwarded for potential future use by presenter.
        """
        del kwargs
        primary = data.get(PRIMARY, {})
        data_key, da = next(iter(primary.items()))

        # Store data-dependent info for edit handler
        self._data_key = data_key
        self._x_unit = (
            str(da.coords['x'].unit)
            if 'x' in da.coords and da.coords['x'].unit
            else None
        )
        self._y_unit = (
            str(da.coords['y'].unit)
            if 'y' in da.coords and da.coords['y'].unit
            else None
        )

        # Params hold bare numbers; they mean coordinates on the plot's axes, so
        # they can only be turned into ROIs once the axis units are known.
        # Seeding here rather than in __init__ is safe because no presenter can
        # exist yet: SessionComponents.create builds one only once the layer has
        # a displayable plot, which this call is what produces.
        if not self._seeded_from_params:
            self._current_rois = self._parse_initial_geometry()
            self._seeded_from_params = True

        # Forward data (presenter may use in future)
        self._set_cached_state(primary)
        return primary

    def _create_edit_handler(self) -> Callable[[dict], None]:
        """
        Create an edit handler over the plotter's shared live ROI state.

        The handler parses edit stream data, compares with the shared state,
        applies skip logic, publishes changes and writes them back into the
        layer's stored params. All sessions' handlers read and update the same
        ``_current_rois`` under ``_roi_lock``.

        Also emits the initial ROI set once per plotter lifetime (on the first
        presenter creation after ``compute()`` has set ``_data_key``), never
        again on subsequent session/presenter creation.

        Returns
        -------
        :
            Callback function for handling edit events.
        """

        def handle_edit(stream_data: dict) -> list | None:
            new_rois = self._converter.parse_stream_data(
                stream_data,
                x_unit=self._x_unit,
                y_unit=self._y_unit,
                index_offset=self._index_offset,
            )
            with self._roi_lock:
                # Skip if unchanged
                if new_rois == self._current_rois:
                    return None
                # Apply subclass-specific skip logic
                if self._should_skip_edit(new_rois):
                    return None
                self._current_rois = new_rois
                self._publish_rois(new_rois)
                self._params = self._params_with_geometry(new_rois)
                params = self._params
                accepted = self._converter.to_hv_data(new_rois, index_to_color=None)
            # Outside the ROI lock: the persister reaches back into the plot
            # orchestrator, which takes its own locks and writes the config store.
            if self._params_persister is not None:
                self._params_persister(params)
            return accepted

        # Seeds a backend that has never seen this selection: the backend
        # latches ROI requests by stream name, but only for the lifetime of the
        # service process, so a restarted one starts blank. Stored params are
        # therefore the authority and overwrite whatever the backend holds.
        with self._roi_lock:
            if not self._published_initial and self._publish_rois(self._current_rois):
                self._published_initial = True

        return handle_edit

    def _params_with_geometry(self, rois: dict[int, ROIType]) -> ParamsType:
        """Return a copy of the params carrying ``rois`` as their geometry."""
        geometry = type(self._params.geometry).model_validate(
            self._params.geometry.model_dump()
            | {'coordinates': self._format_geometry(rois)}
        )
        return self._params.model_copy(update={'geometry': geometry})

    def _publish_rois(self, rois: dict[int, ROIType]) -> bool:
        """Publish ROIs to Kafka. Returns True if a message was published."""
        if not self._roi_publisher or not self._data_key:
            return False

        geometry = self._roi_mapper.geometry_for_type(self._geometry_type())
        if geometry is None:
            logger.warning("%s geometry not configured", self._geometry_type())
            return False

        self._roi_publisher.publish(
            workflow_id=self._data_key.workflow_id,
            source_name=self._data_key.source_name,
            rois=rois,
            geometry=geometry,
        )
        logger.info(
            "Published %d %s ROI(s) for source %s",
            len(rois),
            self._geometry_type(),
            self._data_key.source_name,
        )
        return True


class RectanglesRequestPlotter(
    BaseROIRequestPlotter[RectangleROI, RectanglesRequestParams, RectangleConverter]
):
    """Interactive plotter for ROI rectangle requests.

    Creates presenters with BoxEdit-enabled DynamicMaps that allow users
    to draw rectangles. Edits are published to Kafka when shapes are modified.
    """

    def _create_converter(self) -> RectangleConverter:
        return RectangleConverter()

    def _geometry_type(self) -> str:
        return "rectangle"

    def _get_index_offset(self) -> int:
        return 0

    def _parse_initial_geometry(self) -> dict[int, RectangleROI]:
        """Parse the rectangles stored in params, in the plot's axis units."""
        coords_str = self._params.geometry.coordinates
        if not coords_str or coords_str.strip() == '':
            return {}

        # Construction is inside the try because the stored string is user-facing
        # and rounded: a hand-edited or degenerate rectangle (min == max after
        # rounding) fails ``Interval`` validation, which must not take the plot
        # down with it.
        rois: dict[int, RectangleROI] = {}
        try:
            for i, (x0, y0, x1, y1) in enumerate(self._params.geometry.parse()):
                rois[self._index_offset + i] = RectangleROI(
                    x=Interval(min=min(x0, x1), max=max(x0, x1), unit=self._x_unit),
                    y=Interval(min=min(y0, y1), max=max(y0, y1), unit=self._y_unit),
                )
        except Exception:
            logger.warning("Failed to parse initial rectangle coordinates")
            return {}
        return rois

    def _format_geometry(self, rois: dict[int, RectangleROI]) -> str:
        """Serialize rectangles as ``[x0,y0,x1,y1], [x0,y0,x1,y1]``."""

        def corners(roi: RectangleROI) -> str:
            return (
                f'{_format_coord(roi.x.min)},{_format_coord(roi.y.min)},'
                f'{_format_coord(roi.x.max)},{_format_coord(roi.y.max)}'
            )

        return ', '.join(f'[{corners(rois[i])}]' for i in sorted(rois))

    def _should_skip_edit(self, new_rois: dict[int, RectangleROI]) -> bool:
        del new_rois  # Rectangles never skip edits
        return False

    def _get_style(self) -> RectanglesRequestStyle:
        return self._params.style

    def _get_max_roi_count(self) -> int:
        return self._params.options.max_roi_count

    def create_presenter(self) -> RectanglesRequestPresenter:
        """Create a presenter for rectangle ROI requests."""
        presenter = RectanglesRequestPresenter(
            plotter=self,
            initial_hv_data=self._converter.to_hv_data(
                self._current_rois, index_to_color=None
            ),
            initial_stream_data=self._converter.to_stream_data(self._current_rois),
            style=self._get_style(),
            max_roi_count=self._get_max_roi_count(),
            on_edit=self._create_edit_handler(),
        )
        self._presenters.add(presenter)
        return presenter

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
        description=(
            'In plot axis units, e.g. [[0,0],[10,0],[5,10]], '
            '[[20,20],[30,20],[30,30],[20,30]]'
        ),
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
        title="Coordinates",
        description="Polygons to start from; rewritten as you draw.",
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

    Creates presenters with PolyDraw-enabled DynamicMaps that allow users
    to draw polygons. Edits are published to Kafka when shapes are modified.
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
        """Parse the polygons stored in params, in the plot's axis units."""
        coords_str = self._params.geometry.coordinates
        if not coords_str or coords_str.strip() == '':
            return {}

        # Construction is inside the try for the same reason as for rectangles:
        # the stored string is user-facing, so a hand-edited polygon that fails
        # ``PolygonROI`` validation must not take the plot down.
        rois: dict[int, PolygonROI] = {}
        try:
            for i, (xs, ys) in enumerate(self._params.geometry.parse()):
                if len(xs) >= 3:
                    rois[self._index_offset + i] = PolygonROI(
                        x=xs, y=ys, x_unit=self._x_unit, y_unit=self._y_unit
                    )
        except Exception:
            logger.warning("Failed to parse initial polygon coordinates")
            return {}
        return rois

    def _format_geometry(self, rois: dict[int, PolygonROI]) -> str:
        """Serialize polygons as ``[[x,y],[x,y],[x,y]], [[x,y],...]``."""

        def vertices(roi: PolygonROI) -> str:
            return ','.join(
                f'[{_format_coord(x)},{_format_coord(y)}]'
                for x, y in zip(roi.x, roi.y, strict=True)
            )

        return ', '.join(f'[{vertices(rois[i])}]' for i in sorted(rois))

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

    def _get_max_roi_count(self) -> int:
        return self._params.options.max_roi_count

    def create_presenter(self) -> PolygonsRequestPresenter:
        """Create a presenter for polygon ROI requests."""
        presenter = PolygonsRequestPresenter(
            plotter=self,
            initial_hv_data=self._converter.to_hv_data(
                self._current_rois, index_to_color=None
            ),
            initial_stream_data=self._converter.to_stream_data(self._current_rois),
            style=self._get_style(),
            max_roi_count=self._get_max_roi_count(),
            on_edit=self._create_edit_handler(),
        )
        self._presenters.add(presenter)
        return presenter

    @classmethod
    def from_params(cls, params: PolygonsRequestParams) -> PolygonsRequestPlotter:
        """Create plotter from params (concrete type hint for registry)."""
        return cls(params)
