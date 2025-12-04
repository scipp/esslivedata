# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Factory for creating ROI detector plots with interactive BoxEdit."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import holoviews as hv
import param
import scipp as sc

from ess.livedata.config.models import ROI, Interval, PolygonROI, RectangleROI
from ess.livedata.config.roi_names import get_roi_mapper
from ess.livedata.config.workflow_spec import ResultKey

from .data_subscriber import (
    DataSubscriber,
    MergingStreamAssembler,
)
from .extractors import LatestValueExtractor
from .plot_params import (
    LayoutParams,
    PlotParamsROIDetector,
    create_extractors_from_params,
)
from .plots import ImagePlotter, LinePlotter, PlotAspect, PlotAspectType
from .roi_publisher import ROIPublisher
from .stream_manager import StreamManager

ROIType = RectangleROI | PolygonROI


class GeometryConverter(Protocol):
    """Protocol for geometry-specific ROI conversion logic."""

    @property
    def geometry_type(self) -> str:
        """Type identifier ('rectangle' or 'polygon')."""
        ...

    @property
    def roi_type(self) -> type[ROIType]:
        """ROI class for this geometry."""
        ...

    def parse_stream_data(
        self,
        data: dict[str, Any],
        x_unit: str | None,
        y_unit: str | None,
        index_offset: int = 0,
    ) -> dict[int, ROIType]:
        """Convert UI stream data to ROI instances."""
        ...

    def to_hv_data(
        self, rois: dict[int, ROIType], index_to_color: dict[int, str] | None
    ) -> list[Any]:
        """Convert ROIs to HoloViews display format."""
        ...

    def to_stream_data(self, rois: dict[int, ROIType]) -> dict[str, Any]:
        """Convert ROIs to edit stream data format."""
        ...


class RectangleConverter:
    """Converter for rectangle ROIs using BoxEdit stream."""

    @property
    def geometry_type(self) -> str:
        return "rectangle"

    @property
    def roi_type(self) -> type[RectangleROI]:
        return RectangleROI

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

    @property
    def geometry_type(self) -> str:
        return "polygon"

    @property
    def roi_type(self) -> type[PolygonROI]:
        return PolygonROI

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
            # Skip polygons with fewer than 3 vertices
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


def parse_readback_by_type(
    roi_data: sc.DataArray,
    roi_type: type[ROIType],
    logger: logging.Logger | None = None,
) -> dict[int, ROIType]:
    """
    Parse ROI readback data from backend, filtering by geometry type.

    Parameters
    ----------
    roi_data:
        Concatenated ROI data array with roi_index coordinate.
    roi_type:
        ROI type class to filter for (e.g., RectangleROI or PolygonROI).
    logger:
        Optional logger for debug messages.

    Returns
    -------
    :
        Dictionary mapping ROI index to ROI instances of the specified type.
        Returns empty dict if parsing fails.
    """
    try:
        rois = ROI.from_concatenated_data_array(roi_data)
        return {idx: roi for idx, roi in rois.items() if isinstance(roi, roi_type)}
    except Exception as e:
        if logger:
            logger.debug("Failed to parse %s readback data: %s", roi_type.__name__, e)
        return {}


class GeometryHandler:
    """
    Encapsulates geometry-specific state and operations for ROI editing.

    Each handler manages a single geometry type (rectangle or polygon),
    including its request/readback state, UI streams, and conversion logic.
    Conversion logic is delegated to a GeometryConverter.

    Parameters
    ----------
    converter:
        Converter implementing geometry-specific parsing and formatting.
    edit_stream:
        HoloViews edit stream (BoxEdit or PolyDraw).
    request_pipe:
        Pipe for updating request layer visuals.
    readback_pipe:
        Pipe for updating readback layer visuals.
    index_offset:
        Starting index for ROIs of this type (0 for rectangles, 4 for polygons).
    initial_rois:
        Optional initial ROI configurations.
    """

    def __init__(
        self,
        converter: GeometryConverter,
        edit_stream: hv.streams.BoxEdit | hv.streams.PolyDraw,
        request_pipe: hv.streams.Pipe,
        readback_pipe: hv.streams.Pipe,
        index_offset: int = 0,
        initial_rois: dict[int, ROIType] | None = None,
    ) -> None:
        self._converter = converter
        self.edit_stream = edit_stream
        self.request_pipe = request_pipe
        self.readback_pipe = readback_pipe
        self.index_offset = index_offset

        # Separate request (user's pending changes) and readback (backend truth)
        self._request_rois: dict[int, ROIType] = (
            initial_rois.copy() if initial_rois else {}
        )
        self._readback_rois: dict[int, ROIType] = (
            initial_rois.copy() if initial_rois else {}
        )

    @property
    def geometry_type(self) -> str:
        """Type identifier ('rectangle' or 'polygon')."""
        return self._converter.geometry_type

    @property
    def roi_type(self) -> type[ROIType]:
        """ROI class for this geometry."""
        return self._converter.roi_type

    @property
    def request_rois(self) -> dict[int, ROIType]:
        """Current request ROIs (user's pending changes)."""
        return self._request_rois

    @property
    def readback_rois(self) -> dict[int, ROIType]:
        """Current readback ROIs (backend truth)."""
        return self._readback_rois

    @property
    def active_indices(self) -> set[int]:
        """Indices of currently active ROIs based on readback (backend truth)."""
        return set(self._readback_rois.keys())

    def parse_stream_data(
        self, data: dict[str, Any], x_unit: str | None, y_unit: str | None
    ) -> dict[int, ROIType]:
        """Convert UI stream data to ROI instances."""
        return self._converter.parse_stream_data(
            data, x_unit, y_unit, self.index_offset
        )

    def to_hv_data(
        self, rois: dict[int, ROIType], index_to_color: dict[int, str] | None
    ) -> list[Any]:
        """Convert ROIs to HoloViews display format."""
        return self._converter.to_hv_data(rois, index_to_color)

    def to_stream_data(self, rois: dict[int, ROIType]) -> dict[str, Any]:
        """Convert ROIs to edit stream data format."""
        return self._converter.to_stream_data(rois)

    def update_request_state_only(self, rois: dict[int, ROIType]) -> None:
        """Update request state without updating the visual pipe.

        Use this when responding to user edits - the polygon is already
        visible via PolyDraw/BoxEdit, so we only need to track state.
        Updating the pipe would disrupt the edit tool's selection state.
        """
        self._request_rois = rois

    def update_request(self, rois: dict[int, ROIType]) -> None:
        """Update request state and send to request pipe.

        Use this for backend-initiated updates where we need to render
        the ROIs visually. For user edits, use update_request_state_only.
        Request ROIs use neutral styling (no colors).
        """
        self._request_rois = rois
        hv_data = self.to_hv_data(rois, index_to_color=None)
        self.request_pipe.send(hv_data)

    def update_readback(
        self, rois: dict[int, ROIType], index_to_color: dict[int, str] | None
    ) -> None:
        """Update readback state and send to readback pipe."""
        self._readback_rois = rois
        hv_data = self.to_hv_data(rois, index_to_color)
        self.readback_pipe.send(hv_data)

    def sync_stream_from_rois(self, rois: dict[int, ROIType]) -> None:
        """Sync the edit stream data from ROIs (for backend updates)."""
        stream_data = self.to_stream_data(rois)
        self.edit_stream.event(data=stream_data)


class ROIPlotState:
    """
    Per-plot state for ROI detector plots.

    Encapsulates state and callbacks for a single ROI detector plot,
    including active ROI tracking, publishing logic, and edit streams for
    both rectangle (BoxEdit) and polygon (PolyDraw) ROIs.

    This class implements bidirectional synchronization with Kafka:
    - User edits trigger publishes to backend (on_box_change, on_poly_change)
    - Backend updates trigger UI updates (on_backend_rect_update, ...)
    - Kafka is the single source of truth; backend state always wins

    Two visual layers are maintained for each geometry type:
    - Request ROIs: Interactive dashed shapes showing user's pending changes
    - Readback ROIs: Non-interactive solid shapes showing backend confirmed state

    Parameters
    ----------
    result_key:
        ResultKey identifying this detector plot.
    box_stream:
        HoloViews BoxEdit stream for rectangle ROIs.
    rect_request_pipe:
        HoloViews Pipe stream for programmatically updating request rectangles.
    rect_readback_pipe:
        HoloViews Pipe stream for programmatically updating readback rectangles.
    poly_stream:
        HoloViews PolyDraw stream for polygon ROIs.
    poly_request_pipe:
        HoloViews Pipe stream for programmatically updating request polygons.
    poly_readback_pipe:
        HoloViews Pipe stream for programmatically updating readback polygons.
    roi_state_stream:
        HoloViews Stream for broadcasting active ROI indices to spectrum plot.
    x_unit:
        Unit for x coordinates.
    y_unit:
        Unit for y coordinates.
    roi_publisher:
        Publisher for ROI updates. If None, publishing is disabled.
    logger:
        Logger instance.
    colors:
        List of colors to use for ROI shapes, indexed by ROI number.
    initial_rect_rois:
        Optional dictionary of initial rectangle ROI configurations.
    initial_poly_rois:
        Optional dictionary of initial polygon ROI configurations.
    """

    def __init__(
        self,
        result_key: ResultKey,
        box_stream: hv.streams.BoxEdit,
        rect_request_pipe: hv.streams.Pipe,
        rect_readback_pipe: hv.streams.Pipe,
        poly_stream: hv.streams.PolyDraw,
        poly_request_pipe: hv.streams.Pipe,
        poly_readback_pipe: hv.streams.Pipe,
        roi_state_stream: hv.streams.Stream,
        x_unit: str | None,
        y_unit: str | None,
        roi_publisher: ROIPublisher | None,
        logger: logging.Logger,
        colors: list[str],
        initial_rect_rois: dict[int, RectangleROI] | None = None,
        initial_poly_rois: dict[int, PolygonROI] | None = None,
        roi_mapper=None,
    ) -> None:
        self.result_key = result_key
        self.roi_state_stream = roi_state_stream
        self.x_unit = x_unit
        self.y_unit = y_unit
        self._roi_publisher = roi_publisher
        self._logger = logger
        self._colors = colors
        self._roi_mapper = roi_mapper or get_roi_mapper()

        # Create geometry handlers
        self._rect_handler = GeometryHandler(
            converter=RectangleConverter(),
            edit_stream=box_stream,
            request_pipe=rect_request_pipe,
            readback_pipe=rect_readback_pipe,
            index_offset=0,
            initial_rois=initial_rect_rois,
        )

        poly_index_offset = self._get_polygon_index_offset()
        self._poly_handler = GeometryHandler(
            converter=PolygonConverter(),
            edit_stream=poly_stream,
            request_pipe=poly_request_pipe,
            readback_pipe=poly_readback_pipe,
            index_offset=poly_index_offset,
            initial_rois=initial_poly_rois,
        )

        # Attach callbacks AFTER initializing state
        self._rect_handler.edit_stream.param.watch(self.on_box_change, "data")
        self._poly_handler.edit_stream.param.watch(self.on_poly_change, "data")

        # Initialize roi_state_stream with the current active ROI indices
        self.roi_state_stream.event(active_rois=self._active_roi_indices)

    @property
    def box_stream(self) -> hv.streams.BoxEdit:
        """BoxEdit stream for rectangle ROIs."""
        return self._rect_handler.edit_stream

    @property
    def poly_stream(self) -> hv.streams.PolyDraw:
        """PolyDraw stream for polygon ROIs."""
        return self._poly_handler.edit_stream

    @property
    def _active_roi_indices(self) -> set[int]:
        """Combined indices of active ROIs from both geometry types (readback)."""
        rect_indices = self._rect_handler.active_indices
        poly_indices = self._poly_handler.active_indices
        return rect_indices | poly_indices

    def _compute_index_to_color(self) -> dict[int, str]:
        """
        Compute color mapping based on ROI index value.

        Colors are assigned by ROI index (not position), providing stable color
        identity. ROI 3 always has the same color regardless of which other ROIs
        are active. This matches the coloring in Overlay1DPlotter for consistency
        between detector overlay and spectrum plots.
        """
        return {
            idx: self._colors[idx % len(self._colors)]
            for idx in self._active_roi_indices
        }

    def _recolor_all_readbacks(self) -> None:
        """
        Recolor all readback ROIs based on their index values.

        Called when active ROIs change to update visual colors.
        """
        index_to_color = self._compute_index_to_color()

        # Update rectangle readback visuals
        rect_hv_data = self._rect_handler.to_hv_data(
            self._rect_handler.readback_rois, index_to_color
        )
        self._rect_handler.readback_pipe.send(rect_hv_data)

        # Update polygon readback visuals
        poly_hv_data = self._poly_handler.to_hv_data(
            self._poly_handler.readback_rois, index_to_color
        )
        self._poly_handler.readback_pipe.send(poly_hv_data)

    def _get_polygon_index_offset(self) -> int:
        """Get the index offset for polygon ROIs from the mapper."""
        poly_geom = next(
            (g for g in self._roi_mapper.geometries if g.geometry_type == "polygon"),
            None,
        )
        return poly_geom.index_offset if poly_geom else 4

    def _publish_geometry(self, handler: GeometryHandler) -> None:
        """Publish ROIs for a geometry type to backend."""
        if not self._roi_publisher:
            return

        geometry = self._roi_mapper.geometry_for_type(handler.geometry_type)
        if geometry is None:
            self._logger.warning("Unknown geometry type: %s", handler.geometry_type)
            return

        self._roi_publisher.publish(
            self.result_key.job_id,
            handler.request_rois,
            geometry,
        )
        self._logger.info(
            "Published %d %s ROI(s) for job %s",
            len(handler.request_rois),
            handler.geometry_type,
            self.result_key.job_id,
        )

    def _on_stream_change(self, handler: GeometryHandler, event) -> None:
        """
        Handle edit stream data changes from UI for any geometry type.

        Updates request layer immediately (optimistic UI) and publishes to backend.
        Only publishes if ROIs differ from current request state to prevent cycles.

        Parameters
        ----------
        handler:
            Geometry handler for the affected geometry type.
        event:
            Event object from edit stream.
        """
        data = event.new if hasattr(event, "new") else event
        if data is None:
            data = {}

        try:
            current_rois = handler.parse_stream_data(data, self.x_unit, self.y_unit)

            # Only update and publish if ROIs changed from current request state
            if current_rois == handler.request_rois:
                return

            # Skip publishing while user is actively drawing a polygon.
            # PolyDraw reports the cursor position as a trailing duplicate vertex
            # (last vertex == second-to-last vertex). We only publish when the
            # user clicks to confirm a vertex, which removes the duplicate.
            # This avoids race conditions with backend updates during drawing.
            #
            # NOTE: This relies on undocumented Bokeh PolyDrawTool behavior.
            # In poly_draw_tool.ts, "new" mode initializes with [x,x]/[y,y] and
            # "add" mode captures-then-pushes the last vertex, creating a brief
            # duplicate after each click until cursor movement updates it.
            # This is fundamental to the rubber-band preview UX but not a
            # documented API guarantee.
            for roi in current_rois.values():
                if isinstance(roi, PolygonROI) and len(roi.x) >= 2:
                    if roi.x[-1] == roi.x[-2] and roi.y[-1] == roi.y[-2]:
                        return  # Still drawing, don't publish yet

            # Update state only - don't touch the visual pipe.
            # The user's edit is already visible via PolyDraw/BoxEdit.
            # Updating the pipe would disrupt the tool's selection state,
            # causing issues like new polygons extending existing ones.
            handler.update_request_state_only(current_rois)

            # Publish to backend
            self._publish_geometry(handler)

        except Exception as e:
            self._logger.error(
                "Failed to publish %s ROI update: %s", handler.geometry_type, e
            )

    def _on_backend_update(
        self, handler: GeometryHandler, backend_rois: dict[int, ROIType]
    ) -> None:
        """
        Handle ROI updates from the backend stream for any geometry type.

        The backend is the single source of truth for ROI state. This method
        updates the readback layer with the authoritative backend state,
        and syncs the request layer to match if needed.

        Parameters
        ----------
        handler:
            Geometry handler for the affected geometry type.
        backend_rois:
            Dictionary mapping ROI index to ROI from backend.
        """
        try:
            readback_changed = backend_rois != handler.readback_rois
            request_needs_sync = backend_rois != handler.request_rois

            if not readback_changed and not request_needs_sync:
                return

            self._logger.debug(
                "Applying backend %s ROI update for job %s",
                handler.geometry_type,
                self.result_key.job_id,
            )

            # Update readback layer if changed (authoritative backend state)
            if readback_changed:
                # Update state first (affects _active_roi_indices computation)
                handler._readback_rois = backend_rois
                # Recolor ALL readbacks based on new global positions
                self._recolor_all_readbacks()
                self.roi_state_stream.event(active_rois=self._active_roi_indices)

            # Sync request layer to match backend if needed
            # TODO: Backend sync is disabled - causes issues in multi-session case.
            if request_needs_sync:
                handler.update_request(backend_rois)
                handler.sync_stream_from_rois(backend_rois)

            self._logger.info(
                "UI updated with %d %s ROI(s) from backend for job %s",
                len(backend_rois),
                handler.geometry_type,
                self.result_key.job_id,
            )
        except Exception:
            self._logger.exception(
                "Failed to update UI from backend %s ROI data", handler.geometry_type
            )

    def on_box_change(self, event) -> None:
        """Handle BoxEdit data changes from UI."""
        self._on_stream_change(self._rect_handler, event)

    def on_poly_change(self, event) -> None:
        """Handle PolyDraw data changes from UI."""
        self._on_stream_change(self._poly_handler, event)

    def on_backend_rect_update(self, backend_rois: dict[int, RectangleROI]) -> None:
        """Handle rectangle ROI updates from the backend stream."""
        self._on_backend_update(self._rect_handler, backend_rois)

    def on_backend_poly_update(self, backend_rois: dict[int, PolygonROI]) -> None:
        """Handle polygon ROI updates from the backend stream."""
        self._on_backend_update(self._poly_handler, backend_rois)

    def is_roi_active(self, key: ResultKey) -> bool:
        """
        Check if the ROI index for this key is currently active.

        Parameters
        ----------
        key:
            ResultKey to check.

        Returns
        -------
        :
            True if the ROI index is active, False otherwise.
        """
        roi_index = self._roi_mapper.parse_roi_index(key.output_name)
        return roi_index is not None and roi_index in self._active_roi_indices


class ROIDetectorPlotFactory:
    """
    Factory for creating ROI detector plots with interactive box editing.

    Handles the creation of interactive detector plots with ROI selection
    via BoxEdit overlays, and manages ROI publishing to Kafka.

    Parameters
    ----------
    stream_manager:
        Manager for creating data streams.
    roi_publisher:
        Publisher for ROI updates to Kafka. If None, ROI publishing is disabled.
    logger:
        Logger instance. If None, creates a logger using the module name.
    """

    def __init__(
        self,
        stream_manager: StreamManager,
        roi_publisher: ROIPublisher | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._stream_manager = stream_manager
        self._roi_publisher = roi_publisher
        self._logger = logger or logging.getLogger(__name__)
        self._roi_mapper = get_roi_mapper()

    def _parse_roi_index(self, output_name: str) -> int | None:
        """
        Extract ROI index from output name.

        Parameters
        ----------
        output_name:
            Output name in format 'roi_current_{index}' or 'roi_cumulative_{index}'.

        Returns
        -------
        :
            ROI index if parsing succeeds, None otherwise.
        """
        return self._roi_mapper.parse_roi_index(output_name)

    def _generate_spectrum_keys(self, detector_key: ResultKey) -> list[ResultKey]:
        """
        Generate spectrum keys for ROI histogram outputs matching detector type.

        Generates only current or cumulative histogram keys based on the detector's
        output_name. If detector shows 'current', generates roi_current_*. If
        detector shows 'cumulative', generates roi_cumulative_*.

        Parameters
        ----------
        detector_key:
            ResultKey identifying the detector output. Must have output_name set
            to either 'current' or 'cumulative'.

        Returns
        -------
        :
            List of ResultKeys for ROI histogram outputs matching detector type.
        """
        # Determine which histogram type based on detector output_name
        if detector_key.output_name == "current":
            histogram_keys = self._roi_mapper.all_current_keys()
        elif detector_key.output_name == "cumulative":
            histogram_keys = self._roi_mapper.all_cumulative_keys()
        else:
            # Fallback for unexpected output_name - generate all keys
            histogram_keys = self._roi_mapper.all_histogram_keys()

        return [
            detector_key.model_copy(update={"output_name": key})
            for key in histogram_keys
        ]

    @staticmethod
    def _extract_unit_for_dim(detector_data: sc.DataArray, dim: str) -> str | None:
        """
        Extract unit for a specific dimension from detector data.

        Parameters
        ----------
        detector_data:
            Detector data array to extract unit from.
        dim:
            Dimension name to extract unit for.

        Returns
        -------
        :
            Unit as string, or None if not present.
        """
        if dim not in detector_data.coords:
            return None

        coord_unit = detector_data.coords[dim].unit
        return str(coord_unit) if coord_unit is not None else None

    def _subscribe_to_readback(
        self,
        roi_readback_key: ResultKey,
        roi_type: type[ROIType],
        update_callback,
    ) -> None:
        """
        Subscribe to ROI readback stream from backend for a specific geometry type.

        Creates a subscription that updates the plot's ROIs when
        the backend publishes ROI readback data, enabling bidirectional sync.

        Parameters
        ----------
        roi_readback_key:
            ResultKey for the ROI readback stream.
        roi_type:
            ROI type class (RectangleROI or PolygonROI) for parsing.
        update_callback:
            Callback to invoke with parsed ROIs.
        """

        def on_data_update(data: dict[ResultKey, sc.DataArray]) -> None:
            """Callback for ROI readback data updates."""
            if roi_readback_key not in data:
                return

            roi_data = data[roi_readback_key]
            parsed_rois = parse_readback_by_type(roi_data, roi_type, self._logger)
            # Always update, even if empty dict - this is needed to sync
            # ROI deletion across multiple plots
            if parsed_rois is not None:
                update_callback(parsed_rois)

        # Subscribe to ROI stream if it exists in DataService
        if roi_readback_key in self._stream_manager.data_service:
            # Initialize with current data
            initial_data = {
                roi_readback_key: self._stream_manager.data_service[roi_readback_key]
            }
            on_data_update(initial_data)

        # Create a custom pipe that calls our callback
        class ReadbackPipe:
            def __init__(self, callback):
                self.callback = callback

            def send(self, data):
                self.callback(data)

        def pipe_factory(data):
            """Factory function to create ReadbackPipe with callback."""
            return ReadbackPipe(on_data_update)

        assembler = MergingStreamAssembler({roi_readback_key})
        extractors = {roi_readback_key: LatestValueExtractor()}
        subscriber = DataSubscriber(assembler, pipe_factory, extractors)
        self._stream_manager.data_service.register_subscriber(subscriber)

    def _subscribe_to_rect_readback(
        self, roi_readback_key: ResultKey, plot_state: ROIPlotState
    ) -> None:
        """Subscribe to rectangle ROI readback stream from backend."""
        self._subscribe_to_readback(
            roi_readback_key, RectangleROI, plot_state.on_backend_rect_update
        )

    def _subscribe_to_polygon_readback(
        self, roi_readback_key: ResultKey, plot_state: ROIPlotState
    ) -> None:
        """Subscribe to polygon ROI readback stream from backend."""
        self._subscribe_to_readback(
            roi_readback_key, PolygonROI, plot_state.on_backend_poly_update
        )

    def create_roi_detector_plot_components(
        self,
        detector_key: ResultKey,
        params: PlotParamsROIDetector,
        detector_pipe: hv.streams.Pipe,
    ) -> tuple[hv.DynamicMap, hv.DynamicMap, ROIPlotState]:
        """
        Create ROI detector plot components without layout assembly.

        This is the testable public API that creates the individual components
        for an ROI detector plot. It returns the detector DynamicMap, ROI spectrum
        DynamicMap, and the plot state, allowing the caller to control layout
        or access components for testing.

        The plot_state is not stored internally - it's kept alive by references
        from the returned plot components (via callbacks and stream filters).

        Initial ROI configurations are automatically loaded from DataService via
        the ROI readback subscription if available.

        Parameters
        ----------
        detector_key:
            ResultKey identifying the detector output.
        params:
            The plotter parameters (PlotParamsROIDetector).
        detector_pipe:
            Pre-configured pipe for detector data with data already present.
            This pipe should have been created by the caller using proper
            extractors (via create_extractors_from_params).

        Returns
        -------
        :
            Tuple of (detector_dmap, roi_dmap, plot_state).
        """
        if not isinstance(params, PlotParamsROIDetector):
            raise TypeError("roi_detector requires PlotParamsROIDetector")

        # Detector subscription is managed by the caller and passed via detector_pipe.
        # This factory only creates spectrum and readback subscriptions.

        detector_plotter = ImagePlotter(
            value_margin_factor=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
        )
        # Use extracted data from pipe for plotter initialization
        detector_plotter.initialize_from_data(detector_pipe.data)

        detector_dmap = hv.DynamicMap(
            detector_plotter, streams=[detector_pipe], cache_size=1
        ).opts(shared_axes=False)

        # Get color cycle for ROI styling
        default_colors = hv.Cycle.default_cycles["default_colors"]
        # Get max counts for each geometry type
        # Use params.roi_options.max_roi_count for rectangles (user override)
        poly_geom = next(
            (g for g in self._roi_mapper.geometries if g.geometry_type == "polygon"),
            None,
        )
        max_rect_count = params.roi_options.max_roi_count
        max_poly_count = poly_geom.num_rois if poly_geom else 4
        total_roi_count = max_rect_count + max_poly_count
        colors_list = default_colors[:total_roi_count]

        # There is a very particular way these components must be created, or else the
        # interactiveity will not work. In particular:
        # - Naively composing and wrapping DynamicMap(Image * Rectangles) will lead to a
        #   defunct BoxEdit that does not respond to clicks.
        # - Wrapping only the image, DynamicMap(Image) * Rectangles, makes the BoxEdit
        #   work, but updating the rectangles programmatically does not work.
        # - The only way we have found that works is to create a Pipe for the rectangles
        #   and use a DynamicMap to wrap the Rectangles, and then compose that with
        #   the DynamicMap for the image. This allows both programmatic updates and
        #   user interaction to work correctly. The key insight is to use the
        #   DynamicMap(Rectangles) as source for BoxEdit, not the Rectangles element.

        # === RECTANGLE ROI SETUP ===
        # Create two separate layers:
        # 1. Readback rectangles (solid, non-interactive) - backend truth
        # 2. Request rectangles (dashed, interactive) - user's pending changes

        rect_readback_pipe = hv.streams.Pipe(data=[])
        rect_request_pipe = hv.streams.Pipe(data=[])

        def make_readback_boxes(data: list):
            return hv.Rectangles(data, vdims=['color']).opts(color='color')

        def make_request_boxes(data: list):
            return hv.Rectangles(data)

        rect_readback_dmap = hv.DynamicMap(
            make_readback_boxes, streams=[rect_readback_pipe]
        )
        rect_request_dmap = hv.DynamicMap(
            make_request_boxes, streams=[rect_request_pipe]
        )

        # Create BoxEdit stream with rect_request_dmap as source
        box_stream = hv.streams.BoxEdit(
            source=rect_request_dmap, num_objects=max_rect_count, data={}
        )

        # === POLYGON ROI SETUP ===
        # Same two-layer pattern for polygons

        poly_readback_pipe = hv.streams.Pipe(data=[])
        poly_request_pipe = hv.streams.Pipe(data=[])

        def make_readback_polygons(data: list):
            if not data:
                return hv.Polygons([])
            return hv.Polygons(data, vdims=['color']).opts(color='color')

        def make_request_polygons(data: list):
            if not data:
                return hv.Polygons([])
            return hv.Polygons(data)

        poly_readback_dmap = hv.DynamicMap(
            make_readback_polygons, streams=[poly_readback_pipe]
        )
        poly_request_dmap = hv.DynamicMap(
            make_request_polygons, streams=[poly_request_pipe]
        )

        # Create PolyDraw stream with poly_request_dmap as source
        poly_stream = hv.streams.PolyDraw(
            source=poly_request_dmap,
            num_objects=max_poly_count,
            drag=True,
            show_vertices=True,
            data={'xs': [], 'ys': []},
        )

        # Extract coordinate units from the extracted detector data in pipe
        detector_data = detector_pipe.data[detector_key]
        x_dim, y_dim = detector_data.dims[1], detector_data.dims[0]
        x_unit = self._extract_unit_for_dim(detector_data, x_dim)
        y_unit = self._extract_unit_for_dim(detector_data, y_dim)

        # Create stream for broadcasting active ROI indices to spectrum plot
        # Use a custom Stream class to avoid parameter name clash with spectrum_pipe
        class ROIStateStream(hv.streams.Stream):
            active_rois = param.Parameter(
                default=set(), doc="Set of active ROI indices"
            )

        roi_state_stream = ROIStateStream()

        # Create plot state (which will attach the watchers to streams)
        # Note: plot_state is kept alive by references from the returned plot:
        # - box_stream/poly_stream hold callback references
        # - roi_state_stream is referenced by the spectrum plot DynamicMap
        # - readback pipes hold plot_state callback references
        plot_state = ROIPlotState(
            result_key=detector_key,
            box_stream=box_stream,
            rect_request_pipe=rect_request_pipe,
            rect_readback_pipe=rect_readback_pipe,
            poly_stream=poly_stream,
            poly_request_pipe=poly_request_pipe,
            poly_readback_pipe=poly_readback_pipe,
            roi_state_stream=roi_state_stream,
            x_unit=x_unit,
            y_unit=y_unit,
            roi_publisher=self._roi_publisher,
            logger=self._logger,
            colors=colors_list,
            initial_rect_rois=None,
            initial_poly_rois=None,
            roi_mapper=self._roi_mapper,
        )

        # Subscribe to ROI readback streams from backend for bidirectional sync.
        # Rectangle readback
        rect_readback_key = detector_key.model_copy(
            update={"output_name": "roi_rectangle"}
        )
        self._subscribe_to_rect_readback(rect_readback_key, plot_state)

        # Polygon readback
        poly_readback_key = detector_key.model_copy(
            update={"output_name": "roi_polygon"}
        )
        self._subscribe_to_polygon_readback(poly_readback_key, plot_state)

        # === STYLE AND COMPOSE LAYERS ===
        # Layer order: detector, rect_readback, rect_request, poly_readback,
        # poly_request
        rect_readback_styled = rect_readback_dmap.opts(
            fill_alpha=0.3, line_width=2, line_dash='solid'
        )
        rect_request_styled = rect_request_dmap.opts(
            color='gray',
            fill_alpha=0,
            line_width=2,
            line_dash='dashed',
            # Bokeh bug: line_dash='dashed' doesn't render with WebGL backend
            backend_opts={'plot.output_backend': 'canvas'},
        )
        poly_readback_styled = poly_readback_dmap.opts(
            fill_alpha=0.3, line_width=2, line_dash='solid'
        )
        poly_request_styled = poly_request_dmap.opts(
            color='gray',
            fill_alpha=0,
            line_width=2,
            line_dash='dashed',
            backend_opts={'plot.output_backend': 'canvas'},
        )

        detector_with_rois = (
            detector_dmap
            * rect_readback_styled
            * rect_request_styled
            * poly_readback_styled
            * poly_request_styled
        )

        # Generate spectrum keys and create ROI spectrum plot
        spectrum_keys = self._generate_spectrum_keys(detector_key)
        roi_spectrum_dmap = self._create_roi_spectrum_plot(
            spectrum_keys, roi_state_stream, params
        )

        return detector_with_rois, roi_spectrum_dmap, plot_state

    def _create_roi_spectrum_plot(
        self,
        spectrum_keys: list[ResultKey],
        roi_state_stream: hv.streams.Stream,
        params: PlotParamsROIDetector,
    ) -> hv.DynamicMap:
        """
        Create ROI spectrum plot that overlays all active ROI spectra.

        Parameters
        ----------
        spectrum_keys:
            List of ResultKeys for ROI spectrum outputs.
        roi_state_stream:
            Stream carrying active ROI indices (set[int]) via 'active_rois' parameter.
        params:
            The plotter parameters (PlotParamsROIDetector).

        Returns
        -------
        :
            DynamicMap for the ROI spectrum plot.
        """
        overlay_layout = LayoutParams(combine_mode="overlay")

        # FIXME: Memory leak - subscribers registered via stream_manager are never
        # unregistered. When this plot is closed, the subscriber remains in
        # DataService._subscribers, preventing garbage collection of plot components.
        extractors = create_extractors_from_params(spectrum_keys, params.window)
        spectrum_pipe = self._stream_manager.make_merging_stream(extractors)

        spectrum_plotter = LinePlotter(
            value_margin_factor=0.1,
            layout_params=overlay_layout,
            aspect_params=PlotAspect(aspect_type=PlotAspectType.square),
            scale_opts=params.plot_scale,
        )

        # Create filtering wrapper that filters spectrum data based on active ROIs
        #
        # LIMITATION: Spectrum curve colors may not match ROI shape colors.
        # Shapes use position-based coloring (sorted by ROI index across all
        # geometry types). HoloViews overlay assigns colors by "order of first
        # appearance" and caches this, ignoring subsequent reordering.
        #
        # Root cause: Rectangles use indices 0-3, polygons use indices 4-7
        # (fixed offset). Drawing rect→poly→rect gives indices 0,4,1 but
        # overlay sees appearance order 0,4,1 and assigns colors 0,1,2.
        # Shapes recolor to sorted positions (0→c0, 1→c1, 4→c2) but overlay
        # keeps its cached assignment.
        #
        # Potential fix: Use contiguous indexing across geometry types (no
        # offset), so drawing order matches index order. Requires backend
        # changes to ROI stream naming and handler logic.
        def filtered_spectrum_plotter(
            data: dict[ResultKey, sc.DataArray], active_rois: set[int]
        ) -> hv.Overlay | hv.Layout | hv.Element:
            """Filter spectrum data to only include active ROIs."""
            filtered_data = {
                key: value
                for key, value in data.items()
                if self._roi_mapper.parse_roi_index(key.output_name) in active_rois
            }
            return spectrum_plotter(filtered_data)

        return hv.DynamicMap(
            filtered_spectrum_plotter,
            streams=[spectrum_pipe, roi_state_stream],
            cache_size=1,
        ).opts(shared_axes=False, max_width=400)
