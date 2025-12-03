# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Factory for creating ROI detector plots with interactive BoxEdit."""

from __future__ import annotations

import logging
from typing import Any

import holoviews as hv
import param
import scipp as sc

from ess.livedata.config.models import ROI, Interval, RectangleROI
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


def boxes_to_rois(
    box_data: dict[str, Any],
    x_unit: str | None = None,
    y_unit: str | None = None,
) -> dict[int, RectangleROI]:
    """
    Convert BoxEdit data dictionary to RectangleROI instances.

    BoxEdit returns data as a dictionary with keys 'x0', 'x1', 'y0', 'y1',
    where each value is a list of coordinates for all boxes.

    Parameters
    ----------
    box_data:
        Dictionary from BoxEdit stream with keys x0, x1, y0, y1.
    x_unit:
        Unit for x coordinates (from the detector data coordinates).
    y_unit:
        Unit for y coordinates (from the detector data coordinates).

    Returns
    -------
    :
        Dictionary mapping box index to RectangleROI. Empty boxes are skipped.
    """
    if not box_data or not box_data.get("x0"):
        return {}

    x0_list = box_data.get("x0", [])
    x1_list = box_data.get("x1", [])
    y0_list = box_data.get("y0", [])
    y1_list = box_data.get("y1", [])

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


def rois_to_rectangles(
    rois: dict[int, RectangleROI], colors: list[str] | None = None
) -> list[tuple[float, ...]]:
    """
    Convert RectangleROI instances to HoloViews Rectangles format.

    Parameters
    ----------
    rois:
        Dictionary mapping ROI index to RectangleROI.
    colors:
        Optional list of colors to assign to rectangles based on ROI index.
        If provided, each rectangle tuple will include the color as a fifth element.

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
        if colors is not None:
            # Add color based on ROI index (cycle if necessary)
            color = colors[idx % len(colors)]
            rect_tuple = (*rect_tuple, color)
        rectangles.append(rect_tuple)
    return rectangles


def rois_to_box_data(rois: dict[int, RectangleROI]) -> dict[str, list[float]]:
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


def parse_roi_readback_data(
    roi_data: sc.DataArray, logger: logging.Logger | None = None
) -> dict[int, RectangleROI]:
    """
    Parse ROI readback data from backend into RectangleROI instances.

    Parameters
    ----------
    roi_data:
        Concatenated ROI data array with roi_index coordinate.
    logger:
        Optional logger for debug messages.

    Returns
    -------
    :
        Dictionary mapping ROI index to RectangleROI. Only rectangle ROIs
        are included. Returns empty dict if parsing fails.
    """
    try:
        rois = ROI.from_concatenated_data_array(roi_data)
        # Filter to only RectangleROI (other types not supported yet)
        return {idx: roi for idx, roi in rois.items() if isinstance(roi, RectangleROI)}
    except Exception as e:
        if logger:
            logger.debug("Failed to parse ROI readback data: %s", e)
        return {}


class ROIPlotState:
    """
    Per-plot state for ROI detector plots.

    Encapsulates state and callbacks for a single ROI detector plot,
    including active ROI tracking, publishing logic, and BoxEdit stream.

    This class implements bidirectional synchronization with Kafka:
    - User edits trigger publishes to backend (via on_box_change)
    - Backend updates trigger UI updates (via on_backend_roi_update)
    - Kafka is the single source of truth; backend state always wins

    Two visual layers are maintained:
    - Request ROIs: Interactive dashed boxes showing user's pending changes
    - Readback ROIs: Non-interactive solid boxes showing backend confirmed state

    Parameters
    ----------
    result_key:
        ResultKey identifying this detector plot.
    box_stream:
        HoloViews BoxEdit stream for this plot.
    request_pipe:
        HoloViews Pipe stream for programmatically updating request ROI rectangles.
    readback_pipe:
        HoloViews Pipe stream for programmatically updating readback ROI rectangles.
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
        List of colors to use for ROI rectangles, indexed by ROI number.
    initial_rois:
        Optional dictionary of initial ROI configurations. Used to establish
        the baseline state for both request and readback layers.
    """

    def __init__(
        self,
        result_key: ResultKey,
        box_stream: hv.streams.BoxEdit,
        request_pipe: hv.streams.Pipe,
        readback_pipe: hv.streams.Pipe,
        roi_state_stream: hv.streams.Stream,
        x_unit: str | None,
        y_unit: str | None,
        roi_publisher: ROIPublisher | None,
        logger: logging.Logger,
        colors: list[str],
        initial_rois: dict[int, RectangleROI] | None = None,
        roi_mapper=None,
    ) -> None:
        self.result_key = result_key
        self.box_stream = box_stream
        self.request_pipe = request_pipe
        self.readback_pipe = readback_pipe
        self.roi_state_stream = roi_state_stream
        self.x_unit = x_unit
        self.y_unit = y_unit
        self._roi_publisher = roi_publisher
        self._logger = logger
        self._colors = colors
        self._roi_mapper = roi_mapper or get_roi_mapper()

        # Separate state for request (user's pending changes) and readback
        # (backend truth). Initialize both from initial_rois (they start in sync)
        self._request_rois: dict[int, RectangleROI] = (
            initial_rois.copy() if initial_rois else {}
        )
        self._readback_rois: dict[int, RectangleROI] = (
            initial_rois.copy() if initial_rois else {}
        )

        # Attach the callback to the stream AFTER initializing state
        self.box_stream.param.watch(self.on_box_change, "data")

        # Initialize roi_state_stream with the current active ROI indices
        self.roi_state_stream.event(active_rois=self._active_roi_indices)

    @property
    def _active_roi_indices(self) -> set[int]:
        """Indices of currently active ROIs based on readback (backend truth)."""
        return set(self._readback_rois.keys())

    def on_box_change(self, event) -> None:
        """
        Handle BoxEdit data changes from UI.

        Updates request layer immediately (optimistic UI) and publishes to backend.
        Only publishes if ROIs differ from current request state to prevent cycles
        when backend updates trigger box_stream.event().

        Parameters
        ----------
        event:
            Event object from BoxEdit stream.
        """
        data = event.new if hasattr(event, "new") else event
        if data is None:
            data = {}

        try:
            current_rois = boxes_to_rois(data, x_unit=self.x_unit, y_unit=self.y_unit)

            # Only update and publish if ROIs changed from current request state
            # This prevents republishing when backend updates trigger box_stream.event()
            if current_rois == self._request_rois:
                return

            # Update request layer immediately (optimistic UI feedback)
            self._request_rois = current_rois
            request_rectangles = rois_to_rectangles(current_rois, colors=self._colors)
            self.request_pipe.send(request_rectangles)

            # Publish to backend
            if self._roi_publisher:
                self._roi_publisher.publish_rois(self.result_key.job_id, current_rois)
                self._logger.info(
                    "Published %d ROI(s) for job %s",
                    len(current_rois),
                    self.result_key.job_id,
                )

        except Exception as e:
            self._logger.error("Failed to publish ROI update: %s", e)

    def on_backend_roi_update(self, backend_rois: dict[int, RectangleROI]) -> None:
        """
        Handle ROI updates from the backend stream.

        The backend is the single source of truth for ROI state. This method
        updates the readback layer (solid lines) with the authoritative backend state,
        and syncs the request layer (dashed lines) to match if needed.

        Parameters
        ----------
        backend_rois:
            Dictionary mapping ROI index to RectangleROI from backend.
        """
        try:
            # Check if any state needs updating
            readback_changed = backend_rois != self._readback_rois
            request_needs_sync = backend_rois != self._request_rois

            if not readback_changed and not request_needs_sync:
                # Nothing to update
                return

            self._logger.debug(
                "Applying backend ROI update for job %s",
                self.result_key.job_id,
            )

            rectangles = rois_to_rectangles(backend_rois, colors=self._colors)

            # Update readback layer if changed (authoritative backend state)
            if readback_changed:
                self._readback_rois = backend_rois
                self.readback_pipe.send(rectangles)

                # Trigger ROI state stream to update spectrum plot filtering
                self.roi_state_stream.event(active_rois=self._active_roi_indices)

            # Sync request layer to match backend if needed
            if request_needs_sync:
                self._request_rois = backend_rois

                # Convert to BoxEdit format (dict with float arrays)
                box_data = rois_to_box_data(backend_rois)

                # Update request rectangles via pipe
                self.request_pipe.send(rectangles)

                # Update BoxEdit stream to enable drag operations
                self.box_stream.event(data=box_data)

            self._logger.info(
                "UI updated with %d ROI(s) from backend for job %s",
                len(backend_rois),
                self.result_key.job_id,
            )
        except Exception:
            self._logger.exception("Failed to update UI from backend ROI data")

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

    def _subscribe_to_roi_readback(
        self, roi_readback_key: ResultKey, plot_state: ROIPlotState
    ) -> None:
        """
        Subscribe to ROI readback stream from backend.

        Creates a subscription that updates the plot's ROI rectangles when
        the backend publishes ROI readback data, enabling bidirectional sync.

        Parameters
        ----------
        roi_readback_key:
            ResultKey for the roi_rectangle stream.
        plot_state:
            ROIPlotState to update when backend publishes ROIs.
        """

        def on_roi_data_update(data: dict[ResultKey, sc.DataArray]) -> None:
            """Callback for ROI readback data updates."""
            if roi_readback_key not in data:
                return

            roi_data = data[roi_readback_key]
            rectangle_rois = parse_roi_readback_data(roi_data, self._logger)
            # Always update, even if empty dict - this is needed to sync
            # ROI deletion across multiple plots
            if rectangle_rois is not None:
                plot_state.on_backend_roi_update(rectangle_rois)

        # Subscribe to roi_rectangle stream if it exists in DataService
        # Note: This only triggers if the stream has data available
        if roi_readback_key in self._stream_manager.data_service:
            # Initialize with current data
            initial_data = {
                roi_readback_key: self._stream_manager.data_service[roi_readback_key]
            }
            on_roi_data_update(initial_data)

        # Create a custom pipe that calls our callback
        class ROIReadbackPipe:
            def __init__(self, callback):
                self.callback = callback

            def send(self, data):
                self.callback(data)

        def roi_pipe_factory(data):
            """Factory function to create ROIReadbackPipe with callback."""
            return ROIReadbackPipe(on_roi_data_update)

        assembler = MergingStreamAssembler({roi_readback_key})
        extractors = {roi_readback_key: LatestValueExtractor()}
        subscriber = DataSubscriber(assembler, roi_pipe_factory, extractors)
        self._stream_manager.data_service.register_subscriber(subscriber)

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
        max_roi_count = params.roi_options.max_roi_count
        colors_list = default_colors[:max_roi_count]

        # Initialize with empty rectangles - actual ROI data will be loaded by
        # _subscribe_to_roi_readback if available in DataService
        initial_rectangles = []
        initial_box_data = {}

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

        # Create two separate layers:
        # 1. Readback rectangles (solid, non-interactive) - backend truth
        # 2. Request rectangles (dashed, interactive) - user's pending changes

        # Create Pipes for programmatic updates to both layers
        readback_pipe = hv.streams.Pipe(data=initial_rectangles)
        request_pipe = hv.streams.Pipe(data=initial_rectangles)

        # Create DynamicMap for readback rectangles (solid lines)
        # This shows the backend state (single source of truth) in sync with current
        # spectrum data
        def make_readback_boxes(data: list):
            return hv.Rectangles(data, vdims=['color']).opts(color='color')

        readback_dmap = hv.DynamicMap(make_readback_boxes, streams=[readback_pipe])

        # Create DynamicMap for request rectangles (dashed lines)
        # This allows programmatic updates via request_pipe.send()
        # and serves as the source for BoxEdit interaction
        def make_request_boxes(data: list):
            return hv.Rectangles(data, vdims=['color']).opts(color='color')

        request_dmap = hv.DynamicMap(make_request_boxes, streams=[request_pipe])

        # Create BoxEdit stream with request_dmap as source
        # This enables user interaction (drag, create, delete) on the request layer
        box_stream = hv.streams.BoxEdit(
            source=request_dmap, num_objects=max_roi_count, data=initial_box_data
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

        # Create plot state (which will attach the watcher to box_stream)
        # Note: plot_state is kept alive by references from the returned plot:
        # - box_stream holds a callback reference to plot_state.on_box_change
        # - roi_state_stream (below) is referenced by the spectrum plot DynamicMap
        # - roi_readback_pipe (below) holds plot_state.on_backend_roi_update
        plot_state = ROIPlotState(
            result_key=detector_key,
            box_stream=box_stream,
            request_pipe=request_pipe,
            readback_pipe=readback_pipe,
            roi_state_stream=roi_state_stream,
            x_unit=x_unit,
            y_unit=y_unit,
            roi_publisher=self._roi_publisher,
            logger=self._logger,
            colors=colors_list,
            initial_rois=None,
            roi_mapper=self._roi_mapper,
        )

        # Subscribe to ROI readback stream from backend for bidirectional sync.
        # This will automatically initialize the plot with existing ROI data if
        # available in DataService.
        roi_readback_key = detector_key.model_copy(
            # Index 0 is the "first" geometry readback key: The rectangle ROIs
            update={"output_name": self._roi_mapper.readback_keys[0]}
        )
        self._subscribe_to_roi_readback(roi_readback_key, plot_state)

        # Create the detector plot with two ROI layers:
        # 1. Readback layer (solid lines) - backend confirmed state
        # 2. Request layer (dashed lines) - user's interactive edits
        # Both layers start overlapping; they diverge (briefly) during user edits
        readback_boxes = readback_dmap.opts(
            fill_alpha=0.3, line_width=2, line_dash='solid'
        )
        request_boxes = request_dmap.opts(
            fill_alpha=0.3,
            line_width=2,
            line_dash='dashed',
            # There is a Bokeh(?) bug where line_dash='dashed' does not render with the
            # WebGL backend, so we force canvas output here.
            backend_opts={'plot.output_backend': 'canvas'},
        )
        # Layer order: detector, then readback (solid), then request (dashed on top)
        detector_with_boxes = detector_dmap * readback_boxes * request_boxes

        # Generate spectrum keys and create ROI spectrum plot
        spectrum_keys = self._generate_spectrum_keys(detector_key)
        roi_spectrum_dmap = self._create_roi_spectrum_plot(
            spectrum_keys, roi_state_stream, params
        )

        return detector_with_boxes, roi_spectrum_dmap, plot_state

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
