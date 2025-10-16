# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Factory for creating ROI detector plots with interactive BoxEdit."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import holoviews as hv
import scipp as sc

from ess.livedata.config.models import ROI, Interval, RectangleROI
from ess.livedata.config.workflow_spec import ResultKey

from .data_subscriber import FilteredMergingStreamAssembler
from .plot_params import LayoutParams, PlotParamsROIDetector
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

    Parameters
    ----------
    result_key:
        ResultKey identifying this detector plot.
    box_stream:
        HoloViews BoxEdit stream for this plot.
    boxes_pipe:
        HoloViews Pipe stream for programmatically updating ROI rectangles.
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
        the baseline state for cycle prevention and to determine which ROI
        indices are initially active.
    """

    def __init__(
        self,
        result_key: ResultKey,
        box_stream: hv.streams.BoxEdit,
        boxes_pipe: hv.streams.Pipe,
        x_unit: str | None,
        y_unit: str | None,
        roi_publisher: ROIPublisher | None,
        logger: logging.Logger,
        colors: list[str],
        initial_rois: dict[int, RectangleROI] | None = None,
    ) -> None:
        self.result_key = result_key
        self.box_stream = box_stream
        self.boxes_pipe = boxes_pipe
        self.x_unit = x_unit
        self.y_unit = y_unit
        self._roi_publisher = roi_publisher
        self._logger = logger
        self._colors = colors

        # Single source of truth for current ROI state
        # Initialize from initial_rois to establish baseline
        self._last_known_rois: dict[int, RectangleROI] = (
            initial_rois.copy() if initial_rois else {}
        )

        # Initialize active indices before attaching watcher to prevent race condition
        self._active_roi_indices: set[int] = (
            set(initial_rois.keys()) if initial_rois else set()
        )

        # Attach the callback to the stream AFTER initializing state
        self.box_stream.param.watch(self.on_box_change, "data")

    def on_box_change(self, event) -> None:
        """
        Handle BoxEdit data changes from UI.

        Only publishes to backend if ROIs differ from the last known state,
        preventing redundant publishes when backend updates trigger UI changes.

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
            current_indices = set(current_rois.keys())

            # Update active ROI indices for filtering
            if current_indices != self._active_roi_indices:
                self._logger.debug(
                    "Active ROI indices changing from %s to %s for job %s",
                    self._active_roi_indices,
                    current_indices,
                    self.result_key.job_id,
                )
            self._active_roi_indices = current_indices

            # Only publish if ROIs changed from last known state
            # This prevents republishing when backend updates trigger UI changes
            if current_rois != self._last_known_rois:
                self._last_known_rois = current_rois

                # Update pipe to assign colors to user-drawn rectangles
                # This ensures newly drawn rectangles get colors immediately
                rectangles = rois_to_rectangles(current_rois, colors=self._colors)
                self.boxes_pipe.send(rectangles)

                if self._roi_publisher:
                    self._roi_publisher.publish_rois(
                        self.result_key.job_id, current_rois
                    )
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

        Updates the UI only if the backend state differs from the current state.
        Updates _last_known_rois BEFORE updating the BoxEdit stream to prevent
        infinite cycles when on_box_change is triggered.

        Parameters
        ----------
        backend_rois:
            Dictionary mapping ROI index to RectangleROI from backend.
        """
        try:
            # Only update UI if backend state differs from current state
            if backend_rois != self._last_known_rois:
                self._logger.debug(
                    "Backend ROI update differs from current state for job %s, "
                    "updating UI",
                    self.result_key.job_id,
                )

                # Update state BEFORE updating UI to break potential cycles
                self._last_known_rois = backend_rois
                self._active_roi_indices = set(backend_rois.keys())

                # Convert to BoxEdit format (dict with float arrays)
                box_data = rois_to_box_data(backend_rois)

                # Update Pipe to refresh visual representation (via DynamicMap)
                rectangles = rois_to_rectangles(backend_rois, colors=self._colors)
                self.boxes_pipe.send(rectangles)

                # Update BoxEdit stream to enable drag operations
                # This must be done AFTER pipe.send() to ensure BoxEdit has correct data
                self.box_stream.event(data=box_data)

                self._logger.info(
                    "UI updated with %d ROI(s) from backend for job %s",
                    len(backend_rois),
                    self.result_key.job_id,
                )
        except Exception as e:
            self._logger.error("Failed to update UI from backend ROI data: %s", e)

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
        roi_index = ROIDetectorPlotFactory._parse_roi_index(key.output_name)
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

    @staticmethod
    def _parse_roi_index(output_name: str | None) -> int | None:
        """
        Extract ROI index from output name.

        Parameters
        ----------
        output_name:
            Output name in format 'roi_{output_name}_{index}'.

        Returns
        -------
        :
            ROI index if parsing succeeds, None otherwise.
        """
        if output_name is None:
            return None

        parts = output_name.rsplit("_", 1)
        if len(parts) != 2:
            return None

        try:
            return int(parts[1])
        except ValueError:
            return None

    @staticmethod
    def _generate_spectrum_keys(
        detector_key: ResultKey, max_roi_count: int
    ) -> list[ResultKey]:
        """
        Generate spectrum keys for ROI outputs.

        Parameters
        ----------
        detector_key:
            ResultKey identifying the detector output. Must have output_name set.
        max_roi_count:
            Maximum number of ROIs to subscribe to.

        Returns
        -------
        :
            List of ResultKeys for ROI spectrum outputs.
        """
        roi_base_name = f"roi_{detector_key.output_name}"
        return [
            detector_key.model_copy(update={"output_name": f"{roi_base_name}_{idx}"})
            for idx in range(max_roi_count)
        ]

    def _extract_initial_rois_from_data_service(
        self, roi_readback_key: ResultKey
    ) -> dict[int, RectangleROI]:
        """
        Extract initial ROI state from DataService if available.

        Parameters
        ----------
        roi_readback_key:
            ResultKey for the roi_rectangle stream.

        Returns
        -------
        :
            Dictionary mapping ROI index to RectangleROI. Returns empty dict
            if no ROI readback data is available.
        """
        if roi_readback_key not in self._stream_manager.data_service:
            return {}

        roi_data = self._stream_manager.data_service[roi_readback_key]
        return parse_roi_readback_data(roi_data, self._logger)

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
            if rectangle_rois:
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

        roi_pipe = ROIReadbackPipe(on_roi_data_update)

        # Subscribe via DataService
        from .data_subscriber import DataSubscriber, MergingStreamAssembler

        assembler = MergingStreamAssembler({roi_readback_key})
        subscriber = DataSubscriber(assembler, roi_pipe)
        self._stream_manager.data_service.register_subscriber(subscriber)

    def create_roi_detector_plot_components(
        self,
        detector_key: ResultKey,
        detector_data: sc.DataArray,
        params: PlotParamsROIDetector,
        initial_rois: dict[int, RectangleROI] | None = None,
    ) -> tuple[hv.DynamicMap, hv.DynamicMap, ROIPlotState]:
        """
        Create ROI detector plot components without layout assembly.

        This is the testable public API that creates the individual components
        for an ROI detector plot. It returns the detector DynamicMap, ROI spectrum
        DynamicMap, and the plot state, allowing the caller to control layout
        or access components for testing.

        The plot_state is not stored internally - it's kept alive by references
        from the returned plot components (via callbacks and stream filters).

        Parameters
        ----------
        detector_key:
            ResultKey identifying the detector output.
        detector_data:
            Initial data for the detector plot.
        params:
            The plotter parameters (PlotParamsROIDetector).
        initial_rois:
            Optional dictionary of initial ROI configurations to display.
            If provided, the Rectangles will be initialized with these shapes
            and the BoxEdit stream will be populated accordingly.

        Returns
        -------
        :
            Tuple of (detector_dmap, roi_dmap, plot_state).
        """
        detector_items = {detector_key: detector_data}
        # FIXME: Memory leak - subscribers registered via stream_manager are never
        # unregistered. When this plot is closed, the subscriber remains in
        # DataService._subscribers, preventing garbage collection of plot components.
        merged_detector_pipe = self._stream_manager.make_merging_stream(detector_items)

        detector_plotter = ImagePlotter(
            value_margin_factor=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
        )
        detector_plotter.initialize_from_data(detector_items)

        detector_dmap = hv.DynamicMap(
            detector_plotter, streams=[merged_detector_pipe], cache_size=1
        ).opts(shared_axes=False)

        # Get color cycle for ROI styling
        default_colors = hv.Cycle.default_cycles["default_colors"]
        max_roi_count = params.roi_options.max_roi_count
        colors_list = default_colors[:max_roi_count]

        # Initialize Rectangles with existing ROI shapes if available
        initial_rectangles = []
        initial_box_data = {}
        if initial_rois:
            initial_rectangles = rois_to_rectangles(initial_rois, colors=colors_list)
            initial_box_data = rois_to_box_data(initial_rois)

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

        # Create Pipe for programmatic updates to rectangles
        boxes_pipe = hv.streams.Pipe(data=initial_rectangles)

        # Create DynamicMap that wraps Rectangles
        # This allows programmatic updates via boxes_pipe.send()
        # Rectangles include color as a vdim to preserve per-rectangle colors
        def make_boxes(data):
            if not data:
                data = []
            return hv.Rectangles(data, vdims=['color']).opts(color='color')

        boxes_dmap = hv.DynamicMap(make_boxes, streams=[boxes_pipe])

        # Create BoxEdit stream with DynamicMap as source
        # This enables user interaction (drag, create, delete)
        box_stream = hv.streams.BoxEdit(
            source=boxes_dmap,
            num_objects=max_roi_count,
            styles={"fill_color": default_colors[:max_roi_count]},
            data=initial_box_data,
        )

        # Extract coordinate units
        x_dim, y_dim = detector_data.dims[1], detector_data.dims[0]
        x_unit = self._extract_unit_for_dim(detector_data, x_dim)
        y_unit = self._extract_unit_for_dim(detector_data, y_dim)

        # Create plot state (which will attach the watcher to box_stream)
        # Note: plot_state is kept alive by references from the returned plot:
        # - box_stream holds a callback reference to plot_state.on_box_change
        # - The spectrum assembler (created below) holds plot_state.is_roi_active
        # - roi_readback_pipe (below) holds plot_state.on_backend_roi_update
        plot_state = ROIPlotState(
            result_key=detector_key,
            box_stream=box_stream,
            boxes_pipe=boxes_pipe,
            x_unit=x_unit,
            y_unit=y_unit,
            roi_publisher=self._roi_publisher,
            logger=self._logger,
            colors=colors_list,
            initial_rois=initial_rois,
        )

        # Subscribe to ROI readback stream from backend for bidirectional sync
        roi_readback_key = detector_key.model_copy(
            update={"output_name": "roi_rectangle"}
        )
        self._subscribe_to_roi_readback(roi_readback_key, plot_state)

        # Create the detector plot with interactive boxes overlay
        interactive_boxes = boxes_dmap.opts(fill_alpha=0.3, line_width=2)
        detector_with_boxes = detector_dmap * interactive_boxes

        # Generate spectrum keys and create ROI spectrum plot
        if detector_key.output_name is None:
            raise ValueError(
                "detector_key.output_name must be set for ROI detector plots"
            )
        spectrum_keys = self._generate_spectrum_keys(detector_key, max_roi_count)
        roi_spectrum_dmap = self._create_roi_spectrum_plot(
            spectrum_keys, plot_state, params
        )

        return detector_with_boxes, roi_spectrum_dmap, plot_state

    def _create_roi_spectrum_plot(
        self,
        spectrum_keys: list[ResultKey],
        plot_state: ROIPlotState,
        params: PlotParamsROIDetector,
    ) -> hv.DynamicMap:
        """
        Create ROI spectrum plot that overlays all active ROI spectra.

        Parameters
        ----------
        spectrum_keys:
            List of ResultKeys for ROI spectrum outputs.
        plot_state:
            ROI plot state for filtering active ROIs.
        params:
            The plotter parameters (PlotParamsROIDetector).

        Returns
        -------
        :
            DynamicMap for the ROI spectrum plot.
        """
        overlay_layout = LayoutParams(combine_mode="overlay")

        assembler_factory = partial(
            FilteredMergingStreamAssembler, filter_fn=plot_state.is_roi_active
        )
        # FIXME: Memory leak - subscribers registered via stream_manager are never
        # unregistered. When this plot is closed, the subscriber remains in
        # DataService._subscribers, preventing garbage collection of plot components.
        spectrum_pipe = self._stream_manager.make_merging_stream_from_keys(
            spectrum_keys, assembler_factory=assembler_factory
        )

        spectrum_plotter = LinePlotter(
            value_margin_factor=0.1,
            layout_params=overlay_layout,
            # These settings are not perfect, but the spectrum-plot height will match
            # that of the detector-plot.
            aspect_params=PlotAspect(
                aspect_type=PlotAspectType.free, fix_width=True, width=500
            ),
            scale_opts=params.plot_scale,
        )

        spectrum_dmap = hv.DynamicMap(
            spectrum_plotter, streams=[spectrum_pipe], cache_size=1
        ).opts(shared_axes=False)

        return spectrum_dmap

    def create_roi_detector_plot(
        self,
        detector_key: ResultKey,
        detector_data: sc.DataArray,
        params: PlotParamsROIDetector,
    ) -> hv.Layout:
        """
        Create ROI detector plot with interactive BoxEdit for a single detector.

        This creates a Layout containing two DynamicMaps side-by-side:
        1. Detector image with BoxEdit overlay for ROI selection
        2. ROI spectrum plot that overlays all active ROI spectra

        When a user selects an output (e.g., 'current' or 'cumulative'), this method
        automatically discovers and subscribes to the corresponding ROI outputs:
        - 'current' -> uses 'roi_current' for 1D spectrum
        - 'cumulative' -> uses 'roi_cumulative' for 1D spectrum

        The ROI outputs may be published later (after ROI is configured in the UI),
        so we subscribe to them even if they don't exist yet.

        Initial ROI configurations are automatically extracted from DataService if
        available (e.g., from previous configurations published by backend).

        For testing or custom layouts, use `create_roi_detector_plot_components()`
        to get the individual components without the layout wrapper.

        Parameters
        ----------
        detector_key:
            ResultKey identifying the detector output.
        detector_data:
            Initial data for the detector plot.
        params:
            The plotter parameters (PlotParamsROIDetector).

        Returns
        -------
        :
            A HoloViews Layout with detector image (with BoxEdit overlay) and
            ROI spectrum plot, arranged in 2 columns.
        """
        if not isinstance(params, PlotParamsROIDetector):
            raise TypeError("roi_detector requires PlotParamsROIDetector")

        # Extract initial ROIs from DataService if available
        roi_readback_key = detector_key.model_copy(
            update={"output_name": "roi_rectangle"}
        )
        initial_rois = self._extract_initial_rois_from_data_service(roi_readback_key)

        detector_with_boxes, roi_spectrum_dmap, _plot_state = (
            self.create_roi_detector_plot_components(
                detector_key, detector_data, params, initial_rois=initial_rois
            )
        )

        return hv.Layout([detector_with_boxes, roi_spectrum_dmap]).cols(2)
