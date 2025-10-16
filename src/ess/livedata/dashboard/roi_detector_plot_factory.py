# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Factory for creating ROI detector plots with interactive BoxEdit."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import holoviews as hv
import scipp as sc

from ess.livedata.config.models import Interval, RectangleROI
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

    # Validate all lists have the same length
    lengths = {len(x0_list), len(x1_list), len(y0_list), len(y1_list)}
    if len(lengths) != 1:
        raise ValueError(
            f"BoxEdit data has inconsistent lengths: "
            f"x0={len(x0_list)}, x1={len(x1_list)}, "
            f"y0={len(y0_list)}, y1={len(y1_list)}"
        )

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


def rois_to_rectangles(rois: dict[int, RectangleROI]) -> list[tuple[float, ...]]:
    """
    Convert RectangleROI instances to HoloViews Rectangles format.

    Parameters
    ----------
    rois:
        Dictionary mapping ROI index to RectangleROI.

    Returns
    -------
    :
        List of (x0, y0, x1, y1) tuples for HoloViews Rectangles.
        Returned in sorted order by ROI index.
    """
    rectangles = []
    for idx in sorted(rois.keys()):
        roi = rois[idx]
        rectangles.append((roi.x.min, roi.y.min, roi.x.max, roi.y.max))
    return rectangles


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
    x_unit:
        Unit for x coordinates.
    y_unit:
        Unit for y coordinates.
    roi_publisher:
        Publisher for ROI updates. If None, publishing is disabled.
    logger:
        Logger instance.
    initial_active_indices:
        Optional set of ROI indices that should be active initially.
        If None, no ROIs are active initially. This must be set before
        attaching the watcher to prevent race conditions.
    """

    def __init__(
        self,
        result_key: ResultKey,
        box_stream: hv.streams.BoxEdit,
        x_unit: str | None,
        y_unit: str | None,
        roi_publisher: ROIPublisher | None,
        logger: logging.Logger,
        initial_active_indices: set[int] | None = None,
    ) -> None:
        self.result_key = result_key
        self.box_stream = box_stream
        self.x_unit = x_unit
        self.y_unit = y_unit
        self._roi_publisher = roi_publisher
        self._logger = logger
        # Initialize active indices before attaching watcher to prevent race condition
        self._active_roi_indices: set[int] = (
            initial_active_indices if initial_active_indices is not None else set()
        )
        self._last_published_rois: dict[int, RectangleROI] = {}

        # Attach the callback to the stream AFTER initializing active indices
        self.box_stream.param.watch(self.on_box_change, "data")

    def on_box_change(self, event) -> None:
        """
        Handle BoxEdit data changes.

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
            # Log changes to help debug race conditions
            if current_indices != self._active_roi_indices:
                self._logger.debug(
                    "Active ROI indices changing from %s to %s for job %s",
                    self._active_roi_indices,
                    current_indices,
                    self.result_key.job_id,
                )
            self._active_roi_indices = current_indices

            # Only publish if ROIs actually changed and publisher is available
            if self._roi_publisher and current_rois != self._last_published_rois:
                self._roi_publisher.publish_rois(self.result_key.job_id, current_rois)
                self._last_published_rois = current_rois
                self._logger.info(
                    "Published %d ROI(s) for job %s",
                    len(current_rois),
                    self.result_key.job_id,
                )

        except Exception as e:
            self._logger.error("Failed to publish ROI update: %s", e)

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

        # Initialize Rectangles with existing ROI shapes if available
        initial_rectangles = []
        initial_box_data = {}
        if initial_rois:
            initial_rectangles = rois_to_rectangles(initial_rois)
            # Convert rectangles to BoxEdit format
            if initial_rectangles:
                initial_box_data = {
                    "x0": [r[0] for r in initial_rectangles],
                    "y0": [r[1] for r in initial_rectangles],
                    "x1": [r[2] for r in initial_rectangles],
                    "y1": [r[3] for r in initial_rectangles],
                }

        boxes = hv.Rectangles(initial_rectangles)
        default_colors = hv.Cycle.default_cycles["default_colors"]
        max_roi_count = params.roi_options.max_roi_count
        box_stream = hv.streams.BoxEdit(
            source=boxes,
            num_objects=max_roi_count,
            styles={"fill_color": default_colors[:max_roi_count]},
            data=initial_box_data,
        )

        # Extract coordinate units
        x_dim, y_dim = detector_data.dims[1], detector_data.dims[0]
        x_unit = self._extract_unit_for_dim(detector_data, x_dim)
        y_unit = self._extract_unit_for_dim(detector_data, y_dim)

        # Determine initial active indices from initial_rois
        initial_active_indices = set(initial_rois.keys()) if initial_rois else None

        # Create plot state (which will attach the watcher to box_stream)
        # Note: plot_state is kept alive by references from the returned plot:
        # - box_stream holds a callback reference to plot_state.on_box_change
        # - The spectrum assembler (created below) holds plot_state.is_roi_active
        # We pass initial_active_indices to prevent race condition where watcher
        # triggers before we set the active indices manually.
        plot_state = ROIPlotState(
            result_key=detector_key,
            box_stream=box_stream,
            x_unit=x_unit,
            y_unit=y_unit,
            roi_publisher=self._roi_publisher,
            logger=self._logger,
            initial_active_indices=initial_active_indices,
        )

        # Create the detector plot with interactive boxes overlay
        interactive_boxes = boxes.opts(fill_alpha=0.3, line_width=2)
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
        initial_rois: dict[int, RectangleROI] | None = None,
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
        initial_rois:
            Optional dictionary of initial ROI configurations to display.
            If provided, the Rectangles will be initialized with these shapes
            and the BoxEdit stream will be populated accordingly.

        Returns
        -------
        :
            A HoloViews Layout with detector image (with BoxEdit overlay) and
            ROI spectrum plot, arranged in 2 columns.
        """
        if not isinstance(params, PlotParamsROIDetector):
            raise TypeError("roi_detector requires PlotParamsROIDetector")

        detector_with_boxes, roi_spectrum_dmap, _plot_state = (
            self.create_roi_detector_plot_components(
                detector_key, detector_data, params, initial_rois=initial_rois
            )
        )

        return hv.Layout([detector_with_boxes, roi_spectrum_dmap]).cols(2)
