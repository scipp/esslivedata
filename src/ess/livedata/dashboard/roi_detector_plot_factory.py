# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Factory for creating ROI detector plots with interactive BoxEdit."""

from __future__ import annotations

import logging
from functools import partial

import holoviews as hv
import scipp as sc

from ess.livedata.config.models import RectangleROI
from ess.livedata.config.workflow_spec import ResultKey

from .data_subscriber import FilteredMergingStreamAssembler
from .plot_params import LayoutParams, PlotParams2d
from .plots import ImagePlotter, LinePlotter, PlotAspect
from .roi_publisher import ROIPublisher, boxes_to_rois
from .stream_manager import StreamManager


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
    """

    def __init__(
        self,
        result_key: ResultKey,
        box_stream: hv.streams.BoxEdit,
        x_unit: str | None,
        y_unit: str | None,
        roi_publisher: ROIPublisher | None,
        logger: logging.Logger,
    ) -> None:
        self.result_key = result_key
        self.box_stream = box_stream
        self.x_unit = x_unit
        self.y_unit = y_unit
        self._roi_publisher = roi_publisher
        self._logger = logger
        self._active_roi_indices: set[int] = set()
        self._last_published_rois: dict[int, RectangleROI] = {}

        # Attach the callback to the stream
        self.box_stream.param.watch(self.on_box_change, 'data')

    def on_box_change(self, event) -> None:
        """
        Handle BoxEdit data changes.

        Parameters
        ----------
        event:
            Event object from BoxEdit stream.
        """
        data = event.new if hasattr(event, 'new') else event
        if data is None:
            data = {}

        try:
            current_rois = boxes_to_rois(data, x_unit=self.x_unit, y_unit=self.y_unit)

            # Update active ROI indices for filtering
            self._active_roi_indices = set(current_rois.keys())

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

    _MAX_ROI_COUNT = 3

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

        parts = output_name.rsplit('_', 1)
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
        roi_base_name = f'roi_{detector_key.output_name}'
        return [
            detector_key.model_copy(update={'output_name': f'{roi_base_name}_{idx}'})
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
        if not hasattr(detector_data, 'coords') or dim not in detector_data.coords:
            return None

        coord_unit = detector_data.coords[dim].unit
        return str(coord_unit) if coord_unit is not None else None

    def create_roi_detector_plot_components(
        self,
        detector_key: ResultKey,
        detector_data: sc.DataArray,
        params: PlotParams2d,
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
            The plotter parameters.

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

        boxes = hv.Rectangles([])
        default_colors = hv.Cycle.default_cycles['default_colors']
        box_stream = hv.streams.BoxEdit(
            source=boxes,
            num_objects=self._MAX_ROI_COUNT,
            styles={'fill_color': default_colors[: self._MAX_ROI_COUNT]},
        )

        # Extract coordinate units
        data_source = detector_items[detector_key]
        if hasattr(data_source, 'data') and hasattr(data_source.data, 'coords'):
            actual_data = data_source.data
        elif hasattr(data_source, 'coords'):
            actual_data = data_source
        else:
            actual_data = data_source

        x_dim, y_dim = actual_data.dims[1], actual_data.dims[0]
        x_unit = self._extract_unit_for_dim(actual_data, x_dim)
        y_unit = self._extract_unit_for_dim(actual_data, y_dim)

        # Create plot state (which will attach the watcher to box_stream)
        # Note: plot_state is kept alive by references from the returned plot:
        # - box_stream holds a callback reference to plot_state.on_box_change
        # - The spectrum assembler (created below) holds plot_state.is_roi_active
        plot_state = ROIPlotState(
            result_key=detector_key,
            box_stream=box_stream,
            x_unit=x_unit,
            y_unit=y_unit,
            roi_publisher=self._roi_publisher,
            logger=self._logger,
        )

        # Create the detector plot with interactive boxes overlay
        interactive_boxes = boxes.opts(fill_alpha=0.3, line_width=2)
        detector_with_boxes = detector_dmap * interactive_boxes

        # Generate spectrum keys and create ROI spectrum plot
        if detector_key.output_name is None:
            raise ValueError(
                "detector_key.output_name must be set for ROI detector plots"
            )
        spectrum_keys = self._generate_spectrum_keys(detector_key, self._MAX_ROI_COUNT)
        roi_spectrum_dmap = self._create_roi_spectrum_plot(
            spectrum_keys, plot_state, params
        )

        return detector_with_boxes, roi_spectrum_dmap, plot_state

    def _create_roi_spectrum_plot(
        self,
        spectrum_keys: list[ResultKey],
        plot_state: ROIPlotState,
        params: PlotParams2d,
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
            The plotter parameters.

        Returns
        -------
        :
            DynamicMap for the ROI spectrum plot.
        """
        overlay_layout = LayoutParams(combine_mode='overlay')

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
            aspect_params=PlotAspect(fix_width=True),
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
        params: PlotParams2d,
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
            The plotter parameters (PlotParams2d).

        Returns
        -------
        :
            A HoloViews Layout with detector image (with BoxEdit overlay) and
            ROI spectrum plot, arranged in 2 columns.
        """
        if not isinstance(params, PlotParams2d):
            raise TypeError("roi_detector requires PlotParams2d")

        detector_with_boxes, roi_spectrum_dmap, _plot_state = (
            self.create_roi_detector_plot_components(
                detector_key, detector_data, params
            )
        )

        return hv.Layout([detector_with_boxes, roi_spectrum_dmap]).cols(2)
