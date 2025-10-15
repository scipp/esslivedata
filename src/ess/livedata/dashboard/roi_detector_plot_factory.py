# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Factory for creating ROI detector plots with interactive BoxEdit."""

from __future__ import annotations

import logging
from functools import partial

import holoviews as hv

from ess.livedata.config.models import RectangleROI
from ess.livedata.config.workflow_spec import JobId, JobNumber, ResultKey, WorkflowId

from .data_subscriber import FilteredMergingStreamAssembler
from .plot_params import LayoutParams, PlotParams2d
from .plots import ImagePlotter, LinePlotter
from .roi_publisher import ROIPublisher, boxes_to_rois
from .stream_manager import StreamManager


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
        self._box_streams: dict[ResultKey, hv.streams.BoxEdit] = {}
        self._last_published_rois: dict[ResultKey, dict[int, RectangleROI]] = {}
        # Track which ROI indices are currently active for each job
        self._active_roi_indices: dict[JobId, set[int]] = {}

    def create_roi_detector_plot(
        self,
        workflow_id: WorkflowId,
        job_number: JobNumber,
        detector_items: dict[ResultKey, hv.streams.Pipe],
        params: PlotParams2d,
    ) -> hv.Layout:
        """
        Create ROI detector plot with interactive BoxEdit.

        This is a special-case implementation that creates separate DynamicMaps
        for detector and spectrum data, with BoxEdit overlay applied at the
        DynamicMap level (not inside the callback) to maintain interactivity.

        When a user selects an output (e.g., 'current' or 'cumulative'), this method
        automatically discovers and subscribes to the corresponding ROI outputs:
        - 'current' -> uses 'roi_current' for 1D spectrum
        - 'cumulative' -> uses 'roi_cumulative' for 1D spectrum

        The ROI outputs may be published later (after ROI is configured in the UI),
        so we subscribe to them even if they don't exist yet.

        Parameters
        ----------
        workflow_id:
            The workflow ID for creating ResultKeys.
        job_number:
            The job number to create the plot for.
        detector_items:
            Dictionary mapping ResultKeys to data pipes for detector outputs.
        params:
            The plotter parameters (PlotParams2d).

        Returns
        -------
        :
            A HoloViews Layout with detector image (with BoxEdit overlay) and
            ROI spectrum plot.
        """

        # Maximum number of ROIs to support (subscribe upfront for dynamic addition)
        max_roi_count = 3

        # Derive spectrum keys from detector items
        # For each detector ResultKey with output_name='current', subscribe to
        # 'roi_current_0', 'roi_current_1', etc.
        spectrum_keys: list[ResultKey] = []
        for detector_key in detector_items.keys():
            if detector_key.output_name:
                # Subscribe to multiple ROI indices upfront
                # (roi_current_0, roi_current_1, etc.)
                # This allows ROIs to be added dynamically after plot creation
                roi_base_name = f'roi_{detector_key.output_name}'

                # Subscribe to all ROI indices (0 through max_roi_count-1)
                for roi_idx in range(max_roi_count):
                    roi_spectrum_name = f'{roi_base_name}_{roi_idx}'
                    spectrum_key = ResultKey(
                        workflow_id=workflow_id,
                        job_id=detector_key.job_id,
                        output_name=roi_spectrum_name,
                    )
                    # Subscribe to the key regardless of whether data exists yet
                    spectrum_keys.append(spectrum_key)

        plots = []

        # Create detector plot with BoxEdit overlay
        if detector_items:
            detector_pipe = self._stream_manager.make_merging_stream(detector_items)
            detector_plotter = ImagePlotter(
                value_margin_factor=0.1,
                layout_params=params.layout,
                aspect_params=params.plot_aspect,
                scale_opts=params.plot_scale,
            )
            detector_plotter.initialize_from_data(detector_items)

            detector_dmap = hv.DynamicMap(
                detector_plotter, streams=[detector_pipe], cache_size=1
            ).opts(shared_axes=False)

            # Create BoxEdit overlay at DynamicMap level (critical for interactivity)
            boxes = hv.Rectangles([])
            # Use HoloViews default color cycle to match LinePlotter's automatic colors
            default_colors = hv.Cycle.default_cycles['default_colors']
            box_stream = hv.streams.BoxEdit(
                source=boxes,
                num_objects=max_roi_count,
                styles={'fill_color': default_colors[:max_roi_count]},
            )

            # Store box stream for later access (e.g., publishing to backend)
            # Use the first detector item's key as the reference
            first_detector_key = next(iter(detector_items.keys()))
            self._box_streams[first_detector_key] = box_stream

            # Set up ROI publishing if publisher is available
            if self._roi_publisher:
                # Extract coordinate units from the detector data
                first_detector_data = next(iter(detector_items.values()))
                x_dim, y_dim = first_detector_data.dims[1], first_detector_data.dims[0]

                # Check each coordinate independently and extract unit if present
                x_unit = None
                if x_dim in first_detector_data.coords:
                    x_coord_unit = first_detector_data.coords[x_dim].unit
                    x_unit = str(x_coord_unit) if x_coord_unit is not None else None

                y_unit = None
                if y_dim in first_detector_data.coords:
                    y_coord_unit = first_detector_data.coords[y_dim].unit
                    y_unit = str(y_coord_unit) if y_coord_unit is not None else None

                self._setup_roi_watcher(box_stream, first_detector_key, x_unit, y_unit)

            # Overlay boxes on DynamicMap (not inside callback - this is crucial!)
            interactive_boxes = boxes.opts(fill_alpha=0.3, line_width=2)
            detector_with_boxes = detector_dmap * interactive_boxes
            plots.append(detector_with_boxes)

        # Create single ROI spectrum plot (overlays all ROIs)
        if spectrum_keys:
            # Override layout params to use overlay mode
            overlay_layout = LayoutParams(combine_mode='overlay')

            # Create filtered stream that only shows active ROI indices
            # Get the job_id from the first spectrum key
            first_spectrum_key = spectrum_keys[0]
            job_id = first_spectrum_key.job_id

            # Create filter function that checks if ROI index is active
            def is_roi_active(key: ResultKey) -> bool:
                """Check if the ROI index for this key is currently active."""
                # Extract ROI index from output_name (e.g., 'roi_current_0' -> 0)
                if key.output_name is None:
                    return False

                # Parse the ROI index from the output name
                # Format: 'roi_{output_name}_{index}'
                parts = key.output_name.rsplit('_', 1)
                if len(parts) != 2:
                    return False

                try:
                    roi_index = int(parts[1])
                except ValueError:
                    return False

                # Check if this ROI index is in the active set
                active_indices = self._active_roi_indices.get(job_id, set())
                return roi_index in active_indices

            # Create filtered assembler factory using partial
            assembler_factory = partial(
                FilteredMergingStreamAssembler, filter_fn=is_roi_active
            )
            spectrum_pipe = self._stream_manager.make_merging_stream_from_keys(
                spectrum_keys, assembler_factory=assembler_factory
            )

            spectrum_plotter = LinePlotter(
                value_margin_factor=0.1,
                layout_params=overlay_layout,
                aspect_params=params.plot_aspect,
                scale_opts=params.plot_scale,
            )

            spectrum_dmap = hv.DynamicMap(
                spectrum_plotter, streams=[spectrum_pipe], cache_size=1
            ).opts(shared_axes=False)
            plots.append(spectrum_dmap)

        if len(plots) == 0:
            return hv.Layout(
                [
                    hv.Text(0.5, 0.5, "No data").opts(
                        text_align='center', text_baseline='middle'
                    )
                ]
            )
        elif len(plots) == 1:
            return hv.Layout(plots)
        else:
            return hv.Layout(plots).cols(2)

    def _setup_roi_watcher(
        self,
        box_stream: hv.streams.BoxEdit,
        result_key: ResultKey,
        x_unit: str | None,
        y_unit: str | None,
    ) -> None:
        """
        Set up a watcher on BoxEdit stream to publish ROI updates.

        Parameters
        ----------
        box_stream:
            The BoxEdit stream to watch.
        result_key:
            The result key for tracking published ROIs (contains job_id).
        x_unit:
            Unit for x coordinates, extracted from the detector data.
        y_unit:
            Unit for y coordinates, extracted from the detector data.
        """

        def on_box_change(event):
            """Callback when BoxEdit data changes."""
            # Extract data from the event object
            data = event.new if hasattr(event, 'new') else event
            if data is None:
                # Empty dict means clear all ROIs
                data = {}

            try:
                # Convert BoxEdit data to ROI dictionary
                current_rois = boxes_to_rois(data, x_unit=x_unit, y_unit=y_unit)

                # Update active ROI indices for filtering
                self._active_roi_indices[result_key.job_id] = set(current_rois.keys())

                # Get previously published ROIs for this result key
                last_rois = self._last_published_rois.get(result_key, {})

                # Only publish if ROIs actually changed
                if current_rois != last_rois:
                    # Always publish ALL ROIs (not just changed ones)
                    # Backend needs full set to detect deletions
                    self._roi_publisher.publish_rois(result_key.job_id, current_rois)
                    # Update tracking
                    self._last_published_rois[result_key] = current_rois
                    self._logger.info(
                        "Published %d ROI(s) for job %s",
                        len(current_rois),
                        result_key.job_id,
                    )

            except Exception as e:
                self._logger.error("Failed to publish ROI update: %s", e)

        # Watch the 'data' parameter of the BoxEdit stream
        box_stream.param.watch(on_box_change, 'data')
