# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Factory for creating ROI detector plots with interactive BoxEdit."""

from __future__ import annotations

import logging
from functools import partial

import holoviews as hv
import scipp as sc

from ess.livedata.config.models import RectangleROI
from ess.livedata.config.workflow_spec import JobId, ResultKey

from .data_subscriber import FilteredMergingStreamAssembler
from .plot_params import LayoutParams, PlotParams2d
from .plots import ImagePlotter, LinePlotter, PlotAspect
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

    def create_single_roi_detector_plot(
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
        # Validate params type
        if not isinstance(params, PlotParams2d):
            raise TypeError("roi_detector requires PlotParams2d")

        # Maximum number of ROIs to support (subscribe upfront for dynamic addition)
        max_roi_count = 3

        # Derive spectrum keys from detector key
        # For detector ResultKey with output_name='current', subscribe to
        # 'roi_current_0', 'roi_current_1', etc.
        spectrum_keys: list[ResultKey] = []
        if detector_key.output_name:
            # Subscribe to multiple ROI indices upfront
            # (roi_current_0, roi_current_1, etc.)
            # This allows ROIs to be added dynamically after plot creation
            roi_base_name = f'roi_{detector_key.output_name}'

            # Subscribe to all ROI indices (0 through max_roi_count-1)
            for roi_idx in range(max_roi_count):
                roi_spectrum_name = f'{roi_base_name}_{roi_idx}'
                spectrum_key = detector_key.model_copy(
                    update={'output_name': roi_spectrum_name}
                )
                # Subscribe to the key regardless of whether data exists yet
                spectrum_keys.append(spectrum_key)

        plots = []

        # Create detector plot with BoxEdit overlay
        # detector_pipe is a Pipe stream - wrap in single-item dict for compatibility
        detector_items = {detector_key: detector_data}

        # Create a merging stream (even for single detector, to be consistent)
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

        # Create BoxEdit overlay at DynamicMap level (critical for interactivity)
        boxes = hv.Rectangles([])
        # Use HoloViews default color cycle to match LinePlotter's automatic colors
        default_colors = hv.Cycle.default_cycles['default_colors']
        box_stream = hv.streams.BoxEdit(
            source=boxes,
            num_objects=max_roi_count,
            styles={'fill_color': default_colors[:max_roi_count]},
        )

        # Store box stream for this detector
        self._box_streams[detector_key] = box_stream

        # Set up ROI publishing if publisher is available
        if self._roi_publisher:
            # Extract coordinate units from the detector data
            # Access data from dict (works for both Pipe streams and raw DataArrays)
            data_source = detector_items[detector_key]
            # Handle Pipe streams (have .data attribute with a DataArray)
            # and raw DataArrays (from tests)
            has_data_attr = hasattr(data_source, 'data')
            if has_data_attr and hasattr(data_source.data, 'coords'):
                detector_data = data_source.data
            elif hasattr(data_source, 'coords'):
                # It's already a DataArray
                detector_data = data_source
            else:
                # Fallback: try to get data attribute (it might be a Variable)
                detector_data = data_source

            x_dim, y_dim = detector_data.dims[1], detector_data.dims[0]

            # Check each coordinate independently and extract unit if present
            x_unit = None
            if hasattr(detector_data, 'coords') and x_dim in detector_data.coords:
                x_coord_unit = detector_data.coords[x_dim].unit
                x_unit = str(x_coord_unit) if x_coord_unit is not None else None

            y_unit = None
            if hasattr(detector_data, 'coords') and y_dim in detector_data.coords:
                y_coord_unit = detector_data.coords[y_dim].unit
                y_unit = str(y_coord_unit) if y_coord_unit is not None else None

            self._setup_roi_watcher(box_stream, detector_key, x_unit, y_unit)

        # Overlay boxes on DynamicMap (not inside callback - this is crucial!)
        interactive_boxes = boxes.opts(fill_alpha=0.3, line_width=2)
        detector_with_boxes = detector_dmap * interactive_boxes
        plots.append(detector_with_boxes)

        # Create single ROI spectrum plot (overlays all ROIs for this detector)
        if spectrum_keys:
            # Override layout params to use overlay mode
            overlay_layout = LayoutParams(combine_mode='overlay')

            # Create filtered stream that only shows active ROI indices
            job_id = detector_key.job_id

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
                # Should we provide independent params for spectrum or just hard-code?
                aspect_params=PlotAspect(fix_width=True),
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
