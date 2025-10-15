# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import TypeVar

import holoviews as hv
import pydantic
import scipp as sc

import ess.livedata.config.keys as keys
from ess.livedata.config.models import RectangleROI
from ess.livedata.config.workflow_spec import (
    JobId,
    JobNumber,
    PersistentWorkflowConfig,
    PersistentWorkflowConfigs,
    ResultKey,
    WorkflowConfig,
    WorkflowId,
)

from .config_service import ConfigService
from .job_service import JobService
from .plot_params import LayoutParams, PlotParams2d
from .plots import ImagePlotter, LinePlotter
from .plotting import PlotterSpec, plotter_registry
from .roi_publisher import ROIPublisher, boxes_to_rois
from .stream_manager import StreamManager

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


class PlottingController:
    """
    Controller for managing plotting operations and configurations.

    Handles the creation of plots from job data, manages persistent plotter
    configurations, and coordinates between job services, stream managers,
    and configuration services.

    Parameters
    ----------
    job_service:
        Service for accessing job data and information.
    stream_manager:
        Manager for creating data streams.
    config_service:
        Service for persisting configurations. If None, configurations
        will not be persisted.
    logger:
        Logger instance. If None, creates a logger using the module name.
    max_persistent_configs:
        Maximum number of persistent configurations to keep.
    cleanup_fraction:
        Fraction of configurations to remove when cleanup is triggered. The oldest
        configurations are removed first.
    roi_publisher:
        Publisher for ROI updates to Kafka. If None, ROI publishing is disabled.
    """

    _plotter_config_key = keys.PERSISTENT_PLOTTING_CONFIGS.create_key()

    def __init__(
        self,
        job_service: JobService,
        stream_manager: StreamManager,
        config_service: ConfigService | None = None,
        logger: logging.Logger | None = None,
        max_persistent_configs: int = 100,
        cleanup_fraction: float = 0.2,
        roi_publisher: ROIPublisher | None = None,
    ) -> None:
        self._job_service = job_service
        self._stream_manager = stream_manager
        self._config_service = config_service
        self._logger = logger or logging.getLogger(__name__)
        self._max_persistent_configs = max_persistent_configs
        self._cleanup_fraction = cleanup_fraction
        self._box_streams: dict[ResultKey, hv.streams.BoxEdit] = {}
        self._roi_publisher = roi_publisher
        self._last_published_rois: dict[ResultKey, dict[int, RectangleROI]] = {}

    def get_available_plotters(
        self, job_number: JobNumber, output_name: str | None
    ) -> dict[str, PlotterSpec]:
        """
        Get all available plotters for a given job and output.

        Parameters
        ----------
        job_number:
            The job number to get plotters for.
        output_name:
            The name of the output to get plotters for.

        Returns
        -------
        :
            Dictionary mapping plotter names to their specifications.
        """
        job_data = self._job_service.job_data[job_number]
        data = {k: v[output_name] for k, v in job_data.items()}
        return plotter_registry.get_compatible_plotters(data)

    def get_spec(self, plot_name: str) -> PlotterSpec:
        """
        Get the parameter model for a given plotter name.

        Parameters
        ----------
        plot_name:
            Name of the plotter to get the specification for.

        Returns
        -------
        :
            The specification for the requested plotter.
        """
        return plotter_registry.get_spec(plot_name)

    def get_result_key(
        self, job_number: JobNumber, source_name: str, output_name: str | None
    ) -> ResultKey:
        """
        Get the ResultKey for a given job number and source name.

        Parameters
        ----------
        job_number:
            The job number.
        source_name:
            The name of the data source.
        output_name:
            The name of the output.

        Returns
        -------
        :
            The result key identifying the specific job output.
        """
        workflow_id = self._job_service.job_info[job_number]
        return ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(job_number=job_number, source_name=source_name),
            output_name=output_name,
        )

    def get_persistent_plotter_config(
        self, job_number: JobNumber, output_name: str | None, plot_name: str
    ) -> PersistentWorkflowConfig | None:
        """
        Get persistent plotter configuration for a given job, output, and plot.

        Parameters
        ----------
        job_number:
            The job number.
        output_name:
            The name of the output.
        plot_name:
            The name of the plotter.

        Returns
        -------
        :
            The persistent configuration if found, None otherwise.
        """
        if self._config_service is None:
            return None

        workflow_id = self._job_service.job_info[job_number]
        all_configs = self._config_service.get_config(
            self._plotter_config_key, PersistentWorkflowConfigs()
        )
        plotter_id = self._create_plotter_id(workflow_id, output_name, plot_name)
        return all_configs.configs.get(plotter_id)

    def _create_plotter_id(
        self, workflow_id: WorkflowId, output_name: str | None, plot_name: str
    ) -> WorkflowId:
        """
        Create a plotting-specific WorkflowId based on the data workflow.

        Parameters
        ----------
        workflow_id:
            The original workflow ID.
        output_name:
            The name of the output.
        plot_name:
            The name of the plotter.

        Returns
        -------
        :
            A unique workflow ID for the plotter configuration.
        """
        suffix_parts = [plot_name]
        if output_name is not None:
            suffix_parts.insert(0, output_name)
        suffix = "_".join(suffix_parts)

        return WorkflowId(
            instrument=workflow_id.instrument,
            namespace="plotting",
            name=f"{workflow_id.name}_{suffix}",
            version=workflow_id.version,
        )

    def _cleanup_old_configs(self, configs: PersistentWorkflowConfigs) -> None:
        """
        Remove oldest configs when limit is exceeded.

        In the case of workflows we simply remove workflows that do not exist anymore.
        This approach would be more difficult here, since for every workflow there can
        be multiple outputs, and for every output multiple applicable plotters, each of
        which should have its config saved. Hence we simply remove the oldest ones.

        Parameters
        ----------
        configs:
            The configuration object to clean up.
        """
        if len(configs.configs) <= self._max_persistent_configs:
            return

        num_to_remove = int(len(configs.configs) * self._cleanup_fraction)
        if num_to_remove == 0:
            num_to_remove = 1

        # Remove oldest configs (dict maintains insertion order, and this should work
        # even across serialized/deserialized states)
        oldest_keys = list(configs.configs.keys())[:num_to_remove]
        for key in oldest_keys:
            del configs.configs[key]

        self._logger.info(
            'Cleaned up %d old plotting configs, %d remaining',
            num_to_remove,
            len(configs.configs),
        )

    def _save_plotting_config(
        self,
        workflow_id: WorkflowId,
        source_names: list[str],
        output_name: str | None,
        plot_name: str,
        params: pydantic.BaseModel,
    ) -> None:
        """
        Save plotting configuration for persistence.

        Parameters
        ----------
        workflow_id:
            The workflow ID.
        source_names:
            List of source names for the configuration.
        output_name:
            The name of the output.
        plot_name:
            The name of the plotter.
        params:
            The plotter parameters to save.
        """
        if self._config_service is None:
            return

        plotter_id = self._create_plotter_id(workflow_id, output_name, plot_name)
        plot_config = WorkflowConfig(identifier=plotter_id, params=params.model_dump())

        current_configs = self._config_service.get_config(
            self._plotter_config_key, PersistentWorkflowConfigs()
        )
        current_configs.configs[plotter_id] = PersistentWorkflowConfig(
            source_names=source_names, config=plot_config
        )

        self._cleanup_old_configs(current_configs)
        self._config_service.update_config(self._plotter_config_key, current_configs)

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

    def _create_roi_detector_plot(
        self,
        job_number: JobNumber,
        source_names: list[str],
        output_name: str | None,
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
        job_number:
            The job number to create the plot for.
        source_names:
            List of data source names to include in the plot.
        output_name:
            The selected output name (e.g., 'current', 'cumulative').
        params:
            The plotter parameters (PlotParams2d).

        Returns
        -------
        :
            A HoloViews Layout with detector image (with BoxEdit overlay) and
            ROI spectrum plot.
        """
        job_data = self._job_service.job_data[job_number]

        # Separate detector data from ROI spectrum data by output name
        detector_items: dict[ResultKey, hv.streams.Pipe] = {}
        # Single dict with all ROI spectrum items (overlaid by LinePlotter)
        spectrum_items: dict[ResultKey, hv.streams.Pipe] = {}

        # Maximum number of ROIs to support (subscribe upfront for dynamic addition)
        max_roi_count = 3

        # Single placeholder for initialization (only used for first ROI)
        dim = 'time_of_arrival'
        placeholder = sc.DataArray(
            data=sc.array(dims=[dim], values=[0.0], unit='counts'),
            coords={dim: sc.array(dims=[dim], values=[0.0], unit='ns')},
        )

        for source_name in source_names:
            source_outputs = job_data[source_name]

            # Get detector image (2D) - the selected output
            has_detector_data = False
            if output_name and output_name in source_outputs:
                result_key = self.get_result_key(
                    job_number=job_number,
                    source_name=source_name,
                    output_name=output_name,
                )
                detector_items[result_key] = source_outputs[output_name]
                has_detector_data = True

            # Get or subscribe to ROI spectra (1D) - may not exist yet
            # Only create spectrum plot if we have detector data
            if output_name and has_detector_data:
                # Subscribe to multiple ROI indices upfront
                # (roi_current_0, roi_current_1, etc.)
                # This allows ROIs to be added dynamically after plot creation
                # LinePlotter will overlay all of them on a single plot
                roi_base_name = f'roi_{output_name}'

                # Subscribe to all ROI indices (0 through max_roi_count-1)
                for roi_idx in range(max_roi_count):
                    roi_spectrum_name = f'{roi_base_name}_{roi_idx}'
                    result_key = self.get_result_key(
                        job_number=job_number,
                        source_name=source_name,
                        output_name=roi_spectrum_name,
                    )

                    if roi_spectrum_name in source_outputs:
                        # Output already exists
                        spectrum_items[result_key] = source_outputs[roi_spectrum_name]
                    else:
                        # The output doesn't exist yet, subscribe anyway
                        # Stream will update when data arrives
                        spectrum_items[result_key] = placeholder

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
                coord_names = list(first_detector_data.dims)
                if len(coord_names) != 2:
                    self._logger.warning(
                        "Expected 2D detector data, got %dD data. "
                        "ROI coordinates may lack units.",
                        len(coord_names),
                    )
                    x_unit, y_unit = None, None
                else:
                    # Extract units from the coordinates
                    x_dim, y_dim = coord_names[1], coord_names[0]
                    x_unit = (
                        str(first_detector_data.coords[x_dim].unit)
                        if x_dim in first_detector_data.coords
                        else None
                    )
                    y_unit = (
                        str(first_detector_data.coords[y_dim].unit)
                        if y_dim in first_detector_data.coords
                        else None
                    )

                self._setup_roi_watcher(box_stream, first_detector_key, x_unit, y_unit)

            # Overlay boxes on DynamicMap (not inside callback - this is crucial!)
            interactive_boxes = boxes.opts(fill_alpha=0.3, line_width=2)
            detector_with_boxes = detector_dmap * interactive_boxes
            plots.append(detector_with_boxes)

        # Create single ROI spectrum plot (overlays all ROIs)
        if spectrum_items:
            # Override layout params to use overlay mode
            overlay_layout = LayoutParams(combine_mode='overlay')
            spectrum_pipe = self._stream_manager.make_merging_stream(spectrum_items)
            spectrum_plotter = LinePlotter(
                value_margin_factor=0.1,
                layout_params=overlay_layout,
                aspect_params=params.plot_aspect,
                scale_opts=params.plot_scale,
            )
            spectrum_plotter.initialize_from_data(spectrum_items)

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

    def create_plot(
        self,
        job_number: JobNumber,
        source_names: list[str],
        output_name: str | None,
        plot_name: str,
        params: pydantic.BaseModel,
    ) -> hv.DynamicMap | hv.Layout:
        """
        Create a plot from job data with the specified parameters.

        Saves the plotting configuration for future use and creates a dynamic
        plot that updates with streaming data.

        Parameters
        ----------
        job_number:
            The job number to create the plot for.
        source_names:
            List of data source names to include in the plot.
        output_name:
            The name of the output to plot.
        plot_name:
            The name of the plotter to use.
        params:
            The plotter parameters.

        Returns
        -------
        :
            A HoloViews DynamicMap that updates with streaming data.
            For plotters with kdims (e.g., SlicerPlotter), the DynamicMap
            includes interactive dimensions that generate widgets when rendered.
            For roi_detector, returns a Layout with separate DynamicMaps.
        """
        self._save_plotting_config(
            workflow_id=self._job_service.job_info[job_number],
            source_names=source_names,
            output_name=output_name,
            plot_name=plot_name,
            params=params,
        )

        # Special case for roi_detector: requires separate DynamicMaps
        # to maintain BoxEdit interactivity
        if plot_name == 'roi_detector':
            if not isinstance(params, PlotParams2d):
                raise TypeError(
                    f"roi_detector requires PlotParams2d, got {type(params).__name__}"
                )
            return self._create_roi_detector_plot(
                job_number=job_number,
                source_names=source_names,
                output_name=output_name,
                params=params,
            )

        items = {
            self.get_result_key(
                job_number=job_number, source_name=source_name, output_name=output_name
            ): self._job_service.job_data[job_number][source_name][output_name]
            for source_name in source_names
        }
        pipe = self._stream_manager.make_merging_stream(items)
        plotter = plotter_registry.create_plotter(plot_name, params=params)

        # Initialize plotter with initial data to determine kdims
        plotter.initialize_from_data(items)

        # Create DynamicMap with kdims (None if plotter doesn't use them)
        dmap = hv.DynamicMap(plotter, streams=[pipe], kdims=plotter.kdims, cache_size=1)

        return dmap.opts(shared_axes=False)
