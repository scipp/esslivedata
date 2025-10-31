# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import TypeVar

import holoviews as hv
import pydantic

from ess.livedata.config.workflow_spec import (
    JobId,
    JobNumber,
    PersistedUIConfig,
    ResultKey,
    WorkflowId,
)

from .config_store import ConfigStore
from .job_service import JobService
from .plotting import PlotterSpec, plotter_registry
from .roi_detector_plot_factory import ROIDetectorPlotFactory
from .roi_publisher import ROIPublisher
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
    config_store:
        Store for persisting plotter configurations across sessions.
        If None, configurations will not be persisted.
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

    def __init__(
        self,
        job_service: JobService,
        stream_manager: StreamManager,
        config_store: ConfigStore | None = None,
        logger: logging.Logger | None = None,
        max_persistent_configs: int = 100,
        cleanup_fraction: float = 0.2,
        roi_publisher: ROIPublisher | None = None,
    ) -> None:
        self._job_service = job_service
        self._stream_manager = stream_manager
        self._config_store = config_store
        self._logger = logger or logging.getLogger(__name__)
        self._max_persistent_configs = max_persistent_configs
        self._cleanup_fraction = cleanup_fraction
        self._roi_detector_plot_factory = ROIDetectorPlotFactory(
            stream_manager=stream_manager, roi_publisher=roi_publisher, logger=logger
        )

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
    ) -> PersistedUIConfig | None:
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
        if self._config_store is None:
            return None

        workflow_id = self._job_service.job_info[job_number]
        plotter_id = self._create_plotter_id(workflow_id, output_name, plot_name)
        return self._config_store.load_plotter_config(plotter_id)

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

    def _cleanup_old_configs(self) -> None:
        """
        Remove oldest plotter configs when limit is exceeded.

        In the case of workflows we simply remove workflows that do not exist anymore.
        This approach would be more difficult here, since for every workflow there can
        be multiple outputs, and for every output multiple applicable plotters, each of
        which should have its config saved. Hence we simply remove the oldest ones.
        """
        if self._config_store is None:
            return

        self._config_store.cleanup_old_plotter_configs(
            max_configs=self._max_persistent_configs,
            cleanup_fraction=self._cleanup_fraction,
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
        if self._config_store is None:
            return

        plotter_id = self._create_plotter_id(workflow_id, output_name, plot_name)

        persistent_config = PersistedUIConfig(
            source_names=source_names,
            aux_source_names={},
            params=params.model_dump(),
        )
        self._config_store.save_plotter_config(plotter_id, persistent_config)
        self._cleanup_old_configs()

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
        items = {
            self.get_result_key(
                job_number=job_number, source_name=source_name, output_name=output_name
            ): self._job_service.job_data[job_number][source_name][output_name]
            for source_name in source_names
        }

        # Special case for roi_detector: call factory once per detector
        if plot_name == 'roi_detector':
            plot_components = [
                self._roi_detector_plot_factory.create_roi_detector_plot_components(
                    detector_key=key, detector_data=data, params=params
                )
                for key, data in items.items()
            ]
            # Each component returns (detector_with_boxes, roi_spectrum, plot_state)
            # Flatten detector and spectrum plots into a layout with 2 columns
            plots = []
            for detector_with_boxes, roi_spectrum, _plot_state in plot_components:
                plots.extend([detector_with_boxes, roi_spectrum])
            return hv.Layout(plots).cols(2).opts(shared_axes=False)

        pipe = self._stream_manager.make_merging_stream(items)
        plotter = plotter_registry.create_plotter(plot_name, params=params)

        # Initialize plotter with initial data to determine kdims
        plotter.initialize_from_data(items)

        # Create DynamicMap with kdims (None if plotter doesn't use them)
        dmap = hv.DynamicMap(plotter, streams=[pipe], kdims=plotter.kdims, cache_size=1)

        return dmap.opts(shared_axes=False)
