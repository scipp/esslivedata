# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

import holoviews as hv
import pydantic

from ess.livedata.config.workflow_spec import (
    ResultKey,
    WorkflowSpec,
)

from .job_service import JobService
from .plot_params import create_extractors_from_params
from .plotting import PlotterSpec, plotter_registry
from .roi_detector_plot_factory import ROIDetectorPlotFactory
from .roi_publisher import ROIPublisher
from .stream_manager import StreamManager

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


class PlottingController:
    """
    Controller for managing plotting operations and configurations.

    Coordinates between job services, stream managers, and plot creation,
    using a two-phase pipeline for creating plots with streaming data.

    Parameters
    ----------
    job_service:
        Service for accessing job data and information.
    stream_manager:
        Manager for creating data streams.
    logger:
        Logger instance. If None, creates a logger using the module name.
    roi_publisher:
        Publisher for ROI updates to Kafka. If None, ROI publishing is disabled.
    """

    def __init__(
        self,
        job_service: JobService,
        stream_manager: StreamManager,
        logger: logging.Logger | None = None,
        roi_publisher: ROIPublisher | None = None,
    ) -> None:
        self._job_service = job_service
        self._stream_manager = stream_manager
        self._logger = logger or logging.getLogger(__name__)
        self._roi_detector_plot_factory = ROIDetectorPlotFactory(
            stream_manager=stream_manager, roi_publisher=roi_publisher, logger=logger
        )

    def get_available_plotters_from_spec(
        self, workflow_spec: WorkflowSpec, output_name: str
    ) -> tuple[dict[str, PlotterSpec], bool]:
        """
        Get available plotters based on workflow spec template (before data exists).

        Uses the output template DataArray from the workflow specification to
        determine compatible plotters. The template is an "empty" DataArray with
        the expected structure (dims, coords, units) that allows full validation
        including custom validators.

        Also checks spec requirements (e.g., aux_sources for ROI support) to filter
        out plotters that require features not supported by the workflow spec.

        When a template is not available, falls back to returning all registered
        plotters. The boolean flag indicates whether a template was available.

        Parameters
        ----------
        workflow_spec:
            WorkflowSpec object containing output templates.
        output_name:
            The name of the output to get plotters for.

        Returns
        -------
        :
            Tuple of (plotters_dict, has_template). If has_template is False,
            all registered plotters are returned as a fallback, and the caller
            should warn the user that some plotters may not work with the data.
        """
        template = workflow_spec.get_output_template(output_name)
        if template is None:
            return plotter_registry.get_specs(), False
        return (
            plotter_registry.get_compatible_plotters_with_spec(
                {output_name: template}, workflow_spec.aux_sources
            ),
            True,
        )

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

    def get_static_plotters(self) -> dict[str, PlotterSpec]:
        """
        Get available static plotters (for overlays without data sources).

        Returns
        -------
        :
            Dictionary of static plotter names to their specifications.
        """
        return plotter_registry.get_static_plotters()

    def setup_pipeline(
        self,
        keys: list[ResultKey],
        plot_name: str,
        params: dict | pydantic.BaseModel,
        on_first_data: Callable[[Any], None],
        ready_condition: Callable[[set[ResultKey]], bool] | None = None,
    ) -> None:
        """
        Set up data pipeline for any plot type.

        This is the unified interface for setting up data pipelines that works
        for both single-source and multi-source layers. PlotOrchestrator should
        use this method exclusively.

        Parameters
        ----------
        keys
            ResultKeys for all data sources (built by LayerSubscription).
        plot_name
            Name of the plotter to use.
        params
            Plotter parameters as a dict or validated Pydantic model.
        on_first_data
            Callback when data is ready for plot creation.
        ready_condition
            Condition for when on_first_data should fire. If None, fires
            when any data is available (single-source default). For multi-source
            layers, LayerSubscription provides a condition requiring data from
            each DataSourceConfig.
        """
        # Validate params if dict, pass through if already a model
        if isinstance(params, dict):
            spec = plotter_registry.get_spec(plot_name)
            params = spec.params(**params) if spec.params else pydantic.BaseModel()

        spec = plotter_registry.get_spec(plot_name)
        window = getattr(params, 'window', None)

        # Special case for roi_detector: create separate subscriptions per detector
        # and coordinate them to invoke callback once when all are ready.
        # This is needed because roi_detector creates separate DynamicMaps per detector
        # rather than a single merged stream.
        if plot_name == 'roi_detector':
            pipes: dict[ResultKey, Any] = {}

            def make_detector_callback(key: ResultKey) -> Callable[[Any], None]:
                """Create a callback that tracks individual detector readiness."""

                def on_detector_ready(pipe: Any) -> None:
                    pipes[key] = pipe
                    if len(pipes) == len(keys):
                        # All detectors ready, invoke user callback with dict of pipes
                        on_first_data(pipes)

                return on_detector_ready

            # Create one subscription per detector
            for key in keys:
                extractors = create_extractors_from_params([key], window, spec)
                self._stream_manager.make_merging_stream(
                    extractors, on_first_data=make_detector_callback(key)
                )
            return

        # Standard path: create single merged subscription
        extractors = create_extractors_from_params(keys, window, spec)

        self._stream_manager.make_merging_stream(
            extractors,
            on_first_data=on_first_data,
            ready_condition=ready_condition,
        )

    def create_plot_from_pipeline(
        self,
        plot_name: str,
        params: dict | pydantic.BaseModel,
        pipe: Any,
    ) -> hv.DynamicMap | hv.Layout:
        """
        Create a plot from an already-initialized data pipeline.

        This is Phase 2 of two-phase plot creation. The pipeline must have
        data already (typically called after setup_data_pipeline's subscriber
        has been triggered).

        Parameters
        ----------
        plot_name:
            The name of the plotter to use.
        params:
            The plotter parameters as a dict or validated Pydantic model.
        pipe:
            The pipe from setup_data_pipeline() with data available.
            For roi_detector, this is a dict of pipes (one per detector).
            For standard plots, this is a single pipe.

        Returns
        -------
        :
            A HoloViews DynamicMap that updates with streaming data.
            For roi_detector, returns a Layout with separate DynamicMaps.
        """
        # Special case for roi_detector: receives dict of pipes (one per detector).
        # TODO: Remove when roi_detector is migrated to the layer system.
        if plot_name == 'roi_detector':
            # pipe is dict[ResultKey, Pipe]
            pipes_dict = pipe

            plots = []
            for key, detector_pipe in pipes_dict.items():
                # Create ROI detector plot with ROI overlays
                factory = self._roi_detector_plot_factory
                detector_with_rois, _plot_state = (
                    factory.create_roi_detector_plot_components(
                        detector_key=key, params=params, detector_pipe=detector_pipe
                    )
                )
                plots.append(detector_with_rois)

            return hv.Layout(plots).opts(shared_axes=False)

        plotter = plotter_registry.create_plotter(plot_name, params=params)

        # Initialize plotter with extracted data from pipe to determine kdims
        plotter.initialize_from_data(pipe.data)

        # Create DynamicMap with kdims (None if plotter doesn't use them)
        dmap = hv.DynamicMap(plotter, streams=[pipe], kdims=plotter.kdims, cache_size=1)

        # Return DynamicMap directly without Layout wrapping.
        # This preserves the plotter's framewise settings when layers are overlaid.
        return dmap
