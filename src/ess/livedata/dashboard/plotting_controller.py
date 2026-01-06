# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

import holoviews as hv
import pydantic

from ess.livedata.config.workflow_spec import (
    JobId,
    JobNumber,
    ResultKey,
    WorkflowId,
    WorkflowSpec,
)

from .job_service import JobService
from .plot_params import create_extractors_from_params
from .plotting import PlotterSpec, plotter_registry
from .roi_publisher import ROIPublisher
from .roi_request_plots import PolygonsRequestPlotter, RectanglesRequestPlotter
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
        self._roi_publisher = roi_publisher

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

    def setup_data_pipeline(
        self,
        job_number: JobNumber,
        workflow_id: WorkflowId,
        source_names: list[str],
        output_name: str | None,
        plot_name: str,
        params: dict | pydantic.BaseModel,
        on_first_data: Callable[[Any], None],
    ) -> None:
        """
        Set up the data pipeline for a plot with callback for first data arrival.

        This is Phase 1 of two-phase plot creation. It creates the data subscriber
        and stream without creating the plotter. When data arrives, the callback
        is invoked with the pipe, which should then be used with
        create_plot_from_pipeline() to create the plot.

        Parameters
        ----------
        job_number:
            The job number to set up the pipeline for.
        workflow_id:
            The workflow ID for this plot.
        source_names:
            List of data source names to include.
        output_name:
            The name of the output.
        plot_name:
            The name of the plotter to use.
        params:
            The plotter parameters as a dict or validated Pydantic model.
        on_first_data:
            Callback invoked when first data arrives, receives the pipe as parameter.
        """
        # Validate params if dict, pass through if already a model
        if isinstance(params, dict):
            spec = plotter_registry.get_spec(plot_name)
            params = spec.params(**params) if spec.params else pydantic.BaseModel()

        # Build result keys for all sources
        keys = [
            ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(job_number=job_number, source_name=source_name),
                output_name=output_name,
            )
            for source_name in source_names
        ]
        spec = plotter_registry.get_spec(plot_name)
        window = getattr(params, 'window', None)
        extractors = create_extractors_from_params(keys, window, spec)

        # Set up data pipeline with callback for first data
        self._stream_manager.make_merging_stream(
            extractors, on_first_data=on_first_data
        )

    def create_plot_from_pipeline(
        self,
        plot_name: str,
        params: dict | pydantic.BaseModel,
        pipe: Any,
    ) -> hv.DynamicMap:
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

        Returns
        -------
        :
            A HoloViews DynamicMap that updates with streaming data.
        """
        plotter = plotter_registry.create_plotter(plot_name, params=params)

        # Special case for ROI request plotters: they return a DynamicMap with
        # BoxEdit/PolyDraw streams already attached. Don't wrap again.
        if isinstance(plotter, RectanglesRequestPlotter | PolygonsRequestPlotter):
            plotter._roi_publisher = self._roi_publisher
            # Get the first data key and its data for the plot() call
            data_key = next(iter(pipe.data.keys()))
            data = pipe.data[data_key]
            # plot() creates and returns the interactive DynamicMap
            return plotter.plot(data, data_key)

        # Initialize plotter with extracted data from pipe to determine kdims
        plotter.initialize_from_data(pipe.data)

        # Create DynamicMap with kdims (None if plotter doesn't use them)
        dmap = hv.DynamicMap(plotter, streams=[pipe], kdims=plotter.kdims, cache_size=1)

        # Return DynamicMap directly without Layout wrapping.
        # This preserves the plotter's framewise settings when layers are overlaid.
        return dmap
