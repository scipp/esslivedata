# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Factory for detector view Sciline workflow creation.

This module provides the DetectorViewScilineFactory for creating detector view
workflows with configurable projection types and parameters.
"""

from __future__ import annotations

from typing import Any

import scipp as sc
from scippnexus import NXdetector

from ess.reduce.nexus.types import NeXusData, SampleRun

from .data_source import DetectorDataSource
from .types import (
    CountsInTOARange,
    CountsTotal,
    CumulativeDetectorImage,
    CumulativeROISpectra,
    CurrentDetectorImage,
    CurrentROISpectra,
    GeometricViewConfig,
    LogicalViewConfig,
    ROIPolygonReadback,
    ROIPolygonRequest,
    ROIRectangleReadback,
    ROIRectangleRequest,
    ViewConfig,
)
from .workflow import (
    add_geometric_projection,
    add_logical_projection,
    create_accumulators,
    create_base_workflow,
)


class DetectorViewFactory:
    """
    Factory for creating Sciline-based detector view workflows.

    This factory creates StreamProcessorWorkflow instances that use the
    Sciline-based detector view workflow for accumulating detector data
    and producing cumulative and current detector images.

    Supports two projection modes via ViewConfig:
    1. GeometricViewConfig: For xy_plane/cylinder_mantle_z projections
    2. LogicalViewConfig: For fold/slice transforms

    Parameters
    ----------
    data_source:
        Detector data source configuration. Use NeXusDetectorSource for
        loading geometry from a file, or DetectorNumberSource for fast
        file-less startup with logical views.
    view_config:
        View configuration. Can be a single config (applied to all sources)
        or a dict mapping source names to configs (for per-detector settings).
    """

    def __init__(
        self,
        *,
        data_source: DetectorDataSource,
        view_config: ViewConfig | dict[str, ViewConfig],
    ) -> None:
        self._data_source = data_source
        self._view_config = view_config

    def _get_config(self, source_name: str) -> ViewConfig:
        """Get the view config for a given source."""
        if isinstance(self._view_config, dict):
            return self._view_config[source_name]
        return self._view_config

    def make_workflow(
        self,
        source_name: str,
        params: Any | None = None,
    ) -> Any:  # StreamProcessorWorkflow
        """
        Factory method that creates a detector view workflow.

        Parameters
        ----------
        source_name:
            Name of the detector source (e.g., 'panel_0').
        params:
            Workflow parameters (DetectorViewParams or similar with toa_edges).

        Returns
        -------
        :
            StreamProcessorWorkflow wrapping the Sciline-based detector view.
        """
        from ess.livedata.handlers.stream_processor_workflow import (
            StreamProcessorWorkflow,
        )

        if params is None:
            raise ValueError("params is required (must have toa_edges)")

        # Get histogram bins from params
        bins = params.toa_edges.get_edges()

        # Event coordinate to histogram - currently always event_time_offset
        # Future: could be extended via params to support wavelength, etc.
        event_coord = 'event_time_offset'

        # Get histogram slice from params if available
        histogram_slice = None
        if hasattr(params, 'toa_range') and params.toa_range.enabled:
            histogram_slice = params.toa_range.range_ns

        # Create base workflow
        workflow = create_base_workflow(
            bins=bins,
            event_coord=event_coord,
            histogram_slice=histogram_slice,
        )

        # Configure detector data source (EmptyDetector)
        self._data_source.configure_workflow(workflow, source_name)

        # Add projection based on config type
        config = self._get_config(source_name)
        match config:
            case GeometricViewConfig():
                add_geometric_projection(
                    workflow,
                    projection_type=config.projection_type,
                    resolution=config.resolution,
                    pixel_noise=config.pixel_noise,
                )
            case LogicalViewConfig():
                # Bind source_name to the transform if provided
                if config.transform is not None:

                    def bound_transform(
                        da: sc.DataArray, transform=config.transform
                    ) -> sc.DataArray:
                        return transform(da, source_name)

                else:
                    bound_transform = None

                add_logical_projection(
                    workflow,
                    transform=bound_transform,
                    reduction_dim=config.reduction_dim,
                )

        return StreamProcessorWorkflow(
            workflow,
            # Inject preprocessor output as NeXusData; GenericNeXusWorkflow
            # providers will group events by pixel to produce RawDetector.
            dynamic_keys={source_name: NeXusData[NXdetector, SampleRun]},
            # ROI configuration comes from auxiliary streams, updated less frequently
            context_keys={
                'roi_rectangle': ROIRectangleRequest,
                'roi_polygon': ROIPolygonRequest,
            },
            target_keys={
                'cumulative': CumulativeDetectorImage,
                'current': CurrentDetectorImage,
                'counts_total': CountsTotal,
                'counts_in_toa_range': CountsInTOARange,
                # ROI spectra (extracted from accumulated histograms)
                'roi_spectra_cumulative': CumulativeROISpectra,
                'roi_spectra_current': CurrentROISpectra,
                # ROI readbacks (providers ensure correct units from histogram coords)
                'roi_rectangle': ROIRectangleReadback,
                'roi_polygon': ROIPolygonReadback,
            },
            # Window outputs get time, start_time, end_time coords
            window_outputs=(
                'current',
                'counts_total',
                'counts_in_toa_range',
                'roi_spectra_current',
            ),
            accumulators=create_accumulators(),
        )
