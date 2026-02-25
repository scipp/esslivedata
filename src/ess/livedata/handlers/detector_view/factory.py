# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Factory for detector view Sciline workflow creation.

This module provides the DetectorViewScilineFactory for creating detector view
workflows with configurable projection types and parameters.
"""

from __future__ import annotations

import scipp as sc
from ess.reduce.nexus.types import NeXusData, SampleRun
from ess.reduce.time_of_flight.types import TofLookupTableFilename
from scippnexus import NXdetector

from ..accumulators import NoCopyAccumulator, NoCopyWindowAccumulator

# Import types unconditionally for runtime type hint resolution
# (used by workflow_factory.attach_factory to inspect parameter types)
from ..detector_view_specs import DetectorViewParams
from ..stream_processor_workflow import StreamProcessorWorkflow
from .data_source import DetectorDataSource, DetectorNumberSource
from .types import (
    AccumulatedHistogram,
    CountsInRange,
    CountsTotal,
    Cumulative,
    Current,
    DetectorImage,
    GeometricViewConfig,
    LogicalViewConfig,
    ROIPolygonReadback,
    ROIPolygonRequest,
    ROIRectangleReadback,
    ROIRectangleRequest,
    ROISpectra,
    UsePixelWeighting,
    ViewConfig,
)
from .workflow import (
    add_geometric_projection,
    add_logical_projection,
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
        params: DetectorViewParams,
        tof_lookup_table_filename: str | None = None,
    ) -> StreamProcessorWorkflow:
        """
        Factory method that creates a detector view workflow.

        Parameters
        ----------
        source_name:
            Name of the detector source (e.g., 'panel_0').
        params:
            Workflow parameters containing coordinate mode, edges, and ranges.
        tof_lookup_table_filename:
            Path to TOF lookup table file. Required for 'tof' and 'wavelength'
            coordinate modes. The caller (instrument factory) is responsible
            for resolving this from instrument-specific params.

        Returns
        -------
        :
            StreamProcessorWorkflow wrapping the Sciline-based detector view.
        """
        mode = params.coordinate_mode.mode

        # Validate TOF/wavelength mode requirements
        if mode in ('tof', 'wavelength'):
            if tof_lookup_table_filename is None:
                raise ValueError(f"{mode} mode requires tof_lookup_table_filename")
            if isinstance(self._data_source, DetectorNumberSource):
                raise ValueError(
                    f"{mode} mode requires geometry for Ltotal computation; "
                    "use NeXusDetectorSource instead of DetectorNumberSource"
                )

        # Get mode-specific event coordinate
        event_coord = {
            'toa': 'event_time_offset',
            'tof': 'tof',
            'wavelength': 'wavelength',
        }[mode]

        # Get active edges and range for current mode
        bins = params.get_active_edges()
        histogram_slice = params.get_active_range()

        # Get pixel weighting setting from params
        use_pixel_weighting = params.pixel_weighting.enabled

        # Create base workflow with appropriate mode
        workflow = create_base_workflow(
            bins=bins,
            event_coord=event_coord,
            histogram_slice=histogram_slice,
            coordinate_mode=mode,
        )

        # Set lookup table filename for TOF/wavelength modes
        if mode in ('tof', 'wavelength'):
            workflow[TofLookupTableFilename] = tof_lookup_table_filename

        # Configure detector data source (EmptyDetector)
        self._data_source.configure_workflow(workflow, source_name)

        # Set pixel weighting configuration
        workflow[UsePixelWeighting] = use_pixel_weighting

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
                roi_support = True  # Geometric views always support ROI
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
                roi_support = config.roi_support

        # Build target keys - conditionally include ROI outputs
        target_keys: dict[str, type] = {
            'cumulative': DetectorImage[Cumulative],
            'current': DetectorImage[Current],
            'counts_total': CountsTotal[Current],
            'counts_in_toa_range': CountsInRange[Current],
            'counts_total_cumulative': CountsTotal[Cumulative],
            'counts_in_toa_range_cumulative': CountsInRange[Cumulative],
        }
        context_keys: dict[str, type] = {}
        window_outputs = (
            'current',
            'counts_total',
            'counts_in_toa_range',
        )

        if roi_support:
            # Add ROI-related outputs only when supported
            target_keys.update(
                {
                    'roi_spectra_cumulative': ROISpectra[Cumulative],
                    'roi_spectra_current': ROISpectra[Current],
                    'roi_rectangle': ROIRectangleReadback,
                    'roi_polygon': ROIPolygonReadback,
                }
            )
            context_keys.update(
                {
                    'roi_rectangle': ROIRectangleRequest,
                    'roi_polygon': ROIPolygonRequest,
                }
            )
            window_outputs = (
                'current',
                'counts_total',
                'counts_in_toa_range',
                'roi_spectra_current',
            )

        return StreamProcessorWorkflow(
            workflow,
            # Inject preprocessor output as NeXusData; GenericNeXusWorkflow
            # providers will group events by pixel to produce RawDetector.
            dynamic_keys={source_name: NeXusData[NXdetector, SampleRun]},
            context_keys=context_keys,
            target_keys=target_keys,
            window_outputs=window_outputs,
            accumulators={
                AccumulatedHistogram[Cumulative]: NoCopyAccumulator(),
                AccumulatedHistogram[Current]: NoCopyWindowAccumulator(),
            },
        )
