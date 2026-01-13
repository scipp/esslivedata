# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Factory for detector view Sciline workflow creation.

This module provides the DetectorViewScilineFactory for creating detector view
workflows with configurable projection types and parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

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
    ROIPolygonReadback,
    ROIPolygonRequest,
    ROIRectangleReadback,
    ROIRectangleRequest,
)
from .workflow import (
    add_geometric_projection,
    add_logical_projection,
    create_accumulators,
    create_base_workflow,
)


class DetectorViewScilineFactory:
    """
    Factory for creating Sciline-based detector view workflows.

    This factory creates StreamProcessorWorkflow instances that use the
    Sciline-based detector view workflow for accumulating detector data
    and producing cumulative and current detector images.

    Supports two projection modes:
    1. Geometric: Use projection_type + resolution for xy_plane/cylinder_mantle_z
    2. Logical: Use logical_transform + reduction_dim for fold/slice views

    Parameters
    ----------
    data_source:
        Detector data source configuration. Use NeXusDetectorSource for
        loading geometry from a file, or DetectorNumberSource for fast
        file-less startup with logical views.
    tof_bins:
        Default bin edges for TOF histogramming.
    projection_type:
        Type of geometric projection ('xy_plane' or 'cylinder_mantle_z').
        If None, uses logical projection mode.
    resolution:
        Resolution (number of bins) for geometric projection screen dimensions.
    pixel_noise:
        Noise for geometric projection. 'cylindrical' or scalar variance.
    logical_transform:
        Callable to reshape detector data for logical projection.
        Signature: (da: DataArray, source_name: str) -> DataArray.
    reduction_dim:
        Dimension(s) to merge events over for logical projection.
    """

    def __init__(
        self,
        *,
        data_source: DetectorDataSource,
        tof_bins: sc.Variable,
        # Geometric projection params
        projection_type: Literal['xy_plane', 'cylinder_mantle_z'] | None = None,
        resolution: dict[str, int] | None = None,
        pixel_noise: Literal['cylindrical'] | sc.Variable | None = None,
        # Logical projection params
        logical_transform: Callable[[sc.DataArray, str], sc.DataArray] | None = None,
        reduction_dim: str | list[str] | None = None,
    ) -> None:
        self._data_source = data_source
        self._tof_bins = tof_bins
        # Geometric projection
        self._projection_type = projection_type
        self._resolution = resolution
        self._pixel_noise = pixel_noise
        # Logical projection
        self._logical_transform = logical_transform
        self._reduction_dim = reduction_dim

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
            Workflow parameters (for TOA range, etc.).

        Returns
        -------
        :
            StreamProcessorWorkflow wrapping the Sciline-based detector view.
        """
        from ess.livedata.handlers.stream_processor_workflow import (
            StreamProcessorWorkflow,
        )

        # Get TOF slice from params if available
        tof_slice = None
        if params is not None and hasattr(params, 'toa_range'):
            if params.toa_range.enabled:
                tof_slice = params.toa_range.range_ns

        # Create base workflow
        workflow = create_base_workflow(
            tof_bins=self._tof_bins,
            tof_slice=tof_slice,
        )

        # Configure detector data source (EmptyDetector)
        self._data_source.configure_workflow(workflow, source_name)

        # Add projection based on mode
        if self._projection_type is not None:
            # Geometric projection mode
            add_geometric_projection(
                workflow,
                projection_type=self._projection_type,
                resolution=self._resolution or {},
                pixel_noise=self._pixel_noise,
            )
        else:
            # Logical projection mode
            # Bind source_name to the transform if provided
            if self._logical_transform is not None:

                def bound_transform(da: sc.DataArray) -> sc.DataArray:
                    return self._logical_transform(da, source_name)

            else:
                bound_transform = None

            add_logical_projection(
                workflow,
                transform=bound_transform,
                reduction_dim=self._reduction_dim,
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
            accumulators=create_accumulators(),
        )
