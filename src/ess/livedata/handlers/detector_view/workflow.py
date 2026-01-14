# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Detector view workflow construction.

This module provides functions for creating and configuring the Sciline
workflow for detector view data reduction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import sciline
import scipp as sc

from ess.reduce.live.raw import (
    DetectorViewResolution,
    PositionNoiseReplicaCount,
    PositionNoiseSigma,
    gaussian_position_noise,
    pixel_cylinder_axis,
    pixel_cylinder_radius,
    pixel_shape,
    position_noise_for_cylindrical_pixel,
    position_with_noisy_replicas,
)
from ess.reduce.nexus.types import EmptyDetector, SampleRun
from ess.reduce.nexus.workflow import GenericNeXusWorkflow
from ess.reduce.streaming import EternalAccumulator

from .projectors import (
    GeometricProjector,
    Projector,
    make_geometric_projector,
    make_logical_projector,
)
from .providers import (
    compute_detector_histogram_3d,
    counts_in_range,
    counts_total,
    cumulative_detector_image,
    cumulative_histogram,
    current_detector_image,
    project_events,
    window_histogram,
)
from .roi import (
    cumulative_roi_spectra,
    current_roi_spectra,
    precompute_roi_polygon_masks,
    precompute_roi_rectangle_bounds,
    roi_polygon_readback,
    roi_rectangle_readback,
)
from .types import (
    CumulativeHistogram,
    EventCoordName,
    HistogramBins,
    HistogramSlice,
    LogicalTransform,
    ProjectionType,
    ReductionDim,
    ScreenMetadata,
    WindowHistogram,
)


class WindowAccumulator(EternalAccumulator):
    """
    Accumulator that clears its value after each finalize cycle.

    This is useful for computing "current window" values that should not include
    data from previous finalize cycles.
    """

    def on_finalize(self) -> None:
        """Clear accumulated value after finalize retrieves it."""
        self.clear()


def get_screen_metadata(
    projector: Projector,
    empty_detector: EmptyDetector[SampleRun],
) -> ScreenMetadata:
    """
    Extract screen metadata from projector.

    For GeometricProjector, metadata is stored at construction time.
    For LogicalProjector, metadata is computed from the empty detector.

    Parameters
    ----------
    projector:
        The projector instance.
    empty_detector:
        Detector structure without events (used by LogicalProjector).

    Returns
    -------
    :
        Screen metadata with output dimensions and coordinates.
    """
    if isinstance(projector, GeometricProjector):
        return projector.screen_metadata
    return projector.get_screen_metadata(empty_detector)


def create_base_workflow(
    *,
    bins: sc.Variable,
    event_coord: str = 'event_time_offset',
    histogram_slice: tuple[sc.Variable, sc.Variable] | None = None,
) -> sciline.Pipeline:
    """
    Create the base detector view workflow using GenericNeXusWorkflow.

    This creates the core workflow with all providers. The Projector param
    must be set separately via add_geometric_projection or add_logical_projection.

    Parameters
    ----------
    bins:
        Bin edges for histogramming the event coordinate.
    event_coord:
        Name of the event coordinate to histogram.
    histogram_slice:
        Optional (low, high) range for output image slicing.

    Returns
    -------
    :
        Sciline pipeline with detector view providers.
    """
    # Start with GenericNeXusWorkflow for NeXus loading infrastructure
    workflow = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])

    # Add projection provider (unified for geometric and logical)
    workflow.insert(project_events)

    # Add screen metadata provider (bridges projector to ROI providers)
    workflow.insert(get_screen_metadata)

    # Add histogram and downstream providers
    workflow.insert(compute_detector_histogram_3d)
    workflow.insert(cumulative_histogram)
    workflow.insert(window_histogram)
    workflow.insert(cumulative_detector_image)
    workflow.insert(current_detector_image)
    workflow.insert(counts_total)
    workflow.insert(counts_in_range)

    # Add ROI providers
    workflow.insert(cumulative_roi_spectra)
    workflow.insert(current_roi_spectra)
    workflow.insert(roi_rectangle_readback)
    workflow.insert(roi_polygon_readback)
    workflow.insert(precompute_roi_rectangle_bounds)
    workflow.insert(precompute_roi_polygon_masks)

    # Set configuration parameters
    workflow[HistogramBins] = bins
    workflow[EventCoordName] = event_coord
    workflow[HistogramSlice] = histogram_slice

    return workflow


def add_geometric_projection(
    workflow: sciline.Pipeline,
    *,
    projection_type: Literal['xy_plane', 'cylinder_mantle_z'],
    resolution: dict[str, int],
    pixel_noise: Literal['cylindrical'] | sc.Variable | None = None,
) -> None:
    """
    Configure the workflow for geometric projection.

    This adds the geometric projector provider which creates a projector
    based on calibrated pixel positions.

    Parameters
    ----------
    workflow:
        Sciline pipeline to configure.
    projection_type:
        Type of geometric projection.
    resolution:
        Resolution (number of bins) for each screen dimension.
    pixel_noise:
        Noise to add to pixel positions. Can be 'cylindrical' for cylindrical
        detectors or a scalar variance for Gaussian noise. None disables noise.
    """
    # Add geometric projector provider
    workflow.insert(make_geometric_projector)

    # Set projection configuration
    workflow[ProjectionType] = projection_type
    workflow[DetectorViewResolution] = resolution

    # Configure noise generation
    if pixel_noise is None:
        workflow[PositionNoiseSigma] = sc.scalar(0.0, unit='m')
        workflow[PositionNoiseReplicaCount] = 0
    elif isinstance(pixel_noise, sc.Variable):
        workflow.insert(gaussian_position_noise)
        workflow[PositionNoiseSigma] = pixel_noise
        workflow[PositionNoiseReplicaCount] = 4
    elif pixel_noise == 'cylindrical':
        workflow.insert(pixel_shape)
        workflow.insert(pixel_cylinder_axis)
        workflow.insert(pixel_cylinder_radius)
        workflow.insert(position_noise_for_cylindrical_pixel)
        workflow[PositionNoiseReplicaCount] = 4
    else:
        raise ValueError(f"Invalid pixel_noise: {pixel_noise}")

    # Add noise replica generation for position calibration
    workflow.insert(position_with_noisy_replicas)


def add_logical_projection(
    workflow: sciline.Pipeline,
    *,
    transform: Callable[[sc.DataArray], sc.DataArray] | None = None,
    reduction_dim: str | list[str] | None = None,
) -> None:
    """
    Configure the workflow for logical projection.

    This adds the logical projector provider which creates a projector
    based on fold/slice transforms.

    Parameters
    ----------
    workflow:
        Sciline pipeline to configure.
    transform:
        Callable that reshapes detector data (fold/slice). If None, identity.
    reduction_dim:
        Dimension(s) to merge events over. None means no reduction.
    """
    # Add logical projector provider
    workflow.insert(make_logical_projector)

    # Set projection configuration
    workflow[LogicalTransform] = transform
    workflow[ReductionDim] = reduction_dim


def create_accumulators() -> dict[type, Any]:
    """
    Create the accumulator configuration for StreamProcessor.

    Returns
    -------
    :
        Dict mapping accumulator types to accumulator instances.
    """
    return {
        CumulativeHistogram: EternalAccumulator(),
        WindowHistogram: WindowAccumulator(),
    }
