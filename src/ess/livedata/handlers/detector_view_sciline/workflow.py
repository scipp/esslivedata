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
from ess.reduce.nexus.types import SampleRun
from ess.reduce.nexus.workflow import GenericNeXusWorkflow
from ess.reduce.streaming import EternalAccumulator

from .projectors import make_event_projector
from .providers import (
    compute_detector_histogram_3d,
    counts_in_toa_range,
    counts_total,
    cumulative_detector_image,
    cumulative_histogram,
    current_detector_image,
    project_events_geometric,
    project_events_logical,
    screen_coord_info_geometric,
    screen_coord_info_logical,
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
    LogicalTransform,
    ProjectionType,
    ReductionDim,
    TOFBins,
    TOFSlice,
    WindowHistogram,
)

# ============================================================================
# WindowAccumulator - clears after each finalize
# ============================================================================


class WindowAccumulator(EternalAccumulator):
    """
    Accumulator that clears its value after each finalize cycle.

    This is useful for computing "current window" values that should not include
    data from previous finalize cycles.
    """

    def on_finalize(self) -> None:
        """Clear accumulated value after finalize retrieves it."""
        self.clear()


# ============================================================================
# Workflow Construction
# ============================================================================


def create_base_workflow(
    *,
    tof_bins: sc.Variable,
    tof_slice: tuple[sc.Variable, sc.Variable] | None = None,
) -> sciline.Pipeline:
    """
    Create the base detector view workflow using GenericNeXusWorkflow.

    This creates the core workflow with histogram and downstream providers.
    Projection providers must be added separately based on projection type.

    Parameters
    ----------
    tof_bins:
        Bin edges for TOF histogramming.
    tof_slice:
        Optional (low, high) TOF range for output image slicing.

    Returns
    -------
    :
        Sciline pipeline with detector view providers (without projection).
    """
    # Start with GenericNeXusWorkflow for NeXus loading infrastructure
    workflow = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])

    # Add histogram and downstream providers (shared by both projection types)
    workflow.insert(compute_detector_histogram_3d)
    workflow.insert(cumulative_histogram)
    workflow.insert(window_histogram)
    workflow.insert(cumulative_detector_image)
    workflow.insert(current_detector_image)
    workflow.insert(counts_total)
    workflow.insert(counts_in_toa_range)

    # Add ROI providers
    workflow.insert(cumulative_roi_spectra)
    workflow.insert(current_roi_spectra)
    workflow.insert(roi_rectangle_readback)
    workflow.insert(roi_polygon_readback)

    # Set configuration parameters
    workflow[TOFBins] = tof_bins
    workflow[TOFSlice] = tof_slice

    return workflow


def add_geometric_projection(
    workflow: sciline.Pipeline,
    *,
    projection_type: Literal['xy_plane', 'cylinder_mantle_z'],
    resolution: dict[str, int],
    pixel_noise: Literal['cylindrical'] | sc.Variable | None = None,
) -> None:
    """
    Add geometric projection providers to the workflow.

    Parameters
    ----------
    workflow:
        Sciline pipeline to add providers to.
    projection_type:
        Type of geometric projection.
    resolution:
        Resolution (number of bins) for each screen dimension.
    pixel_noise:
        Noise to add to pixel positions. Can be 'cylindrical' for cylindrical
        detectors or a scalar variance for Gaussian noise. None disables noise.
    """
    # Add projection providers
    workflow.insert(make_event_projector)
    workflow.insert(project_events_geometric)

    # Add screen coordinate info provider (used for ROI precomputation)
    workflow.insert(screen_coord_info_geometric)

    # Add ROI precomputation providers (depend on ScreenCoordInfo)
    workflow.insert(precompute_roi_rectangle_bounds)
    workflow.insert(precompute_roi_polygon_masks)

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
    Add logical projection providers to the workflow.

    Parameters
    ----------
    workflow:
        Sciline pipeline to add providers to.
    transform:
        Callable that reshapes detector data (fold/slice). If None, identity.
    reduction_dim:
        Dimension(s) to merge events over. None means no reduction.
    """
    # Add projection provider
    workflow.insert(project_events_logical)

    # Add screen coordinate info provider (derived from EmptyDetector + transform)
    # Since EmptyDetector is static (from NeXus geometry), ROI precomputation
    # is independent of the event stream - same structure as geometric projection.
    workflow.insert(screen_coord_info_logical)

    # Add ROI precomputation providers (depend on ScreenCoordInfo)
    workflow.insert(precompute_roi_rectangle_bounds)
    workflow.insert(precompute_roi_polygon_masks)

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
