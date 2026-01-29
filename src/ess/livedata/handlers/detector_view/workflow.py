# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Detector view workflow construction.

This module provides functions for creating and configuring the Sciline
workflow for detector view data reduction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

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
from ess.reduce.streaming import EternalAccumulator
from ess.reduce.time_of_flight import GenericTofWorkflow

from .projectors import make_geometric_projector, make_logical_projector
from .providers import (
    accumulated_histogram,
    compute_detector_histogram,
    compute_pixel_weights,
    counts_in_range,
    counts_total,
    detector_image,
    get_screen_metadata,
    project_raw_detector,
    project_tof_detector,
)
from .roi import (
    precompute_roi_polygon_masks,
    precompute_roi_rectangle_bounds,
    roi_polygon_readback,
    roi_rectangle_readback,
    roi_spectra,
)
from .types import (
    CoordinateMode,
    EventCoordName,
    HistogramBins,
    HistogramSlice,
    LogicalTransform,
    ProjectionType,
    ReductionDim,
    UsePixelWeighting,
)


class NoCopyAccumulator(EternalAccumulator):
    """
    Accumulator that skips deepcopy on read for better performance.

    The base EternalAccumulator uses deepcopy in _get_value() to ensure safety.
    This accumulator skips that deepcopy, saving ~30ms per read for a 500MB
    histogram.

    The copy on first push is retained to avoid shared references when the same
    value is pushed to multiple accumulators.

    Use only when downstream consumers do not modify or store references to
    the returned value. This constraint is met in the detector view workflow
    where downstream just serializes the data.
    """

    def _get_value(self):
        """Return value directly without deepcopy."""
        return self._value


class NoCopyWindowAccumulator(NoCopyAccumulator):
    """
    Window accumulator without deepcopy that clears after finalize.

    Combines the performance benefits of NoCopyAccumulator with window semantics
    (clearing after each finalize cycle).
    """

    def on_finalize(self) -> None:
        """Clear accumulated value after finalize retrieves it."""
        self.clear()


def create_base_workflow(
    *,
    bins: sc.Variable,
    event_coord: str = 'event_time_offset',
    histogram_slice: tuple[sc.Variable, sc.Variable] | None = None,
    coordinate_mode: CoordinateMode = 'toa',
) -> sciline.Pipeline:
    """
    Create the base detector view workflow.

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
    coordinate_mode:
        Coordinate system for event data:
        - 'toa': Time-of-arrival (uses GenericNeXusWorkflow, RawDetector)
        - 'tof': Time-of-flight (uses GenericTofWorkflow, TofDetector)
        - 'wavelength': Wavelength (uses GenericTofWorkflow, WavelengthDetector)
        For 'tof' and 'wavelength', caller must configure lookup table provider.

    Returns
    -------
    :
        Sciline pipeline with detector view providers.
    """
    # GenericTofWorkflow extends GenericNeXusWorkflow with TOF providers, so it can
    # be used for all coordinate modes. The coordinate mode determines which
    # projection provider to use.
    workflow = GenericTofWorkflow(run_types=[SampleRun], monitor_types=[])

    # Select projection provider based on coordinate mode
    if coordinate_mode == 'toa':
        workflow.insert(project_raw_detector)
    elif coordinate_mode == 'tof':
        workflow.insert(project_tof_detector)
    elif coordinate_mode == 'wavelength':
        # Future: would use WavelengthDetector-based provider
        raise NotImplementedError("wavelength mode is not yet implemented")
    else:
        raise ValueError(f"Unknown coordinate_mode: {coordinate_mode}")

    # Add screen metadata provider (bridges projector to ROI providers)
    workflow.insert(get_screen_metadata)

    # Add pixel weighting provider
    workflow.insert(compute_pixel_weights)
    workflow[UsePixelWeighting] = False  # Default: disabled

    # Add histogram and downstream providers (generic providers for both modes)
    workflow.insert(compute_detector_histogram)
    workflow.insert(accumulated_histogram)
    workflow.insert(detector_image)
    workflow.insert(counts_total)
    workflow.insert(counts_in_range)

    # Add ROI providers (generic provider for both modes)
    workflow.insert(roi_spectra)
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
