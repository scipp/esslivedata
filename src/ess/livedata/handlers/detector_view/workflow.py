# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Detector view workflow construction.

This module provides functions for creating and configuring the Sciline
workflow for detector view data reduction.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Literal

import sciline
import scipp as sc
import scippnexus as snx
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
from ess.reduce.nexus.types import NeXusComponent, NeXusTransformationChain, SampleRun
from ess.reduce.nexus.workflow import get_transformation_chain
from ess.reduce.unwrap import GenericUnwrapWorkflow

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
    project_wavelength_detector,
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
    FlipX,
    HistogramBins,
    HistogramSlice,
    LogicalTransform,
    ProjectionType,
    ReductionDim,
    TransformName,
    TransformValue,
    TransformValueLog,
    UsePixelWeighting,
)


def get_transformation_chain_with_value(
    detector: NeXusComponent[snx.NXdetector, SampleRun],
    transform_value: TransformValue,
) -> NeXusTransformationChain[snx.NXdetector, SampleRun]:
    """Inject a live value into one entry of the detector transformation chain.

    Replaces essreduce's ``get_transformation_chain`` so that a runtime
    f144 stream value drives the detector position. The baked-in value
    from the reference geometry file is intentionally never used: it may
    be stale or invalid, and a wrong result is worse than no result.
    """
    chain = get_transformation_chain(detector)
    if transform_value.name not in chain.transformations:
        raise KeyError(
            f"Transformation entry {transform_value.name!r} not found in chain. "
            f"Available entries: {sorted(chain.transformations.keys())}"
        )
    # Copy so we don't leak changes back into the cached NeXusComponent.
    chain = deepcopy(chain)
    chain.transformations[transform_value.name].value = transform_value.value
    return chain


def transform_value_from_log(
    log: TransformValueLog,
    name: TransformName,
) -> TransformValue:
    """Build a TransformValue from the latest sample of an NXlog DataArray.

    The ``log`` arrives via ``set_context`` from the ``ToNXlog``
    accumulator. We extract the most recent value as a scalar
    ``sc.Variable`` so the downstream ``to_transformation`` time-filter
    branch is bypassed (see ``ess.reduce.nexus.workflow.to_transformation``).

    Before the first ``set_context`` call the parameter is ``None``;
    after it, it is an NXlog that may still be empty if no f144 message
    has arrived yet. Both cases raise, so the workflow reports "no value
    yet" rather than silently falling back to the reference file's
    baked-in value (which may be stale or invalid).

    Raises
    ------
    ValueError
        If the log is ``None`` or has not yet received any samples.
    """
    if log is None or log.sizes.get('time', 0) == 0:
        raise ValueError(
            f"No samples yet for transformation {name!r}: f144 stream has not "
            "produced a value."
        )
    return TransformValue(name=name, value=log['time', -1].data)


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
        - 'wavelength': Wavelength (uses GenericUnwrapWorkflow, WavelengthDetector)
        For 'wavelength', caller must configure lookup table provider.

    Returns
    -------
    :
        Sciline pipeline with detector view providers.
    """
    # GenericUnwrapWorkflow extends GenericNeXusWorkflow with unwrap providers, so it
    # can be used for all coordinate modes. The coordinate mode determines which
    # projection provider to use.
    workflow = GenericUnwrapWorkflow(run_types=[SampleRun], monitor_types=[])

    # Select projection provider based on coordinate mode
    if coordinate_mode == 'toa':
        workflow.insert(project_raw_detector)
    elif coordinate_mode == 'wavelength':
        workflow.insert(project_wavelength_detector)
    else:
        raise ValueError(f"Unsupported coordinate mode: {coordinate_mode!r}")

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


def add_dynamic_transform(
    workflow: sciline.Pipeline,
    *,
    transform_name: str,
) -> None:
    """
    Patch the workflow to drive a NeXus transformation from a live f144 stream.

    Replaces essreduce's ``get_transformation_chain`` provider so that the
    detector's transformation chain picks up the latest value from an
    ``NXlog`` context stream. Only call this for sources that have a
    dynamic geometry configured; otherwise the workflow uses the file's
    baked-in transformation unchanged.

    Parameters
    ----------
    workflow:
        Sciline pipeline to configure.
    transform_name:
        Name of the entry inside the detector's ``depends_on`` chain whose
        value is driven by the f144 stream.
    """
    workflow.insert(get_transformation_chain_with_value)
    workflow.insert(transform_value_from_log)
    workflow[TransformName] = TransformName(transform_name)


def add_geometric_projection(
    workflow: sciline.Pipeline,
    *,
    projection_type: Literal['xy_plane', 'cylinder_mantle_z'],
    resolution: dict[str, int],
    pixel_noise: Literal['cylindrical'] | sc.Variable | None = None,
    flip_x: bool = False,
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
    flip_x:
        Whether to mirror the x-axis for 'view from sample' orientation.
    """
    # Add geometric projector provider
    workflow.insert(make_geometric_projector)

    # Set projection configuration
    workflow[ProjectionType] = projection_type
    workflow[DetectorViewResolution] = resolution
    workflow[FlipX] = flip_x

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
