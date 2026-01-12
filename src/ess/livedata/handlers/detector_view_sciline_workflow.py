# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Sciline-based detector view workflow for live data visualization.

This module implements the detector view workflow using Sciline and StreamProcessor,
providing event-based projection of detector data to screen coordinates while
preserving TOF information for flexible histogramming.

Supports two projection modes:
1. Geometric projections (xy_plane, cylinder_mantle_z) using Histogrammer coordinates
2. Logical views (fold/slice transforms with optional reduction)
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any, Literal, NewType

import numpy as np
import sciline
import scipp as sc
from scippnexus import NXdetector

from ess.reduce.live.raw import (
    CalibratedPositionWithNoisyReplicas,
    DetectorViewResolution,
    PositionNoiseReplicaCount,
    PositionNoiseSigma,
    gaussian_position_noise,
    make_cylinder_mantle_coords,
    make_xy_plane_coords,
    pixel_cylinder_axis,
    pixel_cylinder_radius,
    pixel_shape,
    position_noise_for_cylindrical_pixel,
    position_with_noisy_replicas,
)
from ess.reduce.nexus.types import (
    Filename,
    NeXusData,
    NeXusName,
    RawDetector,
    SampleRun,
)
from ess.reduce.nexus.workflow import GenericNeXusWorkflow
from ess.reduce.streaming import EternalAccumulator

from ..config import models

# ============================================================================
# Type definitions
# ============================================================================

# Configuration types (set once at workflow creation)
TOFBins = NewType('TOFBins', sc.Variable)
"""Bin edges for time-of-flight histogramming."""

# Optional TOF range for slicing output images
TOFSlice = NewType('TOFSlice', tuple[sc.Variable, sc.Variable] | None)
"""Optional (low, high) TOF range for detector image slicing. None means all TOF."""

# Logical transform configuration
LogicalTransform = NewType(
    'LogicalTransform', Callable[[sc.DataArray], sc.DataArray] | None
)
"""Callable that transforms detector data to logical coordinates, or None."""

# Reduction dimension for logical views
ReductionDim = NewType('ReductionDim', str | list[str] | None)
"""Dimension(s) to sum over after applying logical transform."""

# Projection type for geometric views
ProjectionType = NewType(
    'ProjectionType', Literal['xy_plane', 'cylinder_mantle_z'] | None
)
"""Type of geometric projection to use, or None for logical view."""

# Intermediate types for event projection
ScreenBinnedEvents = NewType('ScreenBinnedEvents', sc.DataArray)
"""Events binned by screen coordinates (screen_y, screen_x) with TOF preserved."""

# Shared intermediate - computed once, then split for accumulation
DetectorHistogram3D = NewType('DetectorHistogram3D', sc.DataArray)
"""3D histogram with dims (y, x, tof) - computed once, shared by accumulators."""

# Accumulated data types - use different types for different accumulator behavior
CumulativeHistogram = NewType('CumulativeHistogram', sc.DataArray)
"""3D histogram accumulated forever (EternalAccumulator)."""

WindowHistogram = NewType('WindowHistogram', sc.DataArray)
"""3D histogram for current window (clears after finalize)."""

# Output types
CumulativeDetectorImage = NewType('CumulativeDetectorImage', sc.DataArray)
"""2D detector image summed over all accumulated data."""

CurrentDetectorImage = NewType('CurrentDetectorImage', sc.DataArray)
"""2D detector image for the current window (since last finalize)."""

CountsTotal = NewType('CountsTotal', sc.DataArray)
"""Total event counts as 0D scalar (from current window)."""

CountsInTOARange = NewType('CountsInTOARange', sc.DataArray)
"""Event counts within configured TOA range as 0D scalar (from current window)."""

# ROI configuration types (context keys - updated less frequently than events)
ROIRectangleRequest = NewType('ROIRectangleRequest', sc.DataArray)
"""ROI rectangle configuration as concatenated DataArray (empty if no ROIs)."""

ROIPolygonRequest = NewType('ROIPolygonRequest', sc.DataArray)
"""ROI polygon configuration as concatenated DataArray (empty if no ROIs)."""

# ROI readback types (outputs - echo request with correct units for frontend)
ROIRectangleReadback = NewType('ROIRectangleReadback', sc.DataArray)
"""ROI rectangle readback with coordinate units from histogram."""

ROIPolygonReadback = NewType('ROIPolygonReadback', sc.DataArray)
"""ROI polygon readback with coordinate units from histogram."""

# ROI output types
ROISpectra = NewType('ROISpectra', sc.DataArray)
"""TOF spectra for ROIs with dims (roi, tof).

Computed from histogram, not accumulated.
"""

CumulativeROISpectra = NewType('CumulativeROISpectra', sc.DataArray)
"""ROI spectra extracted from cumulative histogram."""

CurrentROISpectra = NewType('CurrentROISpectra', sc.DataArray)
"""ROI spectra extracted from current window histogram."""


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
# EventProjector - Projects events to screen coordinates
# ============================================================================


class EventProjector:
    """
    Projects events from detector pixels to screen coordinates.

    Reuses Histogrammer's coordinate infrastructure (including noise replicas)
    but bins events instead of histogramming counts, preserving TOF information.

    Parameters
    ----------
    coords:
        Projected coordinates for each detector pixel, with shape
        (replica, detector_number). Created by make_xy_plane_coords or
        make_cylinder_mantle_coords.
    edges:
        Bin edges for screen coordinates, keyed by dimension name.
    """

    def __init__(self, coords: sc.DataGroup, edges: sc.DataGroup) -> None:
        self._coords = coords
        self._edges = edges
        self._replica_dim = 'replica'
        self._replicas = coords.sizes.get(self._replica_dim, 1)
        self._current = 0

    def project_events(self, events: sc.DataArray) -> sc.DataArray:
        """
        Project events from detector pixels to screen coordinates.

        This method broadcasts per-pixel screen coordinates to individual events,
        then re-bins events by screen position. We use manual numpy indexing rather
        than ``sc.bins_like`` for performance reasons:

        - ``sc.bins_like`` has O(n_pixels) overhead that dominates when pixels >> events
        - With 1M pixels and <1M events (typical for live streaming), numpy is 2-10x
          faster
        - ``sc.bins_like`` only becomes faster at high event density (>10 events/pixel)

        Parameters
        ----------
        events:
            Binned event data with detector pixels as the outer dimension.
            Events should have 'event_time_offset' coordinate.

        Returns
        -------
        :
            Binned data with screen coordinates as outer dimensions,
            preserving events with TOF information.
        """
        # Cycle through replicas for smoother visualization
        replica = self._current % self._replicas
        self._current += 1

        # Get coords for this replica
        replica_coords = {
            key: self._coords[key][self._replica_dim, replica]
            for key in self._edges.keys()
        }

        # Extract flat event table from bins, discarding the pixel binning structure.
        # This is more efficient than using bin(dim='detector_number', ...) which
        # would process the bin structure before flattening.
        constituents = events.data.bins.constituents
        begin = constituents['begin'].values
        end = constituents['end'].values
        event_table = constituents['data']

        # Compute event-to-pixel mapping: for each event, which pixel did it come from?
        # This allows broadcasting per-pixel coordinates to per-event coordinates.
        n_events_per_pixel = end - begin
        event_to_pixel = np.repeat(
            np.arange(len(n_events_per_pixel)), n_events_per_pixel
        )

        # Build coordinates dict for flat events
        event_coords = {}

        # Copy existing event coordinates (event_time_offset, etc.)
        for name in event_table.coords:
            event_coords[name] = event_table.coords[name]

        # Add screen coordinates by indexing pixel coords with the event-to-pixel map
        for key, coord in replica_coords.items():
            event_coords[key] = sc.array(
                dims=[event_table.dim],
                values=coord.values[event_to_pixel],
                unit=coord.unit,
            )

        # Create flat event table with screen coordinates
        flat_events = sc.DataArray(
            data=event_table.data,
            coords=event_coords,
        )

        # Bin by screen coordinates (preserving events)
        return flat_events.bin(self._edges)


# ============================================================================
# Providers - Event Projection
# ============================================================================


def make_event_projector(
    coords: CalibratedPositionWithNoisyReplicas,
    projection_type: ProjectionType,
    resolution: DetectorViewResolution,
) -> EventProjector:
    """
    Create an EventProjector for geometric projection.

    Parameters
    ----------
    coords:
        Calibrated position with noisy replicas from NeXus workflow.
    projection_type:
        Type of geometric projection ('xy_plane' or 'cylinder_mantle_z').
    resolution:
        Resolution (number of bins) for each screen dimension.

    Returns
    -------
    :
        EventProjector configured for the specified projection.
    """
    # Use existing projection functions from essreduce
    if projection_type == 'xy_plane':
        projected_coords = make_xy_plane_coords(coords)
    elif projection_type == 'cylinder_mantle_z':
        projected_coords = make_cylinder_mantle_coords(coords)
    else:
        raise ValueError(f"Unknown projection type: {projection_type}")

    # Create bin edges from coordinates and resolution
    edges = sc.DataGroup(
        {
            dim: projected_coords[dim].hist({dim: res}).coords[dim]
            for dim, res in resolution.items()
        }
    )

    return EventProjector(projected_coords, edges)


def project_events_geometric(
    raw_detector: RawDetector[SampleRun],
    projector: EventProjector,
) -> ScreenBinnedEvents:
    """
    Project events to screen coordinates using geometric projection.

    Parameters
    ----------
    raw_detector:
        Detector data with events binned by detector pixel.
    projector:
        EventProjector configured for geometric projection.

    Returns
    -------
    :
        Events binned by screen coordinates with TOF preserved.
    """
    # TODO Can we modify the provider in ess.reduce to not add variances in the first
    # place (optionally)? This is wasteful here, if variances are needed they can be
    # added after histogramming.
    raw_detector = sc.values(raw_detector)
    return ScreenBinnedEvents(projector.project_events(raw_detector))


def project_events_logical(
    raw_detector: RawDetector[SampleRun],
    transform: LogicalTransform,
    reduction_dim: ReductionDim,
) -> ScreenBinnedEvents:
    """
    Project events using logical view (reshape + optional reduction).

    Parameters
    ----------
    raw_detector:
        Detector data with events binned by detector pixel.
    transform:
        Callable that reshapes detector data (fold/slice). Must not reduce dimensions.
    reduction_dim:
        Dimension(s) to merge events over via bins.concat. None means no reduction.

    Returns
    -------
    :
        Events binned by logical coordinates with TOF preserved.
    """
    raw_detector = sc.values(raw_detector)
    if transform is None:
        return ScreenBinnedEvents(raw_detector)

    # Step 1: Apply transform to reshape bin structure
    # e.g., fold('detector_number', {'y': 100, 'x': 100})
    transformed = transform(raw_detector)

    # Step 2: Merge events along reduction dimensions (if any)
    if reduction_dim is not None:
        dims_to_reduce = (
            [reduction_dim] if isinstance(reduction_dim, str) else list(reduction_dim)
        )
        for dim in dims_to_reduce:
            transformed = transformed.bins.concat(dim)

    return ScreenBinnedEvents(transformed)


# ============================================================================
# Providers - Histogram and Downstream
# ============================================================================


def compute_detector_histogram_3d(
    screen_binned_events: ScreenBinnedEvents,
    tof_bins: TOFBins,
) -> DetectorHistogram3D:
    """
    Histogram TOF from screen-binned events.

    Events have already been projected to screen coordinates by the projection
    providers. This function histograms the event_time_offset (TOF) dimension.

    Parameters
    ----------
    screen_binned_events:
        Events binned by screen coordinates (from geometric or logical projection).
    tof_bins:
        Bin edges for time-of-flight histogramming.

    Returns
    -------
    :
        3D histogram with spatial dims and tof.
    """
    if screen_binned_events.bins is None:
        # Already dense data (shouldn't happen in normal flow)
        return DetectorHistogram3D(screen_binned_events)

    # Histogram by event_time_offset
    histogrammed = screen_binned_events.hist(event_time_offset=tof_bins)

    # Rename to tof for consistency
    if 'event_time_offset' in histogrammed.dims:
        histogrammed = histogrammed.rename_dims(event_time_offset='tof')
        if 'event_time_offset' in histogrammed.coords:
            histogrammed.coords['tof'] = histogrammed.coords.pop('event_time_offset')

    return DetectorHistogram3D(histogrammed)


def cumulative_histogram(data: DetectorHistogram3D) -> CumulativeHistogram:
    """
    Identity transform for cumulative accumulation.

    This allows the histogram to be computed once and accumulated with
    EternalAccumulator.
    """
    return CumulativeHistogram(data)


def window_histogram(data: DetectorHistogram3D) -> WindowHistogram:
    """
    Identity transform for window accumulation.

    This allows the histogram to be computed once and accumulated with
    WindowAccumulator (clears after finalize).
    """
    return WindowHistogram(data)


def cumulative_detector_image(
    data_3d: CumulativeHistogram,
    tof_slice: TOFSlice,
) -> CumulativeDetectorImage:
    """
    Compute cumulative 2D detector image by summing over TOF.

    Parameters
    ----------
    data_3d:
        3D histogram (y, x, tof).
    tof_slice:
        Optional (low, high) TOF range for slicing. If None, sum over all TOF.

    Returns
    -------
    :
        2D detector image (y, x).
    """
    return CumulativeDetectorImage(_sum_over_tof(data_3d, tof_slice))


def current_detector_image(
    data_3d: WindowHistogram,
    tof_slice: TOFSlice,
) -> CurrentDetectorImage:
    """
    Compute current 2D detector image by summing over TOF.

    Parameters
    ----------
    data_3d:
        3D histogram (y, x, tof) for current window.
    tof_slice:
        Optional (low, high) TOF range for slicing. If None, sum over all TOF.

    Returns
    -------
    :
        2D detector image (y, x).
    """
    return CurrentDetectorImage(_sum_over_tof(data_3d, tof_slice))


def _sum_over_tof(
    data_3d: sc.DataArray,
    tof_slice: tuple[sc.Variable, sc.Variable] | None,
) -> sc.DataArray:
    """Sum over TOF dimension, optionally slicing first."""
    tof_dim = 'tof' if 'tof' in data_3d.dims else data_3d.dims[-1]

    if tof_slice is not None:
        low, high = tof_slice
        sliced = data_3d[tof_dim, low:high]
    else:
        sliced = data_3d

    return sliced.sum(tof_dim)


def counts_total(data_3d: WindowHistogram) -> CountsTotal:
    """
    Compute total event counts in current window.

    Parameters
    ----------
    data_3d:
        3D histogram (y, x, tof) for current window.

    Returns
    -------
    :
        Total counts as 0D scalar.
    """
    return CountsTotal(data_3d.sum())


def counts_in_toa_range(
    data_3d: WindowHistogram,
    tof_slice: TOFSlice,
) -> CountsInTOARange:
    """
    Compute event counts within TOA range in current window.

    Parameters
    ----------
    data_3d:
        3D histogram (y, x, tof) for current window.
    tof_slice:
        Optional (low, high) TOA range for counting. If None, counts all.

    Returns
    -------
    :
        Counts in TOA range as 0D scalar.
    """
    tof_dim = 'tof' if 'tof' in data_3d.dims else data_3d.dims[-1]

    if tof_slice is not None:
        low, high = tof_slice
        sliced = data_3d[tof_dim, low:high]
    else:
        sliced = data_3d

    return CountsInTOARange(sliced.sum())


# ============================================================================
# Providers - ROI Spectra Extraction
# ============================================================================


def _extract_roi_spectra(
    histogram_3d: sc.DataArray,
    rectangle_request: sc.DataArray | None,
    polygon_request: sc.DataArray | None,
) -> sc.DataArray:
    """
    Extract TOF spectra from 3D histogram for all configured ROIs.

    Parameters
    ----------
    histogram_3d:
        3D histogram with dims (y, x, tof).
    rectangle_request:
        Concatenated DataArray with rectangle ROI definitions, or None.
    polygon_request:
        Concatenated DataArray with polygon ROI definitions, or None.

    Returns
    -------
    :
        2D DataArray with dims (roi, tof) containing spectra for each ROI.
        Returns empty DataArray with shape (0, n_tof) if no ROIs configured.
    """
    tof_dim = 'tof' if 'tof' in histogram_3d.dims else histogram_3d.dims[-1]
    tof_coord = histogram_3d.coords[tof_dim]
    n_tof = histogram_3d.sizes[tof_dim]

    # Get spatial dims (all dims except tof)
    spatial_dims = [d for d in histogram_3d.dims if d != tof_dim]
    if len(spatial_dims) != 2:
        raise ValueError(
            f"Expected 2 spatial dims, got {len(spatial_dims)}: {spatial_dims}"
        )
    y_dim, x_dim = spatial_dims

    spectra: list[sc.DataArray] = []
    roi_indices: list[int] = []

    # Process rectangle ROIs
    if rectangle_request is not None and len(rectangle_request) > 0:
        rois = models.ROI.from_concatenated_data_array(rectangle_request)
        for idx, roi in rois.items():
            if isinstance(roi, models.RectangleROI):
                spectrum = _extract_rectangle_spectrum(
                    histogram_3d, roi, x_dim=x_dim, y_dim=y_dim, tof_dim=tof_dim
                )
                spectra.append(spectrum)
                roi_indices.append(idx)

    # Process polygon ROIs
    if polygon_request is not None and len(polygon_request) > 0:
        rois = models.ROI.from_concatenated_data_array(polygon_request)
        for idx, roi in rois.items():
            if isinstance(roi, models.PolygonROI):
                spectrum = _extract_polygon_spectrum(
                    histogram_3d, roi, x_dim=x_dim, y_dim=y_dim, tof_dim=tof_dim
                )
                spectra.append(spectrum)
                roi_indices.append(idx)

    # Build output DataArray
    if not spectra:
        # Return empty DataArray with correct structure
        return sc.DataArray(
            data=sc.zeros(dims=['roi', tof_dim], shape=[0, n_tof], unit='counts'),
            coords={
                'roi': sc.array(dims=['roi'], values=[], dtype='int32'),
                tof_dim: tof_coord,
            },
        )

    # Stack spectra along roi dimension
    stacked = sc.concat(spectra, dim='roi')
    stacked.coords['roi'] = sc.array(dims=['roi'], values=roi_indices, dtype='int32')
    return stacked


def _extract_rectangle_spectrum(
    histogram_3d: sc.DataArray,
    roi: models.RectangleROI,
    *,
    x_dim: str,
    y_dim: str,
    tof_dim: str,
) -> sc.DataArray:
    """Extract TOF spectrum for a rectangle ROI by slicing and summing."""
    bounds = roi.get_bounds(x_dim=x_dim, y_dim=y_dim)

    # Slice the histogram by the ROI bounds
    x_low, x_high = bounds[x_dim]
    y_low, y_high = bounds[y_dim]
    sliced = histogram_3d[y_dim, y_low:y_high][x_dim, x_low:x_high]

    # Sum over spatial dimensions to get 1D spectrum
    spectrum = sliced.sum(dim=[y_dim, x_dim])
    return spectrum


def _extract_polygon_spectrum(
    histogram_3d: sc.DataArray,
    roi: models.PolygonROI,
    *,
    x_dim: str,
    y_dim: str,
    tof_dim: str,
) -> sc.DataArray:
    """Extract TOF spectrum for a polygon ROI by masking and summing."""
    # Get polygon vertices
    x_vertices = roi.x
    y_vertices = roi.y

    # Get coordinate arrays for the histogram
    x_coords = histogram_3d.coords.get(x_dim)
    y_coords = histogram_3d.coords.get(y_dim)

    # Create mask using point-in-polygon test
    # We need to check each (y, x) bin center against the polygon
    if x_coords is not None and y_coords is not None:
        # Use bin centers for coordinate-based selection
        if roi.x_unit is not None:
            x_centers = sc.midpoints(x_coords).to(unit=roi.x_unit).values
        else:
            x_centers = np.arange(histogram_3d.sizes[x_dim])

        if roi.y_unit is not None:
            y_centers = sc.midpoints(y_coords).to(unit=roi.y_unit).values
        else:
            y_centers = np.arange(histogram_3d.sizes[y_dim])
    else:
        # Fallback to index-based selection
        x_centers = np.arange(histogram_3d.sizes[x_dim])
        y_centers = np.arange(histogram_3d.sizes[y_dim])

    # Create 2D grid of points
    xx, yy = np.meshgrid(x_centers, y_centers)

    # Point-in-polygon test using matplotlib path
    from matplotlib.path import Path

    polygon_path = Path(list(zip(x_vertices, y_vertices, strict=True)))
    points = np.column_stack([xx.ravel(), yy.ravel()])
    mask_flat = polygon_path.contains_points(points)
    mask_2d = mask_flat.reshape(xx.shape)

    # Convert to scipp Variable and broadcast to 3D
    mask_var = sc.array(dims=[y_dim, x_dim], values=mask_2d)

    # Apply mask and sum over spatial dimensions
    # Use where to zero out values outside the polygon
    # Create zero with matching dtype to avoid DTypeError
    zero = sc.zeros_like(histogram_3d)
    masked = sc.where(mask_var, histogram_3d, zero)
    spectrum = masked.sum(dim=[y_dim, x_dim])
    return spectrum


def cumulative_roi_spectra(
    data_3d: CumulativeHistogram,
    rectangle_request: ROIRectangleRequest,
    polygon_request: ROIPolygonRequest,
) -> CumulativeROISpectra:
    """
    Extract ROI spectra from cumulative histogram.

    Parameters
    ----------
    data_3d:
        Cumulative 3D histogram (y, x, tof).
    rectangle_request:
        Rectangle ROI configuration, or None.
    polygon_request:
        Polygon ROI configuration, or None.

    Returns
    -------
    :
        ROI spectra with dims (roi, tof).
    """
    return CumulativeROISpectra(
        _extract_roi_spectra(data_3d, rectangle_request, polygon_request)
    )


def current_roi_spectra(
    data_3d: WindowHistogram,
    rectangle_request: ROIRectangleRequest,
    polygon_request: ROIPolygonRequest,
) -> CurrentROISpectra:
    """
    Extract ROI spectra from current window histogram.

    Parameters
    ----------
    data_3d:
        Current window 3D histogram (y, x, tof).
    rectangle_request:
        Rectangle ROI configuration, or None.
    polygon_request:
        Polygon ROI configuration, or None.

    Returns
    -------
    :
        ROI spectra with dims (roi, tof).
    """
    return CurrentROISpectra(
        _extract_roi_spectra(data_3d, rectangle_request, polygon_request)
    )


def _get_coord_units_from_histogram(
    histogram: sc.DataArray,
) -> dict[str, sc.Unit | None]:
    """Extract coordinate units from histogram for ROI readback.

    Maps histogram spatial coordinate units to ROI 'x' and 'y' coordinates.
    Assumes histogram dims are (y, x, tof) or similar with last dim being TOF.
    """
    tof_dim = 'tof' if 'tof' in histogram.dims else histogram.dims[-1]
    spatial_dims = [d for d in histogram.dims if d != tof_dim]

    if len(spatial_dims) != 2:
        return {'x': None, 'y': None}

    y_dim, x_dim = spatial_dims

    def get_unit_for_dim(dim: str) -> sc.Unit | None:
        coord = histogram.coords.get(dim)
        if coord is not None:
            return coord.unit
        return None

    return {'x': get_unit_for_dim(x_dim), 'y': get_unit_for_dim(y_dim)}


def roi_rectangle_readback(
    request: ROIRectangleRequest,
    histogram: CumulativeHistogram,
) -> ROIRectangleReadback:
    """
    Produce ROI rectangle readback with correct coordinate units.

    If request has ROIs, returns them unchanged. If empty, creates empty
    DataArray with coordinate units from the histogram so the frontend
    knows what units to use when creating ROIs.

    Parameters
    ----------
    request:
        ROI rectangle request from context.
    histogram:
        Cumulative histogram with coordinate units.

    Returns
    -------
    :
        ROI readback with correct coordinate units.
    """
    if request is not None and len(request) > 0:
        return ROIRectangleReadback(request)

    coord_units = _get_coord_units_from_histogram(histogram)
    return ROIRectangleReadback(
        models.RectangleROI.to_concatenated_data_array({}, coord_units=coord_units)
    )


def roi_polygon_readback(
    request: ROIPolygonRequest,
    histogram: CumulativeHistogram,
) -> ROIPolygonReadback:
    """
    Produce ROI polygon readback with correct coordinate units.

    If request has ROIs, returns them unchanged. If empty, creates empty
    DataArray with coordinate units from the histogram so the frontend
    knows what units to use when creating ROIs.

    Parameters
    ----------
    request:
        ROI polygon request from context.
    histogram:
        Cumulative histogram with coordinate units.

    Returns
    -------
    :
        ROI readback with correct coordinate units.
    """
    if request is not None and len(request) > 0:
        return ROIPolygonReadback(request)

    coord_units = _get_coord_units_from_histogram(histogram)
    return ROIPolygonReadback(
        models.PolygonROI.to_concatenated_data_array({}, coord_units=coord_units)
    )


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

    # Set default ROI configuration (empty DataArrays - None can't be serialized)
    workflow[ROIRectangleRequest] = models.RectangleROI.to_concatenated_data_array({})
    workflow[ROIPolygonRequest] = models.PolygonROI.to_concatenated_data_array({})

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


# ============================================================================
# Factory for integration with instrument specs
# ============================================================================


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
    instrument:
        Instrument configuration.
    tof_bins:
        Default bin edges for TOF histogramming.
    nexus_filename:
        Path to the NeXus geometry file for loading detector geometry.
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
        instrument: Any,
        tof_bins: sc.Variable,
        nexus_filename: pathlib.Path,
        # Geometric projection params
        projection_type: Literal['xy_plane', 'cylinder_mantle_z'] | None = None,
        resolution: dict[str, int] | None = None,
        pixel_noise: Literal['cylindrical'] | sc.Variable | None = None,
        # Logical projection params
        logical_transform: Callable[[sc.DataArray, str], sc.DataArray] | None = None,
        reduction_dim: str | list[str] | None = None,
    ) -> None:
        self._instrument = instrument
        self._tof_bins = tof_bins
        self._nexus_filename = nexus_filename
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
        from .stream_processor_workflow import StreamProcessorWorkflow

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

        # Configure GenericNeXusWorkflow
        workflow[Filename[SampleRun]] = self._nexus_filename
        workflow[NeXusName[NXdetector]] = source_name

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

        # Create empty ROI DataArrays for initial context
        empty_rectangle = models.RectangleROI.to_concatenated_data_array({})
        empty_polygon = models.PolygonROI.to_concatenated_data_array({})

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
            # Set initial context so finalize() returns valid values before any
            # ROI is set
            initial_context={
                ROIRectangleRequest: empty_rectangle,
                ROIPolygonRequest: empty_polygon,
            },
            accumulators=create_accumulators(),
        )
