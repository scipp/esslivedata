# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Sciline providers for detector view workflow.

This module provides all Sciline providers for event projection, histogramming,
and image generation in the detector view workflow.
"""

from __future__ import annotations

import scipp as sc

from ess.reduce.nexus.types import EmptyDetector, RawDetector, SampleRun

from .projectors import EventProjector
from .types import (
    CountsInTOARange,
    CountsTotal,
    CumulativeDetectorImage,
    CumulativeHistogram,
    CurrentDetectorImage,
    DetectorHistogram3D,
    LogicalTransform,
    ReductionDim,
    ScreenBinnedEvents,
    ScreenCoordInfo,
    TOFSlice,
    WindowHistogram,
)

# ============================================================================
# Providers - Event Projection
# ============================================================================


def screen_coord_info_geometric(
    projector: EventProjector,
) -> ScreenCoordInfo:
    """
    Extract screen coordinate information from EventProjector.

    This provides the coordinate structure needed for ROI precomputation,
    computed once from static projection configuration (not event data).

    Parameters
    ----------
    projector:
        EventProjector with screen coordinate edges.

    Returns
    -------
    :
        ScreenCoordInfo with dimension names and coordinate edges.
    """
    edges = projector.edges
    # Get dimension names from the edges (typically 'screen_x', 'screen_y')
    dims = list(edges.keys())
    if len(dims) != 2:
        raise ValueError(f"Expected 2 spatial dims from projector, got {len(dims)}")

    # Order: first dim is y, second is x (matching histogram convention)
    y_dim, x_dim = dims[0], dims[1]

    return ScreenCoordInfo(
        y_dim=y_dim,
        x_dim=x_dim,
        y_edges=edges[y_dim],
        x_edges=edges[x_dim],
    )


def screen_coord_info_logical(
    empty_detector: EmptyDetector[SampleRun],
    transform: LogicalTransform,
) -> ScreenCoordInfo:
    """
    Compute screen coordinate info for logical projection from empty detector structure.

    This applies the logical transform to the detector structure (without events)
    to determine output dimensions and coordinates. Since EmptyDetector is static
    (derived from NeXus geometry, not event data), this allows ROI precomputation
    to be independent of the event stream.

    Parameters
    ----------
    empty_detector:
        Detector structure without neutron data.
    transform:
        Callable that reshapes detector data (fold/slice). If None, identity.

    Returns
    -------
    :
        ScreenCoordInfo with dimension names and coordinate edges (if available).
    """
    if transform is None:
        # No transform - use detector as-is
        transformed = empty_detector
    else:
        # Apply transform to get output structure
        transformed = transform(empty_detector)

    # Extract spatial dimensions
    dims = list(transformed.dims)
    if len(dims) < 2:
        raise ValueError(f"Expected at least 2 dims from transform, got {dims}")

    # Assume first two dims are spatial (y, x)
    y_dim, x_dim = dims[0], dims[1]

    # Get coordinates if available from transform
    y_edges = transformed.coords.get(y_dim)
    x_edges = transformed.coords.get(x_dim)

    return ScreenCoordInfo(
        y_dim=y_dim,
        x_dim=x_dim,
        y_edges=y_edges,
        x_edges=x_edges,
    )


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
    tof_bins: sc.Variable,
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
