# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Sciline providers for detector view workflow.

This module provides all Sciline providers for event projection, histogramming,
and image generation in the detector view workflow.
"""

from __future__ import annotations

import scipp as sc

from ess.reduce.nexus.types import RawDetector, SampleRun

from .projectors import Projector
from .types import (
    CountsInTOARange,
    CountsTotal,
    CumulativeDetectorImage,
    CumulativeHistogram,
    CurrentDetectorImage,
    DetectorHistogram3D,
    ScreenBinnedEvents,
    TOFBins,
    TOFSlice,
    WindowHistogram,
)

# ============================================================================
# Providers - Event Projection
# ============================================================================


def project_events(
    raw_detector: RawDetector[SampleRun],
    projector: Projector,
) -> ScreenBinnedEvents:
    """
    Project events to screen coordinates using the configured projector.

    Parameters
    ----------
    raw_detector:
        Detector data with events binned by detector pixel.
    projector:
        Projector instance (geometric or logical).

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
    data_3d: CumulativeHistogram, tof_slice: TOFSlice
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
    data_3d: WindowHistogram, tof_slice: TOFSlice
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
    data_3d: WindowHistogram, tof_slice: TOFSlice
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
