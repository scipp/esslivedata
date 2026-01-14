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
    EventCoordName,
    HistogramBins,
    HistogramSlice,
    ScreenBinnedEvents,
    WindowHistogram,
)


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


def compute_detector_histogram_3d(
    screen_binned_events: ScreenBinnedEvents,
    bins: HistogramBins,
    event_coord: EventCoordName,
) -> DetectorHistogram3D:
    """
    Histogram events by the specified event coordinate.

    Events have already been projected to screen coordinates by the projection
    providers. This function histograms the specified event coordinate dimension.

    Parameters
    ----------
    screen_binned_events:
        Events binned by screen coordinates (from geometric or logical projection).
    bins:
        Bin edges for histogramming.
    event_coord:
        Name of the event coordinate to histogram.

    Returns
    -------
    :
        3D histogram with spatial dims and the event coordinate dimension.
    """
    if screen_binned_events.bins is None:
        # Already dense data (shouldn't happen in normal flow)
        return DetectorHistogram3D(screen_binned_events)

    histogrammed = screen_binned_events.hist({event_coord: bins})
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
    data_3d: CumulativeHistogram, histogram_slice: HistogramSlice
) -> CumulativeDetectorImage:
    """
    Compute cumulative 2D detector image by summing over spectral dimension.

    Parameters
    ----------
    data_3d:
        3D histogram (y, x, spectral).
    histogram_slice:
        Optional (low, high) range for slicing. If None, sum over full range.

    Returns
    -------
    :
        2D detector image (y, x).
    """
    return CumulativeDetectorImage(_sum_over_spectral_dim(data_3d, histogram_slice))


def current_detector_image(
    data_3d: WindowHistogram, histogram_slice: HistogramSlice
) -> CurrentDetectorImage:
    """
    Compute current 2D detector image by summing over spectral dimension.

    Parameters
    ----------
    data_3d:
        3D histogram (y, x, spectral) for current window.
    histogram_slice:
        Optional (low, high) range for slicing. If None, sum over full range.

    Returns
    -------
    :
        2D detector image (y, x).
    """
    return CurrentDetectorImage(_sum_over_spectral_dim(data_3d, histogram_slice))


def _sum_over_spectral_dim(
    data_3d: sc.DataArray,
    histogram_slice: tuple[sc.Variable, sc.Variable] | None,
) -> sc.DataArray:
    """Sum over spectral dimension (last dim), optionally slicing first."""
    spectral_dim = data_3d.dims[-1]

    if histogram_slice is not None:
        low, high = histogram_slice
        sliced = data_3d[spectral_dim, low:high]
    else:
        sliced = data_3d

    return sliced.sum(spectral_dim)


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


def counts_in_range(
    data_3d: WindowHistogram, histogram_slice: HistogramSlice
) -> CountsInTOARange:
    """
    Compute event counts within specified range in current window.

    Parameters
    ----------
    data_3d:
        3D histogram (y, x, spectral) for current window.
    histogram_slice:
        Optional (low, high) range for counting. If None, counts all.

    Returns
    -------
    :
        Counts in range as 0D scalar.
    """
    spectral_dim = data_3d.dims[-1]

    if histogram_slice is not None:
        low, high = histogram_slice
        sliced = data_3d[spectral_dim, low:high]
    else:
        sliced = data_3d

    return CountsInTOARange(sliced.sum())
