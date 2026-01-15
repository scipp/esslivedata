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

from .projectors import GeometricProjector, LogicalProjector, Projector
from .types import (
    AccumulationMode,
    CountsInTOARange,
    CountsTotal,
    DetectorHistogram3D,
    DetectorImage,
    EventCoordName,
    Histogram3D,
    HistogramBins,
    HistogramSlice,
    PixelWeights,
    ScreenBinnedEvents,
    UsePixelWeighting,
    Window,
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


def compute_pixel_weights(
    projector: Projector,
    empty_detector: EmptyDetector[SampleRun],
) -> PixelWeights:
    """
    Compute pixel weights for normalizing screen pixels.

    Returns the number of detector pixels contributing to each screen pixel.
    Used to normalize output images when pixel weighting is enabled.

    Parameters
    ----------
    projector:
        Projector instance (geometric or logical).
    empty_detector:
        Empty detector structure (used by logical projector for shape info).

    Returns
    -------
    :
        2D array of weights matching screen dimensions.
    """
    if isinstance(projector, GeometricProjector):
        return PixelWeights(projector.compute_weights())
    elif isinstance(projector, LogicalProjector):
        return PixelWeights(projector.compute_weights(empty_detector))
    else:
        raise TypeError(f"Unknown projector type: {type(projector)}")


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

    # Convert bins to ns and rename dimension to match event_coord for histogramming.
    # Then restore user's original dimension name and unit for output.
    output_dim = bins.dim
    bins_ns = bins.to(unit='ns').rename_dims({output_dim: event_coord})
    histogrammed = screen_binned_events.hist({event_coord: bins_ns})
    histogrammed = histogrammed.rename_dims({event_coord: output_dim})
    histogrammed.coords[output_dim] = bins
    return DetectorHistogram3D(histogrammed)


def histogram_3d(data: DetectorHistogram3D) -> Histogram3D[AccumulationMode]:
    """
    Route histogram to accumulation-mode-specific accumulator.

    This generic provider allows the histogram to be computed once and
    accumulated differently based on the accumulation mode type parameter:

    - Histogram3D[Cumulative]: Uses EternalAccumulator (accumulates forever)
    - Histogram3D[Window]: Uses NoCopyWindowAccumulator (clears after finalize)

    Sciline instantiates this provider for each concrete type parameter.
    """
    return Histogram3D[AccumulationMode](data)


def detector_image(
    data_3d: Histogram3D[AccumulationMode],
    histogram_slice: HistogramSlice,
    weights: PixelWeights,
    use_weighting: UsePixelWeighting,
) -> DetectorImage[AccumulationMode]:
    """
    Compute 2D detector image by summing over spectral dimension.

    This generic provider works for both accumulation modes:

    - DetectorImage[Cumulative]: Summed over all accumulated data
    - DetectorImage[Window]: Current window only (since last finalize)

    Parameters
    ----------
    data_3d:
        3D histogram (y, x, spectral).
    histogram_slice:
        Optional (low, high) range for slicing. If None, sum over full range.
    weights:
        Pixel weights for normalization.
    use_weighting:
        Whether to apply pixel weighting.

    Returns
    -------
    :
        2D detector image (y, x).
    """
    image = _sum_over_spectral_dim(data_3d, histogram_slice)
    if use_weighting:
        image = image / weights
    return DetectorImage[AccumulationMode](image)


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


def counts_total(data_3d: Histogram3D[Window]) -> CountsTotal:
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
    data_3d: Histogram3D[Window], histogram_slice: HistogramSlice
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
