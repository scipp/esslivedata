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
from ess.reduce.time_of_flight.types import TofDetector

from .projectors import GeometricProjector, LogicalProjector, Projector
from .types import (
    AccumulatedHistogram,
    AccumulationMode,
    CountsInRange,
    CountsTotal,
    DetectorHistogram,
    DetectorImage,
    EventCoordName,
    HistogramBins,
    HistogramSlice,
    PixelWeights,
    ScreenBinnedEvents,
    ScreenMetadata,
    UsePixelWeighting,
)


def _project_detector(
    detector: sc.DataArray, projector: Projector
) -> ScreenBinnedEvents:
    # TODO Can we modify the provider in ess.reduce to not add variances in the first
    # place (optionally)? This is wasteful here, if variances are needed they can be
    # added after histogramming.
    detector = sc.values(detector)
    return ScreenBinnedEvents(projector.project_events(detector))


def project_raw_detector(
    raw_detector: RawDetector[SampleRun], projector: Projector
) -> ScreenBinnedEvents:
    """
    Project TOA events to screen coordinates.

    Parameters
    ----------
    raw_detector:
        Detector data with events binned by detector pixel (event_time_offset coord).
    projector:
        Projector instance (geometric or logical).

    Returns
    -------
    :
        Events binned by screen coordinates with event_time_offset preserved.
    """
    return _project_detector(raw_detector, projector)


def project_tof_detector(
    tof_detector: TofDetector[SampleRun], projector: Projector
) -> ScreenBinnedEvents:
    """
    Project TOF events to screen coordinates.

    Parameters
    ----------
    tof_detector:
        Detector data with events binned by detector pixel (tof coord).
    projector:
        Projector instance (geometric or logical).

    Returns
    -------
    :
        Events binned by screen coordinates with tof preserved.
    """
    return _project_detector(tof_detector, projector)


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


def compute_detector_histogram(
    screen_binned_events: ScreenBinnedEvents,
    bins: HistogramBins,
    event_coord: EventCoordName,
) -> DetectorHistogram:
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
        Histogram with spatial dims and the event coordinate dimension.
    """
    if screen_binned_events.bins is None:
        # Already dense data (shouldn't happen in normal flow)
        return DetectorHistogram(screen_binned_events)

    # Convert bins to ns and rename dimension to match event_coord for histogramming.
    # Then restore user's original dimension name and unit for output.
    output_dim = bins.dim
    bins_ns = bins.to(unit='ns').rename_dims({output_dim: event_coord})
    histogrammed = screen_binned_events.hist({event_coord: bins_ns})
    histogrammed = histogrammed.rename_dims({event_coord: output_dim})
    histogrammed.coords[output_dim] = bins
    return DetectorHistogram(histogrammed)


def accumulated_histogram(
    data: DetectorHistogram,
) -> AccumulatedHistogram[AccumulationMode]:
    """
    Route histogram to accumulation-mode-specific accumulator.

    This generic provider allows the histogram to be computed once and
    accumulated differently based on the accumulation mode type parameter:

    - AccumulatedHistogram[Cumulative]: Uses EternalAccumulator
      (accumulates forever)
    - AccumulatedHistogram[Current]: Uses NoCopyWindowAccumulator
      (clears after finalize)

    Sciline instantiates this provider for each concrete type parameter.
    """
    return AccumulatedHistogram[AccumulationMode](data)


def detector_image(
    histogram: AccumulatedHistogram[AccumulationMode],
    histogram_slice: HistogramSlice,
    weights: PixelWeights,
    use_weighting: UsePixelWeighting,
) -> DetectorImage[AccumulationMode]:
    """
    Compute 2D detector image by summing over spectral dimension.

    This generic provider works for both accumulation modes:

    - DetectorImage[Cumulative]: Summed over all accumulated data
    - DetectorImage[Current]: Current window only (since last finalize)

    Parameters
    ----------
    histogram:
        Histogram with screen dims and spectral dim.
    histogram_slice:
        Optional (low, high) range for slicing. If None, sum over full range.
    weights:
        Pixel weights for normalization.
    use_weighting:
        Whether to apply pixel weighting.

    Returns
    -------
    :
        2D detector image.
    """
    spectral_dim = histogram.dims[-1]
    if histogram_slice is not None:
        low, high = histogram_slice
        sliced = histogram[spectral_dim, low:high]
    else:
        sliced = histogram
    image = sliced.sum(spectral_dim)

    if use_weighting:
        image = image / weights
    return DetectorImage[AccumulationMode](image)


def counts_total(
    histogram: AccumulatedHistogram[AccumulationMode],
) -> CountsTotal[AccumulationMode]:
    """
    Compute total event counts.

    This generic provider works for both accumulation modes.

    Parameters
    ----------
    histogram:
        Histogram for the given accumulation mode.

    Returns
    -------
    :
        Total counts as 0D scalar.
    """
    return CountsTotal[AccumulationMode](histogram.sum())


def counts_in_range(
    histogram: AccumulatedHistogram[AccumulationMode],
    histogram_slice: HistogramSlice,
) -> CountsInRange[AccumulationMode]:
    """
    Compute event counts within specified range.

    This generic provider works for both accumulation modes.

    Parameters
    ----------
    histogram:
        Histogram for the given accumulation mode.
    histogram_slice:
        Optional (low, high) range for counting. If None, counts all.

    Returns
    -------
    :
        Counts in range as 0D scalar.
    """
    spectral_dim = histogram.dims[-1]

    if histogram_slice is not None:
        low, high = histogram_slice
        sliced = histogram[spectral_dim, low:high]
    else:
        sliced = histogram

    return CountsInRange[AccumulationMode](sliced.sum())
