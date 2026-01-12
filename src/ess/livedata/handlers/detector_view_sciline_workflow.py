# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Sciline-based detector view workflow for live data visualization.

This module implements the detector view workflow using Sciline and StreamProcessor,
providing a logical view of detector data that accumulates counts in a 3D
(screen-x, screen-y, TOF) space and computes detector images as downstream projections.

Phase 1 focuses on logical views (direct pixel mapping) without ROI support.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any, NewType

import sciline
import scipp as sc
from scippnexus import NXdetector

from ess.reduce.nexus.types import (
    Filename,
    NeXusData,
    NeXusName,
    RawDetector,
    SampleRun,
)
from ess.reduce.nexus.workflow import GenericNeXusWorkflow
from ess.reduce.streaming import EternalAccumulator

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
# Providers
# ============================================================================


def compute_detector_histogram_3d(
    raw_detector: RawDetector[SampleRun],
    tof_bins: TOFBins,
    transform: LogicalTransform,
) -> DetectorHistogram3D:
    """
    Compute 3D (spatial, tof) histogram from detector data.

    This is the expensive computation that should only happen once.
    Both cumulative and window accumulators will use this result.

    Parameters
    ----------
    raw_detector:
        Detector data from GenericNeXusWorkflow. This is binned event data
        already grouped by detector pixel.
    tof_bins:
        Bin edges for time-of-flight histogramming.
    transform:
        Optional logical transform to reshape detector data to spatial coordinates.

    Returns
    -------
    :
        3D histogram with spatial dims and tof.
    """
    data = raw_detector if transform is None else transform(raw_detector)

    # RawDetector from GenericNeXusWorkflow is binned by detector pixel with events
    # containing event_time_offset (time of arrival relative to pulse).
    # Histogram into TOF bins.
    if data.bins is not None:
        # Event data - histogram by event_time_offset
        histogrammed = data.hist(event_time_offset=tof_bins)
        # Rename to tof for consistency
        if 'event_time_offset' in histogrammed.dims:
            histogrammed = histogrammed.rename_dims(event_time_offset='tof')
            if 'event_time_offset' in histogrammed.coords:
                histogrammed.coords['tof'] = histogrammed.coords.pop(
                    'event_time_offset'
                )
    else:
        # Already dense data
        histogrammed = data

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
# Workflow Construction
# ============================================================================


def create_base_workflow(
    *,
    tof_bins: sc.Variable,
    tof_slice: tuple[sc.Variable, sc.Variable] | None = None,
    logical_transform: Callable[[sc.DataArray], sc.DataArray] | None = None,
) -> sciline.Pipeline:
    """
    Create the base detector view workflow using GenericNeXusWorkflow.

    Parameters
    ----------
    tof_bins:
        Bin edges for TOF histogramming.
    tof_slice:
        Optional (low, high) TOF range for output image slicing.
    logical_transform:
        Optional transform to apply to convert detector data to logical coordinates.
        If None, an identity transform is used.

    Returns
    -------
    :
        Sciline pipeline with detector view providers.
    """
    # Start with GenericNeXusWorkflow for NeXus loading infrastructure
    workflow = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])

    # Add our detector view providers
    workflow.insert(compute_detector_histogram_3d)
    workflow.insert(cumulative_histogram)
    workflow.insert(window_histogram)
    workflow.insert(cumulative_detector_image)
    workflow.insert(current_detector_image)
    workflow.insert(counts_total)
    workflow.insert(counts_in_toa_range)

    # Set configuration parameters
    workflow[TOFBins] = tof_bins
    workflow[TOFSlice] = tof_slice
    workflow[LogicalTransform] = logical_transform

    return workflow


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

    Parameters
    ----------
    instrument:
        Instrument configuration (used for getting detector_number).
    tof_bins:
        Default bin edges for TOF histogramming.
    nexus_filename:
        Path to the NeXus geometry file for loading detector geometry.
    logical_transform:
        Optional callable to transform detector data to logical coordinates.
        Signature: (da: DataArray, source_name: str) -> DataArray.
        If None, an identity transform is used.
    """

    def __init__(
        self,
        *,
        instrument: Any,  # Instrument type from config
        tof_bins: sc.Variable,
        nexus_filename: pathlib.Path,
        logical_transform: Callable[[sc.DataArray, str], sc.DataArray] | None = None,
    ) -> None:
        self._instrument = instrument
        self._tof_bins = tof_bins
        self._nexus_filename = nexus_filename
        self._logical_transform = logical_transform

    def make_workflow(
        self,
        source_name: str,
        params: Any | None = None,  # DetectorViewParams from specs
    ) -> Any:  # StreamProcessorWorkflow
        """
        Factory method that creates a detector view workflow.

        Parameters
        ----------
        source_name:
            Name of the detector source (e.g., 'panel_0').
        params:
            Workflow parameters (currently unused, for future extension).

        Returns
        -------
        :
            StreamProcessorWorkflow wrapping the Sciline-based detector view.
        """
        from .stream_processor_workflow import StreamProcessorWorkflow

        # Bind source_name to the transform if provided
        if self._logical_transform is not None:

            def bound_transform(da: sc.DataArray) -> sc.DataArray:
                return self._logical_transform(da, source_name)
        else:
            bound_transform = None

        # Get TOF slice from params if available
        tof_slice = None
        if params is not None and hasattr(params, 'toa_range'):
            if params.toa_range.enabled:
                tof_slice = params.toa_range.range_ns

        # Create base workflow
        workflow = create_base_workflow(
            tof_bins=self._tof_bins,
            tof_slice=tof_slice,
            logical_transform=bound_transform,
        )

        # Configure GenericNeXusWorkflow
        workflow[Filename[SampleRun]] = self._nexus_filename
        workflow[NeXusName[NXdetector]] = source_name

        return StreamProcessorWorkflow(
            workflow,
            # Inject preprocessor output as NeXusData; GenericNeXusWorkflow
            # providers will group events by pixel to produce RawDetector.
            dynamic_keys={source_name: NeXusData[NXdetector, SampleRun]},
            target_keys={
                'cumulative': CumulativeDetectorImage,
                'current': CurrentDetectorImage,
                'counts_total': CountsTotal,
                'counts_in_toa_range': CountsInTOARange,
            },
            accumulators=create_accumulators(),
        )
