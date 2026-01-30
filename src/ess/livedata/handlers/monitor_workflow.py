# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Sciline workflow for monitor processing using StreamProcessor."""

from __future__ import annotations

from typing import Literal

import sciline
import scipp as sc
from scippnexus import NXmonitor

from ess.reduce.nexus.types import NeXusData, SampleRun

from .monitor_workflow_types import (
    CumulativeMonitorHistogram,
    HistogramEdges,
    HistogramRangeHigh,
    HistogramRangeLow,
    MonitorCountsInRange,
    MonitorCountsTotal,
    MonitorHistogram,
    WindowMonitorHistogram,
)

# Backwards compatibility
TOAEdges = HistogramEdges
TOARangeLow = HistogramRangeLow
TOARangeHigh = HistogramRangeHigh


def _histogram_monitor(
    data: sc.DataArray, edges: sc.Variable, event_coord: str
) -> sc.DataArray:
    """
    Common histogram logic for monitor data.

    Parameters
    ----------
    data:
        Monitor data (binned events or histogram).
    edges:
        Bin edges for histogramming.
    event_coord:
        Name of the event coordinate to histogram by (e.g., 'event_time_offset', 'tof').
    """
    target_dim = edges.dim

    if data.bins is not None:
        # Event-mode: binned events from preprocessor
        # Convert edges to ns and rename to match event coordinate
        edges_ns = edges.to(unit='ns').rename_dims({target_dim: event_coord})
        # Concat bins and histogram by event coordinate
        concatenated = data.bins.concat()
        hist = concatenated.hist({event_coord: edges_ns})
        # Rename dimension and coordinate to target dimension
        hist = hist.rename_dims({event_coord: target_dim})
        hist.coords[target_dim] = hist.coords.pop(event_coord)
    else:
        # Histogram-mode: already histogrammed from Cumulative preprocessor
        # Rename dimension if needed
        if data.dim != target_dim:
            data = data.rename({data.dim: target_dim})
        # Convert coordinate unit to match target edges
        coord = data.coords.get(target_dim).to(unit=edges.unit, dtype=edges.dtype)
        hist = data.assign_coords({target_dim: coord}).rebin({target_dim: edges})

    return hist


def histogram_raw_monitor(
    data: NeXusData[NXmonitor, SampleRun], edges: HistogramEdges
) -> MonitorHistogram:
    """
    Histogram or rebin monitor data by time-of-arrival (TOA mode).

    Supports two input modes:

    Event-mode:
        Input is sc.DataArray with binned events from ToNXevent_data preprocessor.
        The data has event_time_offset coord containing time-of-arrival values.
        Events are concatenated and histogrammed into the target edges.

    Histogram-mode:
        Input is sc.DataArray histogram from Cumulative preprocessor.
        The histogram is rebinned to the target edges.
    """
    return MonitorHistogram(_histogram_monitor(data, edges, 'event_time_offset'))


def histogram_tof_monitor(
    data: NeXusData[NXmonitor, SampleRun], edges: HistogramEdges
) -> MonitorHistogram:
    """
    Histogram or rebin monitor data by time-of-flight (TOF mode).

    In TOF mode, the data is expected to have a 'tof' coordinate from
    a lookup table conversion applied by an upstream preprocessor.
    """
    return MonitorHistogram(_histogram_monitor(data, edges, 'tof'))


# Backwards compatibility alias
histogram_monitor_data = histogram_raw_monitor


def cumulative_view(hist: MonitorHistogram) -> CumulativeMonitorHistogram:
    """Identity transform for routing to cumulative accumulator."""
    return CumulativeMonitorHistogram(hist)


def window_view(hist: MonitorHistogram) -> WindowMonitorHistogram:
    """Identity transform for routing to window accumulator."""
    return WindowMonitorHistogram(hist)


def counts_total(hist: WindowMonitorHistogram) -> MonitorCountsTotal:
    """Total counts in window."""
    return MonitorCountsTotal(hist.sum())


def counts_in_range(
    hist: WindowMonitorHistogram, low: HistogramRangeLow, high: HistogramRangeHigh
) -> MonitorCountsInRange:
    """Counts within range filter in window."""
    dim = hist.dim
    # Convert range to histogram coordinate unit
    coord_unit = hist.coords[dim].unit
    low_converted = low.to(unit=coord_unit)
    high_converted = high.to(unit=coord_unit)
    return MonitorCountsInRange(hist[dim, low_converted:high_converted].sum())


def build_monitor_workflow(
    coordinate_mode: Literal['toa', 'tof'] = 'toa',
) -> sciline.Pipeline:
    """
    Build the base sciline workflow for monitor processing.

    Parameters
    ----------
    coordinate_mode:
        Coordinate system to use: 'toa' (time-of-arrival) or 'tof' (time-of-flight).
    """
    # Select histogram provider based on coordinate mode
    if coordinate_mode == 'toa':
        histogram_provider = histogram_raw_monitor
    elif coordinate_mode == 'tof':
        histogram_provider = histogram_tof_monitor
    else:
        raise ValueError(f"Unsupported coordinate mode: {coordinate_mode}")

    workflow = sciline.Pipeline(
        [
            histogram_provider,
            cumulative_view,
            window_view,
            counts_total,
            counts_in_range,
        ]
    )
    return workflow


def create_monitor_workflow(
    source_name: str,
    edges: sc.Variable,
    *,
    range_filter: tuple[sc.Variable, sc.Variable] | None = None,
    coordinate_mode: Literal['toa', 'tof'] = 'toa',
):
    """
    Factory for monitor workflow using StreamProcessor.

    Parameters
    ----------
    source_name:
        The monitor name (e.g., 'monitor_1'). Used as dynamic key for stream mapping.
    edges:
        Bin edges for histogramming (TOA or TOF edges depending on mode).
    range_filter:
        Optional (low, high) range for ratemeter counts.
    coordinate_mode:
        Coordinate system to use: 'toa' (time-of-arrival) or 'tof' (time-of-flight).
    """
    from .accumulators import NoCopyAccumulator, NoCopyWindowAccumulator
    from .stream_processor_workflow import StreamProcessorWorkflow

    workflow = build_monitor_workflow(coordinate_mode=coordinate_mode)
    workflow[HistogramEdges] = edges

    if range_filter is not None:
        workflow[HistogramRangeLow] = range_filter[0].to(unit=edges.unit)
        workflow[HistogramRangeHigh] = range_filter[1].to(unit=edges.unit)
    else:
        workflow[HistogramRangeLow] = edges[edges.dim, 0]
        workflow[HistogramRangeHigh] = edges[edges.dim, -1]

    # Only accumulate CumulativeMonitorHistogram and WindowMonitorHistogram.
    # MonitorCountsTotal and MonitorCountsInRange are computed from
    # WindowMonitorHistogram during finalize, not accumulated separately.
    return StreamProcessorWorkflow(
        workflow,
        dynamic_keys={source_name: NeXusData[NXmonitor, SampleRun]},
        target_keys={
            'cumulative': CumulativeMonitorHistogram,
            'current': WindowMonitorHistogram,
            'counts_total': MonitorCountsTotal,
            'counts_in_toa_range': MonitorCountsInRange,
        },
        accumulators={
            CumulativeMonitorHistogram: NoCopyAccumulator(),
            WindowMonitorHistogram: NoCopyWindowAccumulator(),
        },
        window_outputs=['current', 'counts_total', 'counts_in_toa_range'],
    )
