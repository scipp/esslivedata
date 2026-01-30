# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Sciline workflow for monitor processing using StreamProcessor."""

from __future__ import annotations

import sciline
import scipp as sc
from scippnexus import NXmonitor

from ess.reduce.nexus.types import NeXusData, SampleRun

from .monitor_workflow_types import (
    CumulativeMonitorHistogram,
    MonitorCountsInRange,
    MonitorCountsTotal,
    MonitorHistogram,
    TOAEdges,
    TOARangeHigh,
    TOARangeLow,
    WindowMonitorHistogram,
)


def histogram_monitor_data(
    data: NeXusData[NXmonitor, SampleRun], edges: TOAEdges
) -> MonitorHistogram:
    """
    Histogram or rebin monitor data by TOA.

    Supports two input modes:

    Event-mode:
        Input is sc.DataArray with binned events from ToNXevent_data preprocessor.
        The data has event_time_offset coord containing time-of-arrival values.
        Events are concatenated and histogrammed into the target edges.

    Histogram-mode:
        Input is sc.DataArray histogram from Cumulative preprocessor.
        The histogram is rebinned to the target edges.
    """
    target_dim = edges.dim

    if data.bins is not None:
        # Event-mode: binned events from ToNXevent_data
        # Convert edges to ns and rename to match event_time_offset coordinate
        edges_ns = edges.to(unit='ns').rename_dims({target_dim: 'event_time_offset'})
        # Concat bins and histogram by event_time_offset
        concatenated = data.bins.concat()
        hist = concatenated.hist(event_time_offset=edges_ns)
        # Rename dimension and coordinate to target dimension
        hist = hist.rename_dims(event_time_offset=target_dim)
        hist.coords[target_dim] = hist.coords.pop('event_time_offset')
    else:
        # Histogram-mode: already histogrammed from Cumulative preprocessor
        # Rename dimension if needed
        if data.dim != target_dim:
            data = data.rename({data.dim: target_dim})
        # Convert coordinate unit to match target edges
        coord = data.coords.get(target_dim).to(unit=edges.unit, dtype=edges.dtype)
        hist = data.assign_coords({target_dim: coord}).rebin({target_dim: edges})

    return MonitorHistogram(hist)


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
    hist: WindowMonitorHistogram, low: TOARangeLow, high: TOARangeHigh
) -> MonitorCountsInRange:
    """Counts within TOA range in window."""
    dim = hist.dim
    # Convert range to histogram coordinate unit
    coord_unit = hist.coords[dim].unit
    low_ns = low.to(unit=coord_unit)
    high_ns = high.to(unit=coord_unit)
    return MonitorCountsInRange(hist[dim, low_ns:high_ns].sum())


def build_monitor_workflow() -> sciline.Pipeline:
    """Build the base sciline workflow for monitor processing."""
    workflow = sciline.Pipeline(
        [
            histogram_monitor_data,
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
    toa_range: tuple[sc.Variable, sc.Variable] | None = None,
):
    """
    Factory for monitor workflow using StreamProcessor.

    Parameters
    ----------
    source_name:
        The monitor name (e.g., 'monitor_1'). Used as dynamic key for stream mapping.
    edges:
        TOA bin edges for histogramming.
    toa_range:
        Optional (low, high) range for ratemeter counts.
    """
    from .accumulators import NoCopyAccumulator, NoCopyWindowAccumulator
    from .stream_processor_workflow import StreamProcessorWorkflow

    workflow = build_monitor_workflow()
    workflow[TOAEdges] = edges

    if toa_range is not None:
        workflow[TOARangeLow] = toa_range[0].to(unit=edges.unit)
        workflow[TOARangeHigh] = toa_range[1].to(unit=edges.unit)
    else:
        workflow[TOARangeLow] = edges[edges.dim, 0]
        workflow[TOARangeHigh] = edges[edges.dim, -1]

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
