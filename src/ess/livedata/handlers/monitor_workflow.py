# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Sciline workflow for monitor processing using StreamProcessor."""

from __future__ import annotations

from typing import Literal

import sciline
import scipp as sc
from scippnexus import NXmonitor

from ess.reduce.nexus.types import (
    EmptyMonitor,
    Filename,
    NeXusData,
    NeXusName,
    RawMonitor,
    SampleRun,
)
from ess.reduce.time_of_flight import GenericTofWorkflow
from ess.reduce.time_of_flight.types import TofLookupTableFilename, TofMonitor

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

    if data.is_binned:
        # Event-mode: binned events from preprocessor
        # Convert edges to ns and rename to match event coordinate
        edges_ns = edges.to(unit='ns').rename_dims({target_dim: event_coord})
        hist = data.hist({event_coord: edges_ns}, dim=data.dims)
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
    data: RawMonitor[SampleRun, NXmonitor], edges: HistogramEdges
) -> MonitorHistogram:
    """Histogram or rebin monitor data by time-of-arrival (TOA mode)."""
    return MonitorHistogram(_histogram_monitor(data, edges, 'event_time_offset'))


def histogram_tof_monitor(
    data: TofMonitor[SampleRun, NXmonitor], edges: HistogramEdges
) -> MonitorHistogram:
    """Histogram or rebin monitor data by time-of-flight (TOF mode)."""
    return MonitorHistogram(_histogram_monitor(data, edges, 'tof'))


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

    Uses GenericTofWorkflow as the base, which provides TOF conversion via lookup table.
    The coordinate mode determines which histogram provider is used:
    - 'toa': Uses RawMonitor (event_time_offset coordinate)
    - 'tof': Uses TofMonitor (tof coordinate, converted via lookup table)

    Parameters
    ----------
    coordinate_mode:
        Coordinate system to use: 'toa' (time-of-arrival) or 'tof' (time-of-flight).
    """
    # GenericTofWorkflow extends GenericNeXusWorkflow with TOF providers, so it can
    # be used for all coordinate modes. The coordinate mode determines which
    # histogram provider to use.
    workflow = GenericTofWorkflow(run_types=[SampleRun], monitor_types=[NXmonitor])

    # Select histogram provider based on coordinate mode
    if coordinate_mode == 'toa':
        workflow.insert(histogram_raw_monitor)
    elif coordinate_mode == 'tof':
        workflow.insert(histogram_tof_monitor)
    else:
        raise ValueError(f"Unsupported coordinate mode: {coordinate_mode}")

    # Add downstream providers
    workflow.insert(cumulative_view)
    workflow.insert(window_view)
    workflow.insert(counts_total)
    workflow.insert(counts_in_range)

    return workflow


def _create_minimal_empty_monitor() -> sc.DataArray:
    """
    Create a minimal EmptyMonitor for TOA mode when no geometry file is provided.

    This allows the workflow to run without loading monitor geometry from a file.
    The minimal structure is sufficient for TOA mode since no position-dependent
    calculations (like Ltotal) are needed.
    """
    # Minimal empty monitor structure - just needs to be compatible with
    # assemble_monitor_data which combines it with NeXusData
    return sc.DataArray(sc.scalar(0.0, unit='counts'))


def create_monitor_workflow(
    source_name: str,
    edges: sc.Variable,
    *,
    range_filter: tuple[sc.Variable, sc.Variable] | None = None,
    coordinate_mode: Literal['toa', 'tof'] = 'toa',
    geometry_filename: str | None = None,
    tof_lookup_table_filename: str | None = None,
):
    """
    Factory for monitor workflow using StreamProcessor.

    Parameters
    ----------
    source_name:
        The monitor name (e.g., 'monitor_1'). Used as dynamic key for stream mapping
        and as the NeXus component name when loading geometry from file.
    edges:
        Bin edges for histogramming (TOA or TOF edges depending on mode).
    range_filter:
        Optional (low, high) range for ratemeter counts.
    coordinate_mode:
        Coordinate system to use: 'toa' (time-of-arrival) or 'tof' (time-of-flight).
    geometry_filename:
        Path to NeXus file containing monitor geometry. Required for 'tof' mode
        (needed for Ltotal computation). Optional for 'toa' mode.
    tof_lookup_table_filename:
        Path to TOF lookup table file. Required for 'tof' mode.
    """
    from .accumulators import NoCopyAccumulator, NoCopyWindowAccumulator
    from .stream_processor_workflow import StreamProcessorWorkflow

    # Validate TOF mode requirements
    if coordinate_mode == 'tof':
        if tof_lookup_table_filename is None:
            raise ValueError("tof_lookup_table_filename is required for 'tof' mode")
        if geometry_filename is None:
            raise ValueError(
                "geometry_filename is required for 'tof' mode (needed for Ltotal)"
            )

    workflow = build_monitor_workflow(coordinate_mode=coordinate_mode)
    workflow[HistogramEdges] = edges

    if range_filter is not None:
        workflow[HistogramRangeLow] = range_filter[0].to(unit=edges.unit)
        workflow[HistogramRangeHigh] = range_filter[1].to(unit=edges.unit)
    else:
        workflow[HistogramRangeLow] = edges[edges.dim, 0]
        workflow[HistogramRangeHigh] = edges[edges.dim, -1]

    # Configure geometry source
    if geometry_filename is not None:
        # Load geometry from NeXus file (required for TOF mode)
        workflow[Filename[SampleRun]] = geometry_filename
        workflow[NeXusName[NXmonitor]] = source_name
    else:
        # TOA mode without geometry file: provide minimal EmptyMonitor directly
        workflow[EmptyMonitor[SampleRun, NXmonitor]] = _create_minimal_empty_monitor()

    # Configure lookup table for TOF mode
    if coordinate_mode == 'tof':
        workflow[TofLookupTableFilename] = tof_lookup_table_filename

    # Only accumulate CumulativeMonitorHistogram and WindowMonitorHistogram.
    # MonitorCountsTotal and MonitorCountsInRange are computed from
    # WindowMonitorHistogram during finalize, not accumulated separately.
    return StreamProcessorWorkflow(
        workflow,
        # Inject preprocessor output as NeXusData; GenericNeXusWorkflow
        # providers will assemble monitor data to produce RawMonitor.
        # For TOF mode, GenericTofWorkflow providers convert RawMonitor to TofMonitor.
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
