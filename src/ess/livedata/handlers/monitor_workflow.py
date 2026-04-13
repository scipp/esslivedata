# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Sciline workflow for monitor processing using StreamProcessor."""

from __future__ import annotations

from typing import Literal

import sciline
import scipp as sc
from ess.reduce.nexus.types import (
    EmptyDetector,
    EmptyMonitor,
    Filename,
    NeXusData,
    NeXusName,
    RawDetector,
    RawMonitor,
    SampleRun,
)
from ess.reduce.nexus.workflow import GenericNeXusWorkflow
from ess.reduce.unwrap import GenericUnwrapWorkflow, LookupTableFilename
from ess.reduce.unwrap.types import LookupTableRelativeErrorThreshold, WavelengthMonitor
from scippnexus import NXdetector, NXmonitor

from .monitor_workflow_types import (
    CumulativeMonitorHistogram,
    HistogramEdges,
    HistogramRangeHigh,
    HistogramRangeLow,
    MonitorCountsInRange,
    MonitorCountsPerPixel,
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
        # Convert edges to match the event coordinate unit and rename dimension
        event_unit = data.bins.coords[event_coord].unit
        edges_converted = edges.to(unit=event_unit).rename_dims(
            {target_dim: event_coord}
        )
        hist = data.hist({event_coord: edges_converted}, dim=data.dims)
        # Rename dimension and coordinate back to target dimension and unit
        hist = hist.rename_dims({event_coord: target_dim})
        hist.coords[target_dim] = hist.coords.pop(event_coord).to(unit=edges.unit)
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


def histogram_wavelength_monitor(
    data: WavelengthMonitor[SampleRun, NXmonitor], edges: HistogramEdges
) -> MonitorHistogram:
    """Histogram or rebin monitor data by wavelength (wavelength mode)."""
    return MonitorHistogram(_histogram_monitor(data, edges, 'wavelength'))


def histogram_pixellated_monitor(
    data: RawDetector[SampleRun], edges: HistogramEdges
) -> MonitorHistogram:
    """Histogram pixellated monitor data, summing over all dims including pixel.

    Uses the same ``_histogram_monitor`` logic as regular monitors. Because
    ``dim=data.dims`` includes ``detector_number``, the result is a 1D TOA
    histogram identical to what a non-pixellated monitor would produce.
    """
    return MonitorHistogram(_histogram_monitor(data, edges, 'event_time_offset'))


def monitor_counts_per_pixel(data: RawDetector[SampleRun]) -> MonitorCountsPerPixel:
    """Total event counts per pixel for pixellated monitors.

    Returns a 1D DataArray indexed by ``detector_number`` with the total
    event weight in each pixel.
    """
    return MonitorCountsPerPixel(data.bins.sum())


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
    coordinate_mode: Literal['toa', 'wavelength'] = 'toa',
) -> sciline.Pipeline:
    """
    Build the base sciline workflow for monitor processing.

    Uses GenericUnwrapWorkflow as the base, which provides wavelength conversion
    via lookup table. The coordinate mode determines which histogram provider is used:
    - 'toa': Uses RawMonitor (event_time_offset coordinate)
    - 'wavelength': Uses WavelengthMonitor (wavelength coordinate, converted via
      lookup table)

    Parameters
    ----------
    coordinate_mode:
        Coordinate system to use: 'toa' (time-of-arrival) or 'wavelength'.
    """
    # GenericUnwrapWorkflow extends GenericNeXusWorkflow with unwrap providers, so it
    # can be used for all coordinate modes. The coordinate mode determines which
    # histogram provider to use.
    workflow = GenericUnwrapWorkflow(run_types=[SampleRun], monitor_types=[NXmonitor])

    # Select histogram provider based on coordinate mode
    if coordinate_mode == 'toa':
        workflow.insert(histogram_raw_monitor)
    elif coordinate_mode == 'wavelength':
        workflow.insert(histogram_wavelength_monitor)
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


_MONITOR_TARGET_KEYS = {
    'cumulative': CumulativeMonitorHistogram,
    'current': WindowMonitorHistogram,
    'counts_total': MonitorCountsTotal,
    'counts_in_toa_range': MonitorCountsInRange,
}

_MONITOR_WINDOW_OUTPUTS = ['current', 'counts_total', 'counts_in_toa_range']


def create_monitor_workflow(
    source_name: str,
    edges: sc.Variable,
    *,
    range_filter: tuple[sc.Variable, sc.Variable] | None = None,
    coordinate_mode: Literal['toa', 'wavelength'] = 'toa',
    geometry_filename: str | None = None,
    lookup_table_filename: str | None = None,
    context_keys: dict[str, type] | None = None,
    detector_number: sc.Variable | None = None,
):
    """
    Factory for monitor workflow using StreamProcessor.

    Parameters
    ----------
    source_name:
        The monitor name (e.g., 'monitor_1'). Used as dynamic key for stream mapping
        and as the NeXus component name when loading geometry from file.
    edges:
        Bin edges for histogramming (TOA or wavelength edges depending on mode).
    range_filter:
        Optional (low, high) range for ratemeter counts.
    coordinate_mode:
        Coordinate system to use: 'toa' (time-of-arrival) or 'wavelength'.
    geometry_filename:
        Path to NeXus file containing monitor geometry. Required for 'wavelength' mode
        (needed for Ltotal computation). Optional for 'toa' mode.
    lookup_table_filename:
        Path to lookup table file. Required for 'wavelength' mode.
    context_keys:
        Optional mapping from aux source stream names to sciline pipeline keys.
        Used to inject dynamic data (e.g., position streams) into the workflow
        via StreamProcessorWorkflow's context mechanism.
    detector_number:
        Pixel IDs for pixellated monitors. When provided, the workflow uses the
        NXdetector assembly path to group events by pixel and produces an additional
        ``MonitorCountsPerPixel`` output. Only TOA mode is supported.
    """
    from .accumulators import NoCopyAccumulator, make_no_copy_accumulator_pair
    from .stream_processor_workflow import StreamProcessorWorkflow

    if detector_number is not None:
        # Pixellated monitor: NXdetector path with pixel grouping
        if coordinate_mode != 'toa':
            raise ValueError("Pixellated monitors only support 'toa' coordinate mode")
        from .detector_view.data_source import create_empty_detector

        workflow = GenericNeXusWorkflow(
            run_types=[SampleRun], monitor_types=[NXmonitor]
        )
        workflow[EmptyDetector[SampleRun]] = create_empty_detector(detector_number)
        workflow.insert(histogram_pixellated_monitor)
        workflow.insert(monitor_counts_per_pixel)
        workflow.insert(cumulative_view)
        workflow.insert(window_view)
        workflow.insert(counts_total)
        workflow.insert(counts_in_range)
        dynamic_keys = {source_name: NeXusData[NXdetector, SampleRun]}
    else:
        # Standard monitor: NXmonitor path
        if coordinate_mode == 'wavelength':
            if lookup_table_filename is None:
                raise ValueError(
                    "lookup_table_filename is required for 'wavelength' mode"
                )
            if geometry_filename is None:
                raise ValueError(
                    "geometry_filename is required for 'wavelength' mode "
                    "(needed for Ltotal)"
                )

        workflow = build_monitor_workflow(coordinate_mode=coordinate_mode)

        # Configure geometry source
        if geometry_filename is not None:
            workflow[Filename[SampleRun]] = geometry_filename
            workflow[NeXusName[NXmonitor]] = source_name
        else:
            workflow[EmptyMonitor[SampleRun, NXmonitor]] = (
                _create_minimal_empty_monitor()
            )

        # Configure lookup table and error threshold for wavelength mode
        if coordinate_mode == 'wavelength':
            workflow[LookupTableFilename] = lookup_table_filename
            workflow[LookupTableRelativeErrorThreshold] = {source_name: float('inf')}
        dynamic_keys = {source_name: NeXusData[NXmonitor, SampleRun]}

    # Shared parameter setup
    workflow[HistogramEdges] = edges
    if range_filter is not None:
        workflow[HistogramRangeLow] = range_filter[0].to(unit=edges.unit)
        workflow[HistogramRangeHigh] = range_filter[1].to(unit=edges.unit)
    else:
        workflow[HistogramRangeLow] = edges[edges.dim, 0]
        workflow[HistogramRangeHigh] = edges[edges.dim, -1]

    # Build accumulators and targets
    cumulative, window = make_no_copy_accumulator_pair()
    target_keys = dict(_MONITOR_TARGET_KEYS)
    accumulators: dict = {
        CumulativeMonitorHistogram: cumulative,
        WindowMonitorHistogram: window,
    }

    if detector_number is not None:
        target_keys['counts_per_pixel'] = MonitorCountsPerPixel
        accumulators[MonitorCountsPerPixel] = NoCopyAccumulator()

    return StreamProcessorWorkflow(
        workflow,
        dynamic_keys=dynamic_keys,
        context_keys=context_keys,
        target_keys=target_keys,
        accumulators=accumulators,
        window_outputs=_MONITOR_WINDOW_OUTPUTS,
    )
