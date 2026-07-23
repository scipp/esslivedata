# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Sciline workflow for monitor processing using StreamProcessor."""

from __future__ import annotations

from typing import Literal, NewType

import sciline
import scipp as sc
from ess.reduce.nexus.types import (
    EmptyMonitor,
    Filename,
    NeXusData,
    NeXusName,
    NeXusTransformation,
    RawMonitor,
    SampleRun,
)
from ess.reduce.unwrap import GenericUnwrapWorkflow, LookupTableFilename
from ess.reduce.unwrap.types import LookupTableRelativeErrorThreshold, WavelengthMonitor
from scippnexus import NXmonitor

from ..preprocessors.accumulation_mode import AccumulationMode, Cumulative, Current
from .geometry_signal import geometry_signal
from .monitor_workflow_types import (
    AccumulatedMonitorHistogram,
    HistogramEdges,
    HistogramRangeHigh,
    HistogramRangeLow,
    MonitorCountsInRange,
    MonitorCountsTotal,
    MonitorHistogram,
)

MONITOR_TRANSFORM = 'monitor_transform'
"""Coord name carrying :data:`MonitorGeometry` on the accumulated histogram."""

MonitorGeometry = NewType('MonitorGeometry', sc.Variable | None)
"""Scalar geometry signal stamped onto the histogram as the ``MONITOR_TRANSFORM``
coord, used by the cumulative accumulator to reset on a monitor move.

Holds the monitor's resolved ``NeXusTransformation``. Mirrors the detector view's
``DetectorGeometry``: a 0-dim transform is the only signal that survives the
histogram's collapse over the pixel dimension, so it works for both single-point
and pixellated monitors, where a per-pixel ``position`` coord would be dropped.
``None`` when no geometry is loaded (TOA mode without a NeXus file): a move is then
undetectable and no coord is stamped."""


def monitor_geometry(
    transform: NeXusTransformation[NXmonitor, SampleRun],
) -> MonitorGeometry:
    """Expose the monitor's resolved placement as the geometry-change signal.

    Inserted only when a geometry file is loaded; otherwise :data:`MonitorGeometry`
    is set to ``None``. See
    :func:`~ess.livedata.workflows.geometry_signal.geometry_signal`.
    """
    return MonitorGeometry(geometry_signal(transform))


def _histogram_monitor(
    data: sc.DataArray,
    edges: sc.Variable,
    event_coord: str,
    geometry: sc.Variable | None,
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
    geometry:
        Scalar geometry signal stamped as the ``MONITOR_TRANSFORM`` coord so the
        cumulative accumulator can reset on a monitor move. ``None`` stamps nothing.
        Survives the collapse over the pixel dimension, unlike a per-pixel
        ``position`` coord.
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

    if geometry is not None:
        hist.coords[MONITOR_TRANSFORM] = geometry
    return hist


def histogram_raw_monitor(
    data: RawMonitor[SampleRun, NXmonitor],
    edges: HistogramEdges,
    geometry: MonitorGeometry,
) -> MonitorHistogram:
    """Histogram or rebin monitor data by time-of-arrival (TOA mode)."""
    return MonitorHistogram(
        _histogram_monitor(data, edges, 'event_time_offset', geometry)
    )


def histogram_wavelength_monitor(
    data: WavelengthMonitor[SampleRun, NXmonitor],
    edges: HistogramEdges,
    geometry: MonitorGeometry,
) -> MonitorHistogram:
    """Histogram or rebin monitor data by wavelength (wavelength mode)."""
    return MonitorHistogram(_histogram_monitor(data, edges, 'wavelength', geometry))


def accumulated_monitor_histogram(
    hist: MonitorHistogram,
) -> AccumulatedMonitorHistogram[AccumulationMode]:
    """Route the histogram to the accumulation-mode-specific accumulator.

    The histogram is computed once and accumulated differently based on the
    accumulation mode type parameter (cumulative vs window). Sciline
    instantiates this provider for each concrete mode.
    """
    return AccumulatedMonitorHistogram[AccumulationMode](hist)


def counts_total(
    hist: AccumulatedMonitorHistogram[AccumulationMode],
) -> MonitorCountsTotal[AccumulationMode]:
    """Total counts, for the given accumulation mode."""
    return MonitorCountsTotal[AccumulationMode](hist.sum())


def counts_in_range(
    hist: AccumulatedMonitorHistogram[AccumulationMode],
    low: HistogramRangeLow,
    high: HistogramRangeHigh,
) -> MonitorCountsInRange[AccumulationMode]:
    """Counts within range filter, for the given accumulation mode."""
    dim = hist.dim
    # Convert range to histogram coordinate unit
    coord_unit = hist.coords[dim].unit
    low_converted = low.to(unit=coord_unit)
    high_converted = high.to(unit=coord_unit)
    return MonitorCountsInRange[AccumulationMode](
        hist[dim, low_converted:high_converted].sum()
    )


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
    workflow.insert(accumulated_monitor_histogram)
    workflow.insert(counts_total)
    workflow.insert(counts_in_range)

    # No geometry by default; create_monitor_workflow inserts monitor_geometry
    # when a NeXus file is loaded, overriding this.
    workflow[MonitorGeometry] = MonitorGeometry(None)

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
    coordinate_mode: Literal['toa', 'wavelength'] = 'toa',
    geometry_filename: str | None = None,
    lookup_table_filename: str | None = None,
    reset_coord: str | None = MONITOR_TRANSFORM,
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
    reset_coord:
        The cumulative histogram resets when this scalar coord changes, so a
        moving monitor restarts accumulation rather than summing across
        configurations. Defaults to :data:`MONITOR_TRANSFORM`, the monitor's
        resolved placement, stamped whenever a geometry file is loaded (both TOA
        and wavelength modes). This mirrors the detector view's reset-on-move and,
        being a 0-dim transform, survives the histogram's collapse over the pixel
        dimension -- a per-pixel ``position`` coord would be dropped, so a
        pixellated monitor's move would go undetected. When no geometry is loaded
        (TOA mode without a NeXus file) the coord is absent and the reset is a
        no-op, matching the fact that a move is then undetectable. Pass ``None`` to
        disable (e.g. reduction-style accumulation across positions).
    """
    from ..preprocessors.accumulators import make_no_copy_accumulator_pair
    from .stream_processor_workflow import StreamProcessorWorkflow

    # Validate wavelength mode requirements
    if coordinate_mode == 'wavelength':
        if lookup_table_filename is None:
            raise ValueError("lookup_table_filename is required for 'wavelength' mode")
        if geometry_filename is None:
            raise ValueError(
                "geometry_filename is required for 'wavelength' mode "
                "(needed for Ltotal)"
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
        # Geometry available: expose the placement so a move resets accumulation.
        workflow.insert(monitor_geometry)
    else:
        # TOA mode without geometry file: provide minimal EmptyMonitor directly.
        # MonitorGeometry stays None (set by build_monitor_workflow): a move is
        # undetectable without geometry, so there is nothing to reset on.
        workflow[EmptyMonitor[SampleRun, NXmonitor]] = _create_minimal_empty_monitor()

    # Configure lookup table and error threshold for wavelength mode
    if coordinate_mode == 'wavelength':
        workflow[LookupTableFilename] = lookup_table_filename
        workflow[LookupTableRelativeErrorThreshold] = {source_name: float('inf')}

    # Only the histogram is accumulated; the scalar totals are derived from the
    # accumulated histogram per mode during finalize, not accumulated separately.
    cumulative, window = make_no_copy_accumulator_pair(reset_coord=reset_coord)
    return StreamProcessorWorkflow(
        workflow,
        # Inject preprocessor output as NeXusData; GenericNeXusWorkflow
        # providers will assemble monitor data to produce RawMonitor.
        # For wavelength mode, GenericUnwrapWorkflow providers convert RawMonitor to
        # WavelengthMonitor.
        dynamic_keys={source_name: NeXusData[NXmonitor, SampleRun]},
        target_keys={
            'cumulative': AccumulatedMonitorHistogram[Cumulative],
            'current': AccumulatedMonitorHistogram[Current],
            'counts_total': MonitorCountsTotal[Current],
            'counts_in_toa_range': MonitorCountsInRange[Current],
            'counts_total_cumulative': MonitorCountsTotal[Cumulative],
            'counts_in_toa_range_cumulative': MonitorCountsInRange[Cumulative],
        },
        accumulators={
            AccumulatedMonitorHistogram[Cumulative]: cumulative,
            AccumulatedMonitorHistogram[Current]: window,
        },
        window_outputs=['current', 'counts_total', 'counts_in_toa_range'],
    )
