# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Consumption of a streamed wavelength lookup table as workflow context.

The lookup-table workflow publishes two tables as context streams (ADR 0010):
one for the detectors, one for the monitors. A consuming workflow binds the
stream name with a ``ContextBinding`` whose ``workflow_key`` is one of the
context keys here, and inserts the matching provider, which selects the block
covering its own flight path (see
:mod:`~ess.livedata.workflows.lut_blocks`) and reassembles essreduce's public
:class:`~ess.reduce.unwrap.LookupTable` dataclass from it.

Selecting by flight path rather than by component name is what keeps the wire
free of per-component identity: the job already knows its ``Ltotal``, the
table already carries its distances, and the two meet without either side
naming a bank or a monitor.

Reassembling rather than round-tripping through essreduce's file loader is
deliberate. That loader's matching branch is a backwards-compatibility shim for
tables written before the dataclass existed; depending on it would tie us to a
deprecated path. Neither path transports the ``choppers`` field -- the old
format cannot carry it -- so the reassembled dataclass leaves it at its
``None`` default; ADR 0010 assigns chopper provenance to the table's identity
stamp rather than to this field.

The wire value is a single ``DataArray`` because da00 serializes a
``DataArray`` and the dataclass has non-array fields (``pulse_stride`` is an
``int``). The producer attaches the remaining fields as 0-D coords; this module
is the inverse of that.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, NewType

import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import MonitorType, SampleRun
from ess.reduce.unwrap import LookupTable
from ess.reduce.unwrap.types import DetectorLtotal, MonitorLtotal

from ..config.instrument import Instrument
from .lut_blocks import select_block
from .wavelength_lut_workflow_specs import (
    DETECTOR_LUT_OUTPUT,
    LUT_STREAM_NAMES,
    MONITOR_LUT_OUTPUT,
)
from .workflow_factory import SpecHandle

#: Scalar dataclass fields the producer attaches as coords, in field order.
_SCALAR_FIELDS = (
    'pulse_period',
    'pulse_stride',
    'distance_resolution',
    'time_resolution',
)

#: Context key carrying the detectors' streamed table, as it arrives on the wire.
DetectorLutContext = NewType('DetectorLutContext', sc.DataArray)

#: Context key carrying the monitors' streamed table, as it arrives on the wire.
MonitorLutContext = NewType('MonitorLutContext', sc.DataArray)


def _unpack(wire: sc.DataArray, ltotal: sc.Variable) -> dict:
    """Split this job's block of a wire table into ``LookupTable`` fields."""
    if missing := [name for name in _SCALAR_FIELDS if name not in wire.coords]:
        raise ValueError(
            f"Streamed lookup table is missing scalar-field coord(s) {missing}; "
            f"got coords {sorted(wire.coords)}. The producer attaches these, so "
            "this indicates a table from an incompatible producer version."
        )
    block = select_block(wire, ltotal)
    return {
        'array': block.drop_coords(list(_SCALAR_FIELDS)),
        'pulse_period': block.coords['pulse_period'],
        'pulse_stride': int(block.coords['pulse_stride'].value),
        'distance_resolution': block.coords['distance_resolution'],
        'time_resolution': block.coords['time_resolution'],
    }


def detector_lookup_table(
    wire: DetectorLutContext, ltotal: DetectorLtotal[SampleRun]
) -> LookupTable[SampleRun, snx.NXdetector]:
    """Reassemble a detector's lookup table from the detector context stream."""
    return LookupTable[SampleRun, snx.NXdetector](**_unpack(wire, ltotal))


def monitor_lookup_table(
    wire: MonitorLutContext, ltotal: MonitorLtotal[SampleRun, MonitorType]
) -> LookupTable[SampleRun, MonitorType]:
    """Reassemble a monitor's lookup table from the monitor context stream.

    Generic in ``MonitorType`` so one provider serves every role a workflow
    gives a monitor -- the plain ``NXmonitor`` of a monitor view, and the
    incident and transmission monitors of a reduction. Each instantiation picks
    its own block via its own ``MonitorLtotal``, so which monitor fills a role
    is settled by the geometry the job already has, not by the stream it binds.
    """
    return LookupTable[SampleRun, MonitorType](**_unpack(wire, ltotal))


def bind_lookup_tables(
    handle: SpecHandle,
    *,
    instrument: Instrument,
    source_names: Iterable[str],
    is_monitor: bool,
    predicate: Callable[[Any], bool] | None = None,
) -> None:
    """Bind a group's streamed lookup table as gated context.

    One ``ContextBinding`` per spec and group, replacing what used to be one per
    component: the table is shared, so the only per-job question left is which
    jobs gate on it. The predicate keeps time-of-arrival jobs ungated: the table
    is published by an operator-started job that re-emits only on chopper
    change, and making TOA -- the mode you fall back to when everything else is
    broken -- depend on it would be the wrong trade (ADR 0010).

    Two ways a job ends up unbound rather than gated forever: the group has no
    placeable component, so the stream is never published; or the job's own
    source is unplaceable, so no block of the table would cover it.

    Parameters
    ----------
    handle:
        Spec to bind the table to.
    instrument:
        Instrument whose ``lut_components`` decide what is placeable.
    source_names:
        The spec's sources this binding applies to. Not necessarily the
        components reading the table: a reduction's monitors arrive as an aux
        selection, so its monitor-table binding applies to its detector jobs.
    is_monitor:
        Select the monitor table over the detector one.
    predicate:
        Narrows the binding to the jobs that actually read the table. Views
        pass :func:`reads_wavelength`; a reduction passes nothing, since it has
        no coordinate mode to choose and always needs its tables.
    """
    group = instrument.monitors if is_monitor else instrument.detector_names
    if not set(group) & instrument.lut_components:
        return
    if not (gated := sorted(set(source_names) & instrument.lut_components)):
        return
    output = MONITOR_LUT_OUTPUT if is_monitor else DETECTOR_LUT_OUTPUT
    handle.add_context_binding(
        stream_name=LUT_STREAM_NAMES[output],
        workflow_key=MonitorLutContext if is_monitor else DetectorLutContext,
        dependent_sources=gated,
        predicate=predicate,
    )


def reads_wavelength(params: Any) -> bool:
    """Whether a job's params select the wavelength coordinate mode.

    Deliberately not tolerant of a params model without a coordinate mode: such
    a spec has no mode to choose, so routing it through here is a wiring
    mistake. Reading a missing field as "not wavelength" would leave the job
    ungated and hand its providers a table that never arrives.
    """
    return params.coordinate_mode.mode == 'wavelength'
