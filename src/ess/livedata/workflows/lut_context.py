# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Consumption of a streamed wavelength lookup table as workflow context.

The lookup-table workflow publishes two tables as context streams (ADR 0010):
one for the detectors, one for the monitors. A consuming workflow inserts the
matching provider and says nothing else: the provider takes one of the context
keys here as its argument, the instrument declares which stream carries that
key (:func:`declare_lut_context_streams`), and the workflow build derives the
job's gate from whether its graph reaches the provider at all. Each provider
selects the block covering its own flight path (see
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

The wire value is a single ``DataArray``; taking it apart is
:func:`~ess.livedata.workflows.lut_blocks.unpack_block`, which owns that format
jointly with the producer-side packing.
"""

from __future__ import annotations

from typing import NewType

import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import MonitorType, SampleRun
from ess.reduce.unwrap import LookupTable
from ess.reduce.unwrap.types import DetectorLtotal, MonitorLtotal

from ..config.instrument import Instrument
from .lut_blocks import unpack_block
from .wavelength_lut_workflow_specs import (
    DETECTOR_LUT_OUTPUT,
    LUT_STREAM_NAMES,
    MONITOR_LUT_OUTPUT,
)

#: Context key carrying the detectors' streamed table, as it arrives on the wire.
DetectorLutContext = NewType('DetectorLutContext', sc.DataArray)

#: Context key carrying the monitors' streamed table, as it arrives on the wire.
MonitorLutContext = NewType('MonitorLutContext', sc.DataArray)


def detector_lookup_table(
    wire: DetectorLutContext, ltotal: DetectorLtotal[SampleRun]
) -> LookupTable[SampleRun, snx.NXdetector]:
    """Reassemble a detector's lookup table from the detector context stream."""
    return LookupTable[SampleRun, snx.NXdetector](**unpack_block(wire, ltotal))


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
    return LookupTable[SampleRun, MonitorType](**unpack_block(wire, ltotal))


def declare_lut_context_streams(instrument: Instrument) -> None:
    """Declare the streamed tables as context streams consuming graphs can request.

    One declaration per group, and none per spec: a workflow reaches the table
    by inserting :func:`detector_lookup_table` or :func:`monitor_lookup_table`,
    and the build derives the stream from the graph that provider ends up in
    (see :meth:`~ess.livedata.config.instrument.Instrument.declare_context_stream`).
    Time-of-arrival jobs therefore stay ungated without anyone saying so: their
    graph reduces straight from ``event_time_offset``, so the provider is
    unreachable and the table is not requested. That is the property ADR 0010
    asks for -- the mode you fall back to when everything else is broken must
    not wait on an operator-started job -- and it holds by construction rather
    than by something each spec has to remember to say.

    A group with no placeable component publishes no table, so its stream is
    not declared and a job whose graph asks for it fails at job creation on the
    unsatisfied key, rather than waiting forever for a stream nobody publishes.

    A placeable *group* does not make every component in it placeable. A job on
    an unplaceable source gates normally, receives the table, and fails when
    :func:`~ess.livedata.workflows.lut_blocks.select_block` finds no block
    covering its flight path, reporting that distance against the table's
    coverage. A consumer that can rule this out earlier should: LOKI's I(Q)
    rejects an unplaceable monitor selection in its factory, where the aux
    selection is in hand.
    """
    for group, key, output in (
        (instrument.detector_names, DetectorLutContext, DETECTOR_LUT_OUTPUT),
        (instrument.monitors, MonitorLutContext, MONITOR_LUT_OUTPUT),
    ):
        if set(group) & instrument.lut_components:
            instrument.declare_context_stream(
                workflow_key=key, stream_name=LUT_STREAM_NAMES[output]
            )
