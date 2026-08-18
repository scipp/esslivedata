# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Consumption of a streamed wavelength lookup table as workflow context.

The lookup-table workflow publishes one table per component as a context stream
(ADR 0010). A consuming workflow binds the rendered stream name with a
``ContextBinding`` whose ``workflow_key`` is one of the context keys here, and
inserts the matching provider, which reassembles essreduce's public
:class:`~ess.reduce.unwrap.LookupTable` dataclass.

Reassembling rather than round-tripping through essreduce's file loader is
deliberate. That loader's matching branch is a backwards-compatibility shim for
tables written before the dataclass existed; depending on it would tie us to a
deprecated path and silently drop the ``choppers`` field, which the old format
cannot carry. Chopper provenance travels in the table's identity instead.

The wire value is a single ``DataArray`` because da00 serializes a
``DataArray`` and the dataclass has non-array fields (``pulse_stride`` is an
``int``). The producer attaches the remaining fields as 0-D coords; this module
is the inverse of that.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, NewType

import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import SampleRun
from ess.reduce.unwrap import LookupTable

from .wavelength_lut_workflow_specs import lut_stream_name
from .workflow_factory import SpecHandle

#: Provenance scalars the producer attaches as coords, in dataclass field order.
_PROVENANCE = ('pulse_period', 'pulse_stride', 'distance_resolution', 'time_resolution')

#: Context key carrying a detector's streamed table, as it arrives on the wire.
DetectorLutContext = NewType('DetectorLutContext', sc.DataArray)

#: Context key carrying a monitor's streamed table, as it arrives on the wire.
MonitorLutContext = NewType('MonitorLutContext', sc.DataArray)


def _unpack(array: sc.DataArray) -> dict:
    """Split a wire table into the ``LookupTable`` dataclass fields."""
    if missing := [name for name in _PROVENANCE if name not in array.coords]:
        raise ValueError(
            f"Streamed lookup table is missing provenance coord(s) {missing}; "
            f"got coords {sorted(array.coords)}. The producer attaches these, so "
            "this indicates a table from an incompatible producer version."
        )
    return {
        'array': array.drop_coords(list(_PROVENANCE)),
        'pulse_period': array.coords['pulse_period'],
        'pulse_stride': int(array.coords['pulse_stride'].value),
        'distance_resolution': array.coords['distance_resolution'],
        'time_resolution': array.coords['time_resolution'],
    }


def detector_lookup_table(
    wire: DetectorLutContext,
) -> LookupTable[SampleRun, snx.NXdetector]:
    """Reassemble a detector's lookup table from its context stream."""
    return LookupTable[SampleRun, snx.NXdetector](**_unpack(wire))


def monitor_lookup_table(
    wire: MonitorLutContext,
) -> LookupTable[SampleRun, snx.NXmonitor]:
    """Reassemble a monitor's lookup table from its context stream."""
    return LookupTable[SampleRun, snx.NXmonitor](**_unpack(wire))


def bind_lookup_tables(
    handle: SpecHandle, *, source_names: Iterable[str], is_monitor: bool
) -> None:
    """Bind each source's streamed lookup table as gated context.

    One ``ContextBinding`` per source, selected by ``dependent_sources`` so a
    job receives its own component's table and no other. The predicate keeps
    time-of-arrival jobs ungated: the table is published by an operator-started
    job that re-emits only on chopper change, and making TOA -- the mode you
    fall back to when everything else is broken -- depend on it would be the
    wrong trade (ADR 0010).

    Parameters
    ----------
    handle:
        Spec to bind the tables to.
    source_names:
        Sources the spec runs on; each gets its own binding.
    is_monitor:
        Select the monitor context key over the detector one.
    """
    key = MonitorLutContext if is_monitor else DetectorLutContext
    for source_name in source_names:
        handle.add_context_binding(
            stream_name=lut_stream_name(source_name),
            workflow_key=key,
            dependent_sources=[source_name],
            predicate=_reads_wavelength,
        )


def _reads_wavelength(params: Any) -> bool:
    """Whether a job's params select the wavelength coordinate mode."""
    mode = getattr(params, 'coordinate_mode', None)
    return mode is not None and mode.mode == 'wavelength'
