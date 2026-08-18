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

from collections.abc import Callable, Iterable
from typing import Any, NewType

import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import SampleRun
from ess.reduce.unwrap import LookupTable

from ..config.instrument import Instrument
from .dynamic_transforms import synthesise_provider
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
    handle: SpecHandle,
    *,
    instrument: Instrument,
    source_names: Iterable[str],
    is_monitor: bool,
    predicate: Callable[[Any], bool] | None = None,
) -> None:
    """Bind each source's streamed lookup table as gated context.

    One ``ContextBinding`` per source, selected by ``dependent_sources`` so a
    job receives its own component's table and no other. The predicate keeps
    time-of-arrival jobs ungated: the table is published by an operator-started
    job that re-emits only on chopper change, and making TOA -- the mode you
    fall back to when everything else is broken -- depend on it would be the
    wrong trade (ADR 0010).

    Sources the lookup-table workflow cannot place are skipped. Binding a
    stream that is never published would leave the job gated forever, and doing
    so for a component nobody asked about would take unrelated jobs down with
    it.

    Parameters
    ----------
    handle:
        Spec to bind the tables to.
    instrument:
        Instrument whose ``lut_components`` decide which sources have a table.
    source_names:
        Sources the spec runs on; each placeable one gets its own binding.
    is_monitor:
        Select the monitor context key over the detector one.
    predicate:
        Narrows the binding to the jobs that actually read the table. Views
        pass :func:`reads_wavelength`; a reduction passes nothing, since it has
        no coordinate mode to choose and always needs its tables.
    """
    key = MonitorLutContext if is_monitor else DetectorLutContext
    for source_name in sorted(set(source_names) & instrument.lut_components):
        handle.add_context_binding(
            stream_name=lut_stream_name(source_name),
            workflow_key=key,
            dependent_sources=[source_name],
            predicate=predicate,
        )


def component_lut_context(component: str) -> Any:
    """Context key carrying one *named* component's table.

    Views bind by role (:data:`DetectorLutContext`,
    :data:`MonitorLutContext`) because the component is fixed. A reduction
    cannot: which physical monitor plays the incident or transmission role is a
    per-job aux selection, while a ``ContextBinding`` is declared at import
    time. So every candidate binds its own key here and the factory -- which
    does see the selection -- maps the chosen one onto the role.
    """
    return NewType(f'LutContext_{component}', sc.DataArray)


def bind_all_lookup_tables(
    handle: SpecHandle, *, instrument: Instrument, source_names: Iterable[str]
) -> dict[str, Any]:
    """Bind every candidate component's table, each to its own context key.

    Gating on all of them rather than on the selected one costs nothing: every
    table comes from the same lookup-table job, split out of one result, so
    they arrive together and the gate opens at the same instant either way. The
    unselected ones are dead parameters, which
    :meth:`StreamProcessor.set_context` stores and computes nothing from.

    Returns
    -------
    :
        The context key per component, for the factory to map onto role keys.
    """
    keys = {
        name: component_lut_context(name)
        for name in sorted(set(source_names) & instrument.lut_components)
    }
    for name, key in keys.items():
        # No predicate: a reduction has no coordinate mode to choose. It always
        # reduces to wavelength, so it always needs its tables.
        handle.add_context_binding(stream_name=lut_stream_name(name), workflow_key=key)
    return keys


def role_lookup_table_provider(component_type: type, context_key: Any) -> Any:
    """Provider mapping one component's context key onto a role's table key."""

    def _impl(wire: sc.DataArray) -> Any:
        return LookupTable[SampleRun, component_type](**_unpack(wire))

    return synthesise_provider(
        f'_provide_lut_{component_type.__name__}',
        _impl,
        {'wire': context_key, 'return': LookupTable[SampleRun, component_type]},
    )


def reads_wavelength(params: Any) -> bool:
    """Whether a job's params select the wavelength coordinate mode.

    Deliberately not tolerant of a params model without a coordinate mode: such
    a spec has no mode to choose, so routing it through here is a wiring
    mistake. Reading a missing field as "not wavelength" would leave the job
    ungated and hand its providers a table that never arrives.
    """
    return params.coordinate_mode.mode == 'wavelength'
