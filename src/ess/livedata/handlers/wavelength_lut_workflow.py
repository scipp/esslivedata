# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Wavelength lookup-table workflow.

Wraps :func:`ess.reduce.unwrap.lut.LookupTableWorkflow` as a livedata
``Workflow``. The synthetic ``chopper_cascade`` trigger drives recomputation;
per-chopper aux setpoints arrive as cached cumulative NXlogs and are
assembled into a ``DiskChoppers`` ``DataGroup`` outside sciline so the graph
shape does not depend on the chopper count.
"""

from __future__ import annotations

from typing import Any, NewType

import sciline
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus import GenericNeXusWorkflow
from ess.reduce.nexus.types import (
    AnyRun,
    DiskChoppers,
    Filename,
    Position,
    RawChoppers,
    SampleRun,
)
from ess.reduce.unwrap.lut import (
    DistanceResolution,
    LookupTable,
    LookupTableWorkflow,
    LtotalRange,
    NumberOfSimulatedNeutrons,
    PulsePeriod,
    PulseStride,
    SourcePosition,
    TimeResolution,
)
from scippneutron.chopper import DiskChopper

from ..core.timestamp import Timestamp
from .stream_processor_workflow import StreamProcessorWorkflow
from .wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    WAVELENGTH_LUT_OUTPUT,
    WavelengthLutParams,
    delay_setpoint_input,
    speed_setpoint_input,
)

# Placeholder source position. Only used inside the per-chopper loop in the
# upstream simulator, which is empty for chopperless instruments.
_PLACEHOLDER_SOURCE_POSITION = sc.vector([0.0, 0.0, 0.0], unit='m')

# Anti-clockwise angular offset where the beam crosses the chopper disk. The
# upstream essreduce convention (its tests, docs, and fakes) is to store
# slit_edges already measured from the beam-crossing point, so this is zero.
# See plan dynamic-tof-lookup-table.md for context.
_BEAM_POSITION = sc.scalar(0.0, unit='deg')

#: The chopper-cascade trigger payload as it reaches the workflow: the
#: cumulative ``ToNXlog`` timeseries for the synthetic ``chopper_cascade``
#: stream. Its value is unused — only its arrival drives recomputation.
ChopperCascadeTrigger = NewType('ChopperCascadeTrigger', sc.DataArray)

#: Wavelength lookup-table with provenance coords attached.
WavelengthLut = NewType('WavelengthLut', sc.DataArray)

#: Sciline key for the user-facing parameter bundle.
ParamsKey = NewType('ParamsKey', WavelengthLutParams)


def _attach_provenance(table: LookupTable, params: ParamsKey) -> WavelengthLut:
    """Attach the four scalar input parameters as 0-D coords on the result.

    Pulls from ``params`` (not ``table``) to keep units user-facing — the
    upstream pipeline may convert internally. Makes the published da00
    message self-describing.
    """
    arr = table.array.copy()
    arr.coords['pulse_period'] = params.pulse.get_period()
    arr.coords['pulse_stride'] = sc.scalar(int(params.pulse.stride))
    arr.coords['distance_resolution'] = params.distance_resolution.get()
    arr.coords['time_resolution'] = params.time_resolution.get()
    return WavelengthLut(arr)


def _build_pipeline(
    params: WavelengthLutParams, *, source_position: sc.Variable
) -> sciline.Pipeline:
    wf = LookupTableWorkflow()
    wf[PulsePeriod] = params.pulse.get_period()
    wf[PulseStride] = int(params.pulse.stride)
    wf[DistanceResolution] = params.distance_resolution.get()
    wf[TimeResolution] = params.time_resolution.get()
    wf[LtotalRange] = (
        params.distance_range.get_start(),
        params.distance_range.get_stop(),
    )
    wf[NumberOfSimulatedNeutrons] = int(params.simulation.num_simulated_neutrons)
    wf[SourcePosition] = source_position
    wf[ParamsKey] = params
    wf.insert(_attach_provenance)
    return wf


def _load_static_geometry(
    filename: str,
) -> tuple[sc.DataGroup, sc.Variable]:
    """Load static chopper geometry and source position from a NeXus file.

    Uses ``GenericNeXusWorkflow`` so depends_on chains are resolved and
    ``RawChoppers`` is produced in the layout :meth:`DiskChopper.from_nexus`
    consumes. NXlog placeholders for streamed quantities (``rotation_speed``,
    ``delay``, etc.) come through as length-0 ``DataArray``\\ s and are
    overridden later with cached scalar setpoints.
    """
    wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])
    wf[Filename[SampleRun]] = filename
    raw_choppers = wf.compute(RawChoppers[SampleRun])
    source_position = wf.compute(Position[snx.NXsource, SampleRun])
    return raw_choppers, source_position


class _WavelengthLutWorkflow(StreamProcessorWorkflow):
    """Workflow that assembles ``DiskChoppers`` outside sciline.

    Per-chopper aux setpoint NXlogs arrive via the spec's ``aux_sources``
    routing (logical names ``<chopper>_rotation_speed_setpoint`` /
    ``<chopper>_delay_setpoint``). They are not registered as sciline
    ``context_keys``; instead this class caches the latest scalar per
    chopper, builds a ``DiskChoppers`` ``DataGroup`` via
    ``DiskChopper.from_nexus`` against the static
    :class:`RawChoppers` loaded from the geometry artifact, and exposes it
    to the pipeline via a closure provider keyed off the trigger. This
    keeps the sciline graph chopper-count-agnostic.
    """

    def __init__(
        self,
        *,
        params: WavelengthLutParams,
        chopper_names: list[str],
        nexus_filename: str | None = None,
    ) -> None:
        self._chopper_names = list(chopper_names)
        self._cached_speed: dict[str, sc.Variable] = {}
        self._cached_delay: dict[str, sc.Variable] = {}
        self._cached_choppers: sc.DataGroup = sc.DataGroup({})

        if self._chopper_names:
            if nexus_filename is None:
                raise ValueError(
                    "nexus_filename is required for chopper-equipped instruments"
                )
            raw_choppers, source_position = _load_static_geometry(nexus_filename)
            missing = set(self._chopper_names) - set(raw_choppers)
            if missing:
                raise ValueError(
                    f"Geometry artifact {nexus_filename!r} is missing "
                    f"NXdisk_chopper groups for: {sorted(missing)}"
                )
            self._raw_choppers = raw_choppers
        else:
            source_position = _PLACEHOLDER_SOURCE_POSITION
            self._raw_choppers = sc.DataGroup({})

        def _provide_choppers(_: ChopperCascadeTrigger) -> DiskChoppers[AnyRun]:
            return DiskChoppers[AnyRun](self._cached_choppers)

        pipeline = _build_pipeline(params, source_position=source_position)
        pipeline.insert(_provide_choppers)
        super().__init__(
            pipeline,
            dynamic_keys={CHOPPER_CASCADE_SOURCE: ChopperCascadeTrigger},
            target_keys={WAVELENGTH_LUT_OUTPUT: WavelengthLut},
            accumulators={},
            allow_bypass=True,
        )

    def accumulate(
        self, data: dict[str, Any], *, start_time: Timestamp, end_time: Timestamp
    ) -> None:
        for name in self._chopper_names:
            if (key := speed_setpoint_input(name)) in data:
                self._cached_speed[name] = data[key]['time', -1].data
            if (key := delay_setpoint_input(name)) in data:
                self._cached_delay[name] = data[key]['time', -1].data

        if self._all_choppers_ready():
            self._cached_choppers = sc.DataGroup(
                {name: self._build_chopper(name) for name in self._chopper_names}
            )
        super().accumulate(data, start_time=start_time, end_time=end_time)

    def _all_choppers_ready(self) -> bool:
        n = len(self._chopper_names)
        return len(self._cached_speed) == n and len(self._cached_delay) == n

    def _build_chopper(self, name: str) -> DiskChopper:
        # Override the empty NXlog placeholders with the cached aux scalars
        # and inject ``beam_position`` (not present in real source files,
        # zero by convention — see module-level comment). ``from_nexus``
        # then derives ``phase`` from ``delay`` and ``rotation_speed_setpoint``.
        merged = sc.DataGroup(dict(self._raw_choppers[name]))
        merged['rotation_speed_setpoint'] = self._cached_speed[name]
        merged['delay'] = self._cached_delay[name]
        merged['beam_position'] = _BEAM_POSITION
        return DiskChopper.from_nexus(merged)


def create_wavelength_lut_workflow(
    *,
    params: WavelengthLutParams,
    chopper_names: list[str] | None = None,
    nexus_filename: str | None = None,
) -> StreamProcessorWorkflow:
    """Create a wavelength lookup-table workflow.

    Empty ``chopper_names`` ⇒ chopperless: the trigger fires once on the
    cached vacuous tick and the pipeline runs against an empty
    ``DiskChoppers``. Non-empty: per-chopper aux setpoints feed the cached
    ``DiskChoppers`` assembly; the pipeline refires whenever a new trigger
    arrives with all choppers locked.

    ``nexus_filename`` supplies the geometry artifact carrying static
    ``NXdisk_chopper`` groups (slit edges, radius, axle position) and the
    source position. Required when ``chopper_names`` is non-empty; ignored
    for chopperless instruments.
    """
    return _WavelengthLutWorkflow(
        params=params,
        chopper_names=list(chopper_names or []),
        nexus_filename=nexus_filename,
    )
