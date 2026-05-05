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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NewType

import sciline
import scipp as sc
from ess.reduce.nexus.types import AnyRun, DiskChoppers
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
    phase_setpoint_input,
    speed_setpoint_input,
)

# Placeholder source position. Only used inside the per-chopper loop in the
# upstream simulator, which is empty for chopperless instruments.
_PLACEHOLDER_SOURCE_POSITION = sc.vector([0.0, 0.0, 0.0], unit='m')

#: The chopper-cascade trigger payload as it reaches the workflow: the
#: cumulative ``ToNXlog`` timeseries for the synthetic ``chopper_cascade``
#: stream. Its value is unused — only its arrival drives recomputation.
ChopperCascadeTrigger = NewType('ChopperCascadeTrigger', sc.DataArray)

#: Wavelength lookup-table with provenance coords attached.
WavelengthLut = NewType('WavelengthLut', sc.DataArray)

#: Sciline key for the user-facing parameter bundle.
ParamsKey = NewType('ParamsKey', WavelengthLutParams)


@dataclass(frozen=True)
class HardcodedChopperGeometry:
    """Static geometry stand-in until the NeXus geometry artifact carries
    ``NXdisk_chopper`` groups; replaced by ``DiskChopper.from_nexus``.
    """

    axle_position: sc.Variable
    beam_position: sc.Variable
    slit_begin: sc.Variable
    slit_end: sc.Variable
    radius: sc.Variable


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


def _build_pipeline(params: WavelengthLutParams) -> sciline.Pipeline:
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
    wf[SourcePosition] = _PLACEHOLDER_SOURCE_POSITION
    wf[ParamsKey] = params
    wf.insert(_attach_provenance)
    return wf


class _WavelengthLutWorkflow(StreamProcessorWorkflow):
    """Workflow that assembles ``DiskChoppers`` outside sciline.

    Per-chopper aux setpoint NXlogs arrive via the spec's ``aux_sources``
    routing (logical names ``<chopper>_rotation_speed_setpoint`` /
    ``<chopper>_phase_setpoint``). They are not registered as sciline
    ``context_keys``; instead this class caches the latest scalar per
    chopper, builds a ``DiskChoppers`` ``DataGroup``, and exposes it to
    the pipeline via a closure provider keyed off the trigger. This keeps
    the sciline graph chopper-count-agnostic.
    """

    def __init__(
        self,
        *,
        params: WavelengthLutParams,
        chopper_names: list[str],
        chopper_geometry: Mapping[str, HardcodedChopperGeometry] | None = None,
    ) -> None:
        self._chopper_names = list(chopper_names)
        self._chopper_geometry = dict(chopper_geometry or {})
        self._cached_speed: dict[str, sc.Variable] = {}
        self._cached_phase: dict[str, sc.Variable] = {}
        self._cached_choppers: sc.DataGroup = sc.DataGroup({})
        missing = set(self._chopper_names) - self._chopper_geometry.keys()
        if missing:
            raise ValueError(
                f"Missing hardcoded geometry for choppers: {sorted(missing)}"
            )

        def _provide_choppers(_: ChopperCascadeTrigger) -> DiskChoppers[AnyRun]:
            return DiskChoppers[AnyRun](self._cached_choppers)

        pipeline = _build_pipeline(params)
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
            if (key := phase_setpoint_input(name)) in data:
                self._cached_phase[name] = data[key]['time', -1].data

        if self._all_choppers_ready():
            self._cached_choppers = sc.DataGroup(
                {name: self._build_chopper(name) for name in self._chopper_names}
            )
        super().accumulate(data, start_time=start_time, end_time=end_time)

    def _all_choppers_ready(self) -> bool:
        n = len(self._chopper_names)
        return len(self._cached_speed) == n and len(self._cached_phase) == n

    def _build_chopper(self, name: str) -> DiskChopper:
        geom = self._chopper_geometry[name]
        return DiskChopper(
            axle_position=geom.axle_position,
            beam_position=geom.beam_position,
            frequency=self._cached_speed[name],
            phase=self._cached_phase[name],
            slit_begin=geom.slit_begin,
            slit_end=geom.slit_end,
            radius=geom.radius,
        )


def create_wavelength_lut_workflow(
    *,
    params: WavelengthLutParams,
    chopper_names: list[str] | None = None,
    chopper_geometry: Mapping[str, HardcodedChopperGeometry] | None = None,
) -> StreamProcessorWorkflow:
    """Create a wavelength lookup-table workflow.

    Empty ``chopper_names`` ⇒ chopperless: the trigger fires once on the
    cached vacuous tick and the pipeline runs against an empty
    ``DiskChoppers``. Non-empty: per-chopper aux setpoints feed the cached
    ``DiskChoppers`` assembly; the pipeline refires whenever a new trigger
    arrives with all choppers locked.

    ``chopper_geometry`` supplies static (non-aux) per-chopper fields. It is
    a stand-in until ``RawChoppers[SampleRun]`` from the NeXus geometry
    artifact replaces it — see plan ``dynamic-tof-lookup-table.md``.
    """
    return _WavelengthLutWorkflow(
        params=params,
        chopper_names=list(chopper_names or []),
        chopper_geometry=chopper_geometry,
    )
