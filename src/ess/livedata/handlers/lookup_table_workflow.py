# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""TOF lookup-table workflow.

Wraps :func:`ess.reduce.unwrap.lut.LookupTableWorkflow` for use as a livedata
``Workflow``. The chopperless v0 always supplies an empty ``DiskChoppers``;
chopper-equipped instruments will require a translation provider that
combines static NeXus geometry with cached chopper setpoints.
"""

from __future__ import annotations

from typing import Any

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

from ..core.timestamp import Timestamp
from .lookup_table_workflow_specs import CHOPPER_CASCADE_SOURCE, LookupTableParams
from .workflow_factory import Workflow

# Placeholder source position. Only used inside the per-chopper loop in the
# upstream simulator, which is empty for chopperless instruments.
_PLACEHOLDER_SOURCE_POSITION = sc.vector([0.0, 0.0, 0.0], unit='m')


def _attach_provenance_coords(
    array: sc.DataArray, params: LookupTableParams
) -> sc.DataArray:
    """Attach the four scalar input parameters as 0-D coords on the result.

    Makes the published da00 message self-describing: a consumer can
    reconstruct the upstream ``LookupTable`` dataclass from the array alone,
    without out-of-band coordination on parameter values.
    """
    out = array.copy()
    out.coords['pulse_period'] = params.pulse_period.get()
    out.coords['pulse_stride'] = sc.scalar(int(params.simulation.pulse_stride))
    out.coords['distance_resolution'] = params.distance_resolution.get()
    out.coords['time_resolution'] = params.time_resolution.get()
    return out


def _build_pipeline(params: LookupTableParams, choppers: sc.DataGroup) -> Any:
    wf = LookupTableWorkflow()
    wf[DiskChoppers[AnyRun]] = choppers
    wf[PulsePeriod] = params.pulse_period.get()
    wf[PulseStride] = int(params.simulation.pulse_stride)
    wf[DistanceResolution] = params.distance_resolution.get()
    wf[TimeResolution] = params.time_resolution.get()
    wf[LtotalRange] = (
        params.Ltotal_range.get_start(),
        params.Ltotal_range.get_stop(),
    )
    wf[NumberOfSimulatedNeutrons] = int(params.simulation.num_simulated_neutrons)
    wf[SourcePosition] = _PLACEHOLDER_SOURCE_POSITION
    return wf


class LookupTableComputeWorkflow(Workflow):
    """Workflow that computes the TOF lookup table on the first trigger.

    The synthetic ``chopper_cascade`` stream is the trigger; its value is
    ignored. After the first computation the result is cached so re-finalize
    calls (e.g., after orchestrator-level retries) return the same table
    without recomputing. ``clear`` discards the cache so a job reset
    recomputes on the next trigger.
    """

    def __init__(self, params: LookupTableParams) -> None:
        self._params = params
        self._triggered = False
        self._result: sc.DataArray | None = None

    def accumulate(
        self, data: dict[str, Any], *, start_time: Timestamp, end_time: Timestamp
    ) -> None:
        del start_time, end_time
        if CHOPPER_CASCADE_SOURCE in data:
            self._triggered = True

    def finalize(self) -> dict[str, sc.DataArray]:
        if not self._triggered:
            raise RuntimeError(
                f"Workflow finalize() called before any '{CHOPPER_CASCADE_SOURCE}' "
                "trigger was received."
            )
        if self._result is None:
            self._result = self._compute()
        return {'lookup_table': self._result}

    def clear(self) -> None:
        self._triggered = False
        self._result = None

    def _compute(self) -> sc.DataArray:
        wf = _build_pipeline(self._params, choppers=sc.DataGroup({}))
        table: LookupTable = wf.compute(LookupTable)
        return _attach_provenance_coords(table.array, self._params)


def create_chopperless_lookup_table_workflow(*, params: LookupTableParams) -> Workflow:
    """Factory for the chopperless lookup-table workflow.

    Always supplies ``DiskChoppers = {}`` to the upstream pipeline. The
    workflow's only stream input is the synthetic ``chopper_cascade`` trigger.
    """
    return LookupTableComputeWorkflow(params)
