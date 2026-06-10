# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Wavelength lookup-table workflow.

Builds a Sciline pipeline from :func:`ess.reduce.unwrap.GenericUnwrapWorkflow`
in *analytical* mode (chopper-cascade polygon geometry, no neutron simulation)
and wraps it as a livedata ``Workflow`` via :class:`StreamProcessorWorkflow`.
The synthetic ``chopper_cascade`` trigger is exposed as a sciline dynamic key
whose value is consumed by an internal provider that produces ``DiskChoppers``.

For chopperless instruments the provider returns an empty ``DiskChoppers``;
v1 will replace it with one that assembles real choppers from chopper-PV
context streams.
"""

from __future__ import annotations

from typing import NewType

import sciline
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import AnyRun, DiskChoppers, Position
from ess.reduce.unwrap import GenericUnwrapWorkflow
from ess.reduce.unwrap.lut import (
    DistanceResolution,
    LookupTable,
    LtotalRange,
    PulsePeriod,
    PulseStride,
    TimeResolution,
)

from .stream_processor_workflow import StreamProcessorWorkflow
from .wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    WAVELENGTH_LUT_OUTPUT,
    WavelengthLutParams,
)
from .workflow_factory import Workflow

#: Component the standalone table is parametrised by. The table is not tied to a
#: real detector; the choice only fixes the internal Sciline keys (LtotalRange,
#: LookupTable) and does not appear in the published array.
_LUT_COMPONENT = snx.NXdetector

# Placeholder source position. With no choppers the analytical frame sequence
# is the full source pulse, so the absolute position is immaterial.
_PLACEHOLDER_SOURCE_POSITION = sc.vector([0.0, 0.0, 0.0], unit='m')

#: The chopper-cascade trigger payload as it reaches the workflow: the
#: cumulative ``ToNXlog`` timeseries for the synthetic ``chopper_cascade``
#: stream. The value is ignored by the chopperless provider; only its
#: presence matters (it drives ``is_active()`` in the orchestrator).
ChopperCascadeTrigger = NewType('ChopperCascadeTrigger', sc.DataArray)

#: Wavelength lookup-table with provenance coords attached.
WavelengthLut = NewType('WavelengthLut', sc.DataArray)

#: Sciline key for the user-facing parameter bundle, used by the provenance
#: provider. Distinct from the individual parameter keys (PulsePeriod, etc.)
#: that the upstream pipeline consumes (and may unit-convert internally).
ParamsKey = NewType('ParamsKey', WavelengthLutParams)


def _empty_choppers(_: ChopperCascadeTrigger) -> DiskChoppers[AnyRun]:
    """Provide an empty chopper cascade for chopperless instruments.

    The trigger value is intentionally unused: it acts purely as a "compute
    now" signal. v1 will replace this provider with one that takes chopper
    PV context streams as additional inputs and assembles real choppers.
    """
    return DiskChoppers[AnyRun](sc.DataGroup({}))


def _attach_provenance(
    table: LookupTable[AnyRun, _LUT_COMPONENT], params: ParamsKey
) -> WavelengthLut:
    """Attach the four scalar input parameters as 0-D coords on the result.

    Makes the published da00 message self-describing: a consumer can
    reconstruct the upstream ``LookupTable`` dataclass from the array alone,
    without out-of-band coordination on parameter values. Pulling from
    ``params`` (not ``table``) keeps the units user-facing — the upstream
    pipeline may convert internally.
    """
    arr = table.array.copy()
    arr.coords['pulse_period'] = params.pulse.get_period()
    arr.coords['pulse_stride'] = sc.scalar(int(params.pulse.stride))
    arr.coords['distance_resolution'] = params.distance_resolution.get()
    arr.coords['time_resolution'] = params.time_resolution.get()
    return WavelengthLut(arr)


def _build_pipeline(params: WavelengthLutParams) -> sciline.Pipeline:
    wf = GenericUnwrapWorkflow(
        run_types=[AnyRun], monitor_types=[], wavelength_from='analytical'
    )
    wf[PulsePeriod] = params.pulse.get_period()
    wf[PulseStride[AnyRun]] = int(params.pulse.stride)
    wf[DistanceResolution] = params.distance_resolution.get()
    wf[TimeResolution] = params.time_resolution.get()
    wf[LtotalRange[AnyRun, _LUT_COMPONENT]] = (
        params.distance_range.get_start(),
        params.distance_range.get_stop(),
    )
    wf[Position[snx.NXsource, AnyRun]] = _PLACEHOLDER_SOURCE_POSITION
    wf[ParamsKey] = params
    wf.insert(_empty_choppers)
    wf.insert(_attach_provenance)
    return wf


def create_chopperless_wavelength_lut_workflow(
    *, params: WavelengthLutParams
) -> Workflow:
    """Factory for the chopperless wavelength lookup-table workflow.

    The workflow's only stream input is the synthetic ``chopper_cascade``
    trigger, exposed to sciline as a dynamic key. ``allow_bypass=True`` is
    required because the trigger value flows directly to a provider rather
    than through an accumulator.
    """
    return StreamProcessorWorkflow(
        _build_pipeline(params),
        dynamic_keys={CHOPPER_CASCADE_SOURCE: ChopperCascadeTrigger},
        target_keys={WAVELENGTH_LUT_OUTPUT: WavelengthLut},
        accumulators={},
        allow_bypass=True,
    )
