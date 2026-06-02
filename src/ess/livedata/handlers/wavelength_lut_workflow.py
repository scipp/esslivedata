# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Wavelength lookup-table workflow.

Builds a Sciline pipeline from :func:`ess.reduce.unwrap.lut.providers` in
*analytical* mode (chopper-cascade polygon geometry, no neutron simulation) and
wraps it as a livedata ``Workflow`` via :class:`StreamProcessorWorkflow`. The
synthetic
``chopper_cascade`` trigger is a sciline dynamic key consumed by a provider
that produces ``DiskChoppers``; its value is ignored — only its arrival
drives a recompute (the trigger is the job's ``allow_bypass`` primary).

Two flavours:

- **Chopperless** (``create_chopperless_wavelength_lut_workflow``): the
  provider returns an empty ``DiskChoppers``. The trigger fires once.
- **With choppers** (``create_wavelength_lut_workflow``): static
  ``NXdisk_chopper`` geometry is loaded from the NeXus artifact, and the
  provider merges per-chopper rotation-speed and delay **setpoints** —
  delivered as Sciline *context* keys via :meth:`StreamProcessor.set_context`,
  gated at the JobManager (ADR 0002/0003) — onto that geometry via
  ``DiskChopper.from_nexus``. The provider is synthesised at factory time
  (chopper count is known then) with one parameter per setpoint, reusing
  :func:`~ess.livedata.handlers.dynamic_transforms.synthesise_provider`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
from ess.reduce.unwrap import GenericUnwrapWorkflow
from ess.reduce.unwrap.lut import (
    DistanceResolution,
    LookupTable,
    LtotalRange,
    PulsePeriod,
    PulseStride,
    TimeResolution,
)
from scippneutron.chopper import DiskChopper

from .dynamic_transforms import synthesise_provider
from .stream_processor_workflow import StreamProcessorWorkflow
from .wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    WAVELENGTH_LUT_OUTPUT,
    WavelengthLutParams,
)
from .workflow_factory import Workflow

# Placeholder source position for chopperless instruments (the upstream
# per-chopper simulation loop is empty, so the value is unused).
_PLACEHOLDER_SOURCE_POSITION = sc.vector([0.0, 0.0, 0.0], unit='m')

# Anti-clockwise angular offset where the beam crosses the chopper disk. The
# upstream essreduce convention stores ``slit_edges`` already measured from the
# beam-crossing point, so this is zero. Production NeXus files carry no
# ``beam_position`` field, so it is injected here.
_BEAM_POSITION = sc.scalar(0.0, unit='deg')

#: Component the standalone table is parametrised by. The table is not tied to a
#: real detector; the choice only fixes the internal Sciline keys (LtotalRange,
#: LookupTable) and does not appear in the published array.
_LUT_COMPONENT = snx.NXdetector

#: The chopper-cascade trigger payload as it reaches the workflow: the
#: cumulative ``ToNXlog`` timeseries for the synthetic ``chopper_cascade``
#: stream. The value is ignored; only its presence drives recomputation.
ChopperCascadeTrigger = NewType('ChopperCascadeTrigger', sc.DataArray)

#: Wavelength lookup-table with provenance coords attached.
WavelengthLut = NewType('WavelengthLut', sc.DataArray)

#: Sciline key for the user-facing parameter bundle, used by the provenance
#: provider. Distinct from the individual parameter keys (PulsePeriod, etc.)
#: that the upstream pipeline consumes (and may unit-convert internally).
ParamsKey = NewType('ParamsKey', WavelengthLutParams)


@dataclass(frozen=True)
class ChopperSetpointKeys:
    """The pair of Sciline context keys feeding one chopper's setpoints.

    Each key is a distinct ``NewType`` so the synthesised ``DiskChoppers``
    provider has one uniquely-typed parameter per chopper per quantity. The
    *same* objects must be used for both the provider's parameter annotations
    and the ``ContextBinding.workflow_key`` declared by the factory, so the
    binding's ``set_context`` value reaches the right provider parameter —
    hence they are created once (see :func:`make_chopper_setpoint_keys`) and
    shared.
    """

    speed: Any
    delay: Any


def make_chopper_setpoint_keys(chopper: str) -> ChopperSetpointKeys:
    """Create the rotation-speed and delay context keys for one chopper."""
    return ChopperSetpointKeys(
        speed=NewType(f'RotationSpeedSetpoint_{chopper}', sc.DataArray),
        delay=NewType(f'DelaySetpoint_{chopper}', sc.DataArray),
    )


def _latest(container: sc.DataArray) -> sc.Variable:
    """Latest sample of a cumulative NXlog context value."""
    return container['time', -1].data


def build_disk_choppers_provider(
    setpoint_keys: Mapping[str, ChopperSetpointKeys],
    *,
    raw_choppers: sc.DataGroup,
    beam_position: sc.Variable = _BEAM_POSITION,
) -> Callable[..., DiskChoppers[AnyRun]]:
    """Synthesise the provider assembling ``DiskChoppers`` from live setpoints.

    The returned provider consumes :data:`ChopperCascadeTrigger` (so a new
    trigger drives a recompute) plus, per chopper, its rotation-speed and delay
    setpoint context keys. At evaluation it overrides the static geometry's
    NXlog placeholders with the latest setpoint samples and builds each chopper
    via ``DiskChopper.from_nexus``, which derives ``phase`` from ``delay`` and
    ``rotation_speed_setpoint``.

    The arity is fixed at synthesis time from ``setpoint_keys``; sciline reads a
    provider's ``__code__`` and ignores ``__signature__``, so a real function
    with N named typed parameters is built via
    :func:`~ess.livedata.handlers.dynamic_transforms.synthesise_provider`.
    """
    names = list(setpoint_keys)
    order = [(name, quantity) for name in names for quantity in ('speed', 'delay')]

    def _impl(_trigger: Any, *containers: sc.DataArray) -> DiskChoppers[AnyRun]:
        latest: dict[tuple[str, str], sc.Variable] = {
            key: _latest(container)
            for key, container in zip(order, containers, strict=True)
        }
        choppers = {}
        for name in names:
            merged = sc.DataGroup(dict(raw_choppers[name]))
            merged['rotation_speed_setpoint'] = latest[name, 'speed']
            merged['delay'] = latest[name, 'delay']
            merged['beam_position'] = beam_position
            choppers[name] = DiskChopper.from_nexus(merged)
        return DiskChoppers[AnyRun](sc.DataGroup(choppers))

    annotations: dict[str, Any] = {'trigger': ChopperCascadeTrigger}
    for name in names:
        annotations[f'speed_{name}'] = setpoint_keys[name].speed
        annotations[f'delay_{name}'] = setpoint_keys[name].delay
    annotations['return'] = DiskChoppers[AnyRun]
    return synthesise_provider('_provide_disk_choppers', _impl, annotations)


def _empty_choppers(_: ChopperCascadeTrigger) -> DiskChoppers[AnyRun]:
    """Provide an empty chopper cascade for chopperless instruments."""
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


def _build_pipeline(
    params: WavelengthLutParams, *, source_position: sc.Variable
) -> sciline.Pipeline:
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
    wf[Position[snx.NXsource, AnyRun]] = source_position
    wf[ParamsKey] = params
    wf.insert(_attach_provenance)
    return wf


def _make_workflow(pipeline: sciline.Pipeline) -> StreamProcessorWorkflow:
    return StreamProcessorWorkflow(
        pipeline,
        dynamic_keys={CHOPPER_CASCADE_SOURCE: ChopperCascadeTrigger},
        target_keys={WAVELENGTH_LUT_OUTPUT: WavelengthLut},
        accumulators={},
        # The trigger flows straight to the DiskChoppers provider rather than
        # through an accumulator.
        allow_bypass=True,
    )


def _load_static_geometry(filename: str) -> tuple[sc.DataGroup, sc.Variable]:
    """Load static chopper geometry and source position from a NeXus file.

    Uses ``GenericNeXusWorkflow`` so ``depends_on`` chains are resolved and
    ``RawChoppers`` is produced in the layout ``DiskChopper.from_nexus``
    consumes. NXlog placeholders for streamed quantities (``rotation_speed``,
    ``delay``, etc.) come through as length-0 arrays and are overridden later
    with the live setpoints.
    """
    wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])
    wf[Filename[SampleRun]] = filename
    raw_choppers = wf.compute(RawChoppers[SampleRun])
    source_position = wf.compute(Position[snx.NXsource, SampleRun])
    return raw_choppers, source_position


def create_chopperless_wavelength_lut_workflow(
    *, params: WavelengthLutParams
) -> Workflow:
    """Factory for the chopperless wavelength lookup-table workflow.

    The workflow's only stream input is the synthetic ``chopper_cascade``
    trigger. The empty-chopper provider ignores its value and returns an empty
    cascade.
    """
    pipeline = _build_pipeline(params, source_position=_PLACEHOLDER_SOURCE_POSITION)
    pipeline.insert(_empty_choppers)
    return _make_workflow(pipeline)


def create_wavelength_lut_workflow(
    *,
    params: WavelengthLutParams,
    setpoint_keys: Mapping[str, ChopperSetpointKeys],
    nexus_filename: str,
    beam_position: sc.Variable = _BEAM_POSITION,
) -> Workflow:
    """Factory for the chopper-equipped wavelength lookup-table workflow.

    ``setpoint_keys`` maps each chopper name to the context keys its setpoints
    arrive on; the same keys must back the spec-scope ``ContextBinding``\\ s the
    instrument declares. ``nexus_filename`` is the geometry artifact carrying
    the static ``NXdisk_chopper`` groups (slit edges, radius, axle position)
    and the source position.
    """
    raw_choppers, source_position = _load_static_geometry(nexus_filename)
    missing = set(setpoint_keys) - set(raw_choppers)
    if missing:
        raise ValueError(
            f"Geometry artifact {nexus_filename!r} is missing NXdisk_chopper "
            f"groups for: {sorted(missing)}"
        )
    pipeline = _build_pipeline(params, source_position=source_position)
    pipeline.insert(
        build_disk_choppers_provider(
            setpoint_keys, raw_choppers=raw_choppers, beam_position=beam_position
        )
    )
    return _make_workflow(pipeline)
