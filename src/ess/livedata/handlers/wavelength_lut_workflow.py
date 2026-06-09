# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Wavelength lookup-table workflow.

Builds a Sciline pipeline from :func:`ess.reduce.unwrap.GenericUnwrapWorkflow`
in *analytical* mode (chopper-cascade polygon geometry, no neutron simulation)
and wraps it as a livedata ``Workflow`` via :class:`StreamProcessorWorkflow`.
The synthetic ``chopper_cascade`` trigger is a sciline dynamic key consumed by a
provider that produces ``DiskChoppers``; its value is ignored — only its arrival
drives a recompute (the trigger is the job's ``allow_bypass`` primary).

The pipeline loads static ``NXdisk_chopper`` geometry from the NeXus artifact
itself — via ``Filename`` and the ``GenericNeXusWorkflow`` chopper providers,
producing ``RawChoppers`` — and the synthesised provider consumes those raw
choppers, merging per-chopper rotation-speed and delay **setpoints** —
delivered as Sciline *context* keys via :meth:`StreamProcessor.set_context`,
gated at the JobManager (ADR 0002/0003) — onto that geometry, then delegating
the ``RawChoppers`` → ``DiskChoppers`` conversion to essreduce's
``to_disk_choppers``. It thereby replaces the workflow's own call to that
provider. The provider is synthesised at factory time (chopper count is known
then) with one parameter per setpoint, reusing
:func:`~ess.livedata.handlers.dynamic_transforms.synthesise_provider`.

An instrument with no choppers simply supplies a geometry artifact whose
``NXsource`` is present but that has no ``NXdisk_chopper`` groups: the empty
``RawChoppers`` yields empty ``DiskChoppers``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NewType

import sciline
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import (
    AnyRun,
    DiskChoppers,
    Filename,
    RawChoppers,
)
from ess.reduce.nexus.workflow import to_disk_choppers
from ess.reduce.unwrap import GenericUnwrapWorkflow
from ess.reduce.unwrap.lut import (
    ChopperFrameSequence,
    DistanceResolution,
    LookupTable,
    LtotalRange,
    PulsePeriod,
    PulseStride,
    TimeResolution,
    _estimate_wavelength_by_polygon_centers,
)

from ..config.chopper import delay_setpoint_stream, speed_setpoint_stream
from .dynamic_transforms import synthesise_provider
from .stream_processor_workflow import StreamProcessorWorkflow
from .wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    WAVELENGTH_BANDS_OUTPUT,
    WAVELENGTH_LUT_OUTPUT,
    WavelengthLutParams,
)
from .workflow_factory import SpecHandle, Workflow

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

#: Per-component wavelength bands evaluated at exact chopper distances, indexed
#: by a ``distance`` dimension (source + one row per chopper).
WavelengthBands = NewType('WavelengthBands', sc.DataArray)

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
) -> Callable[..., DiskChoppers[AnyRun]]:
    """Synthesise the provider assembling ``DiskChoppers`` from live setpoints.

    The returned provider consumes the workflow's file-loaded
    ``RawChoppers`` (static geometry) and :data:`ChopperCascadeTrigger` (so a new
    trigger drives a recompute), plus, per chopper, its rotation-speed and delay
    setpoint context keys. At evaluation it overrides the static geometry's
    NXlog placeholders with the latest setpoint samples, then delegates the
    ``RawChoppers`` → ``DiskChoppers`` conversion (``DiskChopper.from_nexus`` per
    chopper, including the default zero ``beam_position`` for ESS files) to
    essreduce's :func:`~ess.reduce.nexus.workflow.to_disk_choppers`. Producing
    ``DiskChoppers`` directly replaces the workflow's own call to that provider.

    The arity is fixed at synthesis time from ``setpoint_keys``; sciline reads a
    provider's ``__code__`` and ignores ``__signature__``, so a real function
    with N named typed parameters is built via
    :func:`~ess.livedata.handlers.dynamic_transforms.synthesise_provider`.

    The ``rotation_speed_setpoint``/``delay`` field names hardcoded here are one
    of the sites that move if the streamed chopper quantities change; see
    :class:`~ess.livedata.kafka.chopper_synthesizer.ChopperSynthesizer` for the
    full list.
    """
    names = list(setpoint_keys)
    order = [(name, quantity) for name in names for quantity in ('speed', 'delay')]

    def _impl(
        raw_choppers: sc.DataGroup, _trigger: Any, *containers: sc.DataArray
    ) -> DiskChoppers[AnyRun]:
        latest: dict[tuple[str, str], sc.Variable] = {
            key: _latest(container)
            for key, container in zip(order, containers, strict=True)
        }
        patched = {}
        for name in names:
            merged = sc.DataGroup(dict(raw_choppers[name]))
            merged['rotation_speed_setpoint'] = latest[name, 'speed']
            merged['delay'] = latest[name, 'delay']
            patched[name] = merged
        return to_disk_choppers(RawChoppers[AnyRun](sc.DataGroup(patched)))

    annotations: dict[str, Any] = {
        'raw_choppers': RawChoppers[AnyRun],
        'trigger': ChopperCascadeTrigger,
    }
    for name in names:
        annotations[f'speed_{name}'] = setpoint_keys[name].speed
        annotations[f'delay_{name}'] = setpoint_keys[name].delay
    annotations['return'] = DiskChoppers[AnyRun]
    return synthesise_provider('_provide_disk_choppers', _impl, annotations)


def _attach_provenance(
    table: LookupTable[AnyRun, _LUT_COMPONENT], params: ParamsKey
) -> WavelengthLut:
    """Attach the four scalar input parameters as 0-D coords on the result.

    Makes the published da00 message self-describing: a consumer can
    reconstruct the upstream ``LookupTable`` dataclass from the array alone,
    without out-of-band coordination on parameter values. Pulling from
    ``params`` (not ``table``) keeps the units user-facing — the upstream
    pipeline may convert internally. The stride is the exception: it is read
    back from ``table`` so the coord reflects the value actually used, which
    with auto-detection is guessed from the choppers rather than supplied by
    ``params``.
    """
    arr = table.array.copy()
    arr.coords['pulse_period'] = params.pulse.get_period()
    arr.coords['pulse_stride'] = sc.scalar(int(table.pulse_stride))
    arr.coords['distance_resolution'] = params.distance_resolution.get()
    arr.coords['time_resolution'] = params.time_resolution.get()
    return WavelengthLut(arr)


def make_wavelength_bands_from_frames(
    time_resolution: TimeResolution,
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride[AnyRun],
    frames: ChopperFrameSequence[AnyRun],
) -> WavelengthBands:
    """Wavelength band transmitted at each chopper-cascade frame.

    Evaluates the surviving wavelength vs ``event_time_offset`` at the exact
    distance of every frame in the cascade — the source pulse (distance 0)
    followed by one row per chopper, ordered by ascending distance. A row that
    is entirely NaN means no neutrons are transmitted at that component, i.e.
    the corresponding chopper blocks the beam.

    Unlike :func:`make_wavelength_lut_from_polygons`, this does not rasterize
    onto a uniform distance grid, so closely-spaced choppers stay individually
    resolved regardless of the table's distance resolution — letting one read
    off which chopper in a tight cascade blocks the beam.

    Reuses essreduce's (currently private)
    ``_estimate_wavelength_by_polygon_centers``; once this diagnostic proves its
    design, the whole function should move upstream into ``ess.reduce.unwrap``.
    """
    time_unit = 'us'
    wavelength_unit = 'angstrom'
    pulse_period = pulse_period.to(unit=time_unit)
    frame_period = pulse_period * pulse_stride

    nbins = 10 * int(frame_period / time_resolution.to(unit=time_unit)) + 1
    time_edges = sc.linspace(
        'event_time_offset', 0.0, frame_period.value, nbins + 1, unit=pulse_period.unit
    )

    bands = [
        _estimate_wavelength_by_polygon_centers(
            subframes=frame.subframes,
            time_edges=time_edges,
            time_unit=time_unit,
            wavelength_unit=wavelength_unit,
            frame_period=frame_period,
        )
        for frame in frames
    ]
    distances = sc.concat(
        [frame.distance.to(unit='m') for frame in frames], dim='distance'
    )
    table = sc.DataArray(
        data=sc.concat(bands, dim='distance'),
        coords={'distance': distances, 'event_time_offset': time_edges},
    )
    return WavelengthBands(table)


def _build_pipeline(params: WavelengthLutParams) -> sciline.Pipeline:
    wf = GenericUnwrapWorkflow(
        run_types=[AnyRun], monitor_types=[], wavelength_from='analytical'
    )
    wf[PulsePeriod] = params.pulse.get_period()
    if not params.pulse.auto_stride:
        # Otherwise the workflow's guess_pulse_stride_from_choppers provider
        # derives the stride from the chopper rotation frequencies.
        wf[PulseStride[AnyRun]] = int(params.pulse.stride)
    wf[DistanceResolution] = params.distance_resolution.get()
    wf[TimeResolution] = params.time_resolution.get()
    wf[LtotalRange[AnyRun, _LUT_COMPONENT]] = (
        params.distance_range.get_start(),
        params.distance_range.get_stop(),
    )
    wf[ParamsKey] = params
    wf.insert(_attach_provenance)
    # Per-component diagnostic evaluated at exact chopper distances, sidestepping
    # the table's distance resolution. Reuses the analytical ChopperFrameSequence.
    wf.insert(make_wavelength_bands_from_frames)
    return wf


def _make_workflow(pipeline: sciline.Pipeline) -> StreamProcessorWorkflow:
    return StreamProcessorWorkflow(
        pipeline,
        dynamic_keys={CHOPPER_CASCADE_SOURCE: ChopperCascadeTrigger},
        target_keys={
            WAVELENGTH_LUT_OUTPUT: WavelengthLut,
            WAVELENGTH_BANDS_OUTPUT: WavelengthBands,
        },
        accumulators={},
        # The trigger flows straight to the DiskChoppers provider rather than
        # through an accumulator.
        allow_bypass=True,
    )


def create_wavelength_lut_workflow(
    *,
    params: WavelengthLutParams,
    setpoint_keys: Mapping[str, ChopperSetpointKeys],
    nexus_filename: str,
) -> Workflow:
    """Factory for the chopper-equipped wavelength lookup-table workflow.

    ``setpoint_keys`` maps each chopper name to the context keys its setpoints
    arrive on; the same keys must back the spec-scope ``ContextBinding``\\ s the
    instrument declares. ``nexus_filename`` is the geometry artifact carrying
    the static ``NXdisk_chopper`` groups (slit edges, radius, axle position)
    and the source position; the pipeline loads both from it. A configured
    chopper missing from the artifact surfaces as a ``KeyError`` at recompute.
    """
    pipeline = _build_pipeline(params)
    pipeline[Filename[AnyRun]] = nexus_filename
    pipeline.insert(build_disk_choppers_provider(setpoint_keys))
    return _make_workflow(pipeline)


def attach_wavelength_lut_factory(
    handle: SpecHandle, *, choppers: Sequence[str], nexus_filename: str
) -> None:
    """Bind per-chopper setpoint context and attach the LUT factory.

    The single per-instrument entry point: from ``factories.py`` an instrument
    that has choppers calls this on its registered spec handle. It declares one
    spec-scope ``ContextBinding`` per chopper per setpoint quantity
    (rotation-speed, delay), then attaches the workflow factory.

    The setpoint keys are created once and shared *by reference* between the
    bindings and the ``DiskChoppers`` provider the factory inserts, so each
    ``set_context`` value reaches the matching provider parameter. Sharing them
    here enforces that invariant by construction rather than leaving each
    instrument's ``factories.py`` to wire matching keys by hand.
    """
    setpoint_keys = {
        chopper: make_chopper_setpoint_keys(chopper) for chopper in choppers
    }
    for chopper, keys in setpoint_keys.items():
        handle.add_context_binding(
            stream_name=speed_setpoint_stream(chopper), workflow_key=keys.speed
        )
        handle.add_context_binding(
            stream_name=delay_setpoint_stream(chopper), workflow_key=keys.delay
        )

    @handle.attach_factory()
    def _factory(params: WavelengthLutParams) -> Workflow:
        return create_wavelength_lut_workflow(
            params=params, setpoint_keys=setpoint_keys, nexus_filename=nexus_filename
        )
