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
:func:`~ess.livedata.workflows.dynamic_transforms.synthesise_provider`.

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
import structlog
from ess.reduce.nexus.types import (
    AnyRun,
    DiskChoppers,
    Filename,
    Position,
    RawChoppers,
)
from ess.reduce.nexus.workflow import to_disk_choppers
from ess.reduce.unwrap import GenericUnwrapWorkflow
from ess.reduce.unwrap.lut import (
    ChopperFrameSequence,
    DistanceResolution,
    PulsePeriod,
    PulseStride,
    TimeResolution,
    _estimate_wavelength_by_polygon_centers,
    make_wavelength_lut_from_polygons,
)

from ..config.chopper import delay_setpoint_stream, speed_setpoint_stream
from ..config.stream import AxisRange
from .dynamic_transforms import synthesise_provider
from .lut_blocks import Range, blocks_by_gap, one_block, pack_blocks
from .lut_ranges import LtotalRangeError, component_ltotal_range
from .stream_processor_workflow import StreamProcessorWorkflow
from .wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    DETECTOR_LUT_OUTPUT,
    MONITOR_LUT_OUTPUT,
    WAVELENGTH_BANDS_OUTPUT,
    WavelengthLutParams,
)
from .workflow_factory import SpecHandle, Workflow

logger = structlog.get_logger(__name__)

#: The chopper-cascade trigger payload as it reaches the workflow: the
#: cumulative ``ToNXlog`` timeseries for the synthetic ``chopper_cascade``
#: stream. The value is ignored; only its presence drives recomputation.
ChopperCascadeTrigger = NewType('ChopperCascadeTrigger', sc.DataArray)

#: Per-component wavelength bands evaluated at exact chopper distances, indexed
#: by a ``distance`` dimension (source + one row per chopper, plus one row per
#: configured cut distance).
WavelengthBands = NewType('WavelengthBands', sc.DataArray)

#: The two published tables, each a concatenation of uniform blocks (see
#: :mod:`~ess.livedata.workflows.lut_blocks`). Two keys, not one per component:
#: a table is a function of ``distance`` and ``event_time_offset`` alone, so
#: components sharing a stretch of beamline share a table.
DetectorLut = NewType('DetectorLut', sc.DataArray)
MonitorLut = NewType('MonitorLut', sc.DataArray)

#: The flight-path ranges each table must cover, one per placeable component.
#: Sciline parameters rather than closure state, so the table providers stay
#: plain module-level functions that read as what they compute.
DetectorLtotalRanges = NewType('DetectorLtotalRanges', tuple)
MonitorLtotalRanges = NewType('MonitorLtotalRanges', tuple)

#: Sciline key for the user-facing parameter bundle, used by the chopper-cascade
#: bands provider for settings the upstream pipeline knows nothing about.
#: Distinct from the individual parameter keys (PulsePeriod, etc.) that the
#: upstream pipeline consumes (and may unit-convert internally).
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
    :func:`~ess.livedata.workflows.dynamic_transforms.synthesise_provider`.

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


def make_wavelength_bands_from_frames(
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride[AnyRun],
    frames: ChopperFrameSequence[AnyRun],
    params: ParamsKey,
) -> WavelengthBands:
    """Wavelength band transmitted at points along the beamline.

    Evaluates the surviving wavelength vs ``event_time_offset`` at the exact
    distance of every frame in the cascade — the source pulse (distance 0)
    followed by one row per chopper — plus a row at each user-configured cut
    distance (typically monitor and detector positions). A row that is entirely
    NaN means no neutrons are transmitted there, i.e. the upstream chopper
    blocks the beam. Rows are ordered by ascending distance.

    Cut-distance rows reuse :meth:`FrameSequence.__getitem__`, which propagates
    the last cascade frame *before* the requested distance forward to it, so a
    point between or beyond the choppers reflects the physically correct beam
    state.

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

    time_edges = sc.linspace(
        'event_time_offset',
        0.0,
        frame_period.value,
        params.cascade_bands.num_bins + 1,
        unit=pulse_period.unit,
    )

    def band(frame) -> sc.Variable:
        return _estimate_wavelength_by_polygon_centers(
            subframes=frame.subframes,
            time_edges=time_edges,
            time_unit=time_unit,
            wavelength_unit=wavelength_unit,
            frame_period=frame_period,
        )

    # Source + choppers sit at their own frame distances; cut-distance rows
    # propagate the cascade forward to each configured point.
    rows = [(frame.distance.to(unit='m'), frame) for frame in frames]
    rows.extend(
        (distance, frames[distance])
        for distance in params.cascade_bands.get_distances().to(unit='m')
    )

    # Round to whole millimetres so curve labels read cleanly (e.g. 6.145 m)
    # rather than carrying float-propagation noise.
    distances = sc.round(sc.concat([d for d, _ in rows], dim='distance'), decimals=3)
    table = sc.DataArray(
        data=sc.concat([band(frame) for _, frame in rows], dim='distance'),
        coords={'distance': distances, 'event_time_offset': time_edges},
    )
    return WavelengthBands(sc.sort(table, 'distance'))


def _build_table(
    blocks: Sequence[Range],
    *,
    distance_resolution: sc.Variable,
    time_resolution: sc.Variable,
    pulse_period: sc.Variable,
    pulse_stride: int,
    frames: Any,
) -> sc.DataArray:
    """Rasterize the cascade onto each block and concatenate the results."""
    return pack_blocks(
        [
            make_wavelength_lut_from_polygons(
                ltotal_range=block,
                distance_resolution=distance_resolution,
                time_resolution=time_resolution,
                pulse_period=pulse_period,
                pulse_stride=pulse_stride,
                frames=frames,
            )
            for block in blocks
        ]
    )


def make_detector_lut(
    ranges: DetectorLtotalRanges,
    distance_resolution: DistanceResolution,
    time_resolution: TimeResolution,
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride[AnyRun],
    frames: ChopperFrameSequence[AnyRun],
) -> DetectorLut:
    """The detectors' shared table: one dense block spanning every bank."""
    return DetectorLut(
        _build_table(
            one_block(ranges),
            distance_resolution=distance_resolution,
            time_resolution=time_resolution,
            pulse_period=pulse_period,
            pulse_stride=pulse_stride,
            frames=frames,
        )
    )


def make_monitor_lut(
    ranges: MonitorLtotalRanges,
    distance_resolution: DistanceResolution,
    time_resolution: TimeResolution,
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride[AnyRun],
    frames: ChopperFrameSequence[AnyRun],
) -> MonitorLut:
    """The monitors' shared table: one block per monitor, nothing in between."""
    return MonitorLut(
        _build_table(
            blocks_by_gap(ranges, distance_resolution),
            distance_resolution=distance_resolution,
            time_resolution=time_resolution,
            pulse_period=pulse_period,
            pulse_stride=pulse_stride,
            frames=frames,
        )
    )


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
    wf[ParamsKey] = params
    # Per-component diagnostic evaluated at exact chopper distances, sidestepping
    # the table's distance resolution. Reuses the analytical ChopperFrameSequence.
    wf.insert(make_wavelength_bands_from_frames)
    return wf


def _make_workflow(
    pipeline: sciline.Pipeline, lut_keys: Mapping[str, Any]
) -> StreamProcessorWorkflow:
    return StreamProcessorWorkflow(
        pipeline,
        dynamic_keys={CHOPPER_CASCADE_SOURCE: ChopperCascadeTrigger},
        target_keys={
            **lut_keys,
            WAVELENGTH_BANDS_OUTPUT: WavelengthBands,
        },
        accumulators={},
        # The trigger flows straight to the DiskChoppers provider rather than
        # through an accumulator.
        allow_bypass=True,
    )


def _component_position(filename: str, nx_class: type) -> sc.Variable | None:
    """Position of the file's unique component of ``nx_class``, or ``None``.

    Returns ``None`` when the file has no such component.
    """
    with snx.File(filename, definitions=snx.base_definitions()) as f:
        groups = f['entry/instrument'][nx_class]
        if not groups:
            return None
        (group,) = groups.values()
        positions = snx.compute_positions(group[...], store_position='position')
    return positions['position']


def create_wavelength_lut_workflow(
    *,
    params: WavelengthLutParams,
    setpoint_keys: Mapping[str, ChopperSetpointKeys],
    nexus_filename: str,
    detector_ranges: Sequence[Range],
    monitor_ranges: Sequence[Range],
) -> Workflow:
    """Factory for the chopper-equipped wavelength lookup-table workflow.

    ``setpoint_keys`` maps each chopper name to the context keys its setpoints
    arrive on; the same keys must back the spec-scope ``ContextBinding``\\ s the
    instrument declares. ``nexus_filename`` is the geometry artifact carrying
    the static ``NXdisk_chopper`` groups (slit edges, radius, axle position)
    and the source position; the pipeline loads both from it. A configured
    chopper missing from the artifact surfaces as a ``KeyError`` at recompute.

    ``detector_ranges`` and ``monitor_ranges`` are the flight-path ranges of the
    placeable components in each group; the group's table covers them as blocks.
    A group with no placeable component produces no table and no output: the
    only consumers are its components, and they bind nothing (ADR 0010).
    """
    pipeline = _build_pipeline(params)
    pipeline[Filename[AnyRun]] = nexus_filename
    _set_source_position(pipeline, nexus_filename, params.source.get())
    pipeline.insert(build_disk_choppers_provider(setpoint_keys))
    lut_keys: dict[str, Any] = {}
    if detector_ranges:
        pipeline[DetectorLtotalRanges] = tuple(detector_ranges)
        pipeline.insert(make_detector_lut)
        lut_keys[DETECTOR_LUT_OUTPUT] = DetectorLut
    if monitor_ranges:
        pipeline[MonitorLtotalRanges] = tuple(monitor_ranges)
        pipeline.insert(make_monitor_lut)
        lut_keys[MONITOR_LUT_OUTPUT] = MonitorLut
    return _make_workflow(pipeline, lut_keys)


def _set_source_position(
    pipeline: sciline.Pipeline, nexus_filename: str, offset: sc.Variable
) -> None:
    """Set the chopper-cascade source reference, shifted by ``offset``.

    essreduce would take the reference from the file's unique ``NXsource``.
    BIFROST instead labels the moderator ``NXmoderator`` and reserves
    ``NXsource`` for accelerator metadata sitting at the origin, so essreduce
    would measure flight distance from the accelerator and reject the upstream
    choppers as lying behind the source. The reference is therefore the
    ``NXmoderator`` position when present, else the ``NXsource`` (e.g. LOKI).

    ``offset`` is a beam-aligned displacement added to that reference; it is
    zero by default, in which case this reproduces the position essreduce would
    have loaded.
    """
    reference = _component_position(nexus_filename, snx.NXmoderator)
    if reference is None:
        reference = _component_position(nexus_filename, snx.NXsource)
    pipeline[Position[snx.NXsource, AnyRun]] = reference + offset.to(
        unit=reference.unit
    )


def attach_wavelength_lut_factory(
    handle: SpecHandle,
    *,
    choppers: Sequence[str],
    nexus_filename: str,
    detectors: Sequence[str],
    monitors: Sequence[str],
    axis_ranges: Mapping[str, AxisRange],
) -> frozenset[str]:
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

    Each component's flight-path range is derived from the geometry artifact.
    A component whose range cannot be derived -- one riding a live f144-driven
    axis with no declared :class:`AxisRange` -- is left without a table.

    Returns
    -------
    :
        The components a block of a published table covers. Consumers bind only
        these, so a component without a table gates nothing rather than blocking
        every job that could have selected it.
    """
    detector_ranges = _derive_ltotal_ranges(
        nexus_filename, detectors, is_monitor=False, axis_ranges=axis_ranges
    )
    monitor_ranges = _derive_ltotal_ranges(
        nexus_filename, monitors, is_monitor=True, axis_ranges=axis_ranges
    )
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
            params=params,
            setpoint_keys=setpoint_keys,
            nexus_filename=nexus_filename,
            detector_ranges=list(detector_ranges.values()),
            monitor_ranges=list(monitor_ranges.values()),
        )

    return frozenset(detector_ranges) | frozenset(monitor_ranges)


def _derive_ltotal_ranges(
    nexus_filename: str,
    components: Sequence[str],
    *,
    is_monitor: bool,
    axis_ranges: Mapping[str, AxisRange],
) -> dict[str, Range]:
    """Derive a group's ranges, skipping components the artifact cannot place."""
    ranges: dict[str, Range] = {}
    for name in components:
        try:
            ranges[name] = component_ltotal_range(
                nexus_filename, name, is_monitor=is_monitor, axis_ranges=axis_ranges
            )
        except LtotalRangeError as exc:
            logger.warning(
                'wavelength_lut_range_underivable', component=name, reason=str(exc)
            )
    return ranges
