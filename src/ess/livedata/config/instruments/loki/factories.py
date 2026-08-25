# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""LOKI instrument factory implementations."""

from ess.livedata.config import Instrument
from ess.livedata.config.value_log import ValueLog

from . import specs
from .specs import SansWorkflowParams, TransmissionMode


class DetectorCarriageLog(ValueLog):
    """Per-binding Sciline key for the LOKI rear-bank carriage f144 NXlog.

    Drives the ``detector_carriage`` transformation in ``loki_detector_0``'s
    ``depends_on`` chain. Distinct subclass so multiple dynamic transforms
    on a workflow remain distinguishable in Sciline.
    """


#: I(Q) aux fields naming the monitors a job selects. Each needs a block in the
#: streamed monitor table, which is what the factory checks before creating a job.
_IQ_MONITOR_ROLES = ('incident_monitor', 'transmission_monitor')


def setup_factories(instrument: Instrument) -> None:
    """Initialize LOKI-specific factories and workflows."""
    from ess.livedata.workflows.lut_context import (
        bind_lookup_tables,
        detector_lookup_table,
        monitor_lookup_table,
        reads_wavelength,
    )

    # Detector and monitor views bind their group's streamed lookup table,
    # gated so only wavelength-mode jobs wait for it.
    bind_lookup_tables(
        specs.xy_projection_handle,
        instrument=instrument,
        source_names=instrument.detector_names,
        is_monitor=False,
        predicate=reads_wavelength,
    )
    bind_lookup_tables(
        specs.monitor_handle,
        instrument=instrument,
        source_names=instrument.monitors,
        is_monitor=True,
        predicate=reads_wavelength,
    )
    # I(Q) reads both tables: its detector and its two monitor roles. No
    # predicate -- a reduction always reduces to wavelength -- and no per-role
    # binding, since a role's monitor is picked out of the shared monitor table
    # by its own flight path rather than by a stream name (ADR 0010).
    for is_monitor in (False, True):
        bind_lookup_tables(
            specs.i_of_q_handle,
            instrument=instrument,
            source_names=instrument.detector_names,
            is_monitor=is_monitor,
        )
    import sciline
    import sciline.typing
    import scipp as sc
    from ess.loki.workflow import LokiWorkflow
    from ess.reduce.nexus.types import (
        EmptyBeamRun,
        NeXusData,
        NeXusDetectorName,
        SampleRun,
        TransmissionRun,
    )
    from ess.sans import types as sans_types
    from ess.sans.types import (
        BeamCenter,
        CorrectForGravity,
        DetectorMasks,
        DirectBeam,
        Filename,
        Incident,
        IntensityQ,
        NeXusMonitorName,
        Numerator,
        ReducedQ,
        ReturnEvents,
        Transmission,
        UncertaintyBroadcastMode,
    )
    from scippnexus import NXdetector

    from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename
    from ess.livedata.workflows.detector_view import (
        DetectorViewFactory,
        GeometricViewConfig,
        NeXusDetectorSource,
    )
    from ess.livedata.workflows.stream_processor_workflow import (
        StreamProcessorWorkflow,
    )

    # The rear bank's NeXus ``depends_on`` chain has a dynamic ``detector_carriage``
    # transformation driven by the live f144 carriage readback. Declared at
    # instrument scope so every spec consuming ``loki_detector_0`` picks it up by
    # default. ``tube_view`` sums over straw/pixel and does not consume bank
    # position, so it opts out — co-located here with the binding it negates.
    instrument.add_context_binding(
        stream_name='detector_carriage',
        dependent_sources={'loki_detector_0'},
        workflow_key=DetectorCarriageLog,
    )
    specs.tube_view_handle.skip_instrument_contexts()

    _nexus_geometry_filename = get_nexus_geometry_filename('loki')

    def _make_base_workflow() -> LokiWorkflow:
        """Create the base LokiWorkflow for I(Q) reduction.

        Called lazily inside the I(Q) factory to avoid triggering pooch downloads
        at setup_factories() time (which would block test collection when external
        servers are unavailable).
        """
        wf = LokiWorkflow()
        wf[Filename[SampleRun]] = _nexus_geometry_filename
        wf[DirectBeam] = None
        wf[CorrectForGravity] = CorrectForGravity(False)
        wf[ReturnEvents] = ReturnEvents(False)
        wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
        wf[DetectorMasks] = DetectorMasks({})
        return wf

    # Sciline-based detector view with XY projection for all detector banks.
    # Resolution values = base resolution * scale (12), matching the legacy setup.
    _bank_resolutions = {
        'loki_detector_0': {'y': 144, 'x': 144},
        # First window frame
        'loki_detector_1': {'y': 36, 'x': 108},
        'loki_detector_2': {'y': 108, 'x': 36},
        'loki_detector_3': {'y': 36, 'x': 108},
        'loki_detector_4': {'y': 108, 'x': 36},
        # Second window frame
        'loki_detector_5': {'y': 36, 'x': 108},
        'loki_detector_6': {'y': 108, 'x': 36},
        'loki_detector_7': {'y': 36, 'x': 108},
        'loki_detector_8': {'y': 108, 'x': 36},
    }
    _xy_projection = DetectorViewFactory(
        data_source=NeXusDetectorSource(_nexus_geometry_filename),
        view_config={
            name: GeometricViewConfig(
                projection_type='xy_plane',
                resolution=res,
                pixel_noise='cylindrical',
                flip_x=True,
            )
            for name, res in _bank_resolutions.items()
        },
    )

    specs.xy_projection_handle.attach_factory()(_xy_projection.make_workflow)

    from ess.livedata.workflows.monitor_workflow import create_monitor_workflow
    from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams

    @specs.monitor_handle.attach_factory()
    def _monitor_workflow_factory(source_name: str, params: MonitorDataParams):
        """Factory for LOKI monitor workflow with lookup table support."""
        mode = params.coordinate_mode.mode
        geometry_filename = _nexus_geometry_filename if mode == 'wavelength' else None

        return create_monitor_workflow(
            source_name=source_name,
            edges=params.get_active_edges(),
            range_filter=params.get_active_range(),
            coordinate_mode=mode,
            geometry_filename=geometry_filename,
        )

    # --- Providers for current_run transmission mode ---
    # Map SampleRun monitors to TransmissionRun[SampleRun] so the standard
    # transmission_fraction provider can use them as if they came from a
    # dedicated transmission run.
    # Workaround: position coords must be dropped because monitor_to_wavelength
    # does not consume them (Ltotal is pre-computed), so they survive to
    # transmission_fraction where the multiply of incident/transmission ratios
    # fails on mismatched positions. See https://github.com/scipp/esssans/issues/244

    def _incident_as_transmission_run(
        mon: sans_types.CorrectedMonitor[SampleRun, Incident],
    ) -> sans_types.CorrectedMonitor[TransmissionRun[SampleRun], Incident]:
        out = sc.values(mon).drop_coords([c for c in mon.coords if c != 'wavelength'])
        return sans_types.CorrectedMonitor[TransmissionRun[SampleRun], Incident](out)

    def _transmission_as_transmission_run(
        mon: sans_types.CorrectedMonitor[SampleRun, Transmission],
    ) -> sans_types.CorrectedMonitor[TransmissionRun[SampleRun], Transmission]:
        out = sc.values(mon).drop_coords([c for c in mon.coords if c != 'wavelength'])
        return sans_types.CorrectedMonitor[TransmissionRun[SampleRun], Transmission](
            out
        )

    def _dynamic_keys(
        source_name: str, aux_source_names: dict[str, str]
    ) -> dict[str, sciline.typing.Key]:
        return {
            source_name: NeXusData[NXdetector, SampleRun],
            aux_source_names['incident_monitor']: NeXusData[Incident, SampleRun],
            aux_source_names['transmission_monitor']: NeXusData[
                Transmission, SampleRun
            ],
        }

    _accumulators = (
        ReducedQ[SampleRun, Numerator],
        sans_types.CorrectedMonitor[SampleRun, Incident],
        sans_types.CorrectedMonitor[SampleRun, Transmission],
    )

    @specs.i_of_q_handle.attach_factory()
    def _i_of_q_factory(
        source_name: str,
        params: SansWorkflowParams,
        aux_source_names: dict[str, str],
    ) -> StreamProcessorWorkflow:
        for aux_field in _IQ_MONITOR_ROLES:
            monitor = aux_source_names[aux_field]
            if monitor not in instrument.lut_components:
                # The monitor table has no block for it, so the job would open
                # its gate and then fail at every recompute; fail here instead.
                raise ValueError(
                    f"Monitor {monitor!r} selected as {aux_field} has no "
                    "streamed lookup table: its flight-path range cannot be "
                    "derived from the geometry artifact (undeclared motion "
                    "axis), so it cannot be used in a wavelength reduction."
                )
        wf = _make_base_workflow()
        wf[NeXusDetectorName] = source_name
        wf[NeXusMonitorName[Incident]] = aux_source_names['incident_monitor']
        wf[NeXusMonitorName[Transmission]] = aux_source_names['transmission_monitor']
        wf.insert(detector_lookup_table)
        # One generic provider serves both monitor roles: sciline instantiates
        # it per role, and each instance selects its monitor's block of the
        # shared table via that role's MonitorLtotal.
        wf.insert(monitor_lookup_table)
        wf[sans_types.QBins] = params.q_edges.get_edges()
        wf[sans_types.WavelengthBins] = params.wavelength_edges.get_edges()
        wf[BeamCenter] = params.beam_center.get_vector()

        target_keys: dict[str, sciline.typing.Key] = {
            'i_of_q': IntensityQ[SampleRun],
        }

        mode = params.transmission.mode
        if mode == TransmissionMode.constant:
            wf[sans_types.TransmissionFraction[SampleRun]] = sc.scalar(1.0)
        elif mode == TransmissionMode.current_run:
            wf.insert(_incident_as_transmission_run)
            wf.insert(_transmission_as_transmission_run)
            # Neutralize the empty-beam normalization in the standard
            # transmission_fraction provider so it simplifies to
            # sample_transmission / sample_incident.
            wf[sans_types.CorrectedMonitor[EmptyBeamRun, Incident]] = sc.scalar(
                1.0, unit='counts'
            )
            wf[sans_types.CorrectedMonitor[EmptyBeamRun, Transmission]] = sc.scalar(
                1.0, unit='counts'
            )
            target_keys['transmission_fraction'] = sans_types.TransmissionFraction[
                SampleRun
            ]

        return StreamProcessorWorkflow(
            wf,
            dynamic_keys=_dynamic_keys(source_name, aux_source_names),
            target_keys=target_keys,
            accumulators=_accumulators,
        )
