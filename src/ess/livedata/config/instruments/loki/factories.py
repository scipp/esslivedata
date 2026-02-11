# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""LOKI instrument factory implementations."""

from ess.livedata.config import Instrument

from . import specs
from .specs import SansWorkflowParams, TransmissionMode


def setup_factories(instrument: Instrument) -> None:
    """Initialize LOKI-specific factories and workflows."""
    import sciline
    import sciline.typing
    import scipp as sc
    from scippnexus import NXdetector

    import ess.loki.data
    from ess.livedata.handlers.detector_data_handler import get_nexus_geometry_filename
    from ess.livedata.handlers.detector_view import (
        DetectorViewFactory,
        GeometricViewConfig,
        NeXusDetectorSource,
    )
    from ess.livedata.handlers.stream_processor_workflow import (
        StreamProcessorWorkflow,
    )
    from ess.loki.workflow import LokiWorkflow
    from ess.reduce.nexus.types import (
        EmptyBeamRun,
        IncidentMonitor,
        NeXusData,
        NeXusDetectorName,
        NeXusName,
        SampleRun,
        TransmissionMonitor,
        TransmissionRun,
    )
    from ess.reduce.time_of_flight.types import TofLookupTableFilename
    from ess.sans import types as sans_types
    from ess.sans.types import (
        BeamCenter,
        CorrectForGravity,
        DetectorMasks,
        DirectBeam,
        Filename,
        Incident,
        IntensityQ,
        Numerator,
        ReducedQ,
        ReturnEvents,
        Transmission,
        UncertaintyBroadcastMode,
    )

    _base_workflow = LokiWorkflow()
    _base_workflow[Filename[SampleRun]] = get_nexus_geometry_filename('loki')
    _base_workflow[TofLookupTableFilename] = str(
        ess.loki.data.loki_tof_lookup_table_no_choppers()
    )
    # Override monitor names to match the geometry file naming convention.
    _base_workflow[NeXusName[IncidentMonitor]] = 'monitor_1'
    _base_workflow[NeXusName[TransmissionMonitor]] = 'monitor_3'
    _base_workflow[DirectBeam] = None
    _base_workflow[CorrectForGravity] = CorrectForGravity(False)
    _base_workflow[ReturnEvents] = ReturnEvents(False)
    _base_workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    _base_workflow[DetectorMasks] = DetectorMasks({})

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
        data_source=NeXusDetectorSource(get_nexus_geometry_filename('loki')),
        view_config={
            name: GeometricViewConfig(
                projection_type='xy_plane',
                resolution=res,
                pixel_noise='cylindrical',
            )
            for name, res in _bank_resolutions.items()
        },
    )

    specs.xy_projection_handle.attach_factory()(_xy_projection.make_workflow)

    # Monitor workflow factory (TOA-only)
    from ess.livedata.handlers.monitor_workflow import create_monitor_workflow
    from ess.livedata.handlers.monitor_workflow_specs import TOAOnlyMonitorDataParams

    @specs.monitor_handle.attach_factory()
    def _monitor_workflow_factory(source_name: str, params: TOAOnlyMonitorDataParams):
        """Factory for LOKI monitor workflow (TOA-only)."""
        return create_monitor_workflow(
            source_name=source_name,
            edges=params.get_active_edges(),
            range_filter=params.get_active_range(),
            coordinate_mode='toa',
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

    def _dynamic_keys(source_name: str) -> dict[str, sciline.typing.Key]:
        return {
            source_name: NeXusData[NXdetector, SampleRun],
            'incident_monitor': NeXusData[Incident, SampleRun],
            'transmission_monitor': NeXusData[Transmission, SampleRun],
        }

    _accumulators = (
        ReducedQ[SampleRun, Numerator],
        sans_types.CorrectedMonitor[SampleRun, Incident],
        sans_types.CorrectedMonitor[SampleRun, Transmission],
    )

    @specs.i_of_q_handle.attach_factory()
    def _i_of_q_factory(
        source_name: str, params: SansWorkflowParams
    ) -> StreamProcessorWorkflow:
        wf = _base_workflow.copy()
        wf[NeXusDetectorName] = source_name
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
            dynamic_keys=_dynamic_keys(source_name),
            target_keys=target_keys,
            accumulators=_accumulators,
        )
