# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""LOKI instrument factory implementations."""

from ess.livedata.config import Instrument

from . import specs
from .specs import SansWorkflowParams, TransmissionMode


def setup_factories(instrument: Instrument) -> None:
    """Initialize LOKI-specific factories and workflows."""
    import ess.loki.data
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
        NeXusMonitorName,
        Numerator,
        ReducedQ,
        ReturnEvents,
        Transmission,
        UncertaintyBroadcastMode,
    )
    from scippnexus import NXdetector

    from ess.livedata.handlers.detector_data_handler import get_nexus_geometry_filename
    from ess.livedata.handlers.detector_view import (
        DetectorViewFactory,
        GeometricViewConfig,
        NeXusDetectorSource,
    )
    from ess.livedata.handlers.stream_processor_workflow import (
        StreamProcessorWorkflow,
    )

    _nexus_geometry_filename = get_nexus_geometry_filename('loki')

    def _resolve_tof_lookup_table_filename() -> str:
        """Resolve TOF lookup table filename lazily to avoid eager downloads."""
        return str(ess.loki.data.loki_tof_lookup_table_no_choppers())

    def _make_base_workflow() -> LokiWorkflow:
        """Create the base LokiWorkflow for I(Q) reduction.

        Called lazily inside the I(Q) factory to avoid triggering pooch downloads
        at setup_factories() time (which would block test collection when external
        servers are unavailable).
        """
        wf = LokiWorkflow()
        wf[Filename[SampleRun]] = _nexus_geometry_filename
        wf[TofLookupTableFilename] = _resolve_tof_lookup_table_filename()
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
        # Drive the rear bank's NeXus 'detector_carriage' link from the
        # live f144 carriage readback. Other banks have no override.
        link_overrides={
            'loki_detector_0': (
                'detector_carriage',
                '/entry/instrument/detector_carriage/value',
            ),
        },
    )

    from ess.livedata.handlers.detector_view_specs import DetectorViewParams

    @specs.xy_projection_handle.attach_factory()
    def _detector_view_workflow_factory(
        source_name: str, params: DetectorViewParams
    ) -> StreamProcessorWorkflow:
        """Factory for LOKI detector view with TOF lookup table support."""
        tof_lookup_table_filename = None
        if params.coordinate_mode.mode in ('tof', 'wavelength'):
            tof_lookup_table_filename = _resolve_tof_lookup_table_filename()

        return _xy_projection.make_workflow(
            source_name, params, tof_lookup_table_filename=tof_lookup_table_filename
        )

    from ess.livedata.handlers.monitor_workflow import create_monitor_workflow
    from ess.livedata.handlers.monitor_workflow_specs import MonitorDataParams

    @specs.monitor_handle.attach_factory()
    def _monitor_workflow_factory(source_name: str, params: MonitorDataParams):
        """Factory for LOKI monitor workflow with TOF lookup table support."""
        mode = params.coordinate_mode.mode
        if mode == 'wavelength':
            raise NotImplementedError(
                "wavelength mode not yet implemented for monitors"
            )

        tof_lookup_table_filename = None
        geometry_filename = None

        if mode == 'tof':
            tof_lookup_table_filename = _resolve_tof_lookup_table_filename()
            geometry_filename = _nexus_geometry_filename

        return create_monitor_workflow(
            source_name=source_name,
            edges=params.get_active_edges(),
            range_filter=params.get_active_range(),
            coordinate_mode=mode,
            tof_lookup_table_filename=tof_lookup_table_filename,
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
        source_name: str,
        params: SansWorkflowParams,
        aux_source_names: dict[str, str],
    ) -> StreamProcessorWorkflow:
        wf = _make_base_workflow()
        wf[NeXusDetectorName] = source_name
        wf[NeXusMonitorName[Incident]] = aux_source_names['incident_monitor']
        wf[NeXusMonitorName[Transmission]] = aux_source_names['transmission_monitor']
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
