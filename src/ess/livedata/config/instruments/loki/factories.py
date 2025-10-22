# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""LOKI instrument factory implementations."""

from ess.livedata.config import Instrument

from . import specs
from .specs import SansWorkflowParams


def setup_factories(instrument: Instrument):
    """Initialize LOKI-specific factories and workflows."""
    import sciline
    import sciline.typing
    from scippnexus import NXdetector

    import ess.loki.live  # noqa: F401
    from ess import loki
    from ess.livedata.handlers.detector_data_handler import (
        DetectorProjection,
        get_nexus_geometry_filename,
    )
    from ess.livedata.handlers.stream_processor_workflow import (
        StreamProcessorWorkflow,
    )
    from ess.reduce.nexus.types import NeXusData, NeXusDetectorName, SampleRun
    from ess.sans import types as sans_types
    from ess.sans.types import (
        Filename,
        Incident,
        IofQ,
        Numerator,
        ReducedQ,
        Transmission,
    )

    # Created once outside workflow wrappers since this configures some files from pooch
    # where a checksum is needed, which takes significant time.
    _base_workflow = loki.live._configured_Larmor_AgBeh_workflow()
    _base_workflow[Filename[SampleRun]] = get_nexus_geometry_filename('loki')

    _xy_projection = DetectorProjection(
        instrument=instrument,
        projection='xy_plane',
        pixel_noise='cylindrical',
        resolution={
            'loki_detector_0': {'y': 12, 'x': 12},
            # First window frame
            'loki_detector_1': {'y': 3, 'x': 9},
            'loki_detector_2': {'y': 9, 'x': 3},
            'loki_detector_3': {'y': 3, 'x': 9},
            'loki_detector_4': {'y': 9, 'x': 3},
            # Second window frame
            'loki_detector_5': {'y': 3, 'x': 9},
            'loki_detector_6': {'y': 9, 'x': 3},
            'loki_detector_7': {'y': 3, 'x': 9},
            'loki_detector_8': {'y': 9, 'x': 3},
        },
        resolution_scale=12,
    )

    specs.xy_projection_handles['view'].attach_factory()(_xy_projection.make_view)
    specs.xy_projection_handles['roi'].attach_factory()(_xy_projection.make_roi)

    def _transmission_from_current_run(
        data: sans_types.CleanMonitor[SampleRun, sans_types.MonitorType],
    ) -> sans_types.CleanMonitor[
        sans_types.TransmissionRun[SampleRun], sans_types.MonitorType
    ]:
        return data

    def _dynamic_keys(source_name: str) -> dict[str, sciline.typing.Key]:
        return {
            source_name: NeXusData[NXdetector, SampleRun],
            'incident_monitor': NeXusData[Incident, SampleRun],
            'transmission_monitor': NeXusData[Transmission, SampleRun],
        }

    _accumulators = (
        ReducedQ[SampleRun, Numerator],
        sans_types.CleanMonitor[SampleRun, Incident],
        sans_types.CleanMonitor[SampleRun, Transmission],
    )

    @specs.i_of_q_handle.attach_factory()
    def _i_of_q_factory(source_name: str) -> StreamProcessorWorkflow:
        wf = _base_workflow.copy()
        wf[NeXusDetectorName] = source_name
        return StreamProcessorWorkflow(
            wf,
            dynamic_keys=_dynamic_keys(source_name),
            target_keys={'i_of_q': IofQ[SampleRun]},
            accumulators=_accumulators,
        )

    @specs.i_of_q_with_params_handle.attach_factory()
    def _i_of_q_with_params_factory(
        source_name: str, params: SansWorkflowParams
    ) -> StreamProcessorWorkflow:
        wf = _base_workflow.copy()
        wf[NeXusDetectorName] = source_name

        wf[sans_types.QBins] = params.q_edges.get_edges()
        wf[sans_types.WavelengthBins] = params.wavelength_edges.get_edges()

        if not params.options.use_transmission_run:
            target_keys = {
                'i_of_q': IofQ[SampleRun],
                'transmission_fraction': sans_types.TransmissionFraction[SampleRun],
            }
            wf.insert(_transmission_from_current_run)
        else:
            # Transmission fraction is static, do not display
            target_keys = {'i_of_q': IofQ[SampleRun]}
        return StreamProcessorWorkflow(
            wf,
            dynamic_keys=_dynamic_keys(source_name),
            target_keys=target_keys,
            accumulators=_accumulators,
        )
