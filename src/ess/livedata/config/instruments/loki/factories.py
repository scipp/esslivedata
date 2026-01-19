# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""LOKI instrument factory implementations."""

from ess.livedata.config import Instrument

from . import specs
from .specs import SansWorkflowParams


def setup_factories(instrument: Instrument) -> None:
    """Initialize LOKI-specific factories and workflows."""
    import sciline
    import sciline.typing
    from scippnexus import NXdetector

    import ess.loki.live  # noqa: F401
    from ess import loki
    from ess.livedata.handlers.detector_data_handler import get_nexus_geometry_filename
    from ess.livedata.handlers.detector_view import (
        DetectorViewFactory,
        GeometricViewConfig,
        NeXusDetectorSource,
    )
    from ess.livedata.handlers.stream_processor_workflow import (
        StreamProcessorWorkflow,
    )
    from ess.reduce.nexus.types import NeXusData, NeXusDetectorName, SampleRun
    from ess.sans import types as sans_types
    from ess.sans.types import (
        Filename,
        Incident,
        IntensityQ,
        Numerator,
        ReducedQ,
        Transmission,
    )

    # Created once outside workflow wrappers since this configures some files from pooch
    # where a checksum is needed, which takes significant time.
    _base_workflow = loki.live._configured_Larmor_AgBeh_workflow()
    _base_workflow[Filename[SampleRun]] = get_nexus_geometry_filename('loki')

    # Sciline-based detector view with XY projection for all detector banks.
    # Resolution values = base resolution * scale (12), matching the legacy setup.
    _xy_projection = DetectorViewFactory(
        data_source=NeXusDetectorSource(get_nexus_geometry_filename('loki')),
        view_config={
            'loki_detector_0': GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'y': 144, 'x': 144},
                pixel_noise='cylindrical',
            ),
            # First window frame
            'loki_detector_1': GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'y': 36, 'x': 108},
                pixel_noise='cylindrical',
            ),
            'loki_detector_2': GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'y': 108, 'x': 36},
                pixel_noise='cylindrical',
            ),
            'loki_detector_3': GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'y': 36, 'x': 108},
                pixel_noise='cylindrical',
            ),
            'loki_detector_4': GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'y': 108, 'x': 36},
                pixel_noise='cylindrical',
            ),
            # Second window frame
            'loki_detector_5': GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'y': 36, 'x': 108},
                pixel_noise='cylindrical',
            ),
            'loki_detector_6': GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'y': 108, 'x': 36},
                pixel_noise='cylindrical',
            ),
            'loki_detector_7': GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'y': 36, 'x': 108},
                pixel_noise='cylindrical',
            ),
            'loki_detector_8': GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'y': 108, 'x': 36},
                pixel_noise='cylindrical',
            ),
        },
    )

    specs.xy_projection_handle.attach_factory()(_xy_projection.make_workflow)

    def _transmission_from_current_run(
        data: sans_types.CorrectedMonitor[SampleRun, sans_types.MonitorType],
    ) -> sans_types.CorrectedMonitor[
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
        sans_types.CorrectedMonitor[SampleRun, Incident],
        sans_types.CorrectedMonitor[SampleRun, Transmission],
    )

    @specs.i_of_q_handle.attach_factory()
    def _i_of_q_factory(source_name: str) -> StreamProcessorWorkflow:
        wf = _base_workflow.copy()
        wf[NeXusDetectorName] = source_name
        return StreamProcessorWorkflow(
            wf,
            dynamic_keys=_dynamic_keys(source_name),
            target_keys={'i_of_q': IntensityQ[SampleRun]},
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
                'i_of_q': IntensityQ[SampleRun],
                'transmission_fraction': sans_types.TransmissionFraction[SampleRun],
            }
            wf.insert(_transmission_from_current_run)
        else:
            # Transmission fraction is static, do not display
            target_keys = {'i_of_q': IntensityQ[SampleRun]}
        return StreamProcessorWorkflow(
            wf,
            dynamic_keys=_dynamic_keys(source_name),
            target_keys=target_keys,
            accumulators=_accumulators,
        )
