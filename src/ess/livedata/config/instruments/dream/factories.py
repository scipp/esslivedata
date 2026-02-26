# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
DREAM instrument factory implementations.
"""

from typing import NewType

import scipp as sc

from ess.livedata.config import Instrument

from . import specs
from .specs import DreamMonitorDataParams, PowderWorkflowParams


def setup_factories(instrument: Instrument) -> None:
    """Initialize DREAM-specific factories and workflows."""
    # Lazy imports - all expensive imports go inside the function
    import ess.powder.types  # noqa: F401
    from ess.dream import DreamPowderWorkflow
    from ess.reduce.nexus.types import (
        Filename,
        NeXusData,
        NeXusName,
        RawDetector,
        RunType,
        SampleRun,
        VanadiumRun,
    )
    from scippnexus import NXdetector

    from ess import dream, powder
    from ess.livedata.handlers.detector_data_handler import get_nexus_geometry_filename
    from ess.livedata.handlers.detector_view import (
        DetectorViewFactory,
        GeometricViewConfig,
        NeXusDetectorSource,
    )
    from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow

    from .specs import DreamDetectorViewParams

    def _resolve_tof_lookup_table_filename(instrument_configuration):
        """Resolve TOF lookup table filename from DREAM instrument configuration."""
        from ess.dream.workflows import _get_lookup_table_filename_from_configuration

        config = getattr(
            dream.InstrumentConfiguration,
            instrument_configuration.value.value,
        )
        return _get_lookup_table_filename_from_configuration(config)

    # Sciline-based detector view workflow with per-detector geometric projections.
    # Resolution values = base resolution * scale (8), matching the legacy setup.
    # Pixel noise is shared across all detectors.
    _pixel_noise = sc.scalar(4.0, unit='mm')
    _detector_view_factory = DetectorViewFactory(
        data_source=NeXusDetectorSource(get_nexus_geometry_filename('dream-no-shape')),
        view_config={
            'mantle_detector': GeometricViewConfig(
                projection_type='cylinder_mantle_z',
                resolution={'arc_length': 80, 'z': 320},
                pixel_noise=_pixel_noise,
            ),
            'endcap_backward_detector': GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'y': 240, 'x': 160},
                pixel_noise=_pixel_noise,
            ),
            'endcap_forward_detector': GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'y': 160, 'x': 160},
                pixel_noise=_pixel_noise,
            ),
            'high_resolution_detector': GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'y': 160, 'x': 160},
                pixel_noise=_pixel_noise,
            ),
            'sans_detector': GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'y': 160, 'x': 160},
                pixel_noise=_pixel_noise,
            ),
        },
    )

    @specs.projection_handle.attach_factory()
    def _detector_view_workflow_factory(
        source_name: str, params: DreamDetectorViewParams
    ) -> StreamProcessorWorkflow:
        """Factory for Sciline-based detector view workflow."""
        tof_lookup_table_filename = None
        if params.coordinate_mode.mode in ('tof', 'wavelength'):
            tof_lookup_table_filename = _resolve_tof_lookup_table_filename(
                params.instrument_configuration
            )

        return _detector_view_factory.make_workflow(
            source_name, params, tof_lookup_table_filename=tof_lookup_table_filename
        )

    # Monitor workflow factory with DREAM-specific TOF configuration
    from ess.livedata.handlers.monitor_workflow import create_monitor_workflow

    @specs.monitor_handle.attach_factory()
    def _monitor_workflow_factory(source_name: str, params: DreamMonitorDataParams):
        """Factory for DREAM monitor workflow with TOF lookup table support."""
        mode = params.coordinate_mode.mode
        if mode == 'wavelength':
            raise NotImplementedError(
                "wavelength mode not yet implemented for monitors"
            )

        # monitor_bunker is only 6.62 m from the source, which is outside the
        # DREAM TOF lookup table range (59.85-80.15 m). Only monitor_cave
        # (Ltotal 72.33 m) is compatible with TOF mode.
        if mode == 'tof' and source_name == 'monitor_bunker':
            raise ValueError(
                "TOF mode is not supported for 'monitor_bunker'. "
                "The bunker monitor's flight path (6.62 m) is outside the "
                "DREAM TOF lookup table range (59.85-80.15 m). "
                "Use TOA mode for bunker monitor, or select 'monitor_cave' "
                "for TOF mode."
            )

        tof_lookup_table_filename = None
        geometry_filename = None

        if mode == 'tof':
            tof_lookup_table_filename = _resolve_tof_lookup_table_filename(
                params.instrument_configuration
            )
            geometry_filename = get_nexus_geometry_filename('dream-no-shape')

        return create_monitor_workflow(
            source_name=source_name,
            edges=params.get_active_edges(),
            range_filter=params.get_active_range(),
            coordinate_mode=mode,
            tof_lookup_table_filename=tof_lookup_table_filename,
            geometry_filename=geometry_filename,
        )

    # Powder reduction workflow setup
    # Normalization to monitors is partially broken due to some wavelength-range check
    # in essdiffraction that does not play with TOA-TOF conversion (I think).
    _reduction_workflow = DreamPowderWorkflow(
        run_norm=powder.RunNormalization.proton_charge
    )

    TotalCounts = NewType('TotalCounts', sc.DataArray)

    def _total_counts(data: RawDetector[SampleRun]) -> TotalCounts:
        """Dummy provider for some plottable result of total counts."""
        return TotalCounts(
            data.nanhist(
                event_time_offset=sc.linspace(
                    'event_time_offset', 0, 71_000_000, num=1000, unit='ns'
                ),
                dim=data.dims,
            )
        )

    def _fake_proton_charge(
        data: powder.types.CorrectedDspacing[RunType],
    ) -> powder.types.AccumulatedProtonCharge[RunType]:
        """
        Fake approximate proton charge for consistent normalization.

        This is not meant for production, but as a workaround until monitor
        normalization is fixed and/or we have setup a proton-charge stream.
        """
        fake_charge = sc.values(data.data).sum()
        return powder.types.AccumulatedProtonCharge[RunType](fake_charge)

    _reduction_workflow.insert(_total_counts)
    _reduction_workflow.insert(_fake_proton_charge)
    _reduction_workflow[powder.types.CalibrationData] = None
    _reduction_workflow = powder.with_pixel_mask_filenames(_reduction_workflow, [])
    _reduction_workflow[powder.types.UncertaintyBroadcastMode] = (
        powder.types.UncertaintyBroadcastMode.drop
    )
    _reduction_workflow[powder.types.KeepEvents[SampleRun]] = powder.types.KeepEvents[
        SampleRun
    ](False)

    # dream-no-shape is a much smaller file without pixel_shape, which is not needed
    # for data reduction.
    _reduction_workflow[Filename[SampleRun]] = get_nexus_geometry_filename(
        'dream-no-shape'
    )

    def _configure_powder_workflow(source_name: str, params: PowderWorkflowParams):
        """Configure common powder workflow settings."""
        wf = _reduction_workflow.copy()
        wf[NeXusName[NXdetector]] = source_name
        wf[dream.InstrumentConfiguration] = getattr(
            dream.InstrumentConfiguration, params.instrument_configuration.value
        )
        wmin = params.wavelength_range.get_start()
        wmax = params.wavelength_range.get_stop()
        wf[powder.types.WavelengthMask] = lambda w: (w < wmin) | (w > wmax)
        wf[powder.types.TwoThetaBins] = params.two_theta_edges.get_edges()
        wf[powder.types.DspacingBins] = params.dspacing_edges.get_edges()
        return wf

    def _powder_dynamic_keys(source_name: str):
        return {
            source_name: NeXusData[NXdetector, SampleRun],
            'cave_monitor': NeXusData[powder.types.CaveMonitor, SampleRun],
        }

    _powder_accumulators = (
        powder.types.CorrectedDspacing[SampleRun],
        powder.types.WavelengthMonitor[SampleRun, powder.types.CaveMonitor],
    )

    _focussed_target_keys = {
        'focussed_data_dspacing': powder.types.FocussedDataDspacing[SampleRun],
        'focussed_data_dspacing_two_theta': (
            powder.types.FocussedDataDspacingTwoTheta[SampleRun]
        ),
    }

    @specs.powder_reduction_handle.attach_factory()
    def _powder_workflow_factory(source_name: str, params: PowderWorkflowParams):
        """Factory for DREAM powder reduction workflow."""
        return StreamProcessorWorkflow(
            _configure_powder_workflow(source_name, params),
            dynamic_keys=_powder_dynamic_keys(source_name),
            target_keys=_focussed_target_keys,
            accumulators=_powder_accumulators,
        )

    @specs.powder_reduction_with_vanadium_handle.attach_factory()
    def _powder_workflow_with_vanadium_factory(
        source_name: str, params: PowderWorkflowParams
    ):
        """Factory for DREAM powder reduction workflow with vanadium normalization."""
        wf = _configure_powder_workflow(source_name, params)
        wf[Filename[VanadiumRun]] = (
            '268227_00024779_Vana_inc_BC_offset_240_deg_wlgth.hdf'
        )
        return StreamProcessorWorkflow(
            wf,
            dynamic_keys=_powder_dynamic_keys(source_name),
            target_keys={
                **_focussed_target_keys,
                'i_of_dspacing': powder.types.IntensityDspacing[SampleRun],
                'i_of_dspacing_two_theta': powder.types.IntensityDspacingTwoTheta[
                    SampleRun
                ],
            },
            accumulators=_powder_accumulators,
        )
