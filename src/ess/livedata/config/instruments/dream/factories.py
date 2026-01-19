# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
DREAM instrument factory implementations.
"""

from typing import NewType

import scipp as sc

from ess.livedata.config import Instrument

from . import specs
from .specs import PowderWorkflowParams


def setup_factories(instrument: Instrument) -> None:
    """Initialize DREAM-specific factories and workflows."""
    # Lazy imports - all expensive imports go inside the function
    from scippnexus import NXdetector

    import ess.powder.types  # noqa: F401
    from ess import dream, powder
    from ess.dream import DreamPowderWorkflow
    from ess.livedata.handlers.detector_data_handler import (
        DetectorProjection,
        get_nexus_geometry_filename,
    )
    from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow
    from ess.reduce.nexus.types import (
        Filename,
        NeXusData,
        NeXusName,
        RawDetector,
        RunType,
        SampleRun,
        VanadiumRun,
    )

    # Unified detector projection with per-detector projection types.
    # We use the arc length instead of phi for mantle as it makes it easier to get
    # a correct aspect ratio for the plot if both axes have the same unit.
    # Order in 'resolution' matters so plots have X as horizontal axis and Y vertical.
    _detector_projection = DetectorProjection(
        instrument=instrument,
        projection=specs._projections,
        pixel_noise=sc.scalar(4.0, unit='mm'),
        resolution={
            'mantle_detector': {'arc_length': 10, 'z': 40},
            'endcap_backward_detector': {'y': 30, 'x': 20},
            'endcap_forward_detector': {'y': 20, 'x': 20},
            'high_resolution_detector': {'y': 20, 'x': 20},
            'sans_detector': {'y': 20, 'x': 20},
        },
        resolution_scale=8,
    )

    # Attach unified detector view factory
    specs.projection_handle.attach_factory()(_detector_projection.make_view)

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
        fake_charge.unit = 'counts/µAh'
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

    @specs.powder_reduction_handle.attach_factory()
    def _powder_workflow_factory(source_name: str, params: PowderWorkflowParams):
        """Factory for DREAM powder reduction workflow."""
        wf = _reduction_workflow.copy()
        wf[NeXusName[NXdetector]] = source_name
        # Convert string to enum
        wf[dream.InstrumentConfiguration] = getattr(
            dream.InstrumentConfiguration, params.instrument_configuration.value
        )
        wmin = params.wavelength_range.get_start()
        wmax = params.wavelength_range.get_stop()
        wf[powder.types.WavelengthMask] = lambda w: (w < wmin) | (w > wmax)
        wf[powder.types.TwoThetaBins] = params.two_theta_edges.get_edges()
        wf[powder.types.DspacingBins] = params.dspacing_edges.get_edges()
        return StreamProcessorWorkflow(
            wf,
            dynamic_keys={
                source_name: NeXusData[NXdetector, SampleRun],
                'cave_monitor': NeXusData[powder.types.CaveMonitor, SampleRun],
            },
            target_keys={
                'focussed_data_dspacing': powder.types.FocussedDataDspacing[SampleRun],
                'focussed_data_dspacing_two_theta': (
                    powder.types.FocussedDataDspacingTwoTheta[SampleRun]
                ),
            },
            accumulators=(
                powder.types.CorrectedDspacing[SampleRun],
                powder.types.WavelengthMonitor[SampleRun, powder.types.CaveMonitor],
            ),
        )

    @specs.powder_reduction_with_vanadium_handle.attach_factory()
    def _powder_workflow_with_vanadium_factory(
        source_name: str, params: PowderWorkflowParams
    ):
        """Factory for DREAM powder reduction workflow with vanadium normalization."""
        wf = _reduction_workflow.copy()
        wf[NeXusName[NXdetector]] = source_name
        wf[Filename[VanadiumRun]] = (
            '268227_00024779_Vana_inc_BC_offset_240_deg_wlgth.hdf'
        )
        # Convert string to enum
        wf[dream.InstrumentConfiguration] = getattr(
            dream.InstrumentConfiguration, params.instrument_configuration.value
        )
        wmin = params.wavelength_range.get_start()
        wmax = params.wavelength_range.get_stop()
        wf[powder.types.WavelengthMask] = lambda w: (w < wmin) | (w > wmax)
        wf[powder.types.TwoThetaBins] = params.two_theta_edges.get_edges()
        wf[powder.types.DspacingBins] = params.dspacing_edges.get_edges()
        return StreamProcessorWorkflow(
            wf,
            dynamic_keys={
                source_name: NeXusData[NXdetector, SampleRun],
                'cave_monitor': NeXusData[powder.types.CaveMonitor, SampleRun],
            },
            target_keys={
                'focussed_data_dspacing': powder.types.FocussedDataDspacing[SampleRun],
                'focussed_data_dspacing_two_theta': (
                    powder.types.FocussedDataDspacingTwoTheta[SampleRun]
                ),
                'i_of_dspacing': powder.types.IntensityDspacing[SampleRun],
                'i_of_dspacing_two_theta': powder.types.IntensityDspacingTwoTheta[
                    SampleRun
                ],
            },
            accumulators=(
                powder.types.CorrectedDspacing[SampleRun],
                powder.types.WavelengthMonitor[SampleRun, powder.types.CaveMonitor],
            ),
        )

    # Sciline-based detector view workflow
    from ess.dream.workflows import _get_lookup_table_filename_from_configuration
    from ess.livedata.handlers.detector_view import (
        DetectorViewFactory,
        GeometricViewConfig,
        NeXusDetectorSource,
    )

    from .specs import DreamDetectorViewParams

    # Per-detector view configuration matching the legacy DetectorProjection setup.
    # Resolution values = base resolution * scale (8), matching _detector_projection
    # above. Pixel noise is shared across all detectors.
    _pixel_noise = sc.scalar(4.0, unit='mm')
    _sciline_detector_view = DetectorViewFactory(
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
        },
    )

    @specs.sciline_detector_view_handle.attach_factory()
    def _sciline_detector_view_factory(
        source_name: str, params: DreamDetectorViewParams
    ) -> StreamProcessorWorkflow:
        """Factory for Sciline-based detector view workflow."""
        # Resolve lookup table filename from DREAM-specific params for TOF modes
        tof_lookup_table_filename = None
        if params.coordinate_mode.mode in ('tof', 'wavelength'):
            # Convert enum to DREAM InstrumentConfiguration and get filename
            config = getattr(
                dream.InstrumentConfiguration,
                params.chopper_settings.configuration.value,
            )
            tof_lookup_table_filename = _get_lookup_table_filename_from_configuration(
                config
            )

        return _sciline_detector_view.make_workflow(
            source_name, params, tof_lookup_table_filename=tof_lookup_table_filename
        )
