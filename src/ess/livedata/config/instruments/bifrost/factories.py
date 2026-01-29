# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Bifrost spectrometer factory implementations.
"""

from functools import cache

import scipp as sc

from ess.livedata.config import Instrument

from . import specs
from .specs import (
    ArcEnergy,
    BifrostCustomElasticQMapParams,
    BifrostElasticQMapParams,
    BifrostQMapParams,
    BifrostWorkflowParams,
    DetectorRatemeterParams,
    DetectorRatemeterRegionParams,
)

# Q-vector basis for Q-map calculations
_Q_VECTORS = {
    'Qx': sc.vector([1, 0, 0]),
    'Qy': sc.vector([0, 1, 0]),
    'Qz': sc.vector([0, 0, 1]),
}


def setup_factories(instrument: Instrument) -> None:
    """Initialize BIFROST-specific factories and workflows."""
    # Lazy imports
    import sciline
    import scippnexus as snx
    from scippnexus import NXdetector

    from ess.bifrost.data import (
        simulated_elastic_incoherent_with_phonon,
        tof_lookup_table_simulation,
    )
    from ess.bifrost.live import (
        BifrostQCutWorkflow,
        CutAxis,
        CutAxis1,
        CutAxis2,
        CutData,
    )
    from ess.livedata.handlers.accumulators import LatestValue
    from ess.livedata.handlers.detector_view import (
        DetectorViewFactory,
        InstrumentDetectorSource,
        LogicalViewConfig,
    )
    from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow
    from ess.reduce.nexus.types import (
        Filename,
        NeXusData,
        SampleRun,
    )
    from ess.reduce.streaming import EternalAccumulator
    from ess.spectroscopy.types import (
        InstrumentAngle,
        PreopenNeXusFile,
        ProtonCharge,
        SampleAngle,
        TimeOfFlightLookupTableFilename,
    )

    from .streams import detector_number

    # Configure detector
    instrument.configure_detector('unified_detector', detector_number=detector_number)

    # Create detector view using Sciline-based factory with transform
    _detector_view_factory = DetectorViewFactory(
        data_source=InstrumentDetectorSource(instrument),
        view_config=LogicalViewConfig(transform=_to_flat_detector_view),
    )

    specs.unified_detector_view_handle.attach_factory()(
        _detector_view_factory.make_workflow
    )

    # Create base reduction workflow
    (
        reduction_workflow,
        DetectorRegionCounts,
        detector_ratemeter,
        SpectrumView,
        SpectrumViewTimeBins,
        SpectrumViewPixelsPerTube,
        make_spectrum_view,
    ) = _create_base_reduction_workflow()

    # Configure workflow with default parameters
    reduction_workflow[SpectrumViewTimeBins] = 500
    reduction_workflow[SpectrumViewPixelsPerTube] = 10
    reduction_workflow.insert(make_spectrum_view)
    reduction_workflow.insert(detector_ratemeter)

    @specs.spectrum_view_handle.attach_factory()
    def _spectrum_view_workflow(
        params: BifrostWorkflowParams,
    ) -> StreamProcessorWorkflow:
        wf = reduction_workflow.copy()
        view_params = params.spectrum_view
        wf[SpectrumViewTimeBins] = view_params.time_bins
        wf[SpectrumViewPixelsPerTube] = view_params.pixels_per_tube
        return StreamProcessorWorkflow(
            wf,
            dynamic_keys={'unified_detector': NeXusData[NXdetector, SampleRun]},
            target_keys={'spectrum_view': SpectrumView},
            accumulators={SpectrumView: EternalAccumulator},
        )

    @specs.detector_ratemeter_handle.attach_factory()
    def _detector_ratemeter_workflow(
        params: DetectorRatemeterParams,
    ) -> StreamProcessorWorkflow:
        wf = reduction_workflow.copy()
        wf[DetectorRatemeterRegionParams] = params.region
        return StreamProcessorWorkflow(
            wf,
            dynamic_keys={'unified_detector': NeXusData[NXdetector, SampleRun]},
            target_keys={'detector_region_counts': DetectorRegionCounts},
            accumulators={DetectorRegionCounts: LatestValue},
        )

    # Q-map workflow factories
    @cache
    def _init_q_cut_workflow() -> sciline.Pipeline:
        """Initialize a Bifrost Q-cut workflow with common parameters."""
        fname = simulated_elastic_incoherent_with_phonon()
        with snx.File(fname) as f:
            detector_names = list(f['entry/instrument'][snx.NXdetector])
        workflow = BifrostQCutWorkflow(detector_names)
        workflow[Filename[SampleRun]] = fname
        workflow[TimeOfFlightLookupTableFilename] = tof_lookup_table_simulation()
        workflow[PreopenNeXusFile] = PreopenNeXusFile(True)
        # ProtonCharge is not used in streaming normalization, set to 1
        workflow[ProtonCharge[SampleRun]] = sc.scalar(1.0, unit='microampere*hour')
        return workflow

    def _get_q_cut_workflow() -> sciline.Pipeline:
        return _init_q_cut_workflow().copy()

    def _make_cut_stream_processor(
        workflow: sciline.Pipeline,
    ) -> StreamProcessorWorkflow:
        return StreamProcessorWorkflow(
            workflow,
            dynamic_keys={'unified_detector': NeXusData[NXdetector, SampleRun]},
            context_keys={
                'detector_rotation': InstrumentAngle[SampleRun],
                'sample_rotation': SampleAngle[SampleRun],
            },
            target_keys={'cut_data': CutData[SampleRun]},
            accumulators=(CutData[SampleRun],),
        )

    @specs.qmap_handle.attach_factory()
    def _qmap_workflow(params: BifrostQMapParams) -> StreamProcessorWorkflow:
        wf = _get_q_cut_workflow()
        q_bins = params.q_edges.get_edges()
        q_cut = CutAxis(
            output=q_bins.dim,
            fn=lambda sample_table_momentum_transfer: sc.norm(
                x=sample_table_momentum_transfer
            ),
            bins=q_bins,
        )
        energy_bins = params.energy_edges.get_edges()
        energy_cut = CutAxis(
            output=energy_bins.dim,
            fn=lambda energy_transfer: energy_transfer,
            bins=energy_bins,
        )
        wf[CutAxis1] = q_cut
        wf[CutAxis2] = energy_cut
        return _make_cut_stream_processor(wf)

    @specs.elastic_qmap_handle.attach_factory()
    def _elastic_qmap_workflow(
        params: BifrostElasticQMapParams,
    ) -> StreamProcessorWorkflow:
        wf = _get_q_cut_workflow()
        # Convert axis1 to CutAxis
        vec1 = _Q_VECTORS[params.axis1.axis.value]
        dim1 = params.axis1.axis.value
        edges1 = params.axis1.get_edges().rename(Q=dim1)
        axis1 = CutAxis.from_q_vector(output=dim1, vec=vec1, bins=edges1)
        # Convert axis2 to CutAxis
        vec2 = _Q_VECTORS[params.axis2.axis.value]
        dim2 = params.axis2.axis.value
        edges2 = params.axis2.get_edges().rename(Q=dim2)
        axis2 = CutAxis.from_q_vector(output=dim2, vec=vec2, bins=edges2)
        wf[CutAxis1] = axis1
        wf[CutAxis2] = axis2
        return _make_cut_stream_processor(wf)

    @specs.elastic_qmap_custom_handle.attach_factory()
    def _custom_elastic_qmap_workflow(
        params: BifrostCustomElasticQMapParams,
    ) -> StreamProcessorWorkflow:
        wf = _get_q_cut_workflow()
        # Convert axis1 to CutAxis
        components1 = [params.axis1.qx, params.axis1.qy, params.axis1.qz]
        vec1 = sc.vector(components1)
        name1 = f'Q_({components1[0]},{components1[1]},{components1[2]})'
        edges1 = params.axis1_edges.get_edges().rename(Q=name1)
        axis1 = CutAxis.from_q_vector(output=name1, vec=vec1, bins=edges1)
        # Convert axis2 to CutAxis
        components2 = [params.axis2.qx, params.axis2.qy, params.axis2.qz]
        vec2 = sc.vector(components2)
        name2 = f'Q_({components2[0]},{components2[1]},{components2[2]})'
        edges2 = params.axis2_edges.get_edges().rename(Q=name2)
        axis2 = CutAxis.from_q_vector(output=name2, vec=vec2, bins=edges2)
        wf[CutAxis1] = axis1
        wf[CutAxis2] = axis2
        return _make_cut_stream_processor(wf)


def _transpose_with_coords(data: sc.DataArray, dims: tuple[str, ...]) -> sc.DataArray:
    """
    Transpose data array and all its coordinates.

    Unlike scipp.DataArray.transpose, this function also transposes all coordinates
    that have more than one dimension. Each coordinate is transposed to match the
    order of dimensions specified in `dims`, considering only the intersection of
    the coordinate's dimensions with `dims`.

    Parameters
    ----------
    data:
        Data array to transpose.
    dims:
        Target dimension order.

    Returns
    -------
    :
        Transposed data array with transposed coordinates.
    """
    result = data.transpose(dims)
    # Transpose all multi-dimensional coordinates
    for name, coord in data.coords.items():
        if coord.ndim > 1:
            # Only transpose dimensions that exist in both the coord and target dims
            coord_dims = coord.dims
            ordered_dims = tuple(d for d in dims if d in coord_dims)
            result.coords[name] = coord.transpose(ordered_dims)
    return result


def _combine_banks(*bank: sc.DataArray) -> sc.DataArray:
    """
    Combine Bifrost banks into a single detector data array.

    Each NXdetetor is a He3 tube triplet with shape=(3, 100). Detector numbers in
    triplet are *not* consecutive:
    - 1...900 with increasing angle (across all sectors)
    - 901 is back to first sector and detector, second tube
    """
    combined = (
        sc.concat(bank, dim='')
        .fold('', sizes={'arc': 5, 'channel': 9})
        .rename_dims(dim_0='tube', dim_1='pixel')
    )
    # Order with consecutive detector_number
    return _transpose_with_coords(combined, ('arc', 'tube', 'channel', 'pixel')).copy()


def _to_flat_detector_view(
    obj: sc.Variable | sc.DataArray, source_name: str
) -> sc.DataArray:
    da = sc.DataArray(obj) if isinstance(obj, sc.Variable) else obj
    da = da.to(dtype='float32')
    return da.flatten(dims=('arc', 'tube'), to='arc/tube').flatten(
        dims=('channel', 'pixel'), to='channel/pixel'
    )


def _create_base_reduction_workflow():
    """
    Create the base Bifrost reduction workflow with all helper types and functions.

    This function is called by setup_factories to create the reduction workflow
    used in production, and can also be called by tests to access the same workflow.

    All imports are lazy to avoid requiring heavy dependencies when this module
    is imported.

    Returns
    -------
    reduction_workflow:
        The base TofWorkflow configured with detector geometry.
    DetectorRegionCounts:
        NewType for detector region counts.
    detector_ratemeter:
        Function that computes detector region counts.
    SpectrumView:
        NewType for spectrum view data.
    SpectrumViewTimeBins:
        NewType for spectrum view time bins parameter.
    SpectrumViewPixelsPerTube:
        NewType for spectrum view pixels per tube parameter.
    make_spectrum_view:
        Function that creates spectrum view from detector data.
    """
    # Lazy imports
    from typing import NewType

    from scippnexus import NXdetector

    from ess.livedata.handlers.detector_data_handler import get_nexus_geometry_filename
    from ess.reduce.nexus.types import (
        EmptyDetector,
        Filename,
        NeXusName,
        RawDetector,
        SampleRun,
    )
    from ess.spectroscopy.indirect.time_of_flight import TofWorkflow

    _detector_names = [
        f'{123 + 4 * (arc - 1) + (5 * 4 + 1) * (channel - 1)}'
        f'_channel_{channel}_{arc}_triplet'
        for arc in range(1, 6)
        for channel in range(1, 10)
    ]

    _arc_energy_to_index = {
        ArcEnergy.ARC_2_7: 0,
        ArcEnergy.ARC_3_2: 1,
        ArcEnergy.ARC_3_8: 2,
        ArcEnergy.ARC_4_4: 3,
        ArcEnergy.ARC_5_0: 4,
    }

    DetectorRegionCounts = NewType('DetectorRegionCounts', sc.DataArray)

    def _detector_ratemeter(
        data: RawDetector[SampleRun], region: DetectorRatemeterRegionParams
    ) -> DetectorRegionCounts:
        """Calculate detector count rate for selected arc and pixel range."""
        arc_idx = _arc_energy_to_index[region.arc]
        arc_data = data['arc', arc_idx]
        flat = arc_data.flatten(dims=('channel', 'pixel'), to='position')
        selected = flat['position', region.pixel_start : region.pixel_stop]
        counts = selected.sum()
        time = selected.bins.coords['event_time_zero'].min()
        counts.coords['time'] = time
        counts.variances = counts.values
        return DetectorRegionCounts(counts)

    SpectrumView = NewType('SpectrumView', sc.DataArray)
    SpectrumViewTimeBins = NewType('SpectrumViewTimeBins', int)
    SpectrumViewPixelsPerTube = NewType('SpectrumViewPixelsPerTube', int)

    def _make_spectrum_view(
        data: RawDetector[SampleRun],
        time_bins: SpectrumViewTimeBins,
        pixels_per_tube: SpectrumViewPixelsPerTube,
    ) -> SpectrumView:
        edges = sc.linspace(
            'event_time_offset', 0, 71_000_000, num=time_bins + 1, unit='ns'
        )
        per_arc = 3 * 900
        detector_number_step = 100 // pixels_per_tube
        detector_number = sc.arange(
            'detector_number', 0, 5 * per_arc, step=detector_number_step, unit=None
        ).fold(dim='detector_number', sizes={'arc': 5, 'detector_number': -1})
        return SpectrumView(
            data.fold('pixel', sizes={'pixel': pixels_per_tube, 'subpixel': -1})
            .drop_coords(tuple(data.coords))
            .bins.concat('subpixel')
            .flatten(dims=('tube', 'channel', 'pixel'), to='detector_number')
            .hist(event_time_offset=edges)
            .assign_coords(
                event_time_offset=edges.to(unit='ms'),
                detector_number=detector_number,
            )
        )

    reduction_workflow = TofWorkflow(run_types=(SampleRun,), monitor_types=())
    reduction_workflow[Filename[SampleRun]] = get_nexus_geometry_filename('bifrost')
    reduction_workflow[EmptyDetector[SampleRun]] = (
        reduction_workflow[EmptyDetector[SampleRun]]
        .map({NeXusName[NXdetector]: _detector_names})
        .reduce(func=_combine_banks)
    )

    return (
        reduction_workflow,
        DetectorRegionCounts,
        _detector_ratemeter,
        SpectrumView,
        SpectrumViewTimeBins,
        SpectrumViewPixelsPerTube,
        _make_spectrum_view,
    )
