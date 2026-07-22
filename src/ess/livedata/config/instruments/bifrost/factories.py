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
    from ess.bifrost.data import (
        lookup_table_simulation,
        simulated_elastic_incoherent_with_phonon,
    )
    from ess.bifrost.live import (
        BifrostQCutWorkflow,
        CutAxis,
        CutAxis1,
        CutAxis2,
        CutData,
    )
    from ess.reduce.nexus.types import (
        Filename,
        NeXusData,
        SampleRun,
    )
    from ess.reduce.uncertainty import UncertaintyBroadcastMode
    from ess.reduce.unwrap import LookupTableFilename
    from ess.reduce.unwrap.types import LookupTableRelativeErrorThreshold
    from ess.spectroscopy.types import (
        InstrumentAngle,
        PreopenNeXusFile,
        ProtonCharge,
        SampleAngle,
    )
    from scippnexus import NXdetector

    from ess.livedata.preprocessors.accumulation_mode import Cumulative, Current
    from ess.livedata.preprocessors.accumulators import make_no_copy_accumulator_pair
    from ess.livedata.preprocessors.stream_processor_workflow import (
        StreamProcessorWorkflow,
    )

    from .streams import detector_number

    # Configure detector
    instrument.configure_detector('unified_detector', detector_number=detector_number)

    # Bifrost device streams (merged RBV/VAL/DMOV) feeding typed Sciline keys
    # on the cut-workflow graph. Direct-bind (no chain patch) — declared at
    # instrument scope so every spec consuming ``unified_detector`` picks them
    # up. Non-consumers opt out below, co-located with the bindings they negate:
    # the detector view sums over banks and the ratemeter is counts-only.
    instrument.add_context_binding(
        stream_name='detector_tank_angle_r0',
        dependent_sources={'unified_detector'},
        workflow_key=InstrumentAngle[SampleRun],
    )
    instrument.add_context_binding(
        stream_name='rotation_stage',
        dependent_sources={'unified_detector'},
        workflow_key=SampleAngle[SampleRun],
    )
    specs.detector_ratemeter_handle.skip_instrument_contexts()
    specs.unified_detector_view_handle.skip_instrument_contexts()

    # Create base reduction workflow
    (
        reduction_workflow,
        DetectorRegionCounts,
        _DetectorRegionCountsRaw,
    ) = _create_base_reduction_workflow()

    @specs.detector_ratemeter_handle.attach_factory()
    def _detector_ratemeter_workflow(
        params: DetectorRatemeterParams,
    ) -> StreamProcessorWorkflow:
        wf = reduction_workflow.copy()
        wf[DetectorRatemeterRegionParams] = params.region
        cumulative, window = make_no_copy_accumulator_pair()
        return StreamProcessorWorkflow(
            wf,
            dynamic_keys={'unified_detector': NeXusData[NXdetector, SampleRun]},
            target_keys={
                'detector_region_counts': DetectorRegionCounts[Current],
                'detector_region_counts_cumulative': DetectorRegionCounts[Cumulative],
            },
            accumulators={
                DetectorRegionCounts[Cumulative]: cumulative,
                DetectorRegionCounts[Current]: window,
            },
            window_outputs=['detector_region_counts'],
        )

    # Q-map workflow factories
    @cache
    def _init_q_cut_workflow() -> sciline.Pipeline:
        """Initialize a Bifrost Q-cut workflow with common parameters."""
        fname = simulated_elastic_incoherent_with_phonon()
        with snx.File(fname) as f:
            detector_names = list(f['entry/instrument'][snx.NXdetector])
            monitor_names = list(f['entry/instrument'][snx.NXmonitor])
        workflow = BifrostQCutWorkflow(detector_names)
        workflow[Filename[SampleRun]] = fname
        workflow[LookupTableFilename] = lookup_table_simulation()
        # BifrostQCutWorkflow looks up the detector threshold under the hardcoded
        # key 'detector' (NeXusDetectorName('detector')); monitors are looked up
        # by their actual NeXus component name.
        workflow[LookupTableRelativeErrorThreshold] = {
            'detector': float('inf'),
            **{name: float('inf') for name in monitor_names},
        }
        workflow[PreopenNeXusFile] = PreopenNeXusFile(True)
        # ProtonCharge is not used in streaming normalization, set to 1. Revisit once
        # there is a established stream for this.
        workflow[ProtonCharge[SampleRun]] = sc.scalar(1.0, unit='pC')
        workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
        return workflow

    def _get_q_cut_workflow() -> sciline.Pipeline:
        return _init_q_cut_workflow().copy()

    def _make_cut_stream_processor(
        workflow: sciline.Pipeline,
    ) -> StreamProcessorWorkflow:
        return StreamProcessorWorkflow(
            workflow,
            dynamic_keys={'unified_detector': NeXusData[NXdetector, SampleRun]},
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
        The base TofWorkflow configured with detector geometry, with the
        ratemeter providers inserted.
    DetectorRegionCounts:
        Accumulation-mode-scoped type for detector region counts.
    DetectorRegionCountsRaw:
        Type for the bare per-update region counts (before accumulation).
    """
    # Lazy imports
    from typing import NewType

    import sciline
    from ess.reduce.nexus.types import (
        EmptyDetector,
        Filename,
        NeXusName,
        RawDetector,
        SampleRun,
    )
    from ess.spectroscopy.indirect.time_of_flight import TofWorkflow
    from scippnexus import NXdetector

    from ess.livedata.preprocessors.accumulation_mode import AccumulationMode
    from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename

    # Detector group names in the pinned (pre-2026-06-08) geometry artifact
    # carry a numeric prefix, e.g. ``123_channel_1_1_triplet``. Newer files
    # drop the prefix (``channel_1_1_triplet``; ``detector_number`` content is
    # identical). Switch to the unprefixed form once detector geometry is
    # unpinned (see https://github.com/scipp/esslivedata/issues/962).
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

    DetectorRegionCountsRaw = NewType('DetectorRegionCountsRaw', sc.DataArray)

    class DetectorRegionCounts(
        sciline.Scope[AccumulationMode, sc.DataArray],
        sc.DataArray,  # type: ignore[misc]
    ):
        """Region counts scalar, parametrized by accumulation mode.

        - ``DetectorRegionCounts[Cumulative]``: accumulated since the start of the
          run (reset only on a run transition).
        - ``DetectorRegionCounts[Current]``: the latest update interval only
          (cleared after each finalize).
        """

    def _detector_ratemeter(
        data: RawDetector[SampleRun], region: DetectorRatemeterRegionParams
    ) -> DetectorRegionCountsRaw:
        """Sum detector counts in the selected arc and pixel range for one update.

        Emits bare per-update counts with no time coords: ``StreamProcessorWorkflow``
        stamps ``start_time``/``end_time`` (needed for the dashboard's rate
        normalization) on the accumulated outputs, per the window/cumulative
        convention. Stamping here instead would suppress that (see
        ``_add_time_coords``) and break coord matching across the accumulator sum.
        """
        arc_idx = _arc_energy_to_index[region.arc]
        arc_data = data['arc', arc_idx]
        flat = arc_data.flatten(dims=('channel', 'pixel'), to='position')
        selected = flat['position', region.pixel_start : region.pixel_stop]
        counts = selected.sum()
        counts.variances = counts.values
        return DetectorRegionCountsRaw(counts)

    def _accumulated_region_counts(
        counts: DetectorRegionCountsRaw,
    ) -> DetectorRegionCounts[AccumulationMode]:
        """Route per-update counts to the accumulation-mode-specific accumulator."""
        return DetectorRegionCounts[AccumulationMode](counts)

    reduction_workflow = TofWorkflow(run_types=(SampleRun,), monitor_types=())
    # Pin detector geometry to the pre-2026-06-08 survey. The 2026-06-08 file
    # (added for the corrected chopper geometry, and the default the chopper
    # factory uses) has a broken ``detector_tank_angle`` depends_on chain:
    # missing ``_t0`` offset, stale ``117_`` prefix. Drop this pin once a file
    # valid for both detectors and choppers is available
    # (https://github.com/scipp/esslivedata/issues/962).
    reduction_workflow[Filename[SampleRun]] = get_nexus_geometry_filename(
        'bifrost', date=sc.datetime('2025-06-01T00:00:00')
    )
    reduction_workflow[EmptyDetector[SampleRun]] = (
        reduction_workflow[EmptyDetector[SampleRun]]
        .map({NeXusName[NXdetector]: _detector_names})
        .reduce(func=_combine_banks)
    )

    reduction_workflow.insert(_detector_ratemeter)
    reduction_workflow.insert(_accumulated_region_counts)
    return reduction_workflow, DetectorRegionCounts, DetectorRegionCountsRaw
