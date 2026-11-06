# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Bifrost spectrometer factory implementations.
"""

from functools import cache

import scipp as sc

from ess.livedata.config import Instrument
from ess.livedata.config.workflow_spec import Temporality

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
    from ess.bifrost.types import NeXusDetectorNames
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
    from ess.livedata.workflows.stream_processor_workflow import (
        StreamProcessorWorkflow,
    )

    from .streams import detector_number

    # Configure detector
    instrument.configure_detector('unified_detector', detector_number=detector_number)

    # Bifrost device streams (merged RBV/VAL/DMOV) feeding typed Sciline keys
    # on the cut-workflow graph. Direct-bind (no chain patch): the keys are
    # ordinary parameters of the cut workflow, so each job's gate follows from
    # whether its graph reaches them — the Q-maps do, the detector view (sums
    # over banks) and the ratemeter (counts-only) do not.
    instrument.offer_context_stream(
        stream_name='detector_tank_angle',
        workflow_key=InstrumentAngle[SampleRun],
    )
    instrument.offer_context_stream(
        stream_name='sample_rotation',
        workflow_key=SampleAngle[SampleRun],
    )

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
            window_outputs=specs.DetectorRatemeterOutputs.fields_with(
                Temporality.window
            ),
        )

    # Q-map workflow factories
    @cache
    def _init_q_cut_workflow() -> sciline.Pipeline:
        """Initialize a Bifrost Q-cut workflow with common parameters."""
        fname = simulated_elastic_incoherent_with_phonon()
        with snx.File(fname) as f:
            detector_names = list(f['entry/instrument'][snx.NXdetector])
            monitor_names = list(f['entry/instrument'][snx.NXmonitor])
        workflow = BifrostQCutWorkflow()
        workflow[NeXusDetectorNames] = detector_names
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
        Pipeline computing the ratemeter region counts from raw detector events.
    DetectorRegionCounts:
        Accumulation-mode-scoped type for detector region counts.
    DetectorRegionCountsRaw:
        Type for the bare per-update region counts (before accumulation).
    """
    # Lazy imports
    from typing import NewType

    import sciline
    from ess.reduce.nexus.types import EmptyDetector, RawDetector, RunType, SampleRun
    from ess.reduce.nexus.workflow import assemble_detector_data

    from ess.livedata.preprocessors.accumulation_mode import AccumulationMode

    from .streams import detector_number

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
        stamps ``start_time``/``time`` (the bounds the dashboard's rate
        normalization needs) on the accumulated outputs, per the window/cumulative
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

    # The ratemeter only counts, so the detector needs no geometry: grouping
    # events by ``detector_number`` is all ``assemble_detector_data`` does here.
    reduction_workflow = sciline.Pipeline(
        (assemble_detector_data, _detector_ratemeter, _accumulated_region_counts),
        params={
            EmptyDetector[SampleRun]: sc.DataArray(
                detector_number, coords={'detector_number': detector_number}
            )
        },
        constraints={RunType: [SampleRun]},
    )
    return reduction_workflow, DetectorRegionCounts, DetectorRegionCountsRaw
