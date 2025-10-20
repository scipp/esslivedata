# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Bifrost Q-map workflows."""

from enum import StrEnum
from functools import cache
from typing import Literal

import pydantic
import sciline
import scipp as sc
import scippnexus as snx
from scippnexus import NXdetector

from ess.bifrost.data import (
    simulated_elastic_incoherent_with_phonon,
    tof_lookup_table_simulation,
)
from ess.bifrost.live import BifrostQCutWorkflow, CutAxis, CutAxis1, CutAxis2, CutData
from ess.livedata.config import Instrument
from ess.livedata.config.workflow_spec import AuxSourcesBase, WorkflowOutputsBase
from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow
from ess.livedata.parameter_models import EnergyEdges, QEdges
from ess.reduce.nexus.types import Filename, NeXusData, SampleRun
from ess.spectroscopy.types import (
    InstrumentAngle,
    PreopenNeXusFile,
    SampleAngle,
    TimeOfFlightLookupTableFilename,
)


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
    return workflow


def _get_q_cut_workflow() -> sciline.Pipeline:
    return _init_q_cut_workflow().copy()


q_vectors = {
    'Qx': sc.vector([1, 0, 0]),
    'Qy': sc.vector([0, 1, 0]),
    'Qz': sc.vector([0, 0, 1]),
}

QMAX_DEFAULT = 3.0
QBIN_DEFAULT = 100


class CustomQAxis(pydantic.BaseModel):
    qx: float = pydantic.Field(
        default=0, title='Qx', description="Custom x component of the cut axis."
    )
    qy: float = pydantic.Field(
        default=0, title='Qy', description="Custom y component of the cut axis."
    )
    qz: float = pydantic.Field(
        default=0, title='Qz', description="Custom z component of the cut axis."
    )


class BifrostQMapParams(pydantic.BaseModel):
    q_edges: QEdges = pydantic.Field(
        default=QEdges(start=0.5, stop=QMAX_DEFAULT, num_bins=QBIN_DEFAULT),
        description="Q bin edges.",
    )
    energy_edges: EnergyEdges = pydantic.Field(
        default=EnergyEdges(start=-1.0, stop=1.0, num_bins=QBIN_DEFAULT),
        description="Energy transfer bin edges.",
    )

    def get_q_cut(self) -> CutAxis:
        bins = self.q_edges.get_edges()
        return CutAxis(
            output=bins.dim,
            fn=lambda sample_table_momentum_transfer: sc.norm(
                x=sample_table_momentum_transfer
            ),
            bins=bins,
        )

    def get_energy_cut(self) -> CutAxis:
        bins = self.energy_edges.get_edges()
        return CutAxis(
            output=bins.dim, fn=lambda energy_transfer: energy_transfer, bins=bins
        )


class QAxisOption(StrEnum):
    Qx = 'Qx'
    Qy = 'Qy'
    Qz = 'Qz'


class QAxisSelection(pydantic.BaseModel):
    axis: QAxisOption = pydantic.Field(description="Cut axis.")


class QAxisParams(QEdges, QAxisSelection):
    def to_cut_axis(self) -> CutAxis:
        vec = q_vectors[self.axis.value]
        dim = self.axis.value
        edges = self.get_edges().rename(Q=dim)
        return CutAxis.from_q_vector(output=dim, vec=vec, bins=edges)


class BifrostElasticQMapParams(pydantic.BaseModel):
    axis1: QAxisParams = pydantic.Field(
        default=QAxisParams(
            axis=QAxisOption.Qx,
            start=-QMAX_DEFAULT,
            stop=QMAX_DEFAULT,
            num_bins=QBIN_DEFAULT,
        ),
        description="First cut axis.",
    )
    axis2: QAxisParams = pydantic.Field(
        default=QAxisParams(
            axis=QAxisOption.Qz,
            start=-QMAX_DEFAULT,
            stop=QMAX_DEFAULT,
            num_bins=QBIN_DEFAULT,
        ),
        description="Second cut axis.",
    )


class BifrostCustomElasticQMapParams(pydantic.BaseModel):
    axis1: CustomQAxis = pydantic.Field(
        default=CustomQAxis(qx=1, qy=0, qz=0),
        description="Custom vector for the first cut axis.",
    )
    axis1_edges: QEdges = pydantic.Field(
        default=QEdges(start=-QMAX_DEFAULT, stop=QMAX_DEFAULT, num_bins=QBIN_DEFAULT),
        description="First cut axis edges.",
    )
    axis2: CustomQAxis = pydantic.Field(
        default=CustomQAxis(qx=0, qy=0, qz=1),
        description="Custom vector for the second cut axis.",
    )
    axis2_edges: QEdges = pydantic.Field(
        default=QEdges(start=-QMAX_DEFAULT, stop=QMAX_DEFAULT, num_bins=QBIN_DEFAULT),
        description="Second cut axis edges.",
    )

    def get_cut_axis1(self) -> CutAxis:
        components = [self.axis1.qx, self.axis1.qy, self.axis1.qz]
        vec = sc.vector(components)
        name = f'Q_({components[0]},{components[1]},{components[2]})'
        edges = self.axis1_edges.get_edges().rename(Q=name)
        return CutAxis.from_q_vector(output=name, vec=vec, bins=edges)

    def get_cut_axis2(self) -> CutAxis:
        components = [self.axis2.qx, self.axis2.qy, self.axis2.qz]
        vec = sc.vector(components)
        name = f'Q_({components[0]},{components[1]},{components[2]})'
        edges = self.axis2_edges.get_edges().rename(Q=name)
        return CutAxis.from_q_vector(output=name, vec=vec, bins=edges)


class BifrostAuxSources(AuxSourcesBase):
    """Auxiliary source names for Bifrost Q-map workflows."""

    detector_rotation: Literal['detector_rotation'] = pydantic.Field(
        default='detector_rotation', description='Detector bank rotation angle.'
    )
    sample_rotation: Literal['sample_rotation'] = pydantic.Field(
        default='sample_rotation', description='Sample rotation angle.'
    )


class QMapOutputs(WorkflowOutputsBase):
    """Outputs for Bifrost Q-map workflows."""

    cut_data: sc.DataArray = pydantic.Field(
        title='Cut Data',
        description='2D cut of intensity in Q-space or Q-E space.',
    )


def register_qmap_workflows(
    instrument: Instrument,
) -> None:
    @instrument.register_workflow(
        name='qmap',
        version=1,
        title='Q map',
        description='Map of scattering intensity as function of Q and energy transfer.',
        source_names=['unified_detector'],
        aux_sources=BifrostAuxSources,
        outputs=QMapOutputs,
    )
    def _qmap_workflow(params: BifrostQMapParams) -> StreamProcessorWorkflow:
        wf = _get_q_cut_workflow()
        wf[CutAxis1] = params.get_q_cut()
        wf[CutAxis2] = params.get_energy_cut()
        return _make_cut_stream_processor(wf)

    @instrument.register_workflow(
        name='elastic_qmap',
        version=1,
        title='Elastic Q map',
        description='Elastic Q map with predefined axes.',
        source_names=['unified_detector'],
        aux_sources=BifrostAuxSources,
        outputs=QMapOutputs,
    )
    def _elastic_qmap_workflow(
        params: BifrostElasticQMapParams,
    ) -> StreamProcessorWorkflow:
        wf = _get_q_cut_workflow()
        wf[CutAxis1] = params.axis1.to_cut_axis()
        wf[CutAxis2] = params.axis2.to_cut_axis()
        return _make_cut_stream_processor(wf)

    @instrument.register_workflow(
        name='elastic_qmap_custom',
        version=1,
        title='Elastic Q map (custom)',
        description='Elastic Q map with custom axes.',
        source_names=['unified_detector'],
        aux_sources=BifrostAuxSources,
        outputs=QMapOutputs,
    )
    def _custom_elastic_qmap_workflow(
        params: BifrostCustomElasticQMapParams,
    ) -> StreamProcessorWorkflow:
        wf = _get_q_cut_workflow()
        wf[CutAxis1] = params.get_cut_axis1()
        wf[CutAxis2] = params.get_cut_axis2()
        return _make_cut_stream_processor(wf)


def _make_cut_stream_processor(workflow: sciline.Pipeline) -> StreamProcessorWorkflow:
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
