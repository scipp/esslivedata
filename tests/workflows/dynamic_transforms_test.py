# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Behavioral tests for :func:`wire_dynamic_transforms`.

These drive a *real* Sciline pipeline: the wiring inserts the fused chain-patch
provider, the pipeline is then computed, and we assert on the patched
``NeXusTransformationChain`` — i.e. the observable effect, not the number of
providers inserted. Grouping/dedup is exercised through Sciline's own rejection
of duplicate-typed parameters, the reason the dedup exists.
"""

from __future__ import annotations

import sciline
import scipp as sc
from ess.reduce.nexus.types import (
    NeXusComponent,
    NeXusData,
    NeXusTransformationChain,
    SampleRun,
)
from ess.reduce.nexus.workflow import get_transformation_chain
from scippnexus import NXdetector
from scippnexus.field import DependsOn
from scippnexus.nxtransformations import Transform, TransformationChain

from ess.livedata.config.stream import ChainPatchBinding
from ess.livedata.config.value_log import ValueLog
from ess.livedata.workflows.dynamic_transforms import wire_dynamic_transforms

ROT_PATH = '/entry/det/transformations/rot'
TILT_PATH = '/entry/det/transformations/tilt'


class _RotLog(ValueLog):
    pass


class _TiltLog(ValueLog):
    pass


def _base_chain(
    component: NeXusComponent[NXdetector, SampleRun],
) -> NeXusTransformationChain[NXdetector, SampleRun]:
    """Concrete fallback so the chain is computable whether or not it is patched.

    Mirrors essreduce's generic ``get_transformation_chain``, pinned to a single
    component type because Sciline cannot hold the unconstrained generic. The
    chain-patch provider, when wired, replaces this on ``insert``.
    """
    return get_transformation_chain(component)


def _placeholder_transform(name: str) -> Transform:
    """A length-0 NXlog placeholder, as the geometry artifact stores it."""
    empty = sc.DataArray(
        sc.array(dims=['time'], values=[], unit='deg', dtype='float64'),
        coords={'time': sc.array(dims=['time'], values=[], unit='ns', dtype='int64')},
    )
    return Transform(
        name=name,
        transformation_type='rotation',
        value=empty,
        vector=sc.vector(value=[0, 0, 1]),
        depends_on=DependsOn(parent='/entry/det/transformations', value='.'),
    )


def _component(*paths: str) -> sc.DataGroup:
    """A NeXus component whose ``depends_on`` chain carries placeholders."""
    chain = TransformationChain(parent='/entry/det', value='transformations/rot')
    chain.transformations = sc.DataGroup(
        {path: _placeholder_transform(path) for path in paths}
    )
    return sc.DataGroup({'depends_on': chain})


def _value_log(cls: type[ValueLog], latest: float) -> ValueLog:
    """A cumulative NXlog whose final sample is ``latest`` deg."""
    return cls(
        values=sc.DataArray(
            sc.array(dims=['time'], values=[1.0, latest], unit='deg'),
            coords={
                'time': sc.array(dims=['time'], values=[0, 1], unit='ns', dtype='int64')
            },
        )
    )


class _Workflow:
    """Minimal ``SupportsDynamicTransforms`` over a real pipeline.

    Exposes only the two protocol accessors; the pipeline it wraps is a genuine
    :class:`sciline.Pipeline` so wiring is observed through real computation.
    """

    def __init__(
        self, pipeline: sciline.Pipeline, dynamic_keys: dict[str, object]
    ) -> None:
        self._pipeline = pipeline
        self._dynamic_keys = dynamic_keys

    @property
    def dynamic_keys(self) -> dict[str, object]:
        return self._dynamic_keys

    @property
    def base_pipeline(self) -> sciline.Pipeline:
        return self._pipeline


def _pipeline(*paths: str) -> sciline.Pipeline:
    pipeline = sciline.Pipeline((_base_chain,))
    pipeline[NeXusComponent[NXdetector, SampleRun]] = _component(*paths)
    return pipeline


def _computed_chain(pipeline: sciline.Pipeline) -> TransformationChain:
    return pipeline.compute(NeXusTransformationChain[NXdetector, SampleRun])


class TestWireDynamicTransforms:
    def test_patches_chain_with_latest_sample(self) -> None:
        pipeline = _pipeline(ROT_PATH)
        pipeline[_RotLog] = _value_log(_RotLog, 42.0)
        workflow = _Workflow(pipeline, {'det1': NeXusData[NXdetector, SampleRun]})
        binding = ChainPatchBinding(
            stream_name='rot',
            transform_path=ROT_PATH,
            workflow_key=_RotLog,
            dependent_sources=frozenset({'det1'}),
        )

        wire_dynamic_transforms(workflow, [binding])

        assert _computed_chain(pipeline).transformations[ROT_PATH].value.value == 42.0

    def test_unmatched_source_leaves_placeholder(self) -> None:
        pipeline = _pipeline(ROT_PATH)
        workflow = _Workflow(pipeline, {'det1': NeXusData[NXdetector, SampleRun]})
        binding = ChainPatchBinding(
            stream_name='rot',
            transform_path=ROT_PATH,
            workflow_key=_RotLog,
            dependent_sources=frozenset({'other_det'}),
        )

        wire_dynamic_transforms(workflow, [binding])

        # No provider inserted: the placeholder length-0 NXlog survives.
        value = _computed_chain(pipeline).transformations[ROT_PATH].value
        assert value.sizes == {'time': 0}

    def test_ignores_non_nexusdata_keys(self) -> None:
        pipeline = _pipeline(ROT_PATH)
        workflow = _Workflow(pipeline, {'det1': int})
        binding = ChainPatchBinding(
            stream_name='rot',
            transform_path=ROT_PATH,
            workflow_key=_RotLog,
            dependent_sources=frozenset({'det1'}),
        )

        wire_dynamic_transforms(workflow, [binding])

        value = _computed_chain(pipeline).transformations[ROT_PATH].value
        assert value.sizes == {'time': 0}

    def test_non_supporting_workflow_is_noop(self) -> None:
        binding = ChainPatchBinding(
            stream_name='rot',
            transform_path=ROT_PATH,
            workflow_key=_RotLog,
            dependent_sources=frozenset({'det1'}),
        )
        # An object without dynamic_keys/base_pipeline must not raise.
        wire_dynamic_transforms(object(), [binding])

    def test_dedups_binding_spanning_same_type_sources(self) -> None:
        # One binding, two sources of the same component type. Without dedup the
        # provider would take two identically-typed _RotLog params and Sciline
        # would reject it; dedup collapses them so the chain computes.
        pipeline = _pipeline(ROT_PATH)
        pipeline[_RotLog] = _value_log(_RotLog, 5.0)
        workflow = _Workflow(
            pipeline,
            {
                'det1': NeXusData[NXdetector, SampleRun],
                'det2': NeXusData[NXdetector, SampleRun],
            },
        )
        binding = ChainPatchBinding(
            stream_name='rot',
            transform_path=ROT_PATH,
            workflow_key=_RotLog,
            dependent_sources=frozenset({'det1', 'det2'}),
        )

        wire_dynamic_transforms(workflow, [binding])

        assert _computed_chain(pipeline).transformations[ROT_PATH].value.value == 5.0

    def test_fuses_multiple_bindings_for_one_component_type(self) -> None:
        # Two distinct streams patching two transforms on the same component:
        # one fused provider with two differently-typed params, both applied.
        pipeline = _pipeline(ROT_PATH, TILT_PATH)
        pipeline[_RotLog] = _value_log(_RotLog, 11.0)
        pipeline[_TiltLog] = _value_log(_TiltLog, 22.0)
        workflow = _Workflow(pipeline, {'det1': NeXusData[NXdetector, SampleRun]})
        bindings = [
            ChainPatchBinding(
                stream_name='rot',
                transform_path=ROT_PATH,
                workflow_key=_RotLog,
                dependent_sources=frozenset({'det1'}),
            ),
            ChainPatchBinding(
                stream_name='tilt',
                transform_path=TILT_PATH,
                workflow_key=_TiltLog,
                dependent_sources=frozenset({'det1'}),
            ),
        ]

        wire_dynamic_transforms(workflow, bindings)

        chain = _computed_chain(pipeline)
        assert chain.transformations[ROT_PATH].value.value == 11.0
        assert chain.transformations[TILT_PATH].value.value == 22.0
