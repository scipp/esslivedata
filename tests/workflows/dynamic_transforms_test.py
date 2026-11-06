# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Behavioral tests for live transforms in :class:`StreamProcessorWorkflow`.

These drive a *real* essreduce pipeline through the workflow's public interface:
which streams a built job requests, and what stream updates do to the positions it
computes.
"""

from __future__ import annotations

from typing import NewType

import pytest
import sciline
import scipp as sc
from ess.reduce.nexus.types import (
    NeXusComponent,
    NeXusData,
    NeXusMultiComponent,
    NeXusMultiTransformation,
    NeXusTransformation,
    SampleRun,
)
from ess.reduce.nexus.workflow import GenericNeXusWorkflow
from scippnexus import NXdetector, NXsample
from scippnexus.field import DependsOn
from scippnexus.nxtransformations import Transform, TransformationChain

from ess.livedata.core.timestamp import Timestamp
from ess.livedata.workflows.stream_processor_workflow import StreamProcessorWorkflow

ROT_PATH = '/entry/det/transformations/rot'
TILT_PATH = '/entry/det/transformations/tilt'
STREAMS = {ROT_PATH: 'rot', TILT_PATH: 'tilt'}
Z = sc.vector([0.0, 0.0, 1.0])
POINT = sc.vector([1.0, 0.0, 0.0], unit='m')

_Both = NewType('_Both', sc.DataGroup)
_Angle = NewType('_Angle', sc.DataArray)


def _log(*values: float) -> sc.DataArray:
    """An NXlog, as the geometry artifact (empty) or a stream (not empty) holds it."""
    return sc.DataArray(
        sc.array(dims=['time'], values=list(values), unit='deg', dtype='float64'),
        coords={
            'time': sc.array(
                dims=['time'], values=list(range(len(values))), unit='ns', dtype='int64'
            )
        },
    )


def _component(**logs: sc.DataArray) -> sc.DataGroup:
    """A NeXus component with a chain through ``logs``, by transform name."""
    names = list(logs)
    chain = TransformationChain(
        parent='/entry/det', value=f'transformations/{names[0]}'
    )
    chain.transformations = sc.DataGroup(
        {
            f'/entry/det/transformations/{name}': Transform(
                name=name,
                transformation_type='rotation',
                value=log,
                vector=Z,
                depends_on=DependsOn(
                    parent='/entry/det/transformations',
                    value=names[i + 1] if i + 1 < len(names) else '.',
                ),
            )
            for i, (name, log) in enumerate(logs.items())
        }
    )
    return sc.DataGroup({'depends_on': chain})


def _pipeline(**logs: sc.DataArray) -> sciline.Pipeline:
    pipeline = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])
    pipeline[NeXusComponent[NXdetector, SampleRun]] = _component(**logs)
    return pipeline


def _build(
    pipeline: sciline.Pipeline,
    target: object = NeXusTransformation[NXdetector, SampleRun],
    **build_args: object,
) -> StreamProcessorWorkflow:
    workflow = StreamProcessorWorkflow(
        pipeline,
        dynamic_keys={},
        target_keys={'out': target},
        accumulators=(),
    )
    workflow.build(**{'transform_streams': STREAMS, **build_args})
    return workflow


def _update(workflow: StreamProcessorWorkflow, **logs: sc.DataArray) -> sc.Variable:
    """Send stream updates and return the rotated :data:`POINT`."""
    workflow.accumulate(
        logs, start_time=Timestamp.from_ns(0), end_time=Timestamp.from_ns(1)
    )
    return workflow.finalize()['out'].value * POINT


def _rotated(angle: float) -> sc.Variable:
    rotation = sc.spatial.rotations_from_rotvecs(sc.scalar(angle, unit='deg') * Z)
    return rotation * POINT


def _assert_close(actual: sc.Variable, expected: sc.Variable) -> None:
    assert sc.allclose(actual, expected, atol=sc.scalar(1e-12, unit='m'))


class TestRequestedStreams:
    def test_requests_the_stream_of_each_placeholder_on_a_chain_it_reads(self) -> None:
        workflow = _build(_pipeline(rot=_log(), tilt=_log()))

        assert workflow.requested_context_streams == {'rot', 'tilt'}

    def test_ignores_placeholders_on_a_chain_its_targets_do_not_read(self) -> None:
        pipeline = _pipeline(rot=_log())
        pipeline[NeXusComponent[NXsample, SampleRun]] = _component(tilt=_log())

        workflow = _build(pipeline)

        assert workflow.requested_context_streams == {'rot'}

    def test_requests_for_each_chain_its_targets_read(self) -> None:
        pipeline = _pipeline(rot=_log())
        pipeline[NeXusComponent[NXsample, SampleRun]] = _component(tilt=_log())

        def both(
            detector: NeXusTransformation[NXdetector, SampleRun],
            sample: NeXusTransformation[NXsample, SampleRun],
        ) -> _Both:
            return _Both(sc.DataGroup())

        pipeline.insert(both)
        workflow = _build(pipeline, target=_Both)

        assert workflow.requested_context_streams == {'rot', 'tilt'}

    def test_a_log_with_values_needs_no_stream(self) -> None:
        workflow = _build(_pipeline(rot=_log(), tilt=_log(1.0, 2.0)))

        assert workflow.requested_context_streams == {'rot'}

    def test_raises_for_a_placeholder_with_no_stream(self) -> None:
        with pytest.raises(ValueError, match=ROT_PATH):
            _build(_pipeline(rot=_log()), transform_streams={})


class TestStreamUpdates:
    def test_latest_sample_sets_the_position(self) -> None:
        workflow = _build(_pipeline(rot=_log()))

        _assert_close(_update(workflow, rot=_log(0.0, 90.0)), _rotated(90.0))
        _assert_close(_update(workflow, rot=_log(0.0, 90.0, 180.0)), _rotated(180.0))

    def test_motor_without_an_update_keeps_its_value(self) -> None:
        workflow = _build(_pipeline(rot=_log(), tilt=_log()))

        _update(workflow, rot=_log(30.0), tilt=_log(40.0))

        _assert_close(_update(workflow, rot=_log(50.0)), _rotated(90.0))

    def test_stream_reaches_every_component_type_it_moves(self) -> None:
        pipeline = _pipeline(rot=_log())
        pipeline[NeXusComponent[NXsample, SampleRun]] = _component(rot=_log())

        def both(
            detector: NeXusTransformation[NXdetector, SampleRun],
            sample: NeXusTransformation[NXsample, SampleRun],
        ) -> _Both:
            return _Both(sc.DataGroup(detector=detector.value, sample=sample.value))

        pipeline.insert(both)
        workflow = _build(pipeline, target=_Both)
        workflow.accumulate(
            {'rot': _log(90.0)},
            start_time=Timestamp.from_ns(0),
            end_time=Timestamp.from_ns(1),
        )

        result = workflow.finalize()['out']

        _assert_close(result['detector'] * POINT, _rotated(90.0))
        _assert_close(result['sample'] * POINT, _rotated(90.0))

    def test_stream_reaches_single_and_multi_chains(self) -> None:
        pipeline = _pipeline(rot=_log())
        pipeline[NeXusMultiComponent[NXdetector, SampleRun]] = sc.DataGroup(
            a=_component(rot=_log()), b=_component(rot=_log())
        )

        def both(
            events: NeXusData[NXdetector, SampleRun],
            single: NeXusTransformation[NXdetector, SampleRun],
            multi: NeXusMultiTransformation[NXdetector, SampleRun],
        ) -> _Both:
            return _Both(
                sc.DataGroup(
                    single=single.value * POINT,
                    a=multi.value['a'].value * POINT,
                    b=multi.value['b'].value * POINT,
                )
            )

        pipeline.insert(both)
        workflow = StreamProcessorWorkflow(
            pipeline,
            dynamic_keys={'det': NeXusData[NXdetector, SampleRun]},
            target_keys={'out': _Both},
            accumulators=(),
        )
        workflow.build(transform_streams=STREAMS)

        for angle in (0.0, 90.0, 180.0):
            workflow.accumulate(
                {'rot': _log(angle), 'det': sc.DataArray(sc.scalar(1.0))},
                start_time=Timestamp.from_ns(0),
                end_time=Timestamp.from_ns(1),
            )
            for position in workflow.finalize()['out'].values():
                _assert_close(position, _rotated(angle))

    def test_stream_can_also_fill_an_offered_key(self) -> None:
        pipeline = _pipeline(rot=_log())

        def both(
            angle: _Angle, transform: NeXusTransformation[NXdetector, SampleRun]
        ) -> _Both:
            return _Both(sc.DataGroup(angle=angle, position=transform.value * POINT))

        pipeline.insert(both)
        workflow = _build(
            pipeline, target=_Both, offered_context_streams={_Angle: 'rot'}
        )
        log = _log(0.0, 90.0)

        workflow.accumulate(
            {'rot': log}, start_time=Timestamp.from_ns(0), end_time=Timestamp.from_ns(1)
        )

        result = workflow.finalize()['out']
        assert sc.identical(result['angle'], log)
        _assert_close(result['position'], _rotated(90.0))


class TestStreamedKeyAboveAChain:
    """A key filled from a stream replaces its provider, and with it what only that
    provider reads: a chain below it needs no motor.
    """

    @staticmethod
    def _pipeline_with_angle_from_chain() -> sciline.Pipeline:
        pipeline = _pipeline(rot=_log())

        def angle(transform: NeXusTransformation[NXdetector, SampleRun]) -> _Angle:
            return _Angle(sc.DataArray(transform.value * POINT))

        pipeline.insert(angle)
        return pipeline

    @staticmethod
    def _target(angle: _Angle) -> _Both:
        return _Both(sc.DataGroup(angle=angle))

    def test_offered_key_cuts_off_the_chain(self) -> None:
        pipeline = self._pipeline_with_angle_from_chain()
        pipeline.insert(self._target)

        workflow = _build(
            pipeline, target=_Both, offered_context_streams={_Angle: 'angle'}
        )

        assert workflow.requested_context_streams == {'angle'}

    def test_bound_key_cuts_off_the_chain(self) -> None:
        pipeline = self._pipeline_with_angle_from_chain()
        pipeline.insert(self._target)

        workflow = _build(pipeline, target=_Both, context_keys={'angle': _Angle})

        assert workflow.requested_context_streams == set()

    def test_chain_read_past_the_key_as_well_is_rejected(self) -> None:
        """The key's value would disagree with the chain it is computed from."""
        pipeline = self._pipeline_with_angle_from_chain()

        def both(
            angle: _Angle, transform: NeXusTransformation[NXdetector, SampleRun]
        ) -> _Both:
            return _Both(sc.DataGroup(angle=angle, transform=transform))

        pipeline.insert(both)

        with pytest.raises(ValueError, match='depends on other dynamic/context keys'):
            _build(pipeline, target=_Both, offered_context_streams={_Angle: 'angle'})


def test_build_loads_each_component_once() -> None:
    pipeline = _pipeline(rot=_log())
    component = pipeline.compute(NeXusComponent[NXdetector, SampleRun])
    loads = []

    def load() -> NeXusComponent[NXdetector, SampleRun]:
        loads.append(1)
        return component

    pipeline.insert(load)

    workflow = _build(pipeline)

    assert len(loads) == 1
    _assert_close(_update(workflow, rot=_log(90.0)), _rotated(90.0))
