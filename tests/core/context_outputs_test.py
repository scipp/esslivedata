# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for republication of workflow outputs as context input streams.

The mapping is derived from ``WorkflowSpec.context_outputs``, so the failure
modes worth pinning are declaration-level: a template that renders the same
stream name twice, and a template with an unknown placeholder. Both must fail
at startup, since a collision would interleave two producers silently at the
consumer.
"""

from __future__ import annotations

import dataclasses
import uuid

import pytest
import scipp as sc
from pydantic import Field, ValidationError
from streaming_data_types import dataarray_da00

from ess.livedata.config.instrument import Instrument, instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.config.workflow_spec import (
    REDUCTION,
    JobId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.core.context_outputs import (
    ContextOutputError,
    ContextOutputExtractor,
    resolve_context_outputs,
)
from ess.livedata.core.job import JobResult
from ess.livedata.core.message import StreamId, StreamKind
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.kafka.message_adapter import FakeKafkaMessage
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.kafka.sink_serializers import make_default_sink_serializer
from ess.livedata.preprocessors.detector_data import DetectorPreprocessorFactory

START_TIME = Timestamp.from_ns(1000)


@pytest.fixture(scope='module')
def dummy_instrument() -> Instrument:
    get_config('dummy')  # Imports spec module, registering workflow specs.
    return instrument_registry['dummy']


class _LutOutputs(WorkflowOutputsBase):
    table: sc.DataArray = Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0.0, unit='angstrom')),
        title='Table',
        description='Wavelength lookup table.',
    )
    diagnostic: sc.DataArray = Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0.0, unit='angstrom')),
        title='Diagnostic',
        description='Not republished.',
    )


def _spec(
    *,
    source_names: list[str],
    context_outputs: dict[str, str],
    name: str = 'wf',
) -> WorkflowSpec:
    return WorkflowSpec(
        instrument='dummy',
        name=name,
        version=1,
        title='WF',
        description='',
        outputs=_LutOutputs,
        context_outputs=context_outputs,
        source_names=source_names,
        group=REDUCTION,
    )


@pytest.fixture
def spec() -> WorkflowSpec:
    return _spec(
        source_names=['chopper_cascade'],
        context_outputs={'table': 'wavelength_lut/chopper_cascade'},
    )


@pytest.fixture
def registry(spec: WorkflowSpec) -> dict:
    return {spec.get_id(): spec}


@pytest.fixture
def extractor(registry: dict) -> ContextOutputExtractor:
    return ContextOutputExtractor(registry=registry)


@pytest.fixture
def result(spec: WorkflowSpec) -> JobResult:
    return JobResult(
        job_id=JobId(source_name='chopper_cascade', job_number=uuid.uuid4()),
        workflow_id=spec.get_id(),
        start_time=START_TIME,
        end_time=Timestamp.from_ns(2000),
        data=sc.DataGroup(
            {
                'table': sc.DataArray(sc.scalar(1.5, unit='angstrom')),
                'diagnostic': sc.DataArray(sc.scalar(9.0, unit='angstrom')),
            }
        ),
    )


def test_context_outputs_validator_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match='unknown output field'):
        _spec(source_names=['a'], context_outputs={'no_such': 'lut/x'})


def test_context_outputs_validator_rejects_multiple_source_names() -> None:
    # A stream name carries no job identity, so two jobs of one spec would
    # publish the same names.
    with pytest.raises(ValidationError, match='exactly one source name'):
        _spec(source_names=['a', 'b'], context_outputs={'table': 'lut/x'})


def test_resolves_declared_names_for_the_spec_source() -> None:
    spec = _spec(source_names=['a'], context_outputs={'table': 'wavelength_lut/x'})

    resolved = resolve_context_outputs({spec.get_id(): spec})

    assert resolved == {(spec.get_id(), 'a'): (('table', 'wavelength_lut/x'),)}


def test_collision_across_specs_is_rejected() -> None:
    # The names share one namespace, so the check must span the whole registry
    # rather than each spec in isolation.
    first = _spec(
        source_names=['a'], context_outputs={'table': 'wavelength_lut/x'}, name='one'
    )
    second = _spec(
        source_names=['b'], context_outputs={'table': 'wavelength_lut/x'}, name='two'
    )

    with pytest.raises(ContextOutputError, match='Duplicate context stream name'):
        resolve_context_outputs({s.get_id(): s for s in (first, second)})


def test_extracts_only_designated_output(
    extractor: ContextOutputExtractor, result: JobResult
) -> None:
    messages = extractor.extract([result])

    assert len(messages) == 1
    (message,) = messages
    assert message.stream.kind == StreamKind.LIVEDATA_CONTEXT
    assert message.stream.name == 'wavelength_lut/chopper_cascade'
    assert message.value.value == 1.5


def test_extraction_uses_result_start_time(
    extractor: ContextOutputExtractor, result: JobResult
) -> None:
    (message,) = extractor.extract([result])

    assert message.timestamp == START_TIME


def test_missing_designated_output_is_skipped(
    extractor: ContextOutputExtractor, result: JobResult
) -> None:
    partial = dataclasses.replace(
        result, data=sc.DataGroup({'diagnostic': sc.DataArray(sc.scalar(1.0))})
    )

    assert extractor.extract([partial]) == []


def test_result_without_data_is_skipped(
    extractor: ContextOutputExtractor, result: JobResult
) -> None:
    assert extractor.extract([dataclasses.replace(result, data=None)]) == []


def test_registry_without_declarations_extracts_nothing(result: JobResult) -> None:
    spec = _spec(source_names=['chopper_cascade'], context_outputs={})
    extractor = ContextOutputExtractor(registry={spec.get_id(): spec})

    assert extractor.extract([result]) == []


def test_serialized_da00_carries_stream_name_as_source_name(
    extractor: ContextOutputExtractor, result: JobResult
) -> None:
    """End-to-end: the message lands on the dedicated context topic as da00 with
    the rendered stream name as ``source_name``, which is the name a consuming
    ``ContextBinding`` declares."""
    (message,) = extractor.extract([result])
    serializer = make_default_sink_serializer(instrument='dummy')

    serialized = serializer.serialize(message)

    assert serialized.topic == 'dummy_livedata_context'
    decoded = dataarray_da00.deserialise_da00(serialized.value)
    assert decoded.source_name == 'wavelength_lut/chopper_cascade'


def test_published_table_round_trips_back_as_a_context_stream(
    extractor: ContextOutputExtractor, result: JobResult, dummy_instrument: Instrument
) -> None:
    """The full seam: a designated output published by one service is read back
    by another under the stream name a ``ContextBinding`` declares.

    The ingest route carries no stream lookup table, so the internal stream name
    is the da00 source name. That is the whole reason a rendered
    ``context_outputs`` template can be named directly by a consumer.
    """
    stream_mapping = get_stream_mapping(instrument='dummy', dev=True)
    (message,) = extractor.extract([result])
    serialized = make_default_sink_serializer(instrument='dummy').serialize(message)
    adapter = (
        RoutingAdapterBuilder(stream_mapping=stream_mapping)
        .with_livedata_context_route()
        .build()
    )

    ingested = adapter.adapt(
        FakeKafkaMessage(value=serialized.value, topic=serialized.topic)
    )

    assert ingested.stream.kind == StreamKind.LIVEDATA_CONTEXT
    assert ingested.stream.name == 'wavelength_lut/chopper_cascade'
    assert ingested.value.value == 1.5


def test_context_stream_is_preprocessed_as_a_retained_context_value(
    dummy_instrument: Instrument,
) -> None:
    """The JobManager gate and the context cache both key off ``is_context``,
    so the accumulator for this kind must be a context accumulator."""
    factory = DetectorPreprocessorFactory(instrument=dummy_instrument)

    accumulator = factory.make_preprocessor(
        StreamId(kind=StreamKind.LIVEDATA_CONTEXT, name='wavelength_lut/x')
    )

    assert accumulator is not None
    assert accumulator.is_context
