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

from ess.livedata.config.workflow_spec import (
    REDUCTION,
    JobId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.core.context_outputs import (
    ContextOutputError,
    ContextOutputExtractor,
    resolve_context_streams,
)
from ess.livedata.core.job import JobResult
from ess.livedata.core.message import StreamKind
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.kafka.sink_serializers import make_default_sink_serializer

START_TIME = Timestamp.from_ns(1000)


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
        context_outputs={'table': 'wavelength_lut/{source_name}'},
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
        _spec(source_names=['a'], context_outputs={'no_such': 'lut/{source_name}'})


def test_renders_one_stream_per_source_name() -> None:
    spec = _spec(
        source_names=['a', 'b'],
        context_outputs={'table': 'wavelength_lut/{source_name}'},
    )

    resolved = resolve_context_streams({spec.get_id(): spec})

    assert resolved[(str(spec.get_id()), 'a')] == (('table', 'wavelength_lut/a'),)
    assert resolved[(str(spec.get_id()), 'b')] == (('table', 'wavelength_lut/b'),)


def test_literal_template_over_multiple_sources_collides() -> None:
    spec = _spec(source_names=['a', 'b'], context_outputs={'table': 'fixed_name'})

    with pytest.raises(ContextOutputError, match='Duplicate context stream name'):
        resolve_context_streams({spec.get_id(): spec})


def test_collision_across_specs_is_rejected() -> None:
    # The rendered names share one namespace, so the check must span the whole
    # registry rather than each spec in isolation.
    first = _spec(
        source_names=['a'], context_outputs={'table': 'wavelength_lut/x'}, name='one'
    )
    second = _spec(
        source_names=['b'], context_outputs={'table': 'wavelength_lut/x'}, name='two'
    )

    with pytest.raises(ContextOutputError, match='Duplicate context stream name'):
        resolve_context_streams({s.get_id(): s for s in (first, second)})


def test_bad_placeholder_in_template_raises() -> None:
    spec = _spec(source_names=['a'], context_outputs={'table': '{nope}/x'})

    with pytest.raises(ContextOutputError, match='placeholder'):
        resolve_context_streams({spec.get_id(): spec})


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
