# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the NICOS device projection.

Uses the real ``dummy`` instrument registry and real workflow specs. The
generation token is carried on each :class:`JobResult` (stamped by the
JobManager), so the projector needs no live registry.
"""

from __future__ import annotations

import dataclasses
import uuid

import pytest
import scipp as sc
from streaming_data_types import dataarray_da00

from ess.livedata.config.device_contract import DeviceContract, DeviceContractEntry
from ess.livedata.config.instrument import Instrument, instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.workflow_spec import JobId, WorkflowId
from ess.livedata.core.job import JobResult
from ess.livedata.core.message import StreamKind
from ess.livedata.core.projection import GENERATION_TOKEN_COORD, Projector
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.kafka.sink_serializers import make_default_sink_serializer

MONITOR_WORKFLOW = 'dummy/monitor_histogram/1'
DEVICE_NAME = 'monitor1_counts_total'
TOKEN = Timestamp.from_ns(123_456_789)


@pytest.fixture(scope='module')
def dummy_instrument() -> Instrument:
    get_config('dummy')  # Imports spec module, registering workflow specs.
    return instrument_registry['dummy']


@pytest.fixture
def contract(dummy_instrument: Instrument) -> DeviceContract:
    entry = DeviceContractEntry(
        workflow_id=MONITOR_WORKFLOW,
        source_name='monitor1',
        output_name='counts_total_cumulative',
        device_name=DEVICE_NAME,
    )
    return DeviceContract.from_entries([entry], dummy_instrument.workflow_factory)


@pytest.fixture
def projector(contract: DeviceContract) -> Projector:
    return Projector(device_contract=contract)


@pytest.fixture
def job_id() -> JobId:
    return JobId(source_name='monitor1', job_number=uuid.uuid4())


@pytest.fixture
def result(job_id: JobId) -> JobResult:
    data = sc.DataGroup(
        {
            # (a) in-contract, scalar-cumulative -> projected.
            'counts_total_cumulative': sc.DataArray(sc.scalar(42, unit='counts')),
            # (b) non-eligible runtime output (has time coord).
            'cumulative': sc.DataArray(
                sc.scalar(7, unit='counts'),
                coords={'time': sc.scalar(0, unit='ns')},
            ),
            # (c) not in the contract.
            'current': sc.DataArray(sc.scalar(3, unit='counts')),
        }
    )
    return JobResult(
        job_id=job_id,
        workflow_id=WorkflowId.from_string(MONITOR_WORKFLOW),
        start_time=Timestamp.from_ns(1000),
        end_time=Timestamp.from_ns(2000),
        data=data,
        generation_token=TOKEN,
    )


def test_projects_only_eligible_contracted_output(
    projector: Projector, result: JobResult
) -> None:
    messages = projector.project([result])

    assert len(messages) == 1
    (message,) = messages
    assert message.stream.kind == StreamKind.LIVEDATA_PROJECTION
    assert message.stream.name == DEVICE_NAME
    assert message.value.value == 42


def test_projection_uses_result_timestamp(
    projector: Projector, result: JobResult
) -> None:
    (message,) = projector.project([result])

    assert message.timestamp == result.start_time


def test_projection_attaches_generation_token_coord(
    projector: Projector, result: JobResult
) -> None:
    (message,) = projector.project([result])

    coord = message.value.coords[GENERATION_TOKEN_COORD]
    assert coord.value == TOKEN.to_ns()
    assert coord.unit == 'ns'


def test_result_without_token_is_skipped(
    projector: Projector, result: JobResult
) -> None:
    # A result whose generation could not be stamped is not projected (avoids
    # publishing a device stream we cannot mark).
    untokened = dataclasses.replace(result, generation_token=None)

    assert projector.project([untokened]) == []


def test_empty_contract_projects_nothing(
    dummy_instrument: Instrument, result: JobResult
) -> None:
    empty = DeviceContract.from_entries([], dummy_instrument.workflow_factory)
    projector = Projector(device_contract=empty)

    assert projector.project([result]) == []


def test_projection_carries_the_results_own_token(
    projector: Projector, result: JobResult
) -> None:
    # The projected token follows the result, not any external lookup: a
    # re-minted (changed) token on the result propagates to the coordinate.
    new_token = Timestamp.from_ns(987_654_321)
    retoned = dataclasses.replace(result, generation_token=new_token)

    (message,) = projector.project([retoned])

    assert message.value.coords[GENERATION_TOKEN_COORD].value == new_token.to_ns()


def test_serialized_da00_carries_device_source_name_and_token(
    projector: Projector, result: JobResult
) -> None:
    """End-to-end: the projected message serializes to da00 with the device name
    as source_name and the generation token as a da00 variable."""
    (message,) = projector.project([result])
    serializer = make_default_sink_serializer(instrument='dummy')

    serialized = serializer.serialize(message)

    assert serialized.topic == 'dummy_livedata_projection'
    decoded = dataarray_da00.deserialise_da00(serialized.value)
    assert decoded.source_name == DEVICE_NAME
    var_names = {var.name for var in decoded.data}
    assert GENERATION_TOKEN_COORD in var_names
    token_var = next(v for v in decoded.data if v.name == GENERATION_TOKEN_COORD)
    assert int(token_var.data) == TOKEN.to_ns()
