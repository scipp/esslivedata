# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Service liveness under hostile wire input, with production batching.

Most service-level tests run with ``NaiveMessageBatcher``, which emits every
poll as a batch and therefore cannot detect the most severe corruption class
found in the backend audit (#1038): inputs that stall the *production*
batcher, silently stopping all output service-wide while the process looks
healthy. This harness runs the monitor service with the production-default
``AdaptiveMessageBatcher`` (wrapping ``SimpleMessageBatcher``, which uses
data-derived timestamps as its only clock) and asserts one invariant:

    After consuming any single hostile payload, the service still publishes
    results for subsequent well-formed data.

Payloads come from ``tests/helpers/hostile_wire``. Cases the current code
survives are regression guards; the reproduced wedge (#1038 finding 1) is a
strict xfail that doubles as the acceptance test for the adapter-boundary
validation proposed in #1047.
"""

from __future__ import annotations

import uuid

import pytest

from ess.livedata.config import instrument_registry, workflow_spec
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.message import StreamKind
from ess.livedata.core.message_batcher import AdaptiveMessageBatcher
from ess.livedata.services.monitor_data import make_monitor_service_builder
from tests.helpers import hostile_wire
from tests.helpers.livedata_app import LivedataApp

SOURCE = 'monitor1'
SECOND_NS = 1_000_000_000


def _monitor_workflow_id(instrument: str) -> workflow_spec.WorkflowId:
    config = instrument_registry[instrument]
    for wid, spec in config.workflow_factory.items():
        if spec.group.name == 'monitor_data':
            return wid
    raise ValueError(f'No monitor_data workflow for {instrument}')


class MonitorServiceHarness:
    """Monitor service with production batching and a running workflow.

    Publishes well-formed monitor events with steadily advancing data-derived
    timestamps (the batcher's clock) and observes published workflow results.
    """

    def __init__(self) -> None:
        builder = make_monitor_service_builder(instrument='dummy')
        # Production default; LivedataApp would otherwise install the naive
        # batcher, which hides batching-level failure modes.
        builder.message_batcher = AdaptiveMessageBatcher()
        self.app = LivedataApp.from_service_builder(
            builder, use_naive_message_batcher=False
        )
        self._time_ns = hostile_wire.REALISTIC_EPOCH_NS
        self._seed = 0
        workflow_config = workflow_spec.WorkflowConfig(
            identifier=_monitor_workflow_id('dummy'),
            job_id=JobId(source_name=SOURCE, job_number=uuid.uuid4()),
        )
        self.app.publish_config_message(workflow_config)
        self.app.step()

    def publish_payload(self, payload: bytes) -> None:
        """Queue a raw payload on the monitor topic without stepping."""
        self.app.publish_data(topic=self.app.monitor_topic, time=0, data=payload)

    def publish_good(self) -> None:
        """Queue a well-formed event message one second after the previous."""
        self._time_ns += SECOND_NS
        self._seed += 1
        self.publish_payload(
            hostile_wire.ev44_events(
                SOURCE, reference_time_ns=self._time_ns, seed=self._seed
            )
        )

    def result_count(self) -> int:
        return sum(
            1
            for m in self.app.sink.messages
            if m.stream.kind == StreamKind.LIVEDATA_DATA
        )

    def run_good_cycles(self, n: int) -> int:
        """Publish+process n good messages; return results gained."""
        before = self.result_count()
        for _ in range(n):
            self.publish_good()
            self.app.step()
        return self.result_count() - before


@pytest.fixture
def harness() -> MonitorServiceHarness:
    return MonitorServiceHarness()


def assert_service_live(harness: MonitorServiceHarness, cycles: int = 10) -> None:
    """The service publishes new results as well-formed data keeps arriving.

    The production batcher holds back the trailing batch until later data
    closes it, so we require growth over the window, not one result per cycle.
    """
    assert harness.run_good_cycles(cycles) > 0


def test_baseline_service_is_live(harness: MonitorServiceHarness) -> None:
    """Harness sanity: without hostile input, results flow."""
    harness.publish_good()
    harness.app.step()
    assert_service_live(harness)


@pytest.mark.parametrize('case', sorted(hostile_wire.malformed_corpus(SOURCE)))
def test_malformed_payload_does_not_stall_service(
    harness: MonitorServiceHarness, case: str
) -> None:
    """Malformed payloads are contained per-message; output keeps flowing."""
    harness.publish_good()
    harness.app.step()
    harness.publish_payload(hostile_wire.malformed_corpus(SOURCE)[case])
    harness.app.step()
    assert_service_live(harness)


def test_far_future_timestamp_mid_stream_does_not_stall_service(
    harness: MonitorServiceHarness,
) -> None:
    """A far-future timestamp *after* the first batch degrades the batcher
    (the message lingers as a pending future message) but must not stop
    output. Guards the boundary of the wedge in #1038 finding 1.
    """
    harness.run_good_cycles(3)
    harness.publish_payload(
        hostile_wire.ev44_events(SOURCE, reference_time_ns=hostile_wire.FAR_FUTURE_NS)
    )
    harness.app.step()
    assert_service_live(harness)


def test_ancient_timestamp_mid_stream_does_not_stall_service(
    harness: MonitorServiceHarness,
) -> None:
    """A near-epoch timestamp mid-stream counts as a late message; it lands in
    the current batch and must not disturb batch progression.
    """
    harness.run_good_cycles(3)
    harness.publish_payload(hostile_wire.ev44_events(SOURCE, reference_time_ns=1))
    harness.app.step()
    assert_service_live(harness)


@pytest.mark.xfail(
    strict=True,
    reason='#1038 finding 1 / #1047: a far-future data timestamp in the first '
    'consumed batch becomes the batch boundary, after which no batch is ever '
    'emitted — the service goes silent while appearing healthy',
)
def test_far_future_timestamp_in_first_batch_does_not_stall_service(
    harness: MonitorServiceHarness,
) -> None:
    harness.publish_payload(
        hostile_wire.ev44_events(SOURCE, reference_time_ns=hostile_wire.FAR_FUTURE_NS)
    )
    harness.publish_good()
    harness.app.step()
    assert_service_live(harness, cycles=20)
