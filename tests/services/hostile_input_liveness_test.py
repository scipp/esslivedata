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

Payloads come from ``tests/helpers/hostile_wire``.

Every test runs against both inner batchers: ``SimpleMessageBatcher`` and
``RateAwareMessageBatcher``, which production selects per instrument via
``--batcher``. The two reach batch closure by different mechanisms (fixed
windows versus per-stream pulse-slot gating), so a hostile input that stalls
one need not stall the other, and both are deployed.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

import pytest

from ess.livedata.config import instrument_registry, workflow_spec
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.message import StreamKind
from ess.livedata.core.message_batcher import (
    AdaptiveMessageBatcher,
    MessageBatcher,
    SimpleMessageBatcher,
)
from ess.livedata.core.rate_aware_batcher import RateAwareMessageBatcher
from ess.livedata.kafka.message_adapter import FakeKafkaMessage
from ess.livedata.services.monitor_data import make_monitor_service_builder
from tests.helpers import hostile_wire
from tests.helpers.livedata_app import LivedataApp

SOURCE = 'monitor1'
SECOND_NS = 1_000_000_000

InnerBatcherFactory = Callable[[float], MessageBatcher]


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

    def __init__(self, inner_factory: InnerBatcherFactory) -> None:
        builder = make_monitor_service_builder(instrument='dummy')
        # Production default; LivedataApp would otherwise install the naive
        # batcher, which hides batching-level failure modes.
        builder.message_batcher = AdaptiveMessageBatcher(inner_factory=inner_factory)
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

    def publish_payload(
        self, payload: bytes, *, broker_time_ms: int | None = None
    ) -> None:
        """Queue a raw payload on the monitor topic without stepping.

        Without ``broker_time_ms`` the message carries no broker timestamp
        (``TIMESTAMP_NOT_AVAILABLE``), which forces the adapter's future-clamp
        onto its wall-clock fallback. Real Kafka messages always carry a
        CreateTime; pass one to exercise the broker-timestamp clamp target.
        """
        if broker_time_ms is None:
            self.app.publish_data(topic=self.app.monitor_topic, time=0, data=payload)
        else:
            self.app.consumer.add_message(
                FakeKafkaMessage(
                    value=payload,
                    topic=self.app.monitor_topic,
                    timestamp=broker_time_ms,
                    timestamp_type=1,
                )
            )

    def next_time_ns(self) -> int:
        """Advance and return the data clock, for hand-built in-sequence payloads."""
        self._time_ns += SECOND_NS
        return self._time_ns

    def publish_good(self) -> None:
        """Queue a well-formed event message one second after the previous."""
        self._seed += 1
        self.publish_payload(
            hostile_wire.ev44_events(
                SOURCE, reference_time_ns=self.next_time_ns(), seed=self._seed
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


@pytest.fixture(
    params=[SimpleMessageBatcher, RateAwareMessageBatcher],
    ids=['simple', 'rate_aware'],
)
def harness(request: pytest.FixtureRequest) -> MonitorServiceHarness:
    return MonitorServiceHarness(inner_factory=request.param)


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
    """A far-future timestamp *after* the first batch must not stop output.

    The adapter clamps the insane value to the wall clock. The harness data
    timeline is a fixed epoch, so the clamped timestamp is still months ahead
    of the stream *from the batcher's point of view* -- this test therefore
    exercises the batcher-level in-band outlier defence end-to-end, the band
    the adapter bound cannot cover. The rate-aware batcher may stay silent
    for up to ``_REANCHOR_STALLED_CALLS`` polls before re-anchoring, so the
    liveness window must exceed that.
    """
    harness.run_good_cycles(3)
    harness.publish_payload(
        hostile_wire.ev44_events(SOURCE, reference_time_ns=hostile_wire.FAR_FUTURE_NS)
    )
    harness.app.step()
    assert_service_live(harness, cycles=40)


@pytest.mark.parametrize(
    'timestamp_ns',
    [1, hostile_wire.PRE_EPOCH_NS],
    ids=['near_epoch', 'pre_epoch'],
)
def test_ancient_timestamp_mid_stream_does_not_stall_service(
    harness: MonitorServiceHarness, timestamp_ns: int
) -> None:
    """An ancient (even pre-epoch) timestamp mid-stream counts as a late
    message; it lands in the current batch and must not disturb progression.
    """
    harness.run_good_cycles(3)
    harness.publish_payload(
        hostile_wire.ev44_events(SOURCE, reference_time_ns=timestamp_ns)
    )
    harness.app.step()
    assert_service_live(harness)


def test_pre_epoch_timestamp_in_first_batch_does_not_stall_service(
    harness: MonitorServiceHarness,
) -> None:
    """The mirror of the far-future wedge: an initial batch opening at a
    pre-epoch time only stretches the first batch backwards; subsequent
    batches align to the newest data and output must keep flowing.
    """
    harness.publish_payload(
        hostile_wire.ev44_events(SOURCE, reference_time_ns=hostile_wire.PRE_EPOCH_NS)
    )
    harness.publish_good()
    harness.app.step()
    assert_service_live(harness)


def test_mismatched_event_vectors_do_not_stall_service(
    harness: MonitorServiceHarness,
) -> None:
    """Disagreeing time_of_flight/pixel_id lengths adapt on the monitor path
    (pixel_id is unused there); nothing downstream may index one vector by
    the other's length.
    """
    harness.run_good_cycles(3)
    harness.publish_payload(
        hostile_wire.ev44_mismatched_event_vectors(
            SOURCE, reference_time_ns=harness.next_time_ns()
        )
    )
    harness.app.step()
    assert_service_live(harness)


def test_far_future_timestamp_in_first_batch_does_not_stall_service(
    harness: MonitorServiceHarness,
) -> None:
    """The startup backlog is where a far-future timestamp is most dangerous:
    it would anchor the first batch boundary, and every real message would
    then look early forever. Two independent layers prevent it — the adapter
    boundary clamps the timestamp, and the batchers anchor on a plausible
    timestamp rather than the maximum.

    The message carries a broker CreateTime in the harness's data epoch, as
    in production consuming a backlog: the clamp must land the message among
    its neighbours rather than at the wall clock, where it would re-poison
    the first anchor from within the adapter's bound.
    """
    harness.publish_payload(
        hostile_wire.ev44_events(SOURCE, reference_time_ns=hostile_wire.FAR_FUTURE_NS),
        broker_time_ms=hostile_wire.REALISTIC_EPOCH_NS // 1_000_000,
    )
    harness.publish_good()
    harness.app.step()
    assert_service_live(harness, cycles=20)
