# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid
from typing import Any

import pydantic
import pytest
import scipp as sc

from ess.livedata.config.instrument import Instrument
from ess.livedata.config.workflow_spec import (
    JobId,
    WorkflowConfig,
    WorkflowId,
    WorkflowOutputsBase,
)
from ess.livedata.core.job import JobResult
from ess.livedata.core.message import (
    COMMANDS_STREAM_ID,
    Message,
    StreamId,
    StreamKind,
)
from ess.livedata.core.message_batcher import MessageBatch, NaiveMessageBatcher
from ess.livedata.core.orchestrating_processor import (
    MessagePreprocessor,
    OrchestratingProcessor,
    _job_result_to_message,
)
from ess.livedata.core.preprocessor import Accumulator
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.fakes import FakeMessageSink, FakeMessageSource
from ess.livedata.preprocessors.accumulators import LogData
from ess.livedata.preprocessors.timeseries import LogdataPreprocessorFactory
from ess.livedata.preprocessors.to_nxlog import ToNXlog
from ess.livedata.preprocessors.wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
)

from .job_test import FakeProcessor


class FakeContextAccumulator(Accumulator[Any, sc.DataArray]):
    """Accumulator with is_context=True for testing."""

    is_context = True

    def __init__(self) -> None:
        self._value: sc.DataArray | None = None

    def add(self, timestamp: int, data: Any) -> None:
        self._value = data

    def get(self) -> sc.DataArray:
        if self._value is None:
            raise ValueError("No data has been added")
        return self._value

    def clear(self) -> None:
        self._value = None


class FakeNonContextAccumulator(Accumulator[Any, sc.DataArray]):
    """Accumulator with is_context=False for testing."""

    is_context = False

    def __init__(self) -> None:
        self._value: sc.DataArray | None = None
        self.get_call_count = 0

    def add(self, timestamp: int, data: Any) -> None:
        self._value = data

    def get(self) -> sc.DataArray:
        self.get_call_count += 1
        if self._value is None:
            raise ValueError("No data has been added")
        result = self._value
        self._value = None  # Simulates clear-on-get
        return result

    def clear(self) -> None:
        self._value = None


class FakeFactory:
    """Factory that returns pre-configured accumulators by stream name."""

    def __init__(self, accumulators: dict[str, Accumulator]) -> None:
        self._accumulators = accumulators
        # OrchestratingProcessor loads the device contract from this instrument
        # at construction, so it must be a package-backed instrument. 'tbl' has
        # no device_contract.yaml, yielding an empty contract — these tests do
        # not schedule jobs, so the contract content is irrelevant.
        self.instrument = Instrument(name='tbl')

    def make_preprocessor(self, key: StreamId) -> Accumulator | None:
        return self._accumulators.get(key.name)


def _make_processor(
    factory: FakeFactory,
    source_messages: list[list[Message]],
) -> tuple[OrchestratingProcessor, FakeMessageSink]:
    """Build an OrchestratingProcessor wired to fake I/O for orchestrator-tick tests."""
    sink = FakeMessageSink()
    processor = OrchestratingProcessor(
        source=FakeMessageSource(source_messages),
        sink=sink,
        preprocessor_factory=factory,
        service_name='test_service',
        message_batcher=NaiveMessageBatcher(),
    )
    return processor, sink


class TestOrchestratingProcessorIdleTick:
    def test_truly_idle_tick_publishes_no_data_messages(self):
        """An empty-input tick (batcher returns None) must not publish data results.

        The idle guard at the bottom of process() should fire and return early
        without calling process_jobs, so the sink only ever receives status messages.
        """
        factory = FakeFactory({})
        processor, sink = _make_processor(factory, source_messages=[[]])

        processor.process()

        data_messages = [
            m for m in sink.messages if m.stream.kind == StreamKind.LIVEDATA_DATA
        ]
        assert data_messages == []

    def test_data_tick_triggers_preprocessing(self):
        """When data messages arrive the preprocess path runs (not the idle branch).

        We verify this by checking that the non-context accumulator's get() was
        called exactly once — which only happens inside preprocess_messages(), not
        in the idle-branch early-return path.
        """
        acc = FakeNonContextAccumulator()
        stream_id = StreamId(kind=StreamKind.DETECTOR_EVENTS, name='events')
        factory = FakeFactory({'events': acc})
        msg = Message(
            timestamp=Timestamp.from_ns(1_000_000_000),
            stream=stream_id,
            value=sc.scalar(1.0),
        )
        processor, _ = _make_processor(factory, source_messages=[[msg]])

        processor.process()

        assert acc.get_call_count == 1

    def test_idle_tick_does_not_call_get_on_accumulator(self):
        """No data in → accumulator's get() must never be invoked on an idle tick."""
        acc = FakeNonContextAccumulator()
        factory = FakeFactory({'events': acc})
        processor, _ = _make_processor(factory, source_messages=[[]])

        processor.process()

        assert acc.get_call_count == 0


class TestGetContext:
    def test_reads_from_context_accumulator(self):
        ctx_acc = FakeContextAccumulator()
        factory = FakeFactory({"temperature": ctx_acc})
        preprocessor = MessagePreprocessor(factory)

        # Feed a message to populate the accumulator
        stream_id = StreamId(kind=StreamKind.LOG, name="temperature")
        batch = MessageBatch(
            start_time=100,
            end_time=200,
            messages=[Message(timestamp=150, stream=stream_id, value=sc.scalar(25.0))],
        )
        preprocessor.preprocess_messages(batch)

        result = preprocessor.get_context({"temperature"})
        assert stream_id in result
        assert sc.identical(result[stream_id], sc.scalar(25.0))

    def test_skips_non_context_accumulator(self):
        non_ctx = FakeNonContextAccumulator()
        factory = FakeFactory({"events": non_ctx})
        preprocessor = MessagePreprocessor(factory)

        stream_id = StreamId(kind=StreamKind.DETECTOR_EVENTS, name="events")
        batch = MessageBatch(
            start_time=100,
            end_time=200,
            messages=[Message(timestamp=150, stream=stream_id, value=sc.scalar(1.0))],
        )
        preprocessor.preprocess_messages(batch)

        # Reset get_call_count after preprocess_messages already called get()
        non_ctx.get_call_count = 0

        result = preprocessor.get_context({"events"})
        assert result == {}
        assert non_ctx.get_call_count == 0  # get() was never called

    def test_skips_unpopulated_accumulator(self):
        ctx_acc = FakeContextAccumulator()
        factory = FakeFactory({"temperature": ctx_acc})
        preprocessor = MessagePreprocessor(factory)

        # Register the accumulator without feeding data
        stream_id = StreamId(kind=StreamKind.LOG, name="temperature")
        preprocessor._get_accumulator(stream_id)

        result = preprocessor.get_context({"temperature"})
        assert result == {}

    def test_returns_empty_for_unknown_stream(self):
        ctx_acc = FakeContextAccumulator()
        factory = FakeFactory({"temperature": ctx_acc})
        preprocessor = MessagePreprocessor(factory)

        # Feed data for temperature
        stream_id = StreamId(kind=StreamKind.LOG, name="temperature")
        batch = MessageBatch(
            start_time=100,
            end_time=200,
            messages=[Message(timestamp=150, stream=stream_id, value=sc.scalar(25.0))],
        )
        preprocessor.preprocess_messages(batch)

        result = preprocessor.get_context({"nonexistent"})
        assert result == {}

    def test_does_not_read_non_matching_context_accumulator(self):
        ctx_acc = FakeContextAccumulator()
        factory = FakeFactory({"temperature": ctx_acc})
        preprocessor = MessagePreprocessor(factory)

        stream_id = StreamId(kind=StreamKind.LOG, name="temperature")
        batch = MessageBatch(
            start_time=100,
            end_time=200,
            messages=[Message(timestamp=150, stream=stream_id, value=sc.scalar(25.0))],
        )
        preprocessor.preprocess_messages(batch)

        # Ask for a different stream name
        result = preprocessor.get_context({"chopper_speed"})
        assert result == {}

    def test_works_with_real_to_nxlog(self):
        """Integration test using the real ToNXlog accumulator."""
        to_nxlog = ToNXlog(attrs={'units': 'K'})
        factory = FakeFactory({"temperature": to_nxlog})
        preprocessor = MessagePreprocessor(factory)

        stream_id = StreamId(kind=StreamKind.LOG, name="temperature")
        batch = MessageBatch(
            start_time=100,
            end_time=200,
            messages=[
                Message(
                    timestamp=150,
                    stream=stream_id,
                    value=LogData(time=150, value=25.0),
                ),
            ],
        )
        preprocessor.preprocess_messages(batch)

        result = preprocessor.get_context({"temperature"})
        assert stream_id in result
        da = result[stream_id]
        assert da.sizes == {'time': 1}
        assert da.unit == 'K'

        # Calling again returns the same data (idempotent)
        result2 = preprocessor.get_context({"temperature"})
        assert sc.identical(result[stream_id], result2[stream_id])


class _ReplayOutputs(WorkflowOutputsBase):
    result: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
        title='Result',
    )


class TestEmptyBatchContextReplay:
    """Replaying a one-shot primary tick that is cached as context.

    ``chopper_cascade`` is a job's *primary* source, yet it is backed by an
    ``is_context`` ``ToNXlog`` accumulator. The tick fires once, before any job
    is scheduled. A job scheduled afterwards must still fire exactly once, from
    the cached tick replayed on an empty batch. That assembly —
    ``peek_pending_streams`` + ``get_context`` enriching the empty batch — lives
    only in ``OrchestratingProcessor.process`` (its parts are unit-tested
    above and in ``job_manager_test.py``, but their integration is not).
    """

    @staticmethod
    def _make_instrument() -> tuple[Instrument, Any]:
        # 'tbl' is a package-backed instrument with no device_contract.yaml, so
        # OrchestratingProcessor loads an empty contract at construction.
        instrument = Instrument(name='tbl')
        handle = instrument.register_spec(
            service='timeseries',
            name='replay_probe',
            version=1,
            title='Replay probe',
            source_names=[CHOPPER_CASCADE_SOURCE],
            params=None,
            outputs=_ReplayOutputs,
        )
        handle.attach_factory()(lambda: FakeProcessor())
        return instrument, handle.workflow_id

    @staticmethod
    def _processor(
        instrument: Instrument, batches: list[list[Message]]
    ) -> tuple[OrchestratingProcessor, FakeMessageSink]:
        sink = FakeMessageSink()
        processor = OrchestratingProcessor(
            source=FakeMessageSource(batches),
            sink=sink,
            preprocessor_factory=LogdataPreprocessorFactory(instrument=instrument),
            service_name='timeseries',
            message_batcher=NaiveMessageBatcher(),
        )
        return processor, sink

    @staticmethod
    def _tick() -> Message:
        return Message(
            timestamp=Timestamp.from_ns(100),
            stream=StreamId(kind=StreamKind.LOG, name=CHOPPER_CASCADE_SOURCE),
            value=LogData(time=100, value=1),
        )

    @staticmethod
    def _schedule(workflow_id: Any) -> Message:
        job_id = JobId(source_name=CHOPPER_CASCADE_SOURCE, job_number=uuid.uuid4())
        config = WorkflowConfig.from_params(
            workflow_id=workflow_id, job_id=job_id, params=None, aux_source_names=None
        )
        return Message(
            timestamp=Timestamp.from_ns(200),
            stream=COMMANDS_STREAM_ID,
            value=config,
        )

    @staticmethod
    def _data_messages(sink: FakeMessageSink) -> list[Message]:
        return [m for m in sink.messages if m.stream.kind is StreamKind.LIVEDATA_DATA]

    def test_tick_cached_before_schedule_fires_job_once(self) -> None:
        instrument, workflow_id = self._make_instrument()
        processor, sink = self._processor(
            instrument, [[self._tick()], [self._schedule(workflow_id)]]
        )

        # First poll: the tick arrives and is cached in the ToNXlog context
        # accumulator. No job is scheduled, so nothing fires.
        processor.process()
        assert self._data_messages(sink) == []

        # Second poll: the operator schedules the job. The data batch is empty,
        # so activation can only come from the cached tick being replayed.
        processor.process()
        assert len(self._data_messages(sink)) == 1

    def test_no_replay_without_a_scheduled_job(self) -> None:
        instrument, _ = self._make_instrument()
        # Tick arrives, then an empty poll: with no job scheduled there is
        # nothing to activate and nothing to publish.
        processor, sink = self._processor(instrument, [[self._tick()], []])

        processor.process()
        processor.process()
        assert self._data_messages(sink) == []


class TestJobResultToMessage:
    def _result(self, *, start_time: Timestamp | None) -> JobResult:
        return JobResult(
            job_id=JobId(source_name="det", job_number=uuid.uuid4()),
            workflow_id=WorkflowId(instrument="test", name="wf", version=1),
            start_time=start_time,
            end_time=start_time,
            data=sc.DataGroup(),
        )

    def test_uses_result_start_time_as_timestamp(self):
        message = _job_result_to_message(self._result(start_time=Timestamp.from_ns(42)))
        assert message.timestamp == Timestamp.from_ns(42)

    def test_raises_when_start_time_is_none(self):
        """Regression test for #1016: never publish an epoch-0 fallback timestamp."""
        with pytest.raises(ValueError, match="no start time"):
            _job_result_to_message(self._result(start_time=None))
