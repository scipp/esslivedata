# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import Any

import scipp as sc

from ess.livedata.core.handler import Accumulator
from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.core.message_batcher import MessageBatch
from ess.livedata.core.orchestrating_processor import MessagePreprocessor
from ess.livedata.handlers.accumulators import LogData
from ess.livedata.handlers.to_nxlog import ToNXlog


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

    def make_preprocessor(self, key: StreamId) -> Accumulator | None:
        return self._accumulators.get(key.name)


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
