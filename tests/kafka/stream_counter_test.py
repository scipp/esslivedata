# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ess.livedata.core.job import StreamStat, StreamStatsProvider
from ess.livedata.kafka.stream_counter import StreamCounter


class TestStreamCounter:
    def test_implements_protocol(self) -> None:
        assert isinstance(StreamCounter(), StreamStatsProvider)

    def test_drain_empty_returns_empty_stats(self) -> None:
        counter = StreamCounter()
        stats = counter.drain(window_seconds=30.0)
        assert stats.window_seconds == 30.0
        assert stats.streams == ()

    def test_record_and_drain_single_stream(self) -> None:
        counter = StreamCounter()
        counter.record("topic_a", "source_1", "stream_1")
        counter.record("topic_a", "source_1", "stream_1")
        counter.record("topic_a", "source_1", "stream_1")

        stats = counter.drain(window_seconds=30.0)
        assert len(stats.streams) == 1
        assert stats.streams[0] == StreamStat(
            topic="topic_a", source_name="source_1", stream="stream_1", count=3
        )

    def test_record_and_drain_multiple_streams(self) -> None:
        counter = StreamCounter()
        counter.record("topic_a", "source_1", "stream_1")
        counter.record("topic_a", "source_1", "stream_1")
        counter.record("topic_b", "source_2", "stream_2")

        stats = counter.drain(window_seconds=30.0)
        assert len(stats.streams) == 2

    def test_drain_resets_counts(self) -> None:
        counter = StreamCounter()
        counter.record("topic_a", "source_1", "stream_1")

        stats1 = counter.drain(window_seconds=30.0)
        assert len(stats1.streams) == 1

        stats2 = counter.drain(window_seconds=30.0)
        assert stats2.streams == ()

    def test_unmapped_stream_recorded_with_none(self) -> None:
        counter = StreamCounter()
        counter.record("topic_a", "unknown_source", None)

        stats = counter.drain(window_seconds=30.0)
        assert len(stats.streams) == 1
        assert stats.streams[0].stream is None
        assert stats.streams[0].count == 1

    def test_streams_sorted_by_topic_and_source(self) -> None:
        counter = StreamCounter()
        counter.record("topic_b", "source_2", "s2")
        counter.record("topic_a", "source_1", "s1")
        counter.record("topic_a", "source_2", "s3")

        stats = counter.drain(window_seconds=30.0)
        topics = [(s.topic, s.source_name) for s in stats.streams]
        assert topics == [
            ("topic_a", "source_1"),
            ("topic_a", "source_2"),
            ("topic_b", "source_2"),
        ]

    def test_window_seconds_passed_through(self) -> None:
        counter = StreamCounter()
        stats = counter.drain(window_seconds=42.5)
        assert stats.window_seconds == 42.5

    def test_ignores_epics_dmov_suffix(self) -> None:
        counter = StreamCounter()
        counter.record("motion", "INST-Dev:Mtr.DMOV", None)
        counter.record("motion", "INST-Dev:Mtr.RBV", "stream_1")
        stats = counter.drain(window_seconds=30.0)
        assert len(stats.streams) == 1
        assert stats.streams[0].source_name == "INST-Dev:Mtr.RBV"

    def test_ignores_epics_val_suffix(self) -> None:
        counter = StreamCounter()
        counter.record("motion", "INST-Dev:Mtr.VAL", None)
        counter.record("motion", "INST-Dev:Mtr.RBV", "stream_1")
        stats = counter.drain(window_seconds=30.0)
        assert len(stats.streams) == 1
        assert stats.streams[0].source_name == "INST-Dev:Mtr.RBV"

    def test_ignores_dmov_and_val_on_any_topic(self) -> None:
        counter = StreamCounter()
        counter.record("other_topic", "PV.DMOV", None)
        counter.record("other_topic", "PV.VAL", None)
        counter.record("other_topic", "PV.RBV", "resolved")
        stats = counter.drain(window_seconds=30.0)
        assert len(stats.streams) == 1
