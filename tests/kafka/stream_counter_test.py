# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import logging

from ess.livedata.core.job import (
    LAG_FUTURE_TOLERANCE_S,
    LAG_WARN_THRESHOLD_S,
    StreamStat,
    StreamStatsProvider,
)
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


class TestStreamCounterLag:
    def test_drain_lag_empty_returns_none(self) -> None:
        assert StreamCounter().drain_lag() is None

    def test_aggregates_min_max_count_per_key(self) -> None:
        counter = StreamCounter()
        for lag in (1.0, 3.0, 2.0):
            counter.record_lag("topic_a", "source_1", "ev44", lag)
        report = counter.drain_lag()
        assert report is not None
        (entry,) = report.streams
        assert (entry.topic, entry.source, entry.schema) == (
            "topic_a",
            "source_1",
            "ev44",
        )
        assert (entry.min_s, entry.max_s, entry.count) == (1.0, 3.0, 3)

    def test_distinct_keys_by_topic_source_schema(self) -> None:
        counter = StreamCounter()
        counter.record_lag("t", "s", "ev44", 1.0)
        counter.record_lag("t", "s", "da00", 1.0)
        counter.record_lag("t", "other", "ev44", 1.0)
        report = counter.drain_lag()
        assert report is not None
        assert len(report.streams) == 3

    def test_drain_lag_resets(self) -> None:
        counter = StreamCounter()
        counter.record_lag("t", "s", "ev44", 1.0)
        counter.drain_lag()
        assert counter.drain_lag() is None

    def test_streams_ordered_by_key(self) -> None:
        counter = StreamCounter()
        counter.record_lag("t_b", "s", "ev44", 1.0)
        counter.record_lag("t_a", "s", "f144", 1.0)
        counter.record_lag("t_a", "s", "ev44", 1.0)
        report = counter.drain_lag()
        assert report is not None
        keys = [(s.topic, s.source, s.schema) for s in report.streams]
        assert keys == [
            ("t_a", "s", "ev44"),
            ("t_a", "s", "f144"),
            ("t_b", "s", "ev44"),
        ]

    def test_ignores_epics_noise_suffixes(self) -> None:
        counter = StreamCounter()
        counter.record_lag("motion", "PV.DMOV", "f144", 1.0)
        counter.record_lag("motion", "PV.VAL", "f144", 1.0)
        counter.record_lag("motion", "PV.RBV", "f144", 1.0)
        report = counter.drain_lag()
        assert report is not None
        assert [s.source for s in report.streams] == ["PV.RBV"]

    def _level(self, lag_s: float) -> int:
        counter = StreamCounter()
        counter.record_lag("t", "s", "ev44", lag_s)
        report = counter.drain_lag()
        assert report is not None
        return report.level(report.streams[0])

    def test_level_info_for_small_positive_lag(self) -> None:
        assert self._level(0.5) == logging.INFO

    def test_level_warning_above_threshold(self) -> None:
        assert self._level(LAG_WARN_THRESHOLD_S + 0.1) == logging.WARNING

    def test_level_error_for_future_payload(self) -> None:
        assert self._level(-(LAG_FUTURE_TOLERANCE_S + 0.1)) == logging.ERROR

    def test_small_negative_within_tolerance_stays_info(self) -> None:
        assert self._level(-LAG_FUTURE_TOLERANCE_S / 2) == logging.INFO

    def test_level_is_per_stream(self) -> None:
        counter = StreamCounter()
        counter.record_lag("t", "stale", "ev44", LAG_WARN_THRESHOLD_S + 5)
        counter.record_lag("t", "future", "ev44", -(LAG_FUTURE_TOLERANCE_S + 1))
        report = counter.drain_lag()
        assert report is not None
        by_source = {lag.source: report.level(lag) for lag in report.streams}
        assert by_source == {"stale": logging.WARNING, "future": logging.ERROR}
