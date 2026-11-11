# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import logging

from ess.livedata.core.job import (
    LAG_FUTURE_TOLERANCE_S,
    LAG_WARN_THRESHOLD_S,
    StreamStat,
    StreamStatsProvider,
)
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.kafka.stream_counter import StreamCounter
from ess.livedata.kafka.stream_mapping import InputStreamKey

T0 = Timestamp.from_ns(1_000_000_000_000)


def _at(seconds: float) -> Timestamp:
    return Timestamp.from_ns(T0.to_ns() + int(seconds * 1e9))


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

    def test_drops_out_of_scope_streams(self) -> None:
        counter = StreamCounter(
            out_of_scope=[InputStreamKey(topic="logs", source_name="other_pv.RBV")]
        )
        counter.record("logs", "other_pv.RBV", None)
        stats = counter.drain(window_seconds=30.0)
        assert stats.streams == ()

    def test_genuinely_unmapped_still_recorded_with_out_of_scope_set(self) -> None:
        counter = StreamCounter(
            out_of_scope=[InputStreamKey(topic="logs", source_name="other_pv.RBV")]
        )
        counter.record("logs", "unexpected_pv.RBV", None)
        stats = counter.drain(window_seconds=30.0)
        assert len(stats.streams) == 1
        assert stats.streams[0].source_name == "unexpected_pv.RBV"
        assert stats.streams[0].stream is None

    def test_out_of_scope_match_is_topic_specific(self) -> None:
        counter = StreamCounter(
            out_of_scope=[InputStreamKey(topic="logs", source_name="pv.RBV")]
        )
        counter.record("other_topic", "pv.RBV", None)
        stats = counter.drain(window_seconds=30.0)
        assert len(stats.streams) == 1
        assert stats.streams[0].topic == "other_topic"


class TestStreamCounterLag:
    def test_drain_lag_empty_returns_none(self) -> None:
        assert StreamCounter().drain_lag() is None

    def test_aggregates_min_max_count_per_key(self) -> None:
        counter = StreamCounter()
        for lag in (1.0, 3.0, 2.0):
            counter.record_lag(
                "topic_a", "source_1", "ev44", lag, timestamp=T0, partition=0
            )
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
        counter.record_lag("t", "s", "ev44", 1.0, timestamp=T0, partition=0)
        counter.record_lag("t", "s", "da00", 1.0, timestamp=T0, partition=0)
        counter.record_lag("t", "other", "ev44", 1.0, timestamp=T0, partition=0)
        report = counter.drain_lag()
        assert report is not None
        assert len(report.streams) == 3

    def test_drain_lag_resets(self) -> None:
        counter = StreamCounter()
        counter.record_lag("t", "s", "ev44", 1.0, timestamp=T0, partition=0)
        counter.drain_lag()
        assert counter.drain_lag() is None

    def test_streams_ordered_by_key(self) -> None:
        counter = StreamCounter()
        counter.record_lag("t_b", "s", "ev44", 1.0, timestamp=T0, partition=0)
        counter.record_lag("t_a", "s", "f144", 1.0, timestamp=T0, partition=0)
        counter.record_lag("t_a", "s", "ev44", 1.0, timestamp=T0, partition=0)
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
        counter.record_lag("motion", "PV.DMOV", "f144", 1.0, timestamp=T0, partition=0)
        counter.record_lag("motion", "PV.VAL", "f144", 1.0, timestamp=T0, partition=0)
        counter.record_lag("motion", "PV.RBV", "f144", 1.0, timestamp=T0, partition=0)
        report = counter.drain_lag()
        assert report is not None
        assert [s.source for s in report.streams] == ["PV.RBV"]

    def _level(self, lag_s: float) -> int:
        counter = StreamCounter()
        counter.record_lag("t", "s", "ev44", lag_s, timestamp=T0, partition=0)
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
        counter.record_lag(
            "t", "stale", "ev44", LAG_WARN_THRESHOLD_S + 5, timestamp=T0, partition=0
        )
        counter.record_lag(
            "t",
            "future",
            "ev44",
            -(LAG_FUTURE_TOLERANCE_S + 1),
            timestamp=T0,
            partition=0,
        )
        report = counter.drain_lag()
        assert report is not None
        by_source = {lag.source: report.level(lag) for lag in report.streams}
        assert by_source == {"stale": logging.WARNING, "future": logging.ERROR}


class TestStreamCounterOutOfOrder:
    def test_in_order_stream_reports_no_reordering(self) -> None:
        counter = StreamCounter()
        for t in (0.0, 1.0, 2.0):
            counter.record_lag("t", "s", "ev44", 1.0, timestamp=_at(t), partition=0)
        report = counter.drain_lag()
        assert report is not None
        (entry,) = report.streams
        assert entry.max_out_of_order_s == 0.0
        assert entry.out_of_order_partition is None

    def test_reports_largest_distance_behind_newest_and_its_partition(self) -> None:
        counter = StreamCounter()
        for t, partition in ((10.0, 0), (12.0, 1), (11.5, 2), (9.0, 3), (12.5, 4)):
            counter.record_lag(
                "t", "s", "ev44", 1.0, timestamp=_at(t), partition=partition
            )
        report = counter.drain_lag()
        assert report is not None
        (entry,) = report.streams
        assert entry.max_out_of_order_s == 3.0
        assert entry.out_of_order_partition == 3

    def test_newest_payload_time_carries_across_drains(self) -> None:
        counter = StreamCounter()
        counter.record_lag("t", "s", "ev44", 1.0, timestamp=_at(10.0), partition=0)
        counter.drain_lag()
        counter.record_lag("t", "s", "ev44", 1.0, timestamp=_at(7.0), partition=5)
        report = counter.drain_lag()
        assert report is not None
        (entry,) = report.streams
        assert entry.max_out_of_order_s == 3.0
        assert entry.out_of_order_partition == 5

    def test_future_payload_time_does_not_become_the_reference(self) -> None:
        """One payload stamped far ahead of its Kafka create time must not make
        every later, correctly stamped message read as out of order."""
        counter = StreamCounter()
        counter.record_lag("t", "s", "ev44", 1.0, timestamp=_at(10.0), partition=0)
        counter.record_lag(
            "t",
            "s",
            "ev44",
            -(LAG_FUTURE_TOLERANCE_S + 1e6),
            timestamp=_at(1e6),
            partition=1,
        )
        counter.record_lag("t", "s", "ev44", 1.0, timestamp=_at(11.0), partition=0)
        report = counter.drain_lag()
        assert report is not None
        (entry,) = report.streams
        assert entry.max_out_of_order_s == 0.0
        assert entry.out_of_order_partition is None

    def test_order_is_tracked_per_stream(self) -> None:
        counter = StreamCounter()
        counter.record_lag("t", "a", "ev44", 1.0, timestamp=_at(10.0), partition=0)
        counter.record_lag("t", "b", "ev44", 1.0, timestamp=_at(5.0), partition=1)
        report = counter.drain_lag()
        assert report is not None
        assert [s.max_out_of_order_s for s in report.streams] == [0.0, 0.0]
