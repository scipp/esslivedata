# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Per-stream message counter and lag tracker for observability."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from ess.livedata.core.job import (
    LAG_FUTURE_TOLERANCE_S,
    LAG_WARN_THRESHOLD_S,
    StreamLag,
    StreamLagReport,
    StreamStat,
    StreamStats,
)
from ess.livedata.kafka.stream_mapping import InputStreamKey

# EPICS PV field suffixes that are noise for reporting.
# Only .RBV (Read Back Value) carries the actual readback; .VAL (setpoint)
# and .DMOV (done-moving flag) are redundant in our dashboards.
_IGNORED_SOURCE_SUFFIXES = ('.DMOV', '.VAL')


@dataclass(slots=True)
class _Entry:
    stream: str | None
    count: int
    clamped: int = 0


@dataclass(slots=True)
class _LagEntry:
    min_s: float
    max_s: float
    count: int


class StreamCounter:
    """Counts messages per (topic, source_name) and tracks per-stream lag.

    Call :meth:`record` from the adapter layer each time a message is mapped
    (or fails to map) to a stream, and :meth:`record_lag` when a payload-derived
    timestamp is available. Call :meth:`drain` / :meth:`drain_lag` from the
    processor on metrics rollover to collect the accumulated values and reset.
    """

    def __init__(
        self,
        *,
        out_of_scope: Iterable[InputStreamKey] = (),
        lag_warn_threshold_s: float = LAG_WARN_THRESHOLD_S,
        lag_future_tolerance_s: float = LAG_FUTURE_TOLERANCE_S,
    ) -> None:
        self._counts: dict[tuple[str, str], _Entry] = defaultdict(
            lambda: _Entry(stream=None, count=0)
        )
        self._out_of_scope = {(k.topic, k.source_name) for k in out_of_scope}
        self._lag: dict[tuple[str, str, str], _LagEntry] = {}
        self._lag_warn_threshold_s = lag_warn_threshold_s
        self._lag_future_tolerance_s = lag_future_tolerance_s

    def record(self, topic: str, source_name: str, stream: str | None) -> None:
        """Increment count for a (topic, source_name) combination.

        Sources whose names end with known EPICS noise suffixes (.DMOV, .VAL)
        are silently dropped to reduce clutter in status displays. Streams known
        to the instrument but routed to another service (``out_of_scope``) are
        also dropped: they only arrive because Kafka subscription is per-topic,
        and counting them as unmapped pollutes the status display.
        """
        if source_name.endswith(_IGNORED_SOURCE_SUFFIXES):
            return
        key = (topic, source_name)
        if key in self._out_of_scope:
            return
        entry = self._counts[key]
        entry.count += 1
        entry.stream = stream

    def record_clamped(self, topic: str, source_name: str) -> bool:
        """Record a message whose implausibly-far-future timestamp was clamped.

        Clamps are a subset of the messages counted by :meth:`record`, so a
        stream whose timestamps are *all* clamped reports ``count == clamped``.
        Sources with known EPICS noise suffixes and out-of-scope streams are
        dropped, mirroring :meth:`record`.

        Returns
        -------
        :
            True if this is the first clamp recorded for this stream in the
            current window, so callers can rate-limit logging to once per
            stream per window instead of once per message.
        """
        if source_name.endswith(_IGNORED_SOURCE_SUFFIXES):
            return False
        key = (topic, source_name)
        if key in self._out_of_scope:
            return False
        entry = self._counts[key]
        is_first = entry.clamped == 0
        entry.clamped += 1
        return is_first

    def record_lag(
        self, topic: str, source_name: str, schema: str, lag_s: float
    ) -> None:
        """Fold one message's lag into the current window.

        Lag is ``kafka_create_time - payload_timestamp`` in seconds. Sources with
        known EPICS noise suffixes are dropped, mirroring :meth:`record`.
        """
        if source_name.endswith(_IGNORED_SOURCE_SUFFIXES):
            return
        key = (topic, source_name, schema)
        entry = self._lag.get(key)
        if entry is None:
            self._lag[key] = _LagEntry(min_s=lag_s, max_s=lag_s, count=1)
        else:
            entry.min_s = min(entry.min_s, lag_s)
            entry.max_s = max(entry.max_s, lag_s)
            entry.count += 1

    def drain(self, window_seconds: float) -> StreamStats:
        """Return accumulated counts and reset."""
        stats = StreamStats(
            window_seconds=window_seconds,
            streams=tuple(
                StreamStat(
                    topic=topic,
                    source_name=source_name,
                    stream=entry.stream,
                    count=entry.count,
                    clamped=entry.clamped,
                )
                for (topic, source_name), entry in sorted(self._counts.items())
            ),
        )
        self._counts.clear()
        return stats

    def drain_lag(self) -> StreamLagReport | None:
        """Return accumulated per-stream lag and reset, or None if empty.

        Streams are ordered by ``(topic, source, schema)`` so that successive
        windows list them in the same positions, easing line-by-line comparison.
        """
        if not self._lag:
            return None
        streams = tuple(
            StreamLag(
                topic=topic,
                source=source,
                schema=schema,
                min_s=entry.min_s,
                max_s=entry.max_s,
                count=entry.count,
            )
            for (topic, source, schema), entry in sorted(self._lag.items())
        )
        self._lag.clear()
        return StreamLagReport(
            streams=streams,
            warn_threshold_s=self._lag_warn_threshold_s,
            future_tolerance_s=self._lag_future_tolerance_s,
        )
