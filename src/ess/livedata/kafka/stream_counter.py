# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Per-stream message counter for observability."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ess.livedata.core.job import StreamStat, StreamStats


@dataclass(slots=True)
class _Entry:
    stream: str | None
    count: int


class StreamCounter:
    """Counts messages per (topic, source_name) for inclusion in service heartbeats.

    Call :meth:`record` from the adapter layer each time a message is mapped
    (or fails to map) to a stream. Call :meth:`drain` from the processor on
    metrics rollover to collect the accumulated counts and reset.
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], _Entry] = defaultdict(
            lambda: _Entry(stream=None, count=0)
        )

    def record(self, topic: str, source_name: str, stream: str | None) -> None:
        """Increment count for a (topic, source_name) combination."""
        key = (topic, source_name)
        entry = self._counts[key]
        entry.count += 1
        entry.stream = stream

    def drain(self, window_seconds: float) -> StreamStats:
        """Return accumulated stats and reset counters."""
        stats = StreamStats(
            window_seconds=window_seconds,
            streams=tuple(
                StreamStat(
                    topic=topic,
                    source_name=source_name,
                    stream=entry.stream,
                    count=entry.count,
                )
                for (topic, source_name), entry in sorted(self._counts.items())
            ),
        )
        self._counts.clear()
        return stats
