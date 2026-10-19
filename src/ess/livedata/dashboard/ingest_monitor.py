# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Measurement of whether the dashboard's ingestion loop keeps up with Kafka.

One thread consumes from Kafka and then rebuilds every visible plot whose data
changed, in a loop. When passes take longer than the input allows, messages pile
up in Kafka and the dashboard shows ever older data while the backend reports
no lag: the backend's ``stream_lag`` ends at publish. The dashboard's consumer
starts at the latest offset, so a restart silently skips the backlog.

This module reports the loop from the loop itself, so a backlog shows up as a
growing ``consumer_lag`` rather than as user reports of a growing lag pill.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import structlog

logger = structlog.get_logger(__name__)

_SUMMARY_INTERVAL_S = 60.0
"""How often the accumulated view of ingestion is logged."""


class IngestMonitor:
    """Reports throughput and backlog of the dashboard's ingestion loop.

    Emits ``dashboard_ingest_metrics`` at INFO once per summary interval:

    - ``messages`` and ``passes``: input rate and how it is split into passes.
    - ``max_messages_per_pass``: the largest single pass.
    - ``pump_seconds`` and ``flush_seconds``: time spent ingesting messages and
      rebuilding plots, which tells which side of the loop limits it. Ingesting
      includes the consumer's poll timeout, so an idle loop reads as pumping.
    - ``consumer_lag``: messages published but not yet consumed, sampled at the
      end of the interval. Dividing by the message rate gives the backlog in
      seconds. ``None`` where the transport cannot tell. The high watermark
      is the one cached from the last fetch, so while librdkafka pauses
      fetching (local queue full) a sample under-reports; a sustained backlog
      still shows as growth across summaries.

    Parameters
    ----------
    consumer_lag:
        Returns the current consumer lag. Called on the loop's thread.
    summary_interval:
        Seconds between summaries.
    clock:
        Monotonic clock in seconds.
    """

    def __init__(
        self,
        *,
        consumer_lag: Callable[[], int | None],
        summary_interval: float = _SUMMARY_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._consumer_lag = consumer_lag
        self._summary_interval = summary_interval
        self._clock = clock
        self._window_start = clock()
        self._reset()

    def _reset(self) -> None:
        self._passes = 0
        self._messages = 0
        self._max_messages_per_pass = 0
        self._pump_seconds = 0.0
        self._flush_seconds = 0.0

    def record_pass(
        self, messages: int, *, pump_seconds: float, flush_seconds: float
    ) -> None:
        """Account for one pass of the loop and log a summary when one is due."""
        self._passes += 1
        self._messages += messages
        self._max_messages_per_pass = max(self._max_messages_per_pass, messages)
        self._pump_seconds += pump_seconds
        self._flush_seconds += flush_seconds
        now = self._clock()
        if now - self._window_start >= self._summary_interval:
            self._log_summary(now)

    def _log_summary(self, now: float) -> None:
        logger.info(
            'dashboard_ingest_metrics',
            passes=self._passes,
            messages=self._messages,
            max_messages_per_pass=self._max_messages_per_pass,
            pump_seconds=round(self._pump_seconds, 3),
            flush_seconds=round(self._flush_seconds, 3),
            consumer_lag=self._consumer_lag(),
            interval_seconds=round(now - self._window_start, 1),
        )
        self._window_start = now
        self._reset()
