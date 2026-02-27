# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Load shedding for backend message processing.

When the backend can't keep up with the Kafka message stream, the LoadShedder
selectively drops bulk event data while preserving control messages and f144 logs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .message import Message, StreamKind

DROPPABLE_KINDS = frozenset(
    {
        StreamKind.DETECTOR_EVENTS,
        StreamKind.MONITOR_EVENTS,
        StreamKind.MONITOR_COUNTS,
        StreamKind.AREA_DETECTOR,
    }
)

# Consecutive non-None batcher results before entering shedding mode
_ACTIVATION_THRESHOLD = 5
# Consecutive idle (None) batcher results before exiting shedding mode
_DEACTIVATION_THRESHOLD = 3


@dataclass(frozen=True, slots=True)
class LoadShedderState:
    """Snapshot of load shedder state for status reporting."""

    is_shedding: bool
    messages_dropped: int


class LoadShedder:
    """Selectively drops bulk event data when the backend falls behind.

    Detection uses consecutive non-None batcher results as the overload signal.
    When active, keeps every 2nd droppable message (50% reduction).
    """

    def __init__(self) -> None:
        self._consecutive_batches: int = 0
        self._consecutive_idle: int = 0
        self._is_shedding: bool = False
        self._messages_dropped: int = 0
        self._subsample_counter: int = 0

    @property
    def state(self) -> LoadShedderState:
        return LoadShedderState(
            is_shedding=self._is_shedding,
            messages_dropped=self._messages_dropped,
        )

    def report_batch_result(self, batch_produced: bool) -> None:
        """Update overload detection counters after a batcher cycle.

        Parameters
        ----------
        batch_produced:
            True if the batcher returned a batch (non-None), False if idle (None).
        """
        if batch_produced:
            self._consecutive_batches += 1
            self._consecutive_idle = 0
            if (
                not self._is_shedding
                and self._consecutive_batches >= _ACTIVATION_THRESHOLD
            ):
                self._is_shedding = True
        else:
            self._consecutive_idle += 1
            self._consecutive_batches = 0
            if self._is_shedding and self._consecutive_idle >= _DEACTIVATION_THRESHOLD:
                self._is_shedding = False
                self._subsample_counter = 0

    def shed(self, messages: list[Message]) -> list[Message]:
        """Filter messages when shedding is active.

        When inactive, returns all messages unchanged.
        When active, drops every other droppable message (50% reduction).
        Non-droppable messages (control, f144 logs) are always preserved.
        """
        if not self._is_shedding:
            return messages
        result: list[Message] = []
        for msg in messages:
            if msg.stream.kind not in DROPPABLE_KINDS:
                result.append(msg)
            else:
                self._subsample_counter += 1
                if self._subsample_counter % 2 == 0:
                    result.append(msg)
                else:
                    self._messages_dropped += 1
        return result
