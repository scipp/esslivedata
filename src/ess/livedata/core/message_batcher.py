# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Number
from typing import Any

import structlog

from ess.livedata.core.message import Message

logger = structlog.get_logger(__name__)


@dataclass(slots=True, kw_only=True)
class MessageBatch:
    start_time: int
    end_time: int
    messages: list[Message[Any]]


class MessageBatcher(ABC):
    @abstractmethod
    def batch(self, messages: list[Message[Any]]) -> MessageBatch | None:
        """Create and return a message batch if possible.

        If no batch can be created (batch incomplete), return None.
        """

    def report_batch(  # noqa: B027
        self,
        message_count: int | None,
        processing_time_s: float = 0.0,
    ) -> None:
        """Report the outcome of the last processing cycle.

        Called by the processor after each cycle. Batchers that support adaptive
        behavior override this to adjust their batch length. The default is a
        no-op.

        Parameters
        ----------
        message_count:
            Number of messages in the processed batch. ``None`` if the batcher
            returned ``None`` (idle cycle). 0 indicates an empty batch from a
            time gap.
        processing_time_s:
            Wall-clock time spent processing the batch (preprocessing, workflow
            execution, serialization). Used by adaptive batchers to detect
            overload. Ignored for idle cycles.
        """

    @property
    def batch_length_s(self) -> float:
        """Current effective batch length in seconds."""
        return 1.0


class NaiveMessageBatcher(MessageBatcher):
    """
    A naive batcher that always returns all messages as a single batch.

    This is mainly useful for testing.
    """

    def __init__(
        self, batch_length_s: float = 1.0, pulse_length_s: float = 1.0 / 14
    ) -> None:
        # Batch length is currently ignored.
        self._batch_length_s = batch_length_s
        self._batch_length_ns = int(batch_length_s * 1_000_000_000)
        self._pulse_length_ns = int(pulse_length_s * 1_000_000_000)

    @property
    def batch_length_s(self) -> float:
        return self._batch_length_s

    def batch(self, messages: list[Message[Any]]) -> MessageBatch | None:
        # Filter messages with incompatible (broken) timestamps to avoid issues below.
        messages = [msg for msg in messages if isinstance(msg.timestamp, Number)]
        if not messages:
            return None
        messages = sorted(messages)
        # start_time is the lower bound of the batch, end_time is the upper bound, both
        # in multiples of the pulse length.
        start_time = (
            messages[0].timestamp // self._pulse_length_ns * self._pulse_length_ns
        )
        end_time = (
            (messages[-1].timestamp + self._pulse_length_ns - 1)
            // self._pulse_length_ns
            * self._pulse_length_ns
        )
        return MessageBatch(start_time=start_time, end_time=end_time, messages=messages)


class SimpleMessageBatcher(MessageBatcher):
    """
    A simple batcher that creates batches of a fixed length.

    The first batch will include all messages received so far, and subsequent batches
    will be aligned to the configured batch length. That is, if the first batch ends at
    time T, the next batch will cover [T, T + batch_length), the next
    [T + batch_length, T + 2*batch_length), etc.

    If messages arrive late, they will be included in the next batch. This means that
    batches may contain messages with timestamps outside (before) the batch time range.

    If no messages are available for a given batch, an empty batch is returned.

    When the first message for the next batch is received, the current batch is closed
    and returned, even if no messages were received for it.

    Attention: This means that if messages arrive very infrequently, the batcher may
    return messages very late (or never, if the stream stops). This might cause issues
    but for now we want to avoid relying on wall-clock time. Instead, raw data messages
    serve as our clock.
    """

    def __init__(self, batch_length_s: float = 1.0) -> None:
        self._batch_length_s_value = batch_length_s
        self._batch_length_ns = int(batch_length_s * 1_000_000_000)
        self._active_batch: MessageBatch | None = None
        self._future_messages: list[Message[Any]] = []

    @property
    def batch_length_s(self) -> float:
        return self._batch_length_s_value

    def set_batch_length(self, batch_length_s: float) -> None:
        """Update the batch length for future batches.

        The current active batch keeps its boundaries and completes normally.
        Only the next batch boundary will use the new length.
        """
        self._batch_length_s_value = batch_length_s
        self._batch_length_ns = int(batch_length_s * 1_000_000_000)

    def batch(self, messages: list[Message[Any]]) -> MessageBatch | None:
        # Filter messages with incompatible (broken) timestamps to avoid issues below.
        messages = [msg for msg in messages if isinstance(msg.timestamp, Number)]

        # Create and return initial batch including everything
        if self._active_batch is None:
            return self._make_initial_batch(messages)

        # We have an active batch, decide which messages belong to it
        new_active, after = self._split_messages(messages, self._active_batch.end_time)
        self._active_batch.messages.extend(new_active)
        self._future_messages.extend(after)

        # No future messages, assume we will get more messages for the active batch.
        # Note this is different from returning an empty batch.
        if not self._future_messages:
            return None

        # We have future messages, i.e., we assume the active batch is done. This may
        # return an empty batch, which is desired behavior, i.e., we advance batch by
        # batch.
        batch = self._active_batch
        new_end_time = batch.end_time + self._batch_length_ns
        new_active, self._future_messages = self._split_messages(
            self._future_messages, new_end_time
        )
        self._active_batch = MessageBatch(
            start_time=batch.end_time, end_time=new_end_time, messages=new_active
        )
        return batch

    def _make_initial_batch(self, messages: list[Message[Any]]) -> MessageBatch | None:
        """Make initial batch that includes everything."""
        if not messages:
            return None
        start_time = min(msg.timestamp for msg in messages)
        end_time = max(msg.timestamp for msg in messages)
        batch = MessageBatch(
            start_time=start_time, end_time=end_time, messages=messages
        )
        # After the initial batch we align to batch boundaries.  The next batch starts
        # immediately after the initial batch ends.
        next_start = end_time
        self._active_batch = MessageBatch(
            start_time=next_start,
            end_time=next_start + self._batch_length_ns,
            messages=[],
        )
        return batch

    def _split_messages(
        self, messages: list[Message[Any]], timestamp: int
    ) -> tuple[list[Message[Any]], list[Message[Any]]]:
        # Note that the batch start time will "lie" if there are late messages, but this
        # is currently considered acceptable and better than dropping messages.
        before = [msg for msg in messages if msg.timestamp < timestamp]
        after = [msg for msg in messages if msg.timestamp >= timestamp]
        return before, after


ESCALATION_OVERLOAD_THRESHOLD = 2
ESCALATION_HALF_STEPS = 2
DEESCALATION_HEADROOM_RATIO = 0.75
DEESCALATION_UNDERLOAD_THRESHOLD = 3
DEESCALATION_IDLE_WINDOWS = 3

_SQRT2 = 2**0.5


@dataclass(frozen=True)
class AdaptiveBatcherState:
    """State snapshot of an AdaptiveMessageBatcher for status reporting."""

    level: int
    batch_length_s: float


class AdaptiveMessageBatcher(MessageBatcher):
    """A message batcher that dynamically adjusts its batch length based on load.

    Wraps a ``SimpleMessageBatcher`` and uses processing-time feedback to detect
    overload.  When processing consistently exceeds the batch window, the batcher
    escalates by doubling the window (+2 half-steps).  When processing completes
    with headroom, it de-escalates by a factor of 1/sqrt(2) (-1 half-step).

    The asymmetric step sizes mean two de-escalation steps undo one escalation,
    providing natural damping.  The batch window is always on the grid
    ``base * sqrt(2)^n``, avoiding floating-point drift.

    Idle periods also trigger de-escalation via a wall-clock fallback.
    """

    def __init__(
        self,
        base_batch_length_s: float = 1.0,
        max_level: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._base_batch_length_s = base_batch_length_s
        self._max_half_steps = max_level * 2
        self._half_step = 0
        self._consecutive_overloaded = 0
        self._consecutive_underloaded = 0
        self._last_nonempty_batch_time: float | None = None
        self._clock = clock
        self._inner = SimpleMessageBatcher(batch_length_s=base_batch_length_s)

    def batch(self, messages: list[Message[Any]]) -> MessageBatch | None:
        return self._inner.batch(messages)

    def report_batch(
        self,
        message_count: int | None,
        processing_time_s: float = 0.0,
    ) -> None:
        if message_count is None:
            # Idle cycle — no load signal, leave consecutive counters
            # untouched.  Genuine idleness is handled by the wall-clock
            # fallback below; resetting counters here would prevent
            # de-escalation under continuous light load where idle polls
            # between batches outnumber real reports.
            if self._half_step > 0 and self._last_nonempty_batch_time is not None:
                idle_s = self._clock() - self._last_nonempty_batch_time
                idle_windows = idle_s / self.batch_length_s
                if idle_windows >= DEESCALATION_IDLE_WINDOWS:
                    self._set_half_step(self._half_step - 1)
                    self._last_nonempty_batch_time = self._clock()
        elif message_count == 0:
            # Empty batch from time gap — not a load signal
            pass
        else:
            # Non-empty batch — use processing time to decide
            self._last_nonempty_batch_time = self._clock()

            if processing_time_s > self.batch_length_s:
                # Overloaded: processing exceeded the batch window
                self._consecutive_overloaded += 1
                self._consecutive_underloaded = 0
                if (
                    self._consecutive_overloaded >= ESCALATION_OVERLOAD_THRESHOLD
                    and self._half_step < self._max_half_steps
                ):
                    new = min(
                        self._half_step + ESCALATION_HALF_STEPS,
                        self._max_half_steps,
                    )
                    self._set_half_step(new)
                    self._consecutive_overloaded = 0
            elif processing_time_s < self.batch_length_s * DEESCALATION_HEADROOM_RATIO:
                # Underloaded: headroom available
                self._consecutive_underloaded += 1
                self._consecutive_overloaded = 0
                if (
                    self._consecutive_underloaded >= DEESCALATION_UNDERLOAD_THRESHOLD
                    and self._half_step > 0
                ):
                    self._set_half_step(self._half_step - 1)
                    self._consecutive_underloaded = 0
            else:
                # In between — processing fits but without much headroom
                self._consecutive_overloaded = 0
                self._consecutive_underloaded = 0

    def _set_half_step(self, new_half_step: int) -> None:
        old_length = self.batch_length_s
        self._half_step = new_half_step
        new_length = self.batch_length_s
        logger.warning(
            'adaptive_batch_level_change',
            old_batch_length_s=old_length,
            new_batch_length_s=new_length,
            level=self._half_step,
        )
        self._inner.set_batch_length(new_length)

    @property
    def batch_length_s(self) -> float:
        return self._base_batch_length_s * _SQRT2**self._half_step

    @property
    def state(self) -> AdaptiveBatcherState:
        return AdaptiveBatcherState(
            level=self._half_step,
            batch_length_s=self.batch_length_s,
        )
