# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog

from ess.livedata.core.constants import PULSE_RATE_HZ
from ess.livedata.core.message import Message
from ess.livedata.core.timestamp import Duration, Timestamp

logger = structlog.get_logger(__name__)


@dataclass(slots=True, kw_only=True)
class MessageBatch:
    start_time: Timestamp
    end_time: Timestamp
    messages: list[Message[Any]]


# Largest jump between consecutive timestamps that still reads as continuous
# traffic, in multiples of the batch length.  Data-derived timestamps are the
# batchers' only clock, so window placement must not follow one outlier
# arbitrarily far into the future: an over-anchored window stalls delivery
# until wall time catches up, while an under-anchored one self-heals forward
# as newer traffic arrives.  The adapter boundary clamps far-future
# timestamps to an arrival estimate; this cap is defense in depth for values
# inside the adapter's bound -- including wall-clock clamps, which land ahead
# of the traffic when consuming a backlog -- and for non-Kafka sources.
MAX_TIMESTAMP_AHEAD_BATCHES = 3


def plausible_anchor(timestamps: list[Timestamp], batch_length: Duration) -> Timestamp:
    """Newest timestamp still connected to the bulk of the given traffic.

    Batch windows are anchored at the newest timestamp seen.  A bare ``max``
    lets a single pathological far-future timestamp pin the window (and with
    it the batcher's data-derived clock), stalling delivery until wall time
    catches up.

    Splitting the sorted timestamps at every jump wider than the horizon
    yields groups of mutually-connected traffic; the anchor is the newest
    timestamp of the largest group.  Membership of the bulk is what marks a
    timestamp as genuine, not distance from an average: a burst spread evenly
    over many batch lengths stays one group and keeps its true maximum, while
    a tight cluster plus one stray yields the cluster -- whether the stray
    sits ahead of the traffic or behind it, as a stream in a disjoint epoch
    does.

    Ties go to the later group, and a lone timestamp is necessarily its own
    anchor.  Neither case has evidence to weigh, and anchoring behind the
    traffic is not the safe default it appears to be: a window only walks
    forward one batch length per call, so an anchor stranded in a disjoint
    past epoch takes astronomically many calls to catch up, where one
    stranded ahead merely waits out its own distance.  Resolving these is
    the adapter boundary's job, which clamps an implausible timestamp to an
    arrival estimate rather than weighing it against its neighbours.
    """
    ordered = sorted(timestamps)
    horizon = MAX_TIMESTAMP_AHEAD_BATCHES * batch_length
    groups = [[ordered[0]]]
    for timestamp in ordered[1:]:
        if timestamp - groups[-1][-1] > horizon:
            groups.append([])
        groups[-1].append(timestamp)
    largest = max(len(group) for group in groups)
    return next(group for group in reversed(groups) if len(group) == largest)[-1]


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
        self,
        batch_length_s: float = 1.0,
        pulse_rate_hz: float = PULSE_RATE_HZ,
    ) -> None:
        # Batch length is currently ignored.
        self._batch_length = Duration.from_seconds(batch_length_s)
        self._pulse_length = Duration.from_seconds(1.0 / pulse_rate_hz)

    @property
    def batch_length_s(self) -> float:
        return self._batch_length.to_seconds()

    def batch(self, messages: list[Message[Any]]) -> MessageBatch | None:
        if not messages:
            return None
        messages = sorted(messages)
        # start_time is the lower bound of the batch, end_time is the upper bound, both
        # in multiples of the pulse length.
        start_time = messages[0].timestamp.quantize(self._pulse_length)
        end_time = messages[-1].timestamp.quantize_up(self._pulse_length)
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
    The initial batch may also contain messages after its end time, since its end
    time is an outlier-robust anchor rather than a bare maximum (see
    :func:`plausible_anchor`).

    If no messages are available for a given batch, an empty batch is returned.

    When the first message for the next batch is received, the current batch is closed
    and returned, even if no messages were received for it.

    Attention: This means that if messages arrive very infrequently, the batcher may
    return messages very late (or never, if the stream stops). This might cause issues
    but for now we want to avoid relying on wall-clock time. Instead, raw data messages
    serve as our clock.
    """

    def __init__(self, batch_length_s: float = 1.0) -> None:
        self._batch_length = Duration.from_seconds(batch_length_s)
        self._active_batch: MessageBatch | None = None
        self._future_messages: list[Message[Any]] = []

    @property
    def batch_length_s(self) -> float:
        return self._batch_length.to_seconds()

    def set_batch_length(self, batch_length_s: float) -> None:
        """Update the batch length for future batches.

        The current active batch keeps its boundaries and completes normally.
        Only the next batch boundary will use the new length.
        """
        self._batch_length = Duration.from_seconds(batch_length_s)

    def batch(self, messages: list[Message[Any]]) -> MessageBatch | None:
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
        new_end_time = batch.end_time + self._batch_length
        if not self._may_advance_to(new_end_time):
            return None
        new_active, self._future_messages = self._split_messages(
            self._future_messages, new_end_time
        )
        self._active_batch = MessageBatch(
            start_time=batch.end_time, end_time=new_end_time, messages=new_active
        )
        return batch

    def _make_initial_batch(self, messages: list[Message[Any]]) -> MessageBatch | None:
        """Make initial batch that includes everything.

        The end time anchors all subsequent batch boundaries, so it must be
        robust against a pathological far-future timestamp in the startup
        backlog (see :func:`plausible_anchor`).
        """
        if not messages:
            return None
        start_time = min(msg.timestamp for msg in messages)
        end_time = plausible_anchor(
            [msg.timestamp for msg in messages], self._batch_length
        )
        batch = MessageBatch(
            start_time=start_time, end_time=end_time, messages=messages
        )
        # After the initial batch we align to batch boundaries.  The next batch starts
        # immediately after the initial batch ends.
        next_start = end_time
        self._active_batch = MessageBatch(
            start_time=next_start,
            end_time=next_start + self._batch_length,
            messages=[],
        )
        return batch

    def _may_advance_to(self, end_time: Timestamp) -> bool:
        """Whether advancing the window to ``end_time`` keeps it near the data.

        Closing a batch is driven by the mere presence of a future message, so
        a message parked implausibly far ahead advances the window on every
        call -- once per poll rather than once per batch length. The window
        then overruns the live data and comes to rest at the outlier, and
        nothing closes a batch until wall time catches up: the #1038 wedge,
        merely bounded by the adapter's future bound. Refusing to advance past
        the frontier of the data in hand keeps the window with the traffic,
        and the frontier moves forward as real messages arrive.

        The active batch's own messages are the frontier: they landed before
        its end time by construction, so they mark where the traffic actually
        is, and no future-dated value can inflate them. An empty active batch
        raises no objection -- that is a genuine silence gap, which advances
        window by window as it always has.
        """
        if not self._active_batch.messages:
            return True
        frontier = max(msg.timestamp for msg in self._active_batch.messages)
        return end_time <= frontier + MAX_TIMESTAMP_AHEAD_BATCHES * self._batch_length

    def _split_messages(
        self, messages: list[Message[Any]], timestamp: Timestamp
    ) -> tuple[list[Message[Any]], list[Message[Any]]]:
        # Note that the batch start time will "lie" if there are late messages, but this
        # is currently considered acceptable and better than dropping messages.
        before = [msg for msg in messages if msg.timestamp < timestamp]
        after = [msg for msg in messages if msg.timestamp >= timestamp]
        return before, after


ESCALATION_OVERLOAD_THRESHOLD = 2
ESCALATION_HALF_STEPS = 2
# A batch is "underloaded" (de-escalation candidate) when its processing time is
# below this fraction of the window; the stable "dead zone" is therefore
# [ratio, 1.0] in utilisation.  De-escalating one half-step shrinks the window by
# 1/sqrt(2), i.e. raises utilisation by the consecutive-level ratio.  For every
# workload to have a stable level (utilisation landing in the dead zone at *some*
# level) the dead zone must span one such step: ratio <= 1 / (level ratio).  The
# nominal ratio is sqrt(2) ~ 1.414, but pulse-quantization rounds windows so the
# widest consecutive ratio is ~1.43 (e.g. round(14*sqrt(2))/14 = 20/14).  Hence
# ratio <= 1/1.43 ~ 0.70.  A larger value (e.g. 0.75) leaves a gap where a
# workload is underloaded at every escalated level yet overloaded one step down,
# causing steady oscillation.
DEESCALATION_HEADROOM_RATIO = 0.70
DEESCALATION_UNDERLOAD_THRESHOLD = 3
DEESCALATION_IDLE_WINDOWS = 3

_SQRT2 = 2**0.5


def _compute_pulse_grid(
    base_batch_length_s: float, max_half_steps: int, pulse_rate_hz: float
) -> list[int]:
    """Precompute pulse-quantized batch lengths for each half-step level.

    Returns a list of integer pulse counts.  The batch length for half-step *n*
    is ``pulse_counts[n] / pulse_rate_hz`` seconds.
    """
    base_pulses = base_batch_length_s * pulse_rate_hz
    return [round(base_pulses * _SQRT2**n) for n in range(max_half_steps + 1)]


@dataclass(frozen=True)
class AdaptiveBatcherState:
    """State snapshot of an AdaptiveMessageBatcher for status reporting."""

    level: int
    batch_length_s: float


class AdaptiveMessageBatcher(MessageBatcher):
    """A message batcher that dynamically adjusts its batch length based on load.

    Wraps an inner batcher (defaulting to :class:`SimpleMessageBatcher`) and uses
    processing-time feedback to detect overload.  When processing consistently
    exceeds the batch window, the batcher escalates by doubling the window
    (+2 half-steps).  When processing completes with headroom, it de-escalates
    by a factor of 1/sqrt(2) (-1 half-step).

    The inner batcher must accept ``batch_length_s`` as its first argument and
    expose ``set_batch_length`` so the outer batcher can adjust its window.

    The asymmetric step sizes mean two de-escalation steps undo one escalation,
    providing natural damping.  Batch lengths are quantized to integer multiples
    of PULSE_RATE_HZ, ensuring clean message counts for
    pulse-aligned producers.

    Idle periods also trigger de-escalation via a wall-clock fallback.
    """

    def __init__(
        self,
        base_batch_length_s: float = 1.0,
        max_level: int = 3,
        clock: Callable[[], float] = time.monotonic,
        pulse_rate_hz: float = PULSE_RATE_HZ,
        inner_factory: Callable[[float], MessageBatcher] = SimpleMessageBatcher,
    ) -> None:
        self._max_half_steps = max_level * 2
        self._pulse_rate_hz = pulse_rate_hz
        self._pulse_counts = _compute_pulse_grid(
            base_batch_length_s, self._max_half_steps, pulse_rate_hz
        )
        self._half_step = 0
        self._consecutive_overloaded = 0
        self._consecutive_underloaded = 0
        self._last_nonempty_batch_time: float | None = None
        self._clock = clock
        self._inner = inner_factory(self.batch_length_s)

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
        return self._pulse_counts[self._half_step] / self._pulse_rate_hz

    @property
    def state(self) -> AdaptiveBatcherState:
        return AdaptiveBatcherState(
            level=self._half_step,
            batch_length_s=self.batch_length_s,
        )
