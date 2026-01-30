# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from ..core.message import MessageSource
from .message_adapter import KafkaMessage

logger = structlog.get_logger(__name__)

# Circuit breaker configuration
DEFAULT_MAX_CONSECUTIVE_ERRORS = 10
DEFAULT_HEALTH_TIMEOUT_SECONDS = 60.0


class KafkaConsumer(Protocol):
    def consume(self, num_messages: int, timeout: float) -> Sequence[KafkaMessage]: ...


class MultiConsumer(KafkaConsumer):
    """
    Message source for multiple Kafka consumers.

    This class allows for consuming messages from multiple Kafka consumers with
    different configuration. In particular, we need to use different topic offsets for
    data topics vs. config/command topics.
    """

    def __init__(self, consumers):
        self._consumers = consumers

    def consume(self, num_messages: int, timeout: float) -> list[KafkaMessage]:
        messages = []
        for consumer in self._consumers:
            messages.extend(consumer.consume(num_messages, timeout))
        return messages

    def assignment(self) -> list:
        """Return combined partition assignments from all consumers."""
        assignments = []
        for consumer in self._consumers:
            if hasattr(consumer, 'assignment'):
                assignments.extend(consumer.assignment())
        return assignments

    def get_watermark_offsets(self, partition, timeout: float = 1.0) -> tuple[int, int]:
        """Get watermark offsets for a partition from the owning consumer."""
        for consumer in self._consumers:
            if not hasattr(consumer, 'assignment') or not hasattr(
                consumer, 'get_watermark_offsets'
            ):
                continue
            if partition in consumer.assignment():
                return consumer.get_watermark_offsets(partition, timeout=timeout)
        raise ValueError(f"No consumer found for partition {partition}")

    def position(self, partitions: list) -> list:
        """Get positions for partitions from their owning consumers."""
        results = []
        for partition in partitions:
            for consumer in self._consumers:
                if not hasattr(consumer, 'assignment') or not hasattr(
                    consumer, 'position'
                ):
                    continue
                if partition in consumer.assignment():
                    results.extend(consumer.position([partition]))
                    break
        return results


class KafkaMessageSource(MessageSource[KafkaMessage]):
    """
    Message source for messages from Kafka.

    Parameters
    ----------
    consumer:
        Kafka consumer instance.
    num_messages:
        Number of messages to consume and return in a single call to `get_messages`.
        Fewer messages may be returned if the timeout is reached.
    timeout:
        Timeout in seconds to wait for messages before returning.
    """

    def __init__(
        self, consumer: KafkaConsumer, num_messages: int = 100, timeout: float = 0.05
    ):
        self._consumer = consumer
        self._num_messages = num_messages
        self._timeout = timeout

    def get_messages(self) -> list[KafkaMessage]:
        return self._consumer.consume(
            num_messages=self._num_messages, timeout=self._timeout
        )


@dataclass
class ConsumerHealthStatus:
    """Health status of a BackgroundMessageSource."""

    is_healthy: bool
    thread_alive: bool
    seconds_since_last_consume: float | None
    consecutive_errors: int
    queue_depth: int
    total_messages_consumed: int
    total_batches_dropped: int
    failure_reason: str | None = None


class BackgroundMessageSource(MessageSource[KafkaMessage]):
    """
    Message source that consumes messages in a background thread.

    This allows the processor to work on expensive operations while messages
    continue to be consumed in the background, reducing the change of the processor
    falling behind.

    The consumer includes a circuit breaker that stops consumption after repeated
    failures, and health monitoring to detect silent failures.

    Parameters
    ----------
    consumer:
        The Kafka consumer to consume from.
    num_messages:
        Number of messages to consume in each batch.
    timeout:
        Timeout in seconds for each consume call.
    max_queue_size:
        Maximum number of message batches to keep in the queue. If the queue
        fills up, older batches will be dropped.
    max_consecutive_errors:
        Maximum number of consecutive errors before stopping consumption.
        Set to 0 to disable the circuit breaker (not recommended).
    health_timeout:
        Seconds without successful consumption before the consumer is considered
        unhealthy.
    """

    def __init__(
        self,
        consumer: KafkaConsumer,
        num_messages: int = 100,
        timeout: float = 0.05,
        max_queue_size: int = 1000,
        max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
        health_timeout: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
    ):
        self._consumer = consumer
        self._num_messages = num_messages
        self._timeout = timeout
        self._queue: queue.Queue[list[KafkaMessage]] = queue.Queue(
            maxsize=max_queue_size
        )
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False

        # Circuit breaker configuration
        self._max_consecutive_errors = max_consecutive_errors
        self._consecutive_errors = 0
        self._failure_reason: str | None = None

        # Health monitoring
        self._health_timeout = health_timeout
        self._last_successful_consume: float | None = None
        self._total_messages_consumed = 0
        self._total_batches_dropped = 0

        # Metrics tracking
        self._metrics_interval = 30.0
        self._last_metrics_time = 0.0
        self._messages_consumed_since_last_metrics = 0
        self._batches_dropped_since_last_metrics = 0

    def __enter__(self):
        """Enter context manager and start background consumption."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and stop background consumption."""
        self.stop()

    def start(self) -> None:
        """Start the background message consumption thread."""
        if self._started:
            return

        self._started = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        logger.info("background_consumer_started")

    def stop(self) -> None:
        """Stop the background message consumption thread."""
        if not self._started:
            return

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("background_consumer_stop_timeout")
        self._started = False
        logger.info("background_consumer_stopped")

    def _consume_loop(self) -> None:
        """Main loop for background message consumption."""
        consumer_ready = False
        try:
            while not self._stop_event.is_set():
                try:
                    messages = self._consumer.consume(self._num_messages, self._timeout)
                    # Reset consecutive errors on successful consume
                    self._consecutive_errors = 0
                    self._last_successful_consume = time.monotonic()

                    if not consumer_ready:
                        # First consume() completes consumer group coordination
                        logger.info("kafka_consumer_ready")
                        consumer_ready = True
                    if messages:
                        self._messages_consumed_since_last_metrics += len(messages)
                        self._total_messages_consumed += len(messages)
                        try:
                            self._queue.put_nowait(messages)
                        except queue.Full:
                            # Drop oldest batch if queue is full
                            self._batches_dropped_since_last_metrics += 1
                            self._total_batches_dropped += 1
                            try:
                                dropped = self._queue.get_nowait()
                                logger.warning(
                                    "message_queue_full",
                                    dropped_messages=len(dropped),
                                )
                                self._queue.put_nowait(messages)
                            except queue.Empty:
                                # Queue became empty between full check and get
                                self._queue.put_nowait(messages)
                    self._maybe_log_metrics()
                except Exception as e:
                    self._consecutive_errors += 1
                    logger.exception(
                        "background_consumer_error",
                        consecutive_errors=self._consecutive_errors,
                        max_consecutive_errors=self._max_consecutive_errors,
                    )

                    # Circuit breaker: stop after too many consecutive errors
                    if (
                        self._max_consecutive_errors > 0
                        and self._consecutive_errors >= self._max_consecutive_errors
                    ):
                        self._failure_reason = (
                            f"Circuit breaker triggered after "
                            f"{self._consecutive_errors} consecutive errors. "
                            f"Last error: {e}"
                        )
                        logger.error(
                            "consumer_circuit_breaker_triggered",
                            consecutive_errors=self._consecutive_errors,
                            last_error=str(e),
                        )
                        break

                    # Brief backoff before retry (adds to librdkafka's internal backoff)
                    backoff = min(self._consecutive_errors * 0.5, 5.0)
                    time.sleep(backoff)
        except Exception as e:
            self._failure_reason = f"Fatal error in consume loop: {e}"
            logger.exception("background_consumer_fatal_error")

    def _maybe_log_metrics(self) -> None:
        """Log metrics if the interval has elapsed."""
        now = time.monotonic()
        if now - self._last_metrics_time >= self._metrics_interval:
            # Get consumer lag if available
            lag_info = self.get_consumer_lag()
            total_lag = lag_info.get("total_lag") if lag_info else None

            logger.info(
                "consumer_metrics",
                messages_consumed=self._messages_consumed_since_last_metrics,
                queue_depth=self._queue.qsize(),
                batches_dropped=self._batches_dropped_since_last_metrics,
                consecutive_errors=self._consecutive_errors,
                consumer_lag=total_lag,
                is_healthy=self.is_healthy(),
                interval_seconds=self._metrics_interval,
            )
            self._messages_consumed_since_last_metrics = 0
            self._batches_dropped_since_last_metrics = 0
            self._last_metrics_time = now

    def get_messages(self) -> list[KafkaMessage]:
        """Get all messages consumed since the last call."""
        if not self._started:
            self.start()

        all_messages = []
        try:
            while True:
                batch = self._queue.get_nowait()
                all_messages.extend(batch)
        except queue.Empty:
            pass

        return all_messages

    def is_healthy(self) -> bool:
        """Check if the consumer is healthy.

        Returns False if:
        - The consumer thread is not alive
        - The circuit breaker has been triggered
        - No successful consume has occurred within the health timeout

        Note that returning True does not guarantee messages are being received,
        only that the consumer infrastructure is functioning. A topic with no
        messages will still be considered healthy.
        """
        if not self._started:
            return True  # Not started yet, not unhealthy

        if self._thread is None or not self._thread.is_alive():
            return False

        if self._failure_reason is not None:
            return False

        if self._last_successful_consume is not None:
            elapsed = time.monotonic() - self._last_successful_consume
            if elapsed > self._health_timeout:
                return False

        return True

    def get_health_status(self) -> ConsumerHealthStatus:
        """Get detailed health status for monitoring and debugging."""
        thread_alive = self._thread is not None and self._thread.is_alive()

        seconds_since_last_consume: float | None = None
        if self._last_successful_consume is not None:
            seconds_since_last_consume = (
                time.monotonic() - self._last_successful_consume
            )

        return ConsumerHealthStatus(
            is_healthy=self.is_healthy(),
            thread_alive=thread_alive,
            seconds_since_last_consume=seconds_since_last_consume,
            consecutive_errors=self._consecutive_errors,
            queue_depth=self._queue.qsize(),
            total_messages_consumed=self._total_messages_consumed,
            total_batches_dropped=self._total_batches_dropped,
            failure_reason=self._failure_reason,
        )

    def get_consumer_lag(self) -> dict[str, Any] | None:
        """Get consumer lag information if the underlying consumer supports it.

        Returns a dictionary with lag information per partition, or None if
        the consumer does not support lag monitoring.
        """
        # Check if the consumer has the required methods for lag monitoring
        if not hasattr(self._consumer, 'assignment') or not hasattr(
            self._consumer, 'get_watermark_offsets'
        ):
            return None

        try:
            lag_info: dict[str, Any] = {}
            total_lag = 0
            assignment = self._consumer.assignment()

            for tp in assignment:
                try:
                    low, high = self._consumer.get_watermark_offsets(tp, timeout=1.0)
                    position = self._consumer.position([tp])[0].offset
                    if position >= 0 and high >= 0:
                        partition_lag = high - position
                        total_lag += partition_lag
                        lag_info[f"{tp.topic}:{tp.partition}"] = {
                            "low": low,
                            "high": high,
                            "position": position,
                            "lag": partition_lag,
                        }
                except Exception:  # noqa: S112
                    # Skip partitions we can't get lag for (e.g., not assigned)
                    continue

            lag_info["total_lag"] = total_lag
            return lag_info
        except Exception:
            return None
