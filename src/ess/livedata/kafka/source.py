# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import logging
import queue
import threading
from collections.abc import Sequence
from typing import Protocol

from ..core.message import MessageSource
from .message_adapter import KafkaMessage


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


class BackgroundMessageSource(MessageSource[KafkaMessage]):
    """
    Message source that consumes messages in a background thread.

    This allows the processor to work on expensive operations while messages
    continue to be consumed in the background, reducing the change of the processor
    falling behind.

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
    """

    def __init__(
        self,
        consumer: KafkaConsumer,
        num_messages: int = 100,
        timeout: float = 0.05,
        max_queue_size: int = 1000,
    ):
        self._consumer = consumer
        self._num_messages = num_messages
        self._timeout = timeout
        self._queue: queue.Queue[list[KafkaMessage]] = queue.Queue(
            maxsize=max_queue_size
        )
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._logger = logging.getLogger(__name__)
        self._started = False

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
        self._logger.info("Background message consumption started")

    def stop(self) -> None:
        """Stop the background message consumption thread."""
        if not self._started:
            return

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                self._logger.warning("Background consumer thread did not stop cleanly")
        self._started = False
        self._logger.info("Background message consumption stopped")

    def _consume_loop(self) -> None:
        """Main loop for background message consumption."""
        consumer_ready = False
        try:
            while not self._stop_event.is_set():
                try:
                    messages = self._consumer.consume(self._num_messages, self._timeout)
                    if not consumer_ready:
                        # First consume() completes consumer group coordination
                        self._logger.info("Kafka consumer ready and polling")
                        consumer_ready = True
                    if messages:
                        try:
                            self._queue.put_nowait(messages)
                        except queue.Full:
                            # Drop oldest batch if queue is full
                            try:
                                dropped = self._queue.get_nowait()
                                self._logger.warning(
                                    "Message queue full, dropped %d messages",
                                    len(dropped),
                                )
                                self._queue.put_nowait(messages)
                            except queue.Empty:
                                # Queue became empty between full check and get
                                self._queue.put_nowait(messages)
                except Exception:
                    self._logger.exception("Error in background message consumption")
                    # Continue running even if there's an error
        except Exception:
            self._logger.exception("Fatal error in background consumer thread")

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
