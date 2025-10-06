# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Generic

from ..handlers.config_handler import ConfigProcessor
from .handler import Accumulator, PreprocessorFactory
from .job import JobResult, JobStatus
from .job_manager import JobFactory, JobManager, WorkflowData
from .job_manager_adapter import JobManagerAdapter
from .message import (
    CONFIG_STREAM_ID,
    STATUS_STREAM_ID,
    Message,
    MessageSink,
    MessageSource,
    StreamId,
    StreamKind,
    Tin,
    Tout,
)
from .message_batcher import MessageBatch, MessageBatcher, SimpleMessageBatcher


class MessagePreprocessor(Generic[Tin, Tout]):
    """Message preprocessor that handles batches of messages."""

    def __init__(
        self,
        factory: PreprocessorFactory[Tin, Tout],
        logger: logging.Logger | None = None,
    ) -> None:
        self._factory = factory
        self._logger = logger or logging.getLogger(__name__)
        self._accumulators: dict[StreamId, Accumulator[Tin, Tout]] = {}

    def _get_accumulator(self, key: StreamId) -> Accumulator[Tin, Tout] | None:
        """Get an accumulator for the given stream ID."""
        if (accumulator := self._accumulators.get(key)) is not None:
            return accumulator
        if (accumulator := self._factory.make_preprocessor(key)) is not None:
            self._accumulators[key] = accumulator
            return accumulator
        return None

    def _preprocess_stream(
        self, messages: list[Message[Tin]], accumulator: Accumulator[Tin, Tout]
    ) -> Tout:
        """Preprocess messages for a single stream using the given accumulator."""
        for message in messages:
            accumulator.add(message.timestamp, message.value)
        # We assume the accumulator is cleared in `get`.
        return accumulator.get()

    def preprocess_messages(self, batch: MessageBatch) -> WorkflowData:
        """
        Preprocess messages before they are sent to the accumulators.
        """
        messages_by_key = defaultdict[StreamId, list[Message]](list)
        for msg in batch.messages:
            messages_by_key[msg.stream].append(msg)

        data: dict[StreamId, Any] = {}
        for key, messages in messages_by_key.items():
            accumulator = self._get_accumulator(key)
            if accumulator is None:
                self._logger.debug('No preprocessor for key %s, skipping messages', key)
                continue
            try:
                data[key] = self._preprocess_stream(messages, accumulator)
            except Exception:
                self._logger.exception('Error pre-processing messages for key %s', key)
        return WorkflowData(
            start_time=batch.start_time, end_time=batch.end_time, data=data
        )


class OrchestratingProcessor(Generic[Tin, Tout]):
    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        source: MessageSource[Message[Tin]],
        sink: MessageSink[Tout],
        preprocessor_factory: PreprocessorFactory[Tin, Tout],
        message_batcher: MessageBatcher | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._source = source
        self._sink = sink
        self._message_preprocessor = MessagePreprocessor(
            factory=preprocessor_factory, logger=self._logger
        )
        self._job_manager = JobManager(
            job_factory=JobFactory(instrument=preprocessor_factory.instrument)
        )
        self._job_manager_adapter = JobManagerAdapter(
            job_manager=self._job_manager, logger=self._logger
        )
        self._message_batcher = message_batcher or SimpleMessageBatcher()
        self._config_processor = ConfigProcessor(
            job_manager_adapter=self._job_manager_adapter, logger=self._logger
        )
        self._last_status_update: int | None = None
        self._status_update_interval = 2_000_000_000  # 2 seconds

    def process(self) -> None:
        messages = self._source.get_messages()
        self._logger.debug('Processing %d messages', len(messages))
        config_messages: list[Message[Tin]] = []
        data_messages: list[Message[Tin]] = []

        for msg in messages:
            if msg.stream == CONFIG_STREAM_ID:
                config_messages.append(msg)
            else:
                data_messages.append(msg)

        # Handle config messages
        result_messages = self._config_processor.process_messages(config_messages)

        self._report_status()

        message_batch = self._message_batcher.batch(data_messages)
        if message_batch is None:
            self._logger.debug('No data messages to process')
            self._sink.publish_messages(result_messages)
            if not config_messages:
                # Avoid busy-waiting if there is no data and no config messages.
                # If there are config messages, we avoid sleeping, since config messages
                # may trigger costly workflow creation.
                time.sleep(0.1)
            return
        self._logger.debug(
            'Processing batch with %d data messages', len(message_batch.messages)
        )

        # Pre-process message batch
        workflow_data = self._message_preprocessor.preprocess_messages(message_batch)

        # Handle data messages with the workflow manager, accumulating data as needed.
        job_errors = self._job_manager.push_data(workflow_data)

        # Log any errors from data processing
        for error in job_errors:
            if error.has_error:
                self._logger.error(self._job_manager.format_job_error(error))

        # We used to compute results only after 1-N accumulation calls, reasoning that
        # processing data (partially) immediately (instead of waiting for more data)
        # would increase the latency. A closer look, the contrary is true, based on
        # a simple model with a constant plus linear (per event) time for preprocessing
        # (including accumulation).
        results = self._job_manager.compute_results()

        # Filter valid results and log errors
        valid_results = []
        for result in results:
            if result.error_message is not None:
                self._logger.error(
                    'Job %s for workflow %s failed: %s',
                    result.job_id,
                    result.workflow_id,
                    result.error_message,
                )
            else:
                valid_results.append(result)

        result_messages.extend(
            [_job_result_to_message(result) for result in valid_results]
        )
        self._sink.publish_messages(result_messages)

    def _report_status(self) -> None:
        timestamp = time.time_ns()
        if self._last_status_update is not None:
            if timestamp - self._last_status_update < self._status_update_interval:
                return
        self._last_status_update = timestamp
        job_statuses = self._job_manager.get_all_job_statuses()
        messages = [
            _job_status_to_message(status, timestamp=timestamp)
            for status in job_statuses
        ]
        self._sink.publish_messages(messages)


def _job_result_to_message(result: JobResult) -> Message:
    """
    Convert a workflow result to a message for publishing.

    JobId is unique on its own, but we include the workflow ID to make it easier to
    identify the job in the frontend.
    """
    return Message(
        timestamp=result.start_time or 0,
        stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name=result.stream_name),
        value=result.data,
    )


def _job_status_to_message(status: JobStatus, timestamp: int) -> Message:
    return Message(timestamp=timestamp, stream=STATUS_STREAM_ID, value=status)
