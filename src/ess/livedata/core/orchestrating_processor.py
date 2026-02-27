# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Any, Generic

import structlog

from ..handlers.config_handler import ConfigProcessor
from .handler import Accumulator, PreprocessorFactory
from .job import JobResult, JobStatus, ServiceState, ServiceStatus
from .job_manager import JobFactory, JobManager, WorkflowData
from .job_manager_adapter import JobManagerAdapter
from .load_shedder import LoadShedder
from .message import (
    COMMANDS_STREAM_ID,
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

logger = structlog.get_logger(__name__)


class MessagePreprocessor(Generic[Tin, Tout]):
    """Message preprocessor that handles batches of messages."""

    def __init__(
        self,
        factory: PreprocessorFactory[Tin, Tout],
    ) -> None:
        self._factory = factory
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
                logger.debug('no_preprocessor', stream_id=str(key))
                continue
            try:
                data[key] = self._preprocess_stream(messages, accumulator)
            except Exception:
                logger.exception('preprocessing_error', stream_id=str(key))
        return WorkflowData(
            start_time=batch.start_time, end_time=batch.end_time, data=data
        )


class OrchestratingProcessor(Generic[Tin, Tout]):
    def __init__(
        self,
        *,
        source: MessageSource[Message[Tin]],
        sink: MessageSink[Tout],
        preprocessor_factory: PreprocessorFactory[Tin, Tout],
        message_batcher: MessageBatcher | None = None,
        enable_load_shedding: bool = True,
    ) -> None:
        self._source = source
        self._sink = sink
        self._message_preprocessor = MessagePreprocessor(factory=preprocessor_factory)
        instrument = preprocessor_factory.instrument
        self._job_manager = JobManager(job_factory=JobFactory(instrument=instrument))
        self._job_manager_adapter = JobManagerAdapter(job_manager=self._job_manager)
        self._message_batcher = message_batcher or SimpleMessageBatcher()
        self._config_processor = ConfigProcessor(
            job_manager_adapter=self._job_manager_adapter
        )
        self._load_shedder = LoadShedder() if enable_load_shedding else None
        self._last_status_update: int | None = None
        self._status_update_interval = 2_000_000_000  # 2 seconds

        # Service heartbeat state
        self._instrument = instrument.name
        self._namespace = instrument.active_namespace or "unknown"
        self._worker_id = str(uuid.uuid4())
        self._started_at = time.time_ns()
        self._messages_processed = 0
        self._service_state = ServiceState.starting
        self._service_error: str | None = None
        self._has_processed_first_batch = False

        # Metrics tracking
        self._metrics_interval = 30_000_000_000  # 30 seconds in nanoseconds
        self._last_metrics_time: int | None = None
        self._batches_processed = 0
        self._empty_batches = 0
        self._errors_since_last_metrics = 0

    def process(self) -> None:
        # Transition from starting to running on first process cycle
        if not self._has_processed_first_batch:
            self._has_processed_first_batch = True
            self._service_state = ServiceState.running
            logger.info('service_running')

        messages = self._source.get_messages()
        self._messages_processed += len(messages)
        config_messages: list[Message[Tin]] = []
        data_messages: list[Message[Tin]] = []

        for msg in messages:
            if msg.stream == COMMANDS_STREAM_ID:
                config_messages.append(msg)
            else:
                data_messages.append(msg)

        # Handle config messages
        result_messages = self._config_processor.process_messages(config_messages)

        self._report_status()

        if self._load_shedder is not None:
            data_messages = self._load_shedder.shed(data_messages)
        message_batch = self._message_batcher.batch(data_messages)
        if self._load_shedder is not None:
            self._load_shedder.report_batch_result(message_batch is not None)
        if message_batch is None:
            self._empty_batches += 1
            self._maybe_log_metrics()
            self._sink.publish_messages(result_messages)
            if not config_messages:
                # Avoid busy-waiting if there is no data and no config messages.
                # If there are config messages, we avoid sleeping, since config messages
                # may trigger costly workflow creation.
                time.sleep(0.1)
            return

        # Pre-process message batch
        workflow_data = self._message_preprocessor.preprocess_messages(message_batch)

        # Handle data messages with the workflow manager, accumulating data as needed.
        job_errors = self._job_manager.push_data(workflow_data)

        # Log any errors from data processing
        for error in job_errors:
            if error.has_error:
                self._errors_since_last_metrics += 1
                logger.error(
                    'job_data_error',
                    job_id=str(error.job_id),
                    error=error.error_message,
                )

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
                self._errors_since_last_metrics += 1
                logger.error(
                    'job_failed',
                    job_id=str(result.job_id),
                    workflow_id=str(result.workflow_id),
                    error=result.error_message,
                )
            else:
                valid_results.append(result)

        self._batches_processed += 1
        self._maybe_log_metrics()

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

        # Publish job statuses
        job_statuses = self._job_manager.get_all_job_statuses()
        messages = [
            _job_status_to_message(status, timestamp=timestamp)
            for status in job_statuses
        ]

        # Publish service heartbeat
        service_status = self._get_service_status(job_statuses)
        messages.append(_service_status_to_message(service_status, timestamp=timestamp))

        self._sink.publish_messages(messages)

    def _get_service_status(self, job_statuses: list[JobStatus]) -> ServiceStatus:
        """Get the current service status for heartbeat publishing."""
        shedder_state = self._load_shedder.state if self._load_shedder else None
        return ServiceStatus(
            instrument=self._instrument,
            namespace=self._namespace,
            worker_id=self._worker_id,
            state=self._service_state,
            started_at=self._started_at,
            active_job_count=len(job_statuses),
            messages_processed=self._messages_processed,
            error=self._service_error,
            is_shedding=shedder_state.is_shedding if shedder_state else False,
            messages_dropped=shedder_state.messages_dropped if shedder_state else 0,
            messages_eligible=shedder_state.messages_eligible if shedder_state else 0,
        )

    def _maybe_log_metrics(self) -> None:
        """Log processor metrics if the interval has elapsed."""
        timestamp = time.time_ns()
        if self._last_metrics_time is None:
            self._last_metrics_time = timestamp
            return

        if timestamp - self._last_metrics_time >= self._metrics_interval:
            active_jobs = len(self._job_manager.active_jobs)
            shedder_state = self._load_shedder.state if self._load_shedder else None
            logger.info(
                'processor_metrics',
                messages=self._messages_processed,
                batches=self._batches_processed,
                empty_batches=self._empty_batches,
                active_jobs=active_jobs,
                errors=self._errors_since_last_metrics,
                shedding=shedder_state.is_shedding if shedder_state else False,
                messages_dropped=(
                    shedder_state.messages_dropped if shedder_state else 0
                ),
                messages_eligible=(
                    shedder_state.messages_eligible if shedder_state else 0
                ),
                interval_seconds=(timestamp - self._last_metrics_time) / 1e9,
            )
            # Reset counters (except messages_processed which is cumulative for service)
            self._batches_processed = 0
            self._empty_batches = 0
            self._errors_since_last_metrics = 0
            self._last_metrics_time = timestamp

    def shutdown(self) -> None:
        """Transition to stopping state and send heartbeat.

        Called by Service at the beginning of graceful shutdown to notify the
        dashboard that this worker is shutting down intentionally.
        """
        logger.info('service_shutting_down')
        self._service_state = ServiceState.stopping
        self._send_final_heartbeat()

    def report_stopped(self) -> None:
        """Transition to stopped state and send final heartbeat.

        Called by Service after worker thread has stopped to notify the
        dashboard that shutdown completed successfully.
        """
        logger.info('service_stopped')
        self._service_state = ServiceState.stopped
        self._send_final_heartbeat()

    def report_error(self, error_message: str) -> None:
        """Transition to error state and send final heartbeat.

        Called by Service when an unhandled exception occurs to notify the
        dashboard that this worker encountered a fatal error.
        """
        logger.error('service_error', error=error_message)
        self._service_state = ServiceState.error
        self._service_error = error_message
        self._send_final_heartbeat()

    def _send_final_heartbeat(self) -> None:
        """Send a final service heartbeat with current state."""
        timestamp = time.time_ns()
        job_statuses = self._job_manager.get_all_job_statuses()
        service_status = self._get_service_status(job_statuses)
        message = _service_status_to_message(service_status, timestamp=timestamp)
        try:
            self._sink.publish_messages([message])
        except Exception:
            logger.exception('Failed to send final heartbeat')


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


def _service_status_to_message(status: ServiceStatus, timestamp: int) -> Message:
    return Message(timestamp=timestamp, stream=STATUS_STREAM_ID, value=status)
