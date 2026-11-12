# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import structlog

from ess.livedata import __version__

from ..config.device_contract import DeviceContract
from ..core.command_dispatcher import CommandDispatcher
from .context_outputs import ContextOutputExtractor
from .job import (
    JobResult,
    JobState,
    JobStatus,
    ServiceState,
    ServiceStatus,
    StreamLagReport,
    StreamStats,
    StreamStatsProvider,
)
from .job_manager import JobFactory, JobManager, WorkflowData
from .job_manager_adapter import JobManagerAdapter
from .message import (
    COMMANDS_STREAM_ID,
    RUN_CONTROL_STREAM_ID,
    STATUS_STREAM_ID,
    Message,
    MessageSink,
    MessageSource,
    RunStart,
    RunStop,
    StreamId,
    StreamKind,
)
from .message_batcher import (
    AdaptiveMessageBatcher,
    MessageBatch,
    MessageBatcher,
)
from .nicos_devices import DeviceExtractor
from .preprocessor import Accumulator, PreprocessorFactory
from .timestamp import Duration, Timestamp

logger = structlog.get_logger(__name__)


class MessagePreprocessor[Tin, Tout]:
    """Message preprocessor that handles batches of messages."""

    def __init__(
        self,
        factory: PreprocessorFactory[Tin, Tout],
    ) -> None:
        self._factory = factory
        self._accumulators: dict[StreamId, Accumulator[Tin, Tout]] = {}
        self._changed_unclocked: set[StreamId] = set()

    def _get_accumulator(self, key: StreamId) -> Accumulator[Tin, Tout] | None:
        """Get an accumulator for the given stream ID."""
        if (accumulator := self._accumulators.get(key)) is not None:
            return accumulator
        if (accumulator := self._factory.make_preprocessor(key)) is not None:
            self._accumulators[key] = accumulator
            return accumulator
        return None

    @staticmethod
    def _add(messages: list[Message[Tin]], accumulator: Accumulator[Tin, Tout]) -> bool:
        """Add messages to the accumulator.

        Returns False if no message was accepted (e.g., all duplicates),
        signalling that the stream should be excluded from workflow data.
        """
        any_accepted = False
        for message in messages:
            if accumulator.add(message.timestamp, message.value) is not False:
                any_accepted = True
        return any_accepted

    def release_buffers(self) -> None:
        """Signal that preprocessed data from the last cycle is no longer in use.

        Accumulators that use zero-copy buffer reuse require this call between
        ``get()`` cycles so they can safely overwrite their internal buffers.
        """
        for accumulator in self._accumulators.values():
            accumulator.release_buffers()

    def get_context(self, names: set[str]) -> dict[StreamId, Any]:
        """Read current values from context accumulators for the given stream names.

        Only reads from accumulators that have ``is_context = True`` and that
        have received at least one message. Accumulators that have no data yet
        are silently skipped — this is a normal startup condition (stream
        mapping registered the accumulator before any message arrived).

        Must run on the orchestrator thread while no job worker threads are in
        flight — either ahead of ``process_jobs`` (the enrichment above) or
        inside it before the per-job fan-out (the JobManager's context gate,
        which holds this as its ``context_reader``).
        """
        result: dict[StreamId, Any] = {}
        for stream_id, accumulator in self._accumulators.items():
            if accumulator.is_context and stream_id.name in names:
                try:
                    result[stream_id] = accumulator.get()
                except (RuntimeError, ValueError):
                    logger.debug(
                        'context_accumulator_empty',
                        stream_id=str(stream_id),
                    )
        return result

    def preprocess_messages(self, batch: MessageBatch) -> WorkflowData:
        """
        Preprocess messages before they are sent to the accumulators.
        """
        data: dict[StreamId, Any] = {}
        for key, messages in _group_by_stream(batch.messages).items():
            accumulator = self._get_accumulator(key)
            if accumulator is None:
                logger.debug('no_preprocessor', stream_id=str(key))
                continue
            try:
                if self._add(messages, accumulator):
                    data[key] = accumulator.get()
            except Exception:
                logger.exception('preprocessing_error', stream_id=str(key))
        return WorkflowData(
            start_time=batch.start_time, end_time=batch.end_time, data=data
        )

    def add_unclocked(self, messages: list[Message[Tin]]) -> None:
        """Add messages of unclocked streams to their accumulators on arrival.

        The accumulator buffers the stream until the next push collects it via
        :meth:`take_unclocked`, which is only valid for a context accumulator:
        its ``get()`` is idempotent and does not consume the data.

        Raises
        ------
        TypeError
            If a stream's accumulator is not a context accumulator.
        """
        for key, stream_messages in _group_by_stream(messages).items():
            accumulator = self._get_accumulator(key)
            if accumulator is None:
                logger.debug('no_preprocessor', stream_id=str(key))
                continue
            if not accumulator.is_context:
                raise TypeError(
                    f"Unclocked stream {key} needs a context accumulator, got "
                    f"{type(accumulator).__name__}"
                )
            try:
                if self._add(stream_messages, accumulator):
                    self._changed_unclocked.add(key)
            except Exception:
                logger.exception('preprocessing_error', stream_id=str(key))

    def take_unclocked(self) -> dict[StreamId, Any]:
        """Current values of the unclocked streams changed since the last call."""
        data: dict[StreamId, Any] = {}
        for key in self._changed_unclocked:
            try:
                data[key] = self._accumulators[key].get()
            except Exception:
                logger.exception('preprocessing_error', stream_id=str(key))
        self._changed_unclocked.clear()
        return data


def _group_by_stream[T](messages: list[Message[T]]) -> dict[StreamId, list[Message[T]]]:
    messages_by_key = defaultdict[StreamId, list[Message[T]]](list)
    for msg in messages:
        messages_by_key[msg.stream].append(msg)
    return messages_by_key


#: Stream kinds whose message timestamps are not data time.  The f144 envelope
#: is Kafka time, or for a static PV an arbitrarily old EPICS time (see
#: ``KafkaToF144Adapter``), and devices are synthesized from f144 logs.  ROI
#: requests are stamped at epoch 0, context values with their producer's time.
#: Taken as data time, any of these misplaces the batch window: on ESTIA, Kafka
#: time runs ~36 s ahead of the detector's pulse times (#1133).
DEFAULT_UNCLOCKED_KINDS = frozenset(
    {
        StreamKind.LOG,
        StreamKind.DEVICE,
        StreamKind.LIVEDATA_ROI,
        StreamKind.LIVEDATA_CONTEXT,
    }
)


#: Batch lengths of wall time without a push before a liveness push.  Batches
#: close about one batch length apart with jitter, so a threshold of one would
#: fire liveness pushes between ordinary batches; each push carrying a changed
#: log reruns the job's context stages.
LIVENESS_BATCHES = 2


class OrchestratingProcessor[Tin, Tout]:
    """Processor driving the jobs of a backend service.

    Data messages are either *clocked* or *unclocked*.  Clocked messages go
    through the message batcher, whose batches define data time.  Unclocked
    messages (``unclocked_kinds``) carry no data time: they bypass the batcher
    and go into their (context) accumulators on arrival.  Each push to the jobs
    includes the unclocked streams that changed since the previous push.

    A push happens when the batcher closes a batch.  If nothing is pushed for
    ``LIVENESS_BATCHES`` batch lengths of wall time, a liveness push spanning no
    time at the end of the last batch follows, so job activation, context gates
    and run-transition resets keep moving while the clocked streams pause.
    Wall time decides only when to push, never the data time pushed.  Before
    the first batch there is no data time and nothing is pushed: scheduled jobs
    wait for the first clocked data, while unclocked values wait in their
    accumulators.
    """

    def __init__(
        self,
        *,
        source: MessageSource[Message[Tin]],
        sink: MessageSink[Tout],
        preprocessor_factory: PreprocessorFactory[Tin, Tout],
        service_name: str,
        message_batcher: MessageBatcher | None = None,
        job_threads: int = 1,
        stream_stats_provider: StreamStatsProvider | None = None,
        unclocked_kinds: frozenset[StreamKind] = DEFAULT_UNCLOCKED_KINDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Parameters
        ----------
        unclocked_kinds:
            Stream kinds that bypass the batcher.  A service whose primary data
            are logs (``timeseries``) passes an empty set.
        clock:
            Monotonic seconds, deciding when a liveness push is due.
        """
        self._source = source
        self._sink = sink
        self._message_preprocessor = MessagePreprocessor(factory=preprocessor_factory)
        instrument = preprocessor_factory.instrument
        self._job_manager = JobManager(
            job_factory=JobFactory(instrument=instrument, service_name=service_name),
            context_reader=self._message_preprocessor.get_context,
            job_threads=job_threads,
        )
        self._job_manager_adapter = JobManagerAdapter(job_manager=self._job_manager)
        self._device_extractor = DeviceExtractor(
            device_contract=DeviceContract.from_instrument(instrument),
        )
        self._context_output_extractor = ContextOutputExtractor(
            registry=instrument.workflow_factory,
        )
        self._message_batcher = message_batcher or AdaptiveMessageBatcher()
        self._unclocked_kinds = unclocked_kinds
        self._clock = clock
        self._last_push = clock()
        self._last_batch_end: Timestamp | None = None
        self._config_processor = CommandDispatcher(
            job_manager_adapter=self._job_manager_adapter
        )
        self._last_status_update: Timestamp | None = None
        self._status_update_interval = Duration.from_seconds(2)

        # Service heartbeat state
        self._instrument = instrument.name
        self._service_name = service_name
        self._worker_id = str(uuid.uuid4())
        self._started_at = Timestamp.now()
        self._service_state = ServiceState.starting
        self._service_error: str | None = None
        self._has_processed_first_batch = False

        # Metrics and stream stats tracking
        self._metrics_interval = Duration.from_seconds(30)
        self._last_metrics_time: Timestamp | None = None
        self._batches_processed = 0
        self._empty_batches = 0
        self._errors_since_last_metrics = 0
        self._stream_stats_provider = stream_stats_provider
        self._pending_stream_stats: StreamStats | None = None
        self._finalized = False

    def process(self) -> None:
        # Transition from starting to running on first process cycle
        if not self._has_processed_first_batch:
            self._has_processed_first_batch = True
            self._service_state = ServiceState.running
            logger.info('service_running')

        messages = self._source.get_messages()
        config_messages: list[Message[Tin]] = []
        run_control_messages: list[Message[RunStart | RunStop]] = []
        data_messages: list[Message[Tin]] = []
        unclocked_messages: list[Message[Tin]] = []

        for msg in messages:
            if msg.stream == COMMANDS_STREAM_ID:
                config_messages.append(msg)
            elif msg.stream == RUN_CONTROL_STREAM_ID:
                run_control_messages.append(msg)
            elif msg.stream.kind in self._unclocked_kinds:
                unclocked_messages.append(msg)
            else:
                data_messages.append(msg)

        # Handle config messages
        result_messages = self._config_processor.process_messages(config_messages)

        # Handle run control messages (run start/stop from filewriter topic)
        for msg in run_control_messages:
            match msg.value:
                case RunStart():
                    self._job_manager.on_run_start(msg.value)
                case RunStop():
                    self._job_manager.on_run_stop(msg.value)

        self._report_status()

        self._message_preprocessor.add_unclocked(unclocked_messages)
        message_batch = self._message_batcher.batch(data_messages)
        batch_start = time.monotonic()
        workflow_data = self._next_workflow_data(message_batch)

        if workflow_data is None:  # Idle cycle: no push due.
            self._message_batcher.report_batch(None, processing_time_s=0.0)
            self._empty_batches += 1
            self._maybe_log_metrics()
            self._sink.publish_messages(result_messages)
            if not config_messages:
                # Avoid busy-waiting if there is no data and no config messages.
                # If there are config messages, we avoid sleeping, since config messages
                # may trigger costly workflow creation.
                time.sleep(0.1)
            return

        # Enrich workflow_data with cached context for streams that scheduled
        # jobs will need on activation. peek_pending_streams uses start_time
        # to predict which jobs will activate in the upcoming process_jobs
        # call (same start_time), covering both auxiliary streams (e.g., log
        # data for detector workflows) and primary streams (e.g., log data
        # for a timeseries job that activates after its data already passed
        # through, or the chopperless wavelength_lut tick that fires only
        # once). Lets a liveness push run jobs from cached context alone.
        # Jobs already gated on context are not handled here: the JobManager
        # refills their gating streams itself inside process_jobs (see
        # JobManager._open_context_gates), so gate correctness does not depend
        # on this enrichment. Keeping gated streams out of workflow_data also
        # keeps liveness pushes from re-presenting cached context every batch
        # length while a job waits on context that never comes.
        needed = self._job_manager.peek_pending_streams(workflow_data.start_time)
        missing = needed - {s.name for s in workflow_data.data}
        if missing:
            workflow_data.data.update(self._message_preprocessor.get_context(missing))

        # Push data into jobs and compute results in a single pass.
        # We used to compute results only after 1-N accumulation calls, reasoning that
        # processing data (partially) immediately (instead of waiting for more data)
        # would increase the latency. A closer look, the contrary is true, based on
        # a simple model with a constant plus linear (per event) time for preprocessing
        # (including accumulation).
        # When job_threads > 1, each job's accumulation and finalization run as
        # a single threaded task, avoiding two fan-out/fan-in cycles.
        job_replies, results = self._job_manager.process_jobs(workflow_data)
        self._message_preprocessor.release_buffers()

        # JobManager logs job failures, rate limited per job. The metric counts
        # every failure, so the journal keeps the true rate.
        self._errors_since_last_metrics += sum(r.has_error for r in job_replies)
        valid_results = []
        for result in results:
            if result.error_message is not None:
                self._errors_since_last_metrics += 1
            else:
                valid_results.append(result)

        processing_time_s = time.monotonic() - batch_start
        self._message_batcher.report_batch(
            len(message_batch.messages) if message_batch is not None else 0,
            processing_time_s=processing_time_s,
        )
        self._batches_processed += 1
        self._maybe_log_metrics()

        result_messages.extend(
            [_job_result_to_message(result) for result in valid_results]
        )
        result_messages.extend(self._device_extractor.extract(valid_results))
        result_messages.extend(self._context_output_extractor.extract(valid_results))
        self._sink.publish_messages(result_messages)

    def _next_workflow_data(self, batch: MessageBatch | None) -> WorkflowData | None:
        """Data to push to the jobs in this cycle, or None if no push is due.

        See the class docstring for when a push is due.
        """
        now = self._clock()
        if batch is not None:
            workflow_data = self._message_preprocessor.preprocess_messages(batch)
            self._last_batch_end = batch.end_time
        elif (
            self._last_batch_end is None
            or now - self._last_push
            < LIVENESS_BATCHES * self._message_batcher.batch_length_s
        ):
            return None
        else:
            workflow_data = WorkflowData(
                start_time=self._last_batch_end, end_time=self._last_batch_end, data={}
            )
        self._last_push = now
        workflow_data.data.update(self._message_preprocessor.take_unclocked())
        return workflow_data

    def _report_status(self) -> None:
        timestamp = Timestamp.now()
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
        stream_stats = self._pending_stream_stats
        self._pending_stream_stats = None
        return ServiceStatus(
            instrument=self._instrument,
            service_name=self._service_name,
            worker_id=self._worker_id,
            state=self._service_state,
            started_at=self._started_at,
            active_job_count=len(job_statuses),
            version=__version__,
            error=self._service_error,
            batch_interval_s=self._message_batcher.batch_length_s,
            stream_stats=stream_stats,
        )

    def _maybe_log_metrics(self) -> None:
        """Log processor metrics if the interval has elapsed."""
        timestamp = Timestamp.now()
        if self._last_metrics_time is None:
            self._last_metrics_time = timestamp
            return

        elapsed = timestamp - self._last_metrics_time
        if elapsed >= self._metrics_interval:
            window_seconds = elapsed.to_seconds()
            active_jobs = len(self._job_manager.active_jobs)

            # Drain stream stats from the counter
            if self._stream_stats_provider is not None:
                self._pending_stream_stats = self._stream_stats_provider.drain(
                    window_seconds
                )
                self._log_stream_lag(self._stream_stats_provider.drain_lag())

            batcher_metrics = self._message_batcher.drain_metrics()
            logger.info(
                'processor_metrics',
                batches=self._batches_processed,
                empty_batches=self._empty_batches,
                active_jobs=active_jobs,
                errors=self._errors_since_last_metrics,
                interval_seconds=window_seconds,
                stream_stats=self._pending_stream_stats,
                # How far behind live data the batcher fell, and what it shed
                # to stay bounded.  A rising backlog is the earliest warning
                # that the service is not keeping up; drops mean data for the
                # shed interval never reached any workflow.
                max_backlog_s=round(batcher_metrics.max_backlog_s, 3),
                dropped_messages=batcher_metrics.dropped_messages,
                dropped_bytes=batcher_metrics.dropped_bytes,
            )
            self._batches_processed = 0
            self._empty_batches = 0
            self._errors_since_last_metrics = 0
            self._last_metrics_time = timestamp

    def _log_stream_lag(self, report: StreamLagReport | None) -> None:
        """Log one line per stream at a severity reflecting its lag.

        ERROR for a payload from the future (clock/logic fault upstream),
        WARNING for data seconds stale at publish time, INFO otherwise.
        """
        if report is None:
            return
        for lag in report.streams:
            logger.log(
                report.level(lag),
                'stream_lag',
                topic=lag.topic,
                source=lag.source,
                schema=lag.schema,
                min_lag_s=round(lag.min_s, 3),
                max_lag_s=round(lag.max_s, 3),
                count=lag.count,
                max_out_of_order_s=round(lag.max_out_of_order_s, 3),
                out_of_order_partition=lag.out_of_order_partition,
            )

    def finalize(self, *, error: str | None = None) -> None:
        """Mark jobs as stopped, send final heartbeat, and shut down.

        Called by Service after the worker thread has joined, so job states
        can be safely transitioned without racing with the processing loop.
        Idempotent — safe to call more than once.
        """
        if self._finalized:
            return
        self._finalized = True

        if error is not None:
            logger.error('service_error', error=error)
            self._service_state = ServiceState.error
            self._service_error = error
        else:
            logger.info('service_stopped')
            self._service_state = ServiceState.stopped

        timestamp = Timestamp.now()
        job_statuses = self._job_manager.get_all_job_statuses()
        messages = [
            _job_status_to_message(
                replace(status, state=JobState.stopped), timestamp=timestamp
            )
            for status in job_statuses
        ]
        service_status = self._get_service_status(job_statuses)
        messages.append(_service_status_to_message(service_status, timestamp=timestamp))
        try:
            self._sink.publish_messages(messages)
        except Exception:
            logger.exception('Failed to send final heartbeat')

        self._job_manager.shutdown()


def _job_result_to_message(result: JobResult) -> Message:
    """
    Convert a workflow result to a message for publishing.

    JobId is unique on its own, but we include the workflow ID to make it easier to
    identify the job in the frontend.
    """
    if result.start_time is None:
        raise ValueError(
            f"Result for job {result.job_id} has no start time. A successful "
            "result must carry the start time of its accumulation window."
        )
    return Message(
        timestamp=result.start_time,
        stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name=result.stream_name),
        value=result.data,
    )


def _job_status_to_message(status: JobStatus, timestamp: Timestamp) -> Message:
    return Message(timestamp=timestamp, stream=STATUS_STREAM_ID, value=status)


def _service_status_to_message(status: ServiceStatus, timestamp: Timestamp) -> Message:
    return Message(timestamp=timestamp, stream=STATUS_STREAM_ID, value=status)
