# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from ..config.acknowledgement import CommandAcknowledgement
from ..config.workflow_spec import ResultKey
from ..core.job import JobStatus, ServiceStatus
from ..core.message import (
    RESPONSES_STREAM_ID,
    STATUS_STREAM_ID,
    MessageSource,
    StreamId,
)
from .data_service import DataService
from .job_service import JobService
from .service_registry import ServiceRegistry

if TYPE_CHECKING:
    from .job_orchestrator import JobOrchestrator


class Orchestrator:
    """
    Orchestrates the flow of data from Kafka to the GUI layer of the dashboard.

    This class consumes messages from a message source and forwards them to a data
    forwarder, which caches them in a local data service. A transaction mechanism
    isolates the potentially frequent updates from Kafka, ensuring that the GUI
    receives a consistent and current view of the data.
    """

    def __init__(
        self,
        message_source: MessageSource,
        data_service: DataService,
        job_service: JobService,
        service_registry: ServiceRegistry,
        job_orchestrator: JobOrchestrator | None = None,
    ) -> None:
        self._message_source = message_source
        self._data_service = data_service
        self._job_service = job_service
        self._service_registry = service_registry
        self._job_orchestrator = job_orchestrator
        self._logger = structlog.get_logger()

    def update(self) -> None:
        """
        Call this periodically to consume data and feed it into the dashboard.
        """
        t0 = time.perf_counter()
        messages = self._message_source.get_messages()
        t_get = time.perf_counter()

        # Log downstream latency for result data messages
        result_timestamps = [
            m.timestamp
            for m in messages
            if m.stream != STATUS_STREAM_ID and m.stream != RESPONSES_STREAM_ID
        ]
        if result_timestamps:
            wall_clock_ns = time.time_ns()
            oldest_latency_s = (wall_clock_ns - min(result_timestamps)) / 1e9
            newest_latency_s = (wall_clock_ns - max(result_timestamps)) / 1e9
            self._logger.info(
                'downstream_latency',
                oldest_result_latency_s=round(oldest_latency_s, 2),
                newest_result_latency_s=round(newest_latency_s, 2),
                num_results=len(result_timestamps),
            )

        if not messages:
            return

        # Batch all updates in a transaction to avoid repeated UI updates. Reason:
        # - Some listeners depend on multiple streams.
        # - There may be multiple messages for the same stream, only the last one
        #   should trigger an update.
        with self._data_service.transaction():
            for message in messages:
                self.forward(stream_id=message.stream, value=message.value)
            t_forward = time.perf_counter()
        t_notify = time.perf_counter()

        self._logger.info(
            'update_timing',
            get_messages_ms=round((t_get - t0) * 1000, 1),
            forward_ms=round((t_forward - t_get) * 1000, 1),
            notify_ms=round((t_notify - t_forward) * 1000, 1),
            total_ms=round((t_notify - t0) * 1000, 1),
            num_messages=len(messages),
        )

    def forward(self, stream_id: StreamId, value: Any) -> None:
        """
        Forward data to the appropriate data service based on the stream name.

        Parameters
        ----------
        stream_name:
            The name of the stream in the format 'source_name/service_name/suffix'. The
            suffix may contain additional '/' characters which will be ignored.
        value:
            The data to be forwarded.
        """
        if stream_id == STATUS_STREAM_ID:
            if isinstance(value, ServiceStatus):
                self._service_registry.status_updated(value)
            elif isinstance(value, JobStatus):
                self._job_service.status_updated(value)
            else:
                self._logger.warning("Unknown status type: %s", type(value))
        elif stream_id == RESPONSES_STREAM_ID:
            self._process_response(value)
        else:
            result_key = ResultKey.model_validate_json(stream_id.name)
            self._data_service[result_key] = value

    def _process_response(self, ack: CommandAcknowledgement) -> None:
        """Process a command acknowledgement from the backend."""
        if self._job_orchestrator is None:
            return

        self._job_orchestrator.process_acknowledgement(
            message_id=ack.message_id,
            response=ack.response.value,
            error_message=ack.message,
        )
