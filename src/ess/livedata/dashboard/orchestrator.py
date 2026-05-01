# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from ..config.acknowledgement import CommandAcknowledgement
from ..config.workflow_spec import JobNumber, ResultKey
from ..core.job import JobStatus, ServiceStatus
from ..core.message import (
    RESPONSES_STREAM_ID,
    STATUS_STREAM_ID,
    MessageSource,
    StreamId,
)
from .active_job_registry import ActiveJobRegistry
from .data_service import DataService
from .job_service import JobService
from .service_registry import ServiceRegistry

if TYPE_CHECKING:
    from .fom_orchestrator import FOMOrchestrator
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
        job_orchestrator: JobOrchestrator,
        active_job_registry: ActiveJobRegistry,
        fom_orchestrator: FOMOrchestrator | None = None,
    ) -> None:
        self._message_source = message_source
        self._data_service = data_service
        self._job_service = job_service
        self._service_registry = service_registry
        self._job_orchestrator = job_orchestrator
        self._fom_orchestrator = fom_orchestrator
        self._active_job_registry = active_job_registry
        self._logger = structlog.get_logger()

    def update(self) -> None:
        """
        Call this periodically to consume data and feed it into the dashboard.
        """
        messages = self._message_source.get_messages()
        self._logger.debug("Consumed %d messages", len(messages))

        if messages:
            # The ingestion guard serializes message processing against
            # active-set mutations and DataService cleanup in
            # ActiveJobRegistry.deactivate() (called from the UI thread).
            # Without this, the background thread could iterate or write to
            # DataService while the UI thread deletes buffers, causing
            # dict-iteration crashes or orphaned buffers.
            #
            # Batch all updates in a transaction to avoid repeated UI
            # updates. Reason:
            # - Some listeners depend on multiple streams.
            # - There may be multiple messages for the same stream, only the
            #   last one should trigger an update.
            with self._active_job_registry.ingestion_guard():
                with self._data_service.transaction():
                    for message in messages:
                        self.forward(stream_id=message.stream, value=message.value)

        if self._fom_orchestrator is not None:
            self._fom_orchestrator.tick()

    def forward(self, stream_id: StreamId, value: Any) -> None:
        """
        Forward data to the appropriate data service based on the stream name.

        Data and job status messages are filtered by active job number via the
        :class:`ActiveJobRegistry`. Messages from unknown or stopped jobs are
        silently discarded.

        Parameters
        ----------
        stream_id:
            The stream identifier.
        value:
            The data to be forwarded.
        """
        if stream_id == STATUS_STREAM_ID:
            if isinstance(value, ServiceStatus):
                self._service_registry.status_updated(value)
            elif isinstance(value, JobStatus):
                if self._is_active_job(value.job_id.job_number):
                    self._job_service.status_updated(value)
            else:
                self._logger.warning("Unknown status type: %s", type(value))
        elif stream_id == RESPONSES_STREAM_ID:
            self._process_response(value)
        else:
            result_key = ResultKey.model_validate_json(stream_id.name)
            if self._is_active_job(result_key.job_id.job_number):
                self._data_service[result_key] = value

    def _is_active_job(self, job_number: JobNumber) -> bool:
        """Check if a job_number belongs to an active job."""
        return self._active_job_registry.is_active(job_number)

    def _process_response(self, ack: CommandAcknowledgement) -> None:
        """Forward a command acknowledgement to all interested orchestrators.

        Each tracker silently ignores ``message_id``s it did not register.
        """
        self._job_orchestrator.process_acknowledgement(
            message_id=ack.message_id,
            response=ack.response.value,
            error_message=ack.message,
        )
        if self._fom_orchestrator is not None:
            self._fom_orchestrator.process_acknowledgement(
                message_id=ack.message_id,
                response=ack.response.value,
                error_message=ack.message,
            )
