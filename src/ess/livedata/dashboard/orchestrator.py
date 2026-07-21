# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from ..config.acknowledgement import CommandAcknowledgement
from ..config.workflow_spec import ResultKey
from ..core.job import JobStatus, ServiceStatus
from ..core.message import (
    RESPONSES_STREAM_ID,
    STATUS_STREAM_ID,
    Message,
    MessageSource,
    StreamId,
)
from .active_job_registry import ActiveJobRegistry
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
        job_orchestrator: JobOrchestrator,
        active_job_registry: ActiveJobRegistry,
    ) -> None:
        self._message_source = message_source
        self._data_service = data_service
        self._job_service = job_service
        self._service_registry = service_registry
        self._job_orchestrator = job_orchestrator
        self._active_job_registry = active_job_registry
        self._logger = structlog.get_logger()

    def update(self) -> None:
        """
        Call this periodically to consume data and feed it into the dashboard.
        """
        messages = self._message_source.get_messages()
        self._logger.debug("Consumed %d messages", len(messages))

        if not messages:
            return

        # The ingestion guard serializes message processing against generation
        # flips in ActiveJobRegistry.begin_generation() (called from the UI
        # thread at commit). Without this, a flip could clear DataService
        # buffers mid-batch, leaving a mix of old- and new-generation data.
        #
        # Control messages (status, command acks) never touch DataService, so
        # they are handled outside the transaction: the transaction holds the
        # DataService lock for its duration, and JobService/ServiceRegistry
        # callbacks must not run under it. Handling them first also lets an
        # ack that activates a job admit the same batch's data for that job
        # instead of dropping it at the active filter.
        #
        # Data updates are batched in a transaction so a pull observes the
        # whole batch atomically and subscribers get one notification per
        # batch. Reason:
        # - Some listeners depend on multiple streams.
        # - There may be multiple messages for the same stream, only the last one
        #   should trigger an update.
        control_streams = (STATUS_STREAM_ID, RESPONSES_STREAM_ID)
        with self._active_job_registry.ingestion_guard():
            self._forward_messages([m for m in messages if m.stream in control_streams])
            with self._data_service.transaction():
                self._forward_messages(
                    [m for m in messages if m.stream not in control_streams]
                )

    def _forward_messages(self, messages: list[Message[Any]]) -> None:
        for message in messages:
            # Isolate per-message failures: a single poisoned stream (e.g.
            # corrupt payload, or data whose shape no longer fits its buffer)
            # must not abort the loop and drop every message sorted after it,
            # which would repeat on every polling cycle. Log and continue.
            try:
                self.forward(stream_id=message.stream, value=message.value)
            except Exception:
                self._logger.exception(
                    "Failed to forward message from stream %s; skipping",
                    message.stream,
                )

    def forward(self, stream_id: StreamId, value: Any) -> None:
        """
        Forward data to the appropriate data service based on the stream name.

        Data messages are admitted only if their wire job_number is the
        workflow's current generation (:class:`ActiveJobRegistry`); stale or
        unknown generations are counted and dropped. Job status messages are
        admitted for any known workflow: they are the observation feed for
        job adoption and desired/observed reconciliation (ADR 0008), so
        unknown job_numbers must reach the orchestrator instead of being
        filtered here.

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
                if self._job_orchestrator.is_known_workflow(value.workflow_id):
                    self._job_service.status_updated(value)
            else:
                self._logger.warning("Unknown status type: %s", type(value))
        elif stream_id == RESPONSES_STREAM_ID:
            self._process_response(value)
        else:
            # The wire key carries the per-commit job_number; the dashboard's
            # data plane is keyed by the stable DataKey. The job_number is
            # consumed here as a generation filter and recorded as the
            # provenance stamp — it never enters the data plane.
            result_key = ResultKey.model_validate_json(stream_id.name)
            job_number = result_key.job_id.job_number
            if self._active_job_registry.is_current(result_key.workflow_id, job_number):
                self._data_service.set_item(
                    result_key.data_key, value, stamp=job_number
                )
            else:
                self._active_job_registry.record_stale(
                    result_key.workflow_id, job_number
                )

    def _process_response(self, ack: CommandAcknowledgement) -> None:
        """Process a command acknowledgement from the backend."""
        self._job_orchestrator.process_acknowledgement(
            message_id=ack.message_id,
            response=ack.response.value,
            error_message=ack.message,
        )
