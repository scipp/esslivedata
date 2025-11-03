# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
from typing import Any

from ..config.workflow_spec import ResultKey
from ..core.message import (
    RESPONSES_STREAM_ID,
    STATUS_STREAM_ID,
    MessageSource,
    StreamId,
)
from .data_service import DataService
from .job_service import JobService
from .workflow_config_service import WorkflowConfigService


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
        workflow_config_service: WorkflowConfigService | None = None,
    ) -> None:
        self._message_source = message_source
        self._data_service = data_service
        self._job_service = job_service
        self._workflow_config_service = workflow_config_service
        self._logger = logging.getLogger(__name__)

    def update(self) -> None:
        """
        Call this periodically to consume data and feed it into the dashboard.
        """
        messages = self._message_source.get_messages()
        self._logger.debug("Consumed %d messages", len(messages))

        if not messages:
            return

        # Batch updates in a transaction to avoid repeated UI updates.
        # Reason:
        # - Some listeners depend on multiple streams.
        # - There may be multiple messages for the same stream, only the last one
        #   should trigger an update.
        with self._data_service.transaction():
            for message in messages:
                self.forward(stream_id=message.stream, value=message.value)

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
            self._job_service.status_updated(value)
        elif stream_id == RESPONSES_STREAM_ID:
            if self._workflow_config_service is not None:
                self._workflow_config_service.process_response(value)
        else:
            result_key = ResultKey.model_validate_json(stream_id.name)
            self._data_service[result_key] = value
