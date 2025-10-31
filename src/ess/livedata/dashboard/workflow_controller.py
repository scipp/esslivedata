# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Workflow controller implementation backed by a config service.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import pydantic

from ess.livedata.config.workflow_spec import (
    ResultKey,
    WorkflowConfig,
    WorkflowId,
    WorkflowSpec,
    WorkflowStatus,
    WorkflowStatusType,
)

from .config_store import ConfigStore
from .configuration_adapter import ConfigurationState
from .correlation_histogram import CorrelationHistogramController, make_workflow_spec
from .data_service import DataService
from .workflow_config_service import ConfigServiceAdapter, WorkflowConfigService
from .workflow_configuration_adapter import WorkflowConfigurationAdapter


class WorkflowController:
    """
    Workflow controller backed by a config service.

    This controller manages workflow operations by interacting with a config service
    for starting/stopping workflows and maintaining local state for tracking.

    Brief overview of what this controller does in the wider context of the "data
    reduction" service Kafka:

    - Workflow specs are defined in the workflow registry passed to the controller.
    - GUI displays available workflows and allows configuring and starting them via
      the controller.
    - Controller persists configs for workflows to allow restoring widget state across
      sessions.
    - Reduction services publish workflow status updates to Kafka.
    - Controller listens for these updates and maintains local state for UI display.
    """

    def __init__(
        self,
        *,
        service: WorkflowConfigService,
        source_names: list[str],
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        config_store: ConfigStore | None = None,
        data_service: DataService[ResultKey, object] | None = None,
        correlation_histogram_controller: CorrelationHistogramController | None = None,
    ) -> None:
        """
        Initialize the workflow controller.

        Parameters
        ----------
        service
            Service for runtime workflow communication with backend services.
        source_names
            List of source names to monitor for workflow status updates.
        workflow_registry
            Registry of available workflows and their specifications.
        config_store
            Optional store for persisting UI configuration state across sessions.
        data_service
            Optional data service for cleaning up workflow data keys.
        correlation_histogram_controller
            Optional controller for correlation histogram workflows.
        """
        self._service = service
        self._config_store = config_store
        self._logger = logging.getLogger(__name__)

        self._source_names = source_names

        # Extend registry with correlation histogram specs if controller provided
        self._workflow_registry = dict(workflow_registry)
        self._correlation_histogram_controller = correlation_histogram_controller
        if correlation_histogram_controller is not None:
            correlation_1d_spec = make_workflow_spec(1)
            correlation_2d_spec = make_workflow_spec(2)
            self._workflow_registry[correlation_1d_spec.get_id()] = correlation_1d_spec
            self._workflow_registry[correlation_2d_spec.get_id()] = correlation_2d_spec

        self._data_service = data_service

        # Initialize all sources with UNKNOWN status
        self._workflow_status: dict[str, WorkflowStatus] = {
            source_name: WorkflowStatus(source_name=source_name)
            for source_name in self._source_names
        }

        # Callbacks
        self._workflow_specs_callbacks: list[
            Callable[[dict[WorkflowId, WorkflowSpec]], None]
        ] = []
        self._workflow_status_callbacks: list[
            Callable[[dict[str, WorkflowStatus]], None]
        ] = []

        # Subscribe to updates
        self._setup_subscriptions()

    @classmethod
    def from_config_service(
        cls,
        *,
        config_service,
        source_names: list[str],
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        config_store: ConfigStore | None = None,
        data_service: DataService[ResultKey, object] | None = None,
        correlation_histogram_controller: CorrelationHistogramController | None = None,
    ) -> WorkflowController:
        """Create WorkflowController from ConfigService and ConfigStore."""
        return cls(
            service=ConfigServiceAdapter(config_service),
            source_names=source_names,
            workflow_registry=workflow_registry,
            config_store=config_store,
            data_service=data_service,
            correlation_histogram_controller=correlation_histogram_controller,
        )

    def _setup_subscriptions(self) -> None:
        """Setup subscriptions to service updates."""
        # Subscribe to workflow status for each source
        for source_name in self._source_names:
            self._service.subscribe_to_workflow_status(
                source_name, self._update_workflow_status
            )

    def _update_workflow_status(self, status: WorkflowStatus) -> None:
        """Handle workflow status updates from service."""
        self._logger.info('Received workflow status update: %s', status)
        self._workflow_status[status.source_name] = status
        for callback in self._workflow_status_callbacks:
            self._notify_workflow_status_update(callback)

    def start_workflow(
        self,
        workflow_id: WorkflowId,
        source_names: list[str],
        config: pydantic.BaseModel,
        aux_source_names: pydantic.BaseModel | None = None,
    ) -> None:
        """Start a workflow with given configuration.

        Raises
        ------
        ValueError
            If the workflow spec is not found.
        """
        self._logger.info(
            'Starting workflow %s on sources %s with config %s and aux_sources %s',
            workflow_id,
            source_names,
            config,
            aux_source_names,
        )

        spec = self.get_workflow_spec(workflow_id)
        if spec is None:
            msg = f'Workflow spec for {workflow_id} not found'
            self._logger.error('%s, cannot start workflow', msg)
            raise ValueError(msg)

        # Create a SINGLE workflow config which will be used for ALL source names (see
        # loop below). WorkflowConfig.from_params generates a new job number which
        # allows for associating multiple jobs with the same workflow run across the
        # different sources.
        workflow_config = WorkflowConfig.from_params(
            workflow_id=workflow_id,
            params=config,
            aux_source_names=aux_source_names,
        )

        # Persist config for this workflow to restore widget state across sessions
        if self._config_store is not None:
            config_state = ConfigurationState(
                source_names=source_names,
                aux_source_names=workflow_config.aux_source_names,
                params=workflow_config.params,
            )
            self._config_store[workflow_id] = config_state.model_dump()

        # Send workflow config to each source
        for source_name in source_names:
            self._service.send_workflow_config(source_name, workflow_config)

            # Set status to STARTING for immediate UI feedback
            self._workflow_status[source_name] = WorkflowStatus(
                source_name=source_name,
                workflow_id=workflow_id,
                status=WorkflowStatusType.STARTING,
            )
        # Notify once, will update whole list of source names
        for callback in self._workflow_status_callbacks:
            self._notify_workflow_status_update(callback)

    def create_workflow_adapter(self, workflow_id: WorkflowId):
        """Create a workflow configuration adapter for the given workflow ID."""

        spec = self.get_workflow_spec(workflow_id)
        if spec is None:
            raise ValueError(f'Workflow {workflow_id} not found')

        # Handle correlation histogram workflows specially
        if (
            self._correlation_histogram_controller is not None
            and spec.namespace == 'correlation'
            and spec.name.startswith('correlation_histogram_')
        ):
            if spec.name == 'correlation_histogram_1d':
                return self._correlation_histogram_controller.create_1d_config()
            elif spec.name == 'correlation_histogram_2d':
                return self._correlation_histogram_controller.create_2d_config()

        # Handle regular workflows
        persistent_config = self.get_workflow_config(workflow_id)

        def start_callback(
            selected_sources: list[str],
            parameter_values: pydantic.BaseModel,
            aux_source_names: pydantic.BaseModel | None = None,
        ) -> None:
            """Bound callback to start this specific workflow."""
            self.start_workflow(
                workflow_id, selected_sources, parameter_values, aux_source_names
            )

        return WorkflowConfigurationAdapter(spec, persistent_config, start_callback)

    def get_workflow_titles(self) -> dict[WorkflowId, str]:
        """Get workflow IDs mapped to their titles, sorted by title."""
        return {
            workflow_id: spec.title
            for workflow_id, spec in sorted(
                self._workflow_registry.items(), key=lambda item: item[1].title
            )
        }

    def get_workflow_description(self, workflow_id: WorkflowId) -> str | None:
        """Get the description for the given workflow ID."""
        spec = self._workflow_registry.get(workflow_id)
        return spec.description if spec else None

    def get_workflow_spec(self, workflow_id: WorkflowId) -> WorkflowSpec | None:
        """Get the current workflow specification for the given Id."""
        return self._workflow_registry.get(workflow_id)

    def get_workflow_config(self, workflow_id: WorkflowId) -> ConfigurationState | None:
        """Load saved workflow configuration."""
        if self._config_store is None:
            return None
        if data := self._config_store.get(workflow_id):
            return ConfigurationState.model_validate(data)
        return None

    def subscribe_to_workflow_status_updates(
        self, callback: Callable[[dict[str, WorkflowStatus]], None]
    ) -> None:
        """Subscribe to workflow status updates."""
        self._workflow_status_callbacks.append(callback)
        self._notify_workflow_status_update(callback)

    def _notify_workflow_status_update(
        self, callback: Callable[[dict[str, WorkflowStatus]], None]
    ):
        try:
            callback(self._workflow_status.copy())
        except Exception as e:
            self._logger.error('Error in workflow status update callback: %s', e)
