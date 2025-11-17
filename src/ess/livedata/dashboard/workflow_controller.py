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
    JobId,
    ResultKey,
    WorkflowId,
    WorkflowSpec,
)

from .command_service import CommandService
from .config_store import ConfigStore
from .configuration_adapter import ConfigurationState
from .correlation_histogram import CorrelationHistogramController, make_workflow_spec
from .data_service import DataService
from .job_orchestrator import JobOrchestrator
from .workflow_config_service import WorkflowConfigService
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
        command_service: CommandService,
        workflow_config_service: WorkflowConfigService,
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
        command_service
            Service for sending workflow commands to backend services.
        workflow_config_service
            Service for receiving workflow status updates from backend services.
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

        # Create job orchestrator for workflow lifecycle management
        self._orchestrator = JobOrchestrator(
            command_service=command_service,
            workflow_config_service=workflow_config_service,
            source_names=source_names,
            workflow_registry=self._workflow_registry,
            config_store=config_store,
        )

        # Callbacks
        self._workflow_specs_callbacks: list[
            Callable[[dict[WorkflowId, WorkflowSpec]], None]
        ] = []

    def start_workflow(
        self,
        workflow_id: WorkflowId,
        source_names: list[str],
        config: pydantic.BaseModel,
        aux_source_names: pydantic.BaseModel | None = None,
    ) -> list[JobId]:
        """Start a workflow with given configuration.

        Parameters
        ----------
        workflow_id:
            The workflow to start
        source_names:
            List of source names to process
        config:
            Workflow configuration parameters
        aux_source_names:
            Optional auxiliary source names

        Returns
        -------
        :
            List of JobIds created (one per source)

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

        # If no sources, nothing to start
        if not source_names:
            return []

        # Clear existing staged configs and stage new ones
        # This ensures only the requested sources are included in the workflow
        self._orchestrator.clear_staged_configs(workflow_id)

        # Convert Pydantic models to dicts for orchestrator
        params_dict = config.model_dump()
        aux_dict = aux_source_names.model_dump() if aux_source_names else {}

        for source_name in source_names:
            self._orchestrator.stage_config(
                workflow_id,
                source_name=source_name,
                params=params_dict,
                aux_source_names=aux_dict,
            )

        # Commit and start workflow
        return self._orchestrator.commit_workflow(workflow_id)

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
        """
        Load saved workflow configuration.

        Returns the staged config from orchestrator, which reflects either:
        - Config loaded from persistent storage on init, or
        - Config from most recent commit (active job config)
        """
        try:
            staged_jobs = self._orchestrator.get_staged_config(workflow_id)
        except KeyError:
            # Workflow not in registry
            return None

        if not staged_jobs:
            return None

        # Convert JobOrchestrator's staged_jobs back to ConfigurationState
        # (see ConfigurationState schema note about expansion/contraction)
        source_names = list(staged_jobs.keys())
        first_job_config = next(iter(staged_jobs.values()))

        return ConfigurationState(
            source_names=source_names,
            params=first_job_config.params,
            aux_source_names=first_job_config.aux_source_names,
        )
