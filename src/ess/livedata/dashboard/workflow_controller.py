# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Workflow controller implementation backed by a config service.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pydantic
import structlog

from ess.livedata.config.workflow_spec import (
    JobId,
    ResultKey,
    WorkflowId,
    WorkflowSpec,
)

from .configuration_adapter import ConfigurationState
from .data_service import DataService
from .job_orchestrator import JobOrchestrator
from .workflow_configuration_adapter import WorkflowConfigurationAdapter

if TYPE_CHECKING:
    from ess.livedata.config import Instrument

logger = structlog.get_logger(__name__)


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
        job_orchestrator: JobOrchestrator,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        data_service: DataService[ResultKey, object] | None = None,
        instrument_config: Instrument | None = None,
    ) -> None:
        """
        Initialize the workflow controller.

        Parameters
        ----------
        job_orchestrator
            Shared job orchestrator for workflow lifecycle management.
        workflow_registry
            Registry of available workflows and their specifications.
        data_service
            Optional data service for cleaning up workflow data keys.
        instrument_config
            Optional instrument configuration for source metadata lookup.
        """
        self._orchestrator = job_orchestrator
        self._instrument_config = instrument_config

        # Extend registry with correlation histogram specs if controller provided
        self._workflow_registry = dict(workflow_registry)
        self._data_service = data_service

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
        logger.info(
            'WorkflowController.start_workflow: workflow_id=%s, sources=%s, '
            'config=%s, aux_sources=%s',
            workflow_id,
            source_names,
            config,
            aux_source_names,
        )

        spec = self.get_workflow_spec(workflow_id)
        if spec is None:
            msg = f'Workflow spec for {workflow_id} not found'
            logger.error('%s, cannot start workflow', msg)
            raise ValueError(msg)

        # If no sources, nothing to start
        if not source_names:
            return []

        # Convert Pydantic models to dicts for orchestrator
        params_dict = config.model_dump(mode='json')
        aux_dict = aux_source_names.model_dump(mode='json') if aux_source_names else {}

        # Replace all staged configs in a transaction (single notification)
        with self._orchestrator.staging_transaction(workflow_id):
            self._orchestrator.clear_staged_configs(workflow_id)
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

        persistent_config = self.get_workflow_config(workflow_id)

        # Determine initial source names from staged config if available
        initial_source_names = None
        staged_jobs = self._orchestrator.get_staged_config(workflow_id)
        if staged_jobs:
            initial_source_names = list(staged_jobs.keys())

        def start_callback(
            selected_sources: list[str],
            parameter_values: pydantic.BaseModel,
            aux_source_names: pydantic.BaseModel | None = None,
        ) -> None:
            """Bound callback to start this specific workflow."""
            self.start_workflow(
                workflow_id, selected_sources, parameter_values, aux_source_names
            )

        return WorkflowConfigurationAdapter(
            spec,
            persistent_config,
            start_callback,
            initial_source_names,
            instrument_config=self._instrument_config,
        )

    def get_workflow_spec(self, workflow_id: WorkflowId) -> WorkflowSpec | None:
        """Get the current workflow specification for the given Id."""
        return self._workflow_registry.get(workflow_id)

    def get_workflow_config(self, workflow_id: WorkflowId) -> ConfigurationState | None:
        """Load saved workflow configuration.

        Returns the reference configuration (from first staged source) for the
        given workflow. This is primarily used for adapter initialization.
        """
        try:
            staged_jobs = self._orchestrator.get_staged_config(workflow_id)
        except KeyError:
            # Workflow not in registry
            return None

        if not staged_jobs:
            return None

        # Return config from first staged source as reference
        first_job = next(iter(staged_jobs.values()))
        return ConfigurationState(
            params=first_job.params,
            aux_source_names=first_job.aux_source_names,
        )
