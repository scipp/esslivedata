# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
JobOrchestrator - Manages workflow job lifecycle and state transitions.

Coordinates workflow execution across multiple sources, handling:
- Configuration staging and commit (two-phase workflow start)
- Job number generation and JobSet lifecycle
- Workflow status tracking and aggregation
- Job transitions and cleanup
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pydantic

import ess.livedata.config.keys as keys
from ess.livedata.config.workflow_spec import (
    JobNumber,
    WorkflowConfig,
    WorkflowId,
    WorkflowSpec,
    WorkflowStatus,
)

from .command_service import CommandService
from .workflow_config_service import WorkflowConfigService

SourceName = str


@dataclass
class JobConfig:
    """Configuration for a single job within a JobSet."""

    params: pydantic.BaseModel
    aux_source_names: pydantic.BaseModel | None


@dataclass
class JobSet:
    """A set of jobs sharing the same job_number.

    Maps source_name to config for each running job.
    JobId can be reconstructed as JobId(source_name, job_number).
    """

    job_number: JobNumber
    jobs: dict[SourceName, JobConfig]


@dataclass
class WorkflowState:
    """State for an active workflow, including transitions."""

    current: JobSet | None = None
    previous: JobSet | None = None
    staged_jobs: dict[SourceName, JobConfig] = field(default_factory=dict)


class JobOrchestrator:
    """Orchestrates workflow job lifecycle and state management."""

    def __init__(
        self,
        command_service: CommandService,
        workflow_config_service: WorkflowConfigService,
        source_names: list[SourceName],
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
    ) -> None:
        """
        Initialize the job orchestrator.

        Parameters
        ----------
        command_service
            Service for sending workflow commands to backend services.
        workflow_config_service
            Service for receiving workflow status updates from backend services.
        source_names
            List of source names to monitor.
        workflow_registry
            Registry of available workflows and their specifications.
        """
        self._command_service = command_service
        self._workflow_config_service = workflow_config_service
        self._source_names = source_names
        self._workflow_registry = workflow_registry
        self._logger = logging.getLogger(__name__)

        # Workflow state tracking
        self._workflows: defaultdict[WorkflowId, WorkflowState] = defaultdict(
            WorkflowState
        )

        # Status callbacks
        self._status_callbacks: list[Callable[[WorkflowStatus], None]] = []

        # Setup subscriptions to workflow status updates
        self._setup_subscriptions()

    def _setup_subscriptions(self) -> None:
        """Subscribe to workflow status updates from all sources."""
        for source_name in self._source_names:
            self._workflow_config_service.subscribe_to_workflow_status(
                source_name, self.handle_response
            )

    def stage_config(
        self,
        workflow_id: WorkflowId,
        source_name: SourceName,
        params: pydantic.BaseModel,
        aux_source_names: pydantic.BaseModel | None = None,
    ) -> None:
        """
        Stage configuration for a source.

        Parameters
        ----------
        workflow_id
            The workflow to configure.
        source_name
            Source name to configure.
        params
            Workflow parameters.
        aux_source_names
            Optional auxiliary source names for this specific job.
        """
        self._workflows[workflow_id].staged_jobs[source_name] = JobConfig(
            params=params, aux_source_names=aux_source_names
        )

    def commit_workflow(self, workflow_id: WorkflowId) -> JobNumber:
        """
        Commit staged configs and start workflow.

        Parameters
        ----------
        workflow_id
            The workflow to start.

        Returns
        -------
        :
            The generated job number for this workflow run.

        Raises
        ------
        ValueError
            If no configs have been staged for this workflow.
        """
        state = self._workflows[workflow_id]
        if not state.staged_jobs:
            msg = f'No staged configs for workflow {workflow_id}'
            raise ValueError(msg)

        # Generate job number for this workflow run
        job_number = uuid.uuid4()

        # TODO: Stop old jobs if any
        if state.current is not None:
            self._logger.info(
                'Workflow %s already has active jobs, should stop: %s',
                workflow_id,
                state.current.job_number,
            )
            # Move current to previous for potential cleanup later
            state.previous = state.current

        # Send workflow configs to all staged sources
        # Note: Currently all jobs use same params, but aux_source_names may differ
        commands = []
        for source_name, job_config in state.staged_jobs.items():
            workflow_config = WorkflowConfig.from_params(
                workflow_id=workflow_id,
                params=job_config.params,
                aux_source_names=job_config.aux_source_names,
            )
            # Override job_number to ensure all jobs in this run share it
            workflow_config.job_number = job_number
            key = keys.WORKFLOW_CONFIG.create_key(source_name=source_name)
            commands.append((key, workflow_config))

        self._command_service.send_batch(commands)

        self._logger.info(
            'Started workflow %s with job_number %s on sources %s',
            workflow_id,
            job_number,
            list(state.staged_jobs.keys()),
        )

        # Create new JobSet
        state.current = JobSet(
            job_number=job_number,
            jobs=state.staged_jobs.copy(),
        )

        # Clear staging area
        state.staged_jobs.clear()

        return job_number

    def handle_response(self, status: WorkflowStatus) -> None:
        """
        Handle workflow status updates from backend services.

        Parameters
        ----------
        status
            Workflow status update from a source.

        Notes
        -----
        Future: Implement state machine for job lifecycle tracking:
        - Aggregate per-source status to JobSet state
        - Confirm job starts/stops
        - Handle failures and retries
        - Manage transitions between old and new JobSets
        """
        self._logger.info('Received workflow response: %s', status)
        # TODO: State machine logic here

        # Notify subscribers
        for callback in self._status_callbacks:
            try:
                callback(status)
            except Exception as e:
                self._logger.error('Error in status callback: %s', e)

    def subscribe_to_status_updates(
        self, callback: Callable[[WorkflowStatus], None]
    ) -> None:
        """Subscribe to workflow status updates."""
        self._status_callbacks.append(callback)

    def get_active_job_number(self, workflow_id: WorkflowId) -> JobNumber | None:
        """Get the active job number for a workflow, if any."""
        state = self._workflows.get(workflow_id)
        if state is None or state.current is None:
            return None
        return state.current.job_number

    def is_workflow_running(self, workflow_id: WorkflowId) -> bool:
        """Check if a workflow has active jobs."""
        return self.get_active_job_number(workflow_id) is not None

    def get_active_sources(self, workflow_id: WorkflowId) -> list[SourceName]:
        """Get list of sources with active jobs for a workflow."""
        state = self._workflows.get(workflow_id)
        if state is None or state.current is None:
            return []
        return list(state.current.jobs.keys())
