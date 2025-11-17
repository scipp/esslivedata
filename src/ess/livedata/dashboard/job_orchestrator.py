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
from collections.abc import Mapping
from dataclasses import dataclass, field

import pydantic

import ess.livedata.config.keys as keys
from ess.livedata.config.models import ConfigKey
from ess.livedata.config.workflow_spec import (
    JobId,
    JobNumber,
    WorkflowConfig,
    WorkflowId,
    WorkflowSpec,
)
from ess.livedata.core.job_manager import JobAction, JobCommand

from .command_service import CommandService
from .config_store import ConfigStore
from .configuration_adapter import ConfigurationState
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
        config_store: ConfigStore | None = None,
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
        config_store
            Optional store for persisting workflow configurations across sessions.
            Orchestrator loads configs on init and persists on commit.
        """
        self._command_service = command_service
        self._workflow_config_service = workflow_config_service
        self._source_names = source_names
        self._workflow_registry = workflow_registry
        self._config_store = config_store
        self._logger = logging.getLogger(__name__)

        # Workflow state tracking
        self._workflows: dict[WorkflowId, WorkflowState] = {}

        # Load persisted configs
        self._load_configs_from_store()

        # Setup subscriptions to workflow status updates
        self._setup_subscriptions()

    def _setup_subscriptions(self) -> None:
        """Subscribe to workflow status updates from all sources."""
        for source_name in self._source_names:
            self._workflow_config_service.subscribe_to_workflow_status(
                source_name, self.handle_response
            )

    def _load_configs_from_store(self) -> None:
        """Initialize all workflows with either loaded configs or defaults from spec."""
        for workflow_id, spec in self._workflow_registry.items():
            # Try to load persisted config
            config_state = None
            if self._config_store is not None:
                config_data = self._config_store.get(workflow_id)
                if config_data is not None:
                    try:
                        config_state = ConfigurationState.model_validate(config_data)
                    except Exception as e:
                        self._logger.warning(
                            'Failed to load config for workflow %s: %s. '
                            'Using defaults.',
                            workflow_id,
                            e,
                        )
                        config_state = None

            # Determine params, aux_source_names, and source_names
            if config_state is not None and config_state.params:
                # Use loaded config
                if spec.params is None:
                    self._logger.warning(
                        'Loaded config for workflow %s has params but '
                        'spec defines none',
                        workflow_id,
                    )
                    # Workflow has no params - just create empty WorkflowState
                    self._workflows[workflow_id] = WorkflowState()
                    continue

                try:
                    params = spec.params.model_validate(config_state.params)
                    aux_source_names = None
                    if config_state.aux_source_names and spec.aux_sources:
                        aux_source_names = spec.aux_sources.model_validate(
                            config_state.aux_source_names
                        )
                    source_names = config_state.source_names
                    self._logger.info(
                        'Loaded config for workflow %s from store: %d sources',
                        workflow_id,
                        len(source_names),
                    )
                except Exception as e:
                    self._logger.warning(
                        'Failed to validate config for workflow %s: %s. '
                        'Using defaults.',
                        workflow_id,
                        e,
                    )
                    # Fall back to defaults
                    params = spec.params() if spec.params is not None else None
                    aux_source_names = (
                        spec.aux_sources() if spec.aux_sources is not None else None
                    )
                    source_names = spec.source_names
            else:
                # Use defaults from spec
                if spec.params is not None:
                    try:
                        params = spec.params()
                    except pydantic.ValidationError:
                        # Params model has required fields without defaults
                        # (e.g., correlation histograms with dynamic defaults)
                        # These workflows don't use JobOrchestrator
                        self._workflows[workflow_id] = WorkflowState()
                        self._logger.debug(
                            'Initialized workflow %s (params cannot be instantiated)',
                            workflow_id,
                        )
                        continue
                else:
                    params = None
                aux_source_names = (
                    spec.aux_sources() if spec.aux_sources is not None else None
                )
                source_names = spec.source_names

            # Initialize staged_jobs for all sources (if we have params)
            if params is not None:
                state = WorkflowState()
                for source_name in source_names:
                    state.staged_jobs[source_name] = JobConfig(
                        params=params, aux_source_names=aux_source_names
                    )
                self._workflows[workflow_id] = state

                if config_state is None:
                    self._logger.debug(
                        'Initialized workflow %s with defaults: %d sources',
                        workflow_id,
                        len(source_names),
                    )
            else:
                # Workflow has no params - just create empty WorkflowState
                self._workflows[workflow_id] = WorkflowState()
                self._logger.debug('Initialized workflow %s (no params)', workflow_id)

    def clear_staged_configs(self, workflow_id: WorkflowId) -> None:
        """
        Clear all staged configs for a workflow.

        Parameters
        ----------
        workflow_id
            The workflow to clear staged configs for.
        """
        self._workflows[workflow_id].staged_jobs.clear()

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

        # Prepare all commands (stop old jobs + start new workflow) in single batch
        commands = []

        # Stop old jobs if any
        if state.current is not None:
            self._logger.info(
                'Workflow %s already has active jobs, stopping: %s',
                workflow_id,
                state.current.job_number,
            )
            # Create stop commands for all old jobs
            old_job_number = state.current.job_number
            for source_name in state.current.jobs:
                job_id = JobId(source_name=source_name, job_number=old_job_number)
                commands.append(
                    (
                        ConfigKey(key=JobCommand.key, source_name=str(job_id)),
                        JobCommand(job_id=job_id, action=JobAction.stop),
                    )
                )
            self._logger.debug(
                'Will stop %d old jobs in batch', len(state.current.jobs)
            )
            # Move current to previous for potential cleanup later
            state.previous = state.current

        # Send workflow configs to all staged sources
        # Note: Currently all jobs use same params, but aux_source_names may differ
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

        # Create new JobSet from staged configs
        state.current = JobSet(job_number=job_number, jobs=state.staged_jobs.copy())

        # Persist staged configs to store (keeping staged_jobs as working copy)
        self._persist_config_to_store(workflow_id, state.staged_jobs)

        return job_number

    def handle_response(self, status: object) -> None:
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

    def _persist_config_to_store(
        self, workflow_id: WorkflowId, staged_jobs: dict[SourceName, JobConfig]
    ) -> None:
        """Persist staged job configs to config store."""
        if self._config_store is None or not staged_jobs:
            return

        # Contract staged_jobs back to ConfigurationState schema
        # (single params, list of sources - see ConfigurationState schema note)
        source_names = list(staged_jobs.keys())
        # Take params from first source (all should be same in current implementation)
        first_job_config = next(iter(staged_jobs.values()))

        config_state = ConfigurationState(
            source_names=source_names,
            params=first_job_config.params.model_dump(),
            aux_source_names=(
                first_job_config.aux_source_names.model_dump()
                if first_job_config.aux_source_names
                else {}
            ),
        )

        self._config_store[workflow_id] = config_state.model_dump()
        self._logger.debug(
            'Persisted config for workflow %s to store: %d sources',
            workflow_id,
            len(source_names),
        )

    def get_staged_config(
        self, workflow_id: WorkflowId, source_name: SourceName | None = None
    ) -> JobConfig | dict[SourceName, JobConfig] | None:
        """
        Get staged configuration for a workflow.

        Parameters
        ----------
        workflow_id
            The workflow to query.
        source_name
            Optional source name. If provided, returns config for that source only.
            If None, returns all staged configs as a dict.

        Returns
        -------
        :
            JobConfig for the specified source, dict of all staged configs,
            or None if source not found.
        """
        state = self._workflows[workflow_id]

        if source_name is not None:
            return state.staged_jobs.get(source_name)

        return state.staged_jobs.copy()

    def get_active_config(
        self, workflow_id: WorkflowId, source_name: SourceName | None = None
    ) -> JobConfig | dict[SourceName, JobConfig] | None:
        """
        Get active (running) configuration for a workflow.

        Parameters
        ----------
        workflow_id
            The workflow to query.
        source_name
            Optional source name. If provided, returns config for that source only.
            If None, returns all active configs as a dict.

        Returns
        -------
        :
            JobConfig for the specified source, dict of all active configs,
            or None if source not found or not running.
        """
        state = self._workflows[workflow_id]
        if state.current is None:
            return None

        if source_name is not None:
            return state.current.jobs.get(source_name)

        return state.current.jobs.copy()

    def get_active_job_number(self, workflow_id: WorkflowId) -> JobNumber | None:
        """Get the active job number for a workflow, if any."""
        state = self._workflows[workflow_id]
        if state.current is None:
            return None
        return state.current.job_number

    def is_workflow_running(self, workflow_id: WorkflowId) -> bool:
        """Check if a workflow has active jobs."""
        return self.get_active_job_number(workflow_id) is not None

    def get_active_sources(self, workflow_id: WorkflowId) -> list[SourceName]:
        """Get list of sources with active jobs for a workflow."""
        state = self._workflows[workflow_id]
        if state.current is None:
            return []
        return list(state.current.jobs.keys())
