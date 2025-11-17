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
from .workflow_config_service import WorkflowConfigService

SourceName = str


@dataclass
class JobConfig:
    """Configuration for a single job within a JobSet."""

    params: dict
    aux_source_names: dict


@dataclass
class JobSet:
    """A set of jobs sharing the same job_number.

    Maps source_name to config for each running job.
    JobId can be reconstructed as JobId(source_name, job_number).
    """

    job_number: JobNumber = field(default_factory=uuid.uuid4)
    jobs: dict[SourceName, JobConfig] = field(default_factory=dict)

    def add_job(self, source_name: SourceName, config: JobConfig) -> None:
        """Add a job to this set.

        Parameters
        ----------
        source_name
            Name of the source for this job.
        config
            Configuration for this job.
        """
        self.jobs[source_name] = config

    def job_ids(self) -> list[JobId]:
        """Create JobIds for all jobs in this set.

        Returns
        -------
        :
            List of JobIds, one per job in this set.
        """
        return [
            JobId(source_name=source_name, job_number=self.job_number)
            for source_name in self.jobs
        ]


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
        *,
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
            # Try to load persisted config directly as dict
            config_data = None
            if self._config_store is not None:
                config_data = self._config_store.get(workflow_id)

            # Get params/aux_source_names/source_names as dicts
            if config_data and config_data.get('params'):
                # Use loaded config (already dicts)
                params = config_data['params']
                aux_source_names = config_data.get('aux_source_names', {})
                source_names = config_data.get('source_names', [])
                self._logger.info(
                    'Loaded config for workflow %s from store: %d sources',
                    workflow_id,
                    len(source_names),
                )
            else:
                # Use defaults from spec, converting to dicts
                params = {}
                if spec.params is not None:
                    try:
                        params = spec.params().model_dump()
                    except pydantic.ValidationError:
                        # Params model has required fields without defaults
                        # These workflows don't use JobOrchestrator staging
                        self._workflows[workflow_id] = WorkflowState()
                        self._logger.debug(
                            'Initialized workflow %s (params cannot be instantiated)',
                            workflow_id,
                        )
                        continue

                aux_source_names = {}
                if spec.aux_sources is not None:
                    aux_source_names = spec.aux_sources().model_dump()

                source_names = spec.source_names
                self._logger.debug(
                    'Initialized workflow %s with defaults: %d sources',
                    workflow_id,
                    len(source_names),
                )

            # Initialize staged_jobs with dict-based configs
            if params:
                state = WorkflowState()
                for source_name in source_names:
                    state.staged_jobs[source_name] = JobConfig(
                        params=params,
                        aux_source_names=aux_source_names,
                    )
                self._workflows[workflow_id] = state
            else:
                # Workflow has no params - create empty WorkflowState
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
        *,
        source_name: SourceName,
        params: dict,
        aux_source_names: dict,
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
            Workflow parameters as dict.
        aux_source_names
            Auxiliary source names as dict.
        """
        self._workflows[workflow_id].staged_jobs[source_name] = JobConfig(
            params=params, aux_source_names=aux_source_names
        )

    def commit_workflow(self, workflow_id: WorkflowId) -> list[JobId]:
        """
        Commit staged configs and start workflow.

        Parameters
        ----------
        workflow_id
            The workflow to start.

        Returns
        -------
        :
            List of JobIds created (one per staged source).

        Raises
        ------
        ValueError
            If no configs have been staged for this workflow.
        """
        state = self._workflows[workflow_id]
        if not state.staged_jobs:
            msg = f'No staged configs for workflow {workflow_id}'
            raise ValueError(msg)

        # Create new JobSet with auto-generated job number
        job_set = JobSet()
        for source_name, job_config in state.staged_jobs.items():
            job_set.add_job(source_name, job_config)

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
            commands.extend(
                (
                    ConfigKey(key=JobCommand.key, source_name=str(job_id)),
                    JobCommand(job_id=job_id, action=JobAction.stop),
                )
                for job_id in state.current.job_ids()
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
            workflow_config.job_number = job_set.job_number
            key = keys.WORKFLOW_CONFIG.create_key(source_name=source_name)
            commands.append((key, workflow_config))

        self._command_service.send_batch(commands)

        self._logger.info(
            'Started workflow %s with job_number %s on sources %s',
            workflow_id,
            job_set.job_number,
            list(state.staged_jobs.keys()),
        )

        # Set as current JobSet
        state.current = job_set

        # Persist staged configs to store (keeping staged_jobs as working copy)
        self._persist_config_to_store(workflow_id, state.staged_jobs)

        # Return JobIds for all created jobs
        return job_set.job_ids()

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

        # Contract staged_jobs to storage format
        # (single params, list of sources - see ConfigurationState schema note)
        source_names = list(staged_jobs.keys())
        # Take params from first source (all should be same in current implementation)
        first_job_config = next(iter(staged_jobs.values()))

        config_dict = {
            'source_names': source_names,
            'params': first_job_config.params,
            'aux_source_names': first_job_config.aux_source_names,
        }

        self._config_store[workflow_id] = config_dict
        self._logger.debug(
            'Persisted config for workflow %s to store: %d sources',
            workflow_id,
            len(source_names),
        )

    def get_staged_config(self, workflow_id: WorkflowId) -> dict[SourceName, JobConfig]:
        """
        Get staged configuration for a workflow.

        Parameters
        ----------
        workflow_id
            The workflow to query.

        Returns
        -------
        :
            Dict mapping source names to their staged configs.
        """
        state = self._workflows[workflow_id]
        return state.staged_jobs.copy()

    def get_active_config(self, workflow_id: WorkflowId) -> dict[SourceName, JobConfig]:
        """
        Get active (running) configuration for a workflow.

        Parameters
        ----------
        workflow_id
            The workflow to query.

        Returns
        -------
        :
            Dict mapping source names to their active configs.
            Returns empty dict if no active jobs.
        """
        state = self._workflows[workflow_id]
        if state.current is None:
            return {}
        return state.current.jobs.copy()
