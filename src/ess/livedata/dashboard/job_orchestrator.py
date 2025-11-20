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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import NewType
from uuid import UUID

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
from .data_service import DataService

SourceName = str
SubscriptionId = NewType('SubscriptionId', UUID)


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
        data_service: DataService,
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
        data_service
            Service for accessing workflow result data. Used to detect when
            workflow data becomes available.
        source_names
            List of source names (for validation when staging configs).
        workflow_registry
            Registry of available workflows and their specifications.
        config_store
            Optional store for persisting workflow configurations across sessions.
            Orchestrator loads configs on init and persists on commit.

        Notes
        -----
        JobOrchestrator receives job status updates via the `status_updated` method,
        which is called by the main Orchestrator when STATUS_STREAM messages arrive.
        """
        self._command_service = command_service
        self._data_service = data_service
        self._source_names = source_names
        self._workflow_registry = workflow_registry
        self._config_store = config_store
        self._logger = logging.getLogger(__name__)

        # Workflow state tracking
        self._workflows: dict[WorkflowId, WorkflowState] = {}

        # Workflow subscription tracking
        self._subscriptions: dict[SubscriptionId, Callable[[JobNumber], None]] = {}
        self._workflow_subscriptions: dict[WorkflowId, set[SubscriptionId]] = {}

        # Track which job_numbers we've notified for each workflow
        # to avoid duplicate notifications
        self._notified_jobs: dict[WorkflowId, set[JobNumber]] = {}

        # Subscribe to data updates to detect when workflow data becomes available
        self._data_service.register_update_callback(self._on_data_updated)

        # Load persisted configs
        self._load_configs_from_store()

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
                        params = spec.params().model_dump(mode='json')
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
                    aux_source_names = spec.aux_sources().model_dump(mode='json')

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
                        params=params.copy(),
                        aux_source_names=aux_source_names.copy(),
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

        Raises
        ------
        ValueError
            If source_name is not in the list of monitored sources.
        """
        if source_name not in self._source_names:
            msg = (
                f'Cannot stage config for unknown source {source_name!r}. '
                f'Available sources: {self._source_names}'
            )
            raise ValueError(msg)

        self._workflows[workflow_id].staged_jobs[source_name] = JobConfig(
            params=params.copy(), aux_source_names=aux_source_names.copy()
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
        job_set = JobSet(jobs=state.staged_jobs.copy())

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
            # Move current to previous for cleanup once stop commands succeed.
            # Related to #445: Future improvements may wait for stop command
            # success responses before removing old job data.
            state.previous = state.current

        # Send workflow configs to all staged sources
        # Note: Currently all jobs use same params, but aux_source_names may differ
        for source_name, job_config in state.staged_jobs.items():
            # We are currently using a single command to mean "configure and start".
            # See #445 for plans on splitting this, in line with the interface of the
            # orchestrator interface.
            workflow_config = WorkflowConfig.from_params(
                workflow_id=workflow_id,
                params=job_config.params,
                aux_source_names=job_config.aux_source_names,
                job_number=job_set.job_number,
            )
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

        # Note: We do NOT notify subscribers here. They will be notified
        # when actual workflow data arrives via _on_data_updated.
        self._logger.info(
            'Workflow %s committed with job_number=%s. Subscribers will be '
            'notified when data arrives.',
            workflow_id,
            job_set.job_number,
        )

        # Persist staged configs to store (keeping staged_jobs as working copy)
        self._persist_config_to_store(workflow_id, state.staged_jobs)

        # Return JobIds for all created jobs
        return job_set.job_ids()

    def _on_data_updated(self, updated_keys: set) -> None:
        """
        Handle data updates from DataService.

        Called when new workflow result data arrives. Notifies subscribers
        when workflow data becomes available for the first time.

        Parameters
        ----------
        updated_keys
            Set of ResultKey objects for data that was updated.
        """
        from ess.livedata.config.workflow_spec import ResultKey

        # Group updates by (workflow_id, job_number) to notify only once per workflow
        workflows_to_notify: dict[WorkflowId, JobNumber] = {}

        for key in updated_keys:
            # Type guard - ensure we got a ResultKey
            if not isinstance(key, ResultKey):
                continue

            workflow_id = key.workflow_id
            job_number = key.job_id.job_number

            # Only track workflows we know about
            if workflow_id not in self._workflows:
                continue

            # Skip if we've already notified for this job_number
            if workflow_id not in self._notified_jobs:
                self._notified_jobs[workflow_id] = set()

            if job_number in self._notified_jobs[workflow_id]:
                continue

            # Check if this job_number matches current state or is a new job
            state = self._workflows[workflow_id]
            if state.current is not None and state.current.job_number != job_number:
                # Data arrived for a different job number - could be a restart
                self._logger.info(
                    'Data arrived for new job_number %s (previous: %s) '
                    'for workflow %s',
                    job_number,
                    state.current.job_number,
                    workflow_id,
                )

            # Mark for notification (collect all updates before notifying)
            workflows_to_notify[workflow_id] = job_number

        # Notify subscribers for all workflows that got new data
        for workflow_id, job_number in workflows_to_notify.items():
            self._logger.info(
                'First data arrived for workflow %s job_number %s, '
                'notifying subscribers',
                workflow_id,
                job_number,
            )
            self._notified_jobs[workflow_id].add(job_number)
            self._notify_workflow_available(workflow_id, job_number)

    def status_updated(self, job_status: object) -> None:
        """
        Process job status updates from the STATUS_STREAM.

        This method is called by the main Orchestrator when STATUS_STREAM messages
        arrive. Updates internal state to track running jobs.

        Parameters
        ----------
        job_status
            JobStatus object from the STATUS_STREAM.

        Notes
        -----
        This method tracks job state but does NOT notify subscribers. Subscribers
        are notified when actual data arrives (via _on_data_updated), not when
        the job status changes.

        Future: Full state machine for job lifecycle tracking:
        - Aggregate per-source status to JobSet state
        - Confirm job starts/stops
        - Handle failures and retries
        - Manage transitions between old and new JobSets
        """
        from ess.livedata.core.job import JobState, JobStatus

        self._logger.debug('Received job status: %s', job_status)

        # Type guard - only process JobStatus objects
        if not isinstance(job_status, JobStatus):
            self._logger.warning(
                'Received non-JobStatus object in status_updated: %s', type(job_status)
            )
            return

        workflow_id = job_status.workflow_id
        job_number = job_status.job_id.job_number
        source_name = job_status.job_id.source_name

        # Only track workflows we know about
        if workflow_id not in self._workflows:
            self._logger.debug('Ignoring status for untracked workflow %s', workflow_id)
            return

        state = self._workflows[workflow_id]

        # If we receive an active/scheduled status and don't have a current job,
        # or have a different job number, update our state
        if job_status.state in (JobState.active, JobState.scheduled):
            # Check if this is a new job we didn't know about
            if state.current is None or state.current.job_number != job_number:
                self._logger.info(
                    'Discovered running job for workflow %s: job_number=%s '
                    'from source=%s (previous job_number=%s)',
                    workflow_id,
                    job_number,
                    source_name,
                    state.current.job_number if state.current else None,
                )

                # Create a JobSet to track this running job
                # Note: We don't know all sources yet, but that's OK - subscribers
                # will request specific sources they need
                job_set = JobSet(job_number=job_number, jobs={})
                state.current = job_set

                # Note: We do NOT notify subscribers here. They will be notified
                # when actual data arrives via _on_data_updated.

    def _persist_config_to_store(
        self, workflow_id: WorkflowId, staged_jobs: dict[SourceName, JobConfig]
    ) -> None:
        """Persist staged job configs to config store.

        Returns silently if config_store is None (persistence is optional)
        or if staged_jobs is empty (nothing to persist).
        """
        if self._config_store is None or not staged_jobs:
            return

        # Contract staged_jobs to storage format
        # (single params, list of sources - see ConfigurationState schema note)
        source_names = list(staged_jobs.keys())
        # Take params from first source (all should be same in current implementation)
        first_job_config = next(iter(staged_jobs.values()))

        # Params and aux_source_names are already JSON-serializable (Enums converted
        # to strings via model_dump(mode='json') in workflow_controller.start_workflow)
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
        return {
            source_name: JobConfig(
                params=job_config.params.copy(),
                aux_source_names=job_config.aux_source_names.copy(),
            )
            for source_name, job_config in state.staged_jobs.items()
        }

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
        return {
            source_name: JobConfig(
                params=job_config.params.copy(),
                aux_source_names=job_config.aux_source_names.copy(),
            )
            for source_name, job_config in state.current.jobs.items()
        }

    def _notify_workflow_available(
        self, workflow_id: WorkflowId, job_number: JobNumber
    ) -> None:
        """
        Notify all subscribers that a workflow is now available.

        Parameters
        ----------
        workflow_id
            The workflow that was committed.
        job_number
            The job number for the new JobSet.
        """
        if workflow_id not in self._workflow_subscriptions:
            self._logger.debug(
                'No subscribers for workflow %s (job_number=%s)',
                workflow_id,
                job_number,
            )
            return

        subscriber_count = len(self._workflow_subscriptions[workflow_id])
        self._logger.info(
            'Notifying %d subscriber(s) that workflow %s is available (job_number=%s)',
            subscriber_count,
            workflow_id,
            job_number,
        )

        for subscription_id in self._workflow_subscriptions[workflow_id]:
            if subscription_id in self._subscriptions:
                try:
                    self._logger.debug(
                        'Calling callback for subscription %s '
                        '(workflow=%s, job_number=%s)',
                        subscription_id,
                        workflow_id,
                        job_number,
                    )
                    self._subscriptions[subscription_id](job_number)
                except Exception:
                    self._logger.exception(
                        'Error in workflow availability callback for workflow %s',
                        workflow_id,
                    )

    def subscribe_to_workflow(
        self, workflow_id: WorkflowId, callback: Callable[[JobNumber], None]
    ) -> SubscriptionId:
        """
        Subscribe to workflow data availability notifications.

        The callback will be called with the job_number when workflow data
        becomes available (i.e., first result data arrives from the workflow).

        If workflow data already exists when you subscribe, the callback
        will be called immediately with the current job_number.

        Parameters
        ----------
        workflow_id
            The workflow to subscribe to.
        callback
            Called with job_number when workflow data becomes available.

        Returns
        -------
        :
            Subscription ID that can be used to unsubscribe.
        """
        subscription_id = SubscriptionId(uuid.uuid4())
        self._subscriptions[subscription_id] = callback

        # Track which workflows have subscriptions
        if workflow_id not in self._workflow_subscriptions:
            self._workflow_subscriptions[workflow_id] = set()
        self._workflow_subscriptions[workflow_id].add(subscription_id)

        # If workflow data already exists, notify immediately
        if workflow_id in self._workflows:
            state = self._workflows[workflow_id]
            if state.current is not None:
                current_job_number = state.current.job_number
                # Check if we have data for this job_number
                has_data = self._has_workflow_data(workflow_id, current_job_number)
                if has_data:
                    self._logger.debug(
                        'Workflow %s already has data for job_number=%s, '
                        'notifying new subscriber %s immediately',
                        workflow_id,
                        current_job_number,
                        subscription_id,
                    )
                    try:
                        callback(current_job_number)
                    except Exception:
                        self._logger.exception(
                            'Error in immediate workflow availability callback '
                            'for subscription %s',
                            subscription_id,
                        )
                else:
                    self._logger.debug(
                        'Workflow %s is running (job_number=%s) but no data yet, '
                        'will notify subscriber %s when data arrives',
                        workflow_id,
                        current_job_number,
                        subscription_id,
                    )

        return subscription_id

    def _has_workflow_data(
        self, workflow_id: WorkflowId, job_number: JobNumber
    ) -> bool:
        """
        Check if DataService has any data for a workflow's job_number.

        Parameters
        ----------
        workflow_id
            The workflow to check.
        job_number
            The job number to check.

        Returns
        -------
        :
            True if DataService has at least one result for this workflow/job.
        """
        from ess.livedata.config.workflow_spec import ResultKey

        # Check if any key in DataService matches this workflow_id and job_number
        for key in self._data_service:
            if isinstance(key, ResultKey):
                if (
                    key.workflow_id == workflow_id
                    and key.job_id.job_number == job_number
                ):
                    return True
        return False

    def unsubscribe(self, subscription_id: SubscriptionId) -> None:
        """
        Unsubscribe from workflow availability notifications.

        Parameters
        ----------
        subscription_id
            The subscription ID returned from subscribe_to_workflow.
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            # Remove from workflow tracking
            for workflow_subs in self._workflow_subscriptions.values():
                workflow_subs.discard(subscription_id)
            self._logger.debug('Removed subscription %s', subscription_id)
        else:
            self._logger.warning(
                'Attempted to unsubscribe from non-existent subscription %s',
                subscription_id,
            )
