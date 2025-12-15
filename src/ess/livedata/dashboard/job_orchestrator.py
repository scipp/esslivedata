# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
JobOrchestrator - Manages workflow job lifecycle and state transitions.

Coordinates workflow execution across multiple sources, handling:
- Configuration staging and commit (two-phase workflow start)
- Job number generation and JobSet lifecycle
- Job transitions and cleanup
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import NewType
from uuid import UUID

import pydantic
from pydantic import BaseModel, Field

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
from .configuration_adapter import ConfigurationState, JobConfigState
from .workflow_config_service import WorkflowConfigService
from .workflow_configuration_adapter import WorkflowConfigurationAdapter

SourceName = str
SubscriptionId = NewType('SubscriptionId', UUID)
WidgetLifecycleSubscriptionId = NewType('WidgetLifecycleSubscriptionId', UUID)


class JobConfig(BaseModel):
    """Configuration for a single job within a JobSet."""

    params: dict
    aux_source_names: dict


class JobSet(BaseModel):
    """A set of jobs sharing the same job_number.

    Maps source_name to config for each running job.
    JobId can be reconstructed as JobId(source_name, job_number).
    """

    job_number: JobNumber = Field(default_factory=uuid.uuid4)
    jobs: dict[SourceName, JobConfig] = Field(default_factory=dict)

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


@dataclass
class WidgetLifecycleCallbacks:
    """Callbacks for widget lifecycle events from JobOrchestrator.

    Widgets use this to stay synchronized across browser sessions.
    All callbacks receive the workflow_id so widgets can filter relevant events.
    """

    on_staged_changed: Callable[[WorkflowId], None] | None = None
    on_workflow_committed: Callable[[WorkflowId], None] | None = None
    on_workflow_stopped: Callable[[WorkflowId], None] | None = None


class JobOrchestrator:
    """Orchestrates workflow job lifecycle and state management."""

    def __init__(
        self,
        *,
        command_service: CommandService,
        workflow_config_service: WorkflowConfigService,
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
        workflow_registry
            Registry of available workflows and their specifications.
        config_store
            Optional store for persisting workflow configurations across sessions.
            Orchestrator loads configs on init and persists on commit.
        """
        self._command_service = command_service
        self._workflow_config_service = workflow_config_service
        self._workflow_registry = workflow_registry
        self._config_store = config_store
        self._logger = logging.getLogger(__name__)

        # Workflow state tracking
        self._workflows: dict[WorkflowId, WorkflowState] = {}

        # Workflow subscription tracking (for PlotOrchestrator)
        self._subscriptions: dict[SubscriptionId, Callable[[JobNumber], None]] = {}
        self._workflow_subscriptions: dict[WorkflowId, set[SubscriptionId]] = {}

        # Widget lifecycle subscription tracking (for cross-session widget sync)
        self._widget_subscriptions: dict[
            WidgetLifecycleSubscriptionId, WidgetLifecycleCallbacks
        ] = {}

        # Transaction state for batching staging operations
        self._transaction_workflow: WorkflowId | None = None
        self._transaction_depth: int = 0

        # Load persisted configs
        self._load_configs_from_store()

    def _load_configs_from_store(self) -> None:
        """Initialize all workflows with either loaded configs or defaults from spec."""
        for workflow_id, spec in self._workflow_registry.items():
            # Try to load persisted config directly as dict
            config_data = None
            if self._config_store is not None:
                config_data = self._config_store.get(str(workflow_id))

            state = WorkflowState()
            available_sources = set(spec.source_names)

            # Load per-source configs from 'jobs' dict (new schema)
            if config_data and config_data.get('jobs'):
                jobs_data = config_data['jobs']
                loaded_count = 0
                for source_name, job_data in jobs_data.items():
                    # Filter to sources that exist in the spec
                    if source_name in available_sources:
                        state.staged_jobs[source_name] = JobConfig(
                            params=job_data.get('params', {}),
                            aux_source_names=job_data.get('aux_source_names', {}),
                        )
                        loaded_count += 1
                self._logger.info(
                    'Loaded config for workflow %s from store: %d sources',
                    workflow_id,
                    loaded_count,
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

                if params:
                    for source_name in spec.source_names:
                        state.staged_jobs[source_name] = JobConfig(
                            params=params.copy(),
                            aux_source_names=aux_source_names.copy(),
                        )
                self._logger.debug(
                    'Initialized workflow %s with defaults: %d sources',
                    workflow_id,
                    len(spec.source_names),
                )

            # Restore active job state if present
            if config_data:
                if current_data := config_data.get('current_job'):
                    try:
                        state.current = JobSet.model_validate(current_data)
                    except (KeyError, ValueError, TypeError) as e:
                        self._logger.warning(
                            'Failed to restore active job for workflow %s: %s',
                            workflow_id,
                            e,
                        )

                if previous_data := config_data.get('previous_job'):
                    try:
                        state.previous = JobSet.model_validate(previous_data)
                    except (KeyError, ValueError, TypeError) as e:
                        self._logger.warning(
                            'Failed to restore previous job for workflow %s: %s',
                            workflow_id,
                            e,
                        )

            self._workflows[workflow_id] = state

            # Note: No need to notify subscribers here - none exist during __init__.
            # Future subscribers are notified via subscribe_to_workflow(), which
            # checks for existing active jobs and notifies immediately.

    @contextmanager
    def staging_transaction(self, workflow_id: WorkflowId):
        """
        Context manager for batching staging operations.

        All staging operations (clear_staged_configs, stage_config) within the
        context will trigger only a single notification when the context exits.
        This prevents N+1 UI rebuild issues when performing multiple staging
        operations together.

        Transactions can be nested - only the outermost transaction triggers
        the notification on exit.

        Parameters
        ----------
        workflow_id
            The workflow to perform staging operations on.

        Yields
        ------
        None

        Raises
        ------
        ValueError
            If attempting to nest transactions for different workflows.
        """
        # Enter transaction
        if self._transaction_workflow is None:
            self._transaction_workflow = workflow_id
        elif self._transaction_workflow != workflow_id:
            msg = (
                f'Cannot nest transactions for different workflows: '
                f'current={self._transaction_workflow}, new={workflow_id}'
            )
            raise ValueError(msg)

        self._transaction_depth += 1

        try:
            yield
        finally:
            # Exit transaction
            self._transaction_depth -= 1
            if self._transaction_depth == 0:
                # Outermost transaction exiting - send notification
                self._transaction_workflow = None
                self._notify_staged_changed(workflow_id)

    def clear_staged_configs(self, workflow_id: WorkflowId) -> None:
        """
        Clear all staged configs for a workflow.

        Parameters
        ----------
        workflow_id
            The workflow to clear staged configs for.
        """
        self._workflows[workflow_id].staged_jobs.clear()
        self._notify_staged_changed(workflow_id)

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
            params=params.copy(), aux_source_names=aux_source_names.copy()
        )
        self._notify_staged_changed(workflow_id)

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

        # Persist full state (staged configs + active jobs) to store
        self._persist_state_to_store(workflow_id)

        # Notify PlotOrchestrator subscribers that job is active
        self._logger.info(
            'Workflow %s committed with job_number=%s, notifying subscribers',
            workflow_id,
            job_set.job_number,
        )
        self._notify_workflow_available(workflow_id, job_set.job_number)

        # Notify widget lifecycle subscribers that workflow was committed
        self._notify_workflow_committed(workflow_id)

        # Return JobIds for all created jobs
        return job_set.job_ids()

    def _persist_state_to_store(self, workflow_id: WorkflowId) -> None:
        """Persist full workflow state (config + active jobs) to config store.

        Returns silently if config_store is None (persistence is optional).
        """
        if self._config_store is None:
            return

        state = self._workflows.get(workflow_id)
        if state is None:
            return

        # Build config dict from staged_jobs if present
        config_dict: dict = {}
        if state.staged_jobs:
            config_dict = {
                'jobs': {
                    source_name: {
                        'params': job_config.params,
                        'aux_source_names': job_config.aux_source_names,
                    }
                    for source_name, job_config in state.staged_jobs.items()
                }
            }

        # Add active job state
        if state.current is not None:
            config_dict['current_job'] = state.current.model_dump(mode='json')

        if state.previous is not None:
            config_dict['previous_job'] = state.previous.model_dump(mode='json')

        # Persist if we have something to save
        if config_dict:
            self._config_store[str(workflow_id)] = config_dict

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

    def get_active_job_number(self, workflow_id: WorkflowId) -> JobNumber | None:
        """
        Get the job number of the currently active job for a workflow.

        Parameters
        ----------
        workflow_id
            The workflow to query.

        Returns
        -------
        :
            The job number if there's an active job, None otherwise.
        """
        state = self._workflows.get(workflow_id)
        if state is None or state.current is None:
            return None
        return state.current.job_number

    def get_workflow_registry(self) -> Mapping[WorkflowId, WorkflowSpec]:
        """
        Get the workflow registry containing all managed workflows.

        Returns
        -------
        :
            Mapping from workflow ID to workflow spec.
        """
        return self._workflow_registry

    def create_workflow_adapter(
        self, workflow_id: WorkflowId
    ) -> WorkflowConfigurationAdapter:
        """
        Create a workflow configuration adapter for the given workflow ID.

        The adapter provides the interface for configuration widgets to display
        workflow parameters and start the workflow.

        Parameters
        ----------
        workflow_id
            The workflow to create an adapter for.

        Returns
        -------
        :
            Configuration adapter for the workflow.

        Raises
        ------
        KeyError
            If the workflow ID is not in the registry.
        """
        spec = self._workflow_registry[workflow_id]

        # Convert staged config to ConfigurationState for the adapter
        config_state = self._get_config_state(workflow_id)

        def start_callback(
            selected_sources: list[str],
            parameter_values,
            aux_source_names=None,
        ) -> None:
            """Stage configs and commit the workflow."""
            params_dict = parameter_values.model_dump(mode='json')
            aux_dict = (
                aux_source_names.model_dump(mode='json') if aux_source_names else {}
            )

            with self.staging_transaction(workflow_id):
                self.clear_staged_configs(workflow_id)
                for source_name in selected_sources:
                    self.stage_config(
                        workflow_id,
                        source_name=source_name,
                        params=params_dict,
                        aux_source_names=aux_dict,
                    )

            self.commit_workflow(workflow_id)

        return WorkflowConfigurationAdapter(spec, config_state, start_callback)

    def _get_config_state(self, workflow_id: WorkflowId) -> ConfigurationState | None:
        """Convert staged jobs to ConfigurationState for adapter initialization."""
        staged_jobs = self.get_staged_config(workflow_id)
        if not staged_jobs:
            return None

        return ConfigurationState(
            jobs={
                source_name: JobConfigState(
                    params=job_config.params,
                    aux_source_names=job_config.aux_source_names,
                )
                for source_name, job_config in staged_jobs.items()
            }
        )

    def get_previous_job_number(self, workflow_id: WorkflowId) -> JobNumber | None:
        """
        Get the job number of the previous job for a workflow.

        Parameters
        ----------
        workflow_id
            The workflow to query.

        Returns
        -------
        :
            The job number if there's a previous job, None otherwise.
        """
        state = self._workflows.get(workflow_id)
        if state is None or state.previous is None:
            return None
        return state.previous.job_number

    def stop_workflow(self, workflow_id: WorkflowId) -> bool:
        """
        Stop all jobs for a workflow.

        Sends stop commands to the backend and clears the local active job state.
        The workflow configuration remains staged for future restarts.

        Parameters
        ----------
        workflow_id
            The workflow to stop.

        Returns
        -------
        :
            True if jobs were stopped, False if no active jobs.
        """
        state = self._workflows[workflow_id]
        if state.current is None:
            self._logger.debug('No active jobs for workflow %s to stop', workflow_id)
            return False

        # Send stop commands to backend
        commands = [
            (
                ConfigKey(key=JobCommand.key, source_name=str(job_id)),
                JobCommand(job_id=job_id, action=JobAction.stop),
            )
            for job_id in state.current.job_ids()
        ]
        self._command_service.send_batch(commands)

        job_number = state.current.job_number
        self._logger.info(
            'Stopped workflow %s (job_number=%s, %d jobs)',
            workflow_id,
            job_number,
            len(state.current.jobs),
        )

        # Clear local state immediately (don't wait for backend confirmation)
        state.previous = state.current
        state.current = None

        # Persist updated state
        self._persist_state_to_store(workflow_id)

        # Notify widget lifecycle subscribers that workflow was stopped
        self._notify_workflow_stopped(workflow_id)

        return True

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
        for subscription_id in self._workflow_subscriptions.get(workflow_id, set()):
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
    ) -> tuple[SubscriptionId, bool]:
        """
        Subscribe to workflow job availability notifications.

        The callback will be called with the job_number when:
        1. A workflow is committed (immediately after commit)
        2. Immediately if subscribing and workflow already has an active job

        Parameters
        ----------
        workflow_id
            The workflow to subscribe to.
        callback
            Called with job_number when a job becomes active.

        Returns
        -------
        :
            Tuple of (subscription_id, callback_invoked_immediately).
            subscription_id can be used to unsubscribe.
            callback_invoked_immediately is True if the workflow was already
            running and the callback was invoked synchronously during this call.
        """
        subscription_id = SubscriptionId(uuid.uuid4())
        self._subscriptions[subscription_id] = callback
        callback_invoked = False

        # Track which workflows have subscriptions
        if workflow_id not in self._workflow_subscriptions:
            self._workflow_subscriptions[workflow_id] = set()
        self._workflow_subscriptions[workflow_id].add(subscription_id)

        # If workflow already has an active job, notify immediately
        if workflow_id in self._workflows:
            state = self._workflows[workflow_id]
            if state.current is not None:
                current_job_number = state.current.job_number
                self._logger.debug(
                    'Workflow %s already has active job (job_number=%s), '
                    'notifying new subscriber %s immediately',
                    workflow_id,
                    current_job_number,
                    subscription_id,
                )
                try:
                    callback(current_job_number)
                    callback_invoked = True
                except Exception:
                    self._logger.exception(
                        'Error in immediate callback for subscription %s',
                        subscription_id,
                    )

        return subscription_id, callback_invoked

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

    def subscribe_to_widget_lifecycle(
        self, callbacks: WidgetLifecycleCallbacks
    ) -> WidgetLifecycleSubscriptionId:
        """
        Subscribe to widget lifecycle events for cross-session synchronization.

        Widgets use this to stay synchronized across browser sessions. When any
        session triggers a state change (staging, commit, stop), all subscribed
        widgets are notified so they can rebuild.

        Parameters
        ----------
        callbacks
            Callbacks to invoke on lifecycle events. Any callback can be None
            if that event type is not needed.

        Returns
        -------
        :
            Subscription ID for unsubscribing later.
        """
        subscription_id = WidgetLifecycleSubscriptionId(uuid.uuid4())
        self._widget_subscriptions[subscription_id] = callbacks
        self._logger.debug('Added widget lifecycle subscription %s', subscription_id)
        return subscription_id

    def unsubscribe_from_widget_lifecycle(
        self, subscription_id: WidgetLifecycleSubscriptionId
    ) -> None:
        """
        Unsubscribe from widget lifecycle events.

        Parameters
        ----------
        subscription_id
            The subscription ID returned from subscribe_to_widget_lifecycle.
        """
        if subscription_id in self._widget_subscriptions:
            del self._widget_subscriptions[subscription_id]
            self._logger.debug(
                'Removed widget lifecycle subscription %s', subscription_id
            )
        else:
            self._logger.warning(
                'Attempted to unsubscribe from non-existent widget subscription %s',
                subscription_id,
            )

    def _notify_staged_changed(self, workflow_id: WorkflowId) -> None:
        """Notify all widget subscribers that staging area changed.

        Notifications are deferred if currently in a staging transaction.
        The transaction context manager will call this on exit.
        """
        if self._transaction_workflow is not None:
            return

        for subscription_id, callbacks in self._widget_subscriptions.items():
            if callbacks.on_staged_changed is not None:
                try:
                    callbacks.on_staged_changed(workflow_id)
                except Exception:
                    self._logger.exception(
                        'Error in on_staged_changed callback for subscription %s',
                        subscription_id,
                    )

    def _notify_workflow_committed(self, workflow_id: WorkflowId) -> None:
        """Notify all widget subscribers that a workflow was committed."""
        for subscription_id, callbacks in self._widget_subscriptions.items():
            if callbacks.on_workflow_committed is not None:
                try:
                    callbacks.on_workflow_committed(workflow_id)
                except Exception:
                    self._logger.exception(
                        'Error in on_workflow_committed callback for subscription %s',
                        subscription_id,
                    )

    def _notify_workflow_stopped(self, workflow_id: WorkflowId) -> None:
        """Notify all widget subscribers that a workflow was stopped."""
        for subscription_id, callbacks in self._widget_subscriptions.items():
            if callbacks.on_workflow_stopped is not None:
                try:
                    callbacks.on_workflow_stopped(workflow_id)
                except Exception:
                    self._logger.exception(
                        'Error in on_workflow_stopped callback for subscription %s',
                        subscription_id,
                    )
