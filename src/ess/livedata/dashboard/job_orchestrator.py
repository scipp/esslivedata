# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
JobOrchestrator - Manages workflow job lifecycle and state transitions.

Coordinates workflow execution across multiple sources, handling:
- Configuration staging and commit (two-phase workflow start)
- Job number generation and JobSet lifecycle
- Job transitions and cleanup
- Dynamic workflow registration from templates
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NewType
from uuid import UUID, uuid4

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
from ess.livedata.config.workflow_template import (
    JobExecutor,
    TemplateInstance,
    WorkflowTemplate,
)
from ess.livedata.core.job_manager import JobAction, JobCommand

from .command_service import CommandService
from .config_store import ConfigStore
from .workflow_config_service import WorkflowConfigService

if TYPE_CHECKING:
    pass

SourceName = str
SubscriptionId = NewType('SubscriptionId', UUID)

# Key used to store template instances in config store
_TEMPLATE_INSTANCES_KEY = '_template_instances'


class JobConfig(BaseModel):
    """Configuration for a single job within a JobSet."""

    params: dict
    aux_source_names: dict


class JobSet(BaseModel):
    """A set of jobs sharing the same job_number.

    Maps source_name to config for each running job.
    JobId can be reconstructed as JobId(source_name, job_number).
    """

    job_number: JobNumber = Field(default_factory=uuid4)
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


class BackendJobExecutor:
    """
    Default executor that sends workflow commands via CommandService.

    This executor is used for regular backend workflows that run in separate
    services and communicate via Kafka.
    """

    def __init__(
        self, command_service: CommandService, workflow_id: WorkflowId
    ) -> None:
        self._command_service = command_service
        self._workflow_id = workflow_id

    def start_jobs(
        self,
        staged_jobs: dict[SourceName, JobConfig],
        job_number: JobNumber,
    ) -> None:
        """Start jobs by sending workflow configs to backend services."""
        commands: list[tuple[ConfigKey, WorkflowConfig]] = []
        for source_name, job_config in staged_jobs.items():
            workflow_config = WorkflowConfig.from_params(
                workflow_id=self._workflow_id,
                params=job_config.params,
                aux_source_names=job_config.aux_source_names,
                job_number=job_number,
            )
            key = keys.WORKFLOW_CONFIG.create_key(source_name=source_name)
            commands.append((key, workflow_config))
        self._command_service.send_batch(commands)

    def stop_jobs(self, job_ids: list[JobId]) -> None:
        """Stop jobs by sending stop commands to backend services."""
        if not job_ids:
            return
        commands: list[tuple[ConfigKey, JobCommand]] = [
            (
                ConfigKey(key=JobCommand.key, source_name=str(job_id)),
                JobCommand(job_id=job_id, action=JobAction.stop),
            )
            for job_id in job_ids
        ]
        self._command_service.send_batch(commands)


# Verify BackendJobExecutor implements JobExecutor protocol
_: type[JobExecutor] = BackendJobExecutor  # type: ignore[type-abstract]


class WorkflowRegistryManager:
    """
    Manages a combined view of static and dynamically-created workflow specs.

    Static workflows come from the instrument's workflow factory. Dynamic workflows
    are created from templates (e.g., correlation histograms) and are persisted
    to the config store for restoration across sessions.
    """

    def __init__(
        self,
        static_registry: Mapping[WorkflowId, WorkflowSpec],
        config_store: ConfigStore | None,
        templates: Sequence[WorkflowTemplate],
        logger: logging.Logger,
    ) -> None:
        """
        Initialize the registry manager.

        Parameters
        ----------
        static_registry:
            Registry of static workflows from instrument configuration.
        config_store:
            Optional store for persisting template instances.
        templates:
            Available workflow templates for creating dynamic workflows.
        logger:
            Logger instance for logging.
        """
        self._static_registry = dict(static_registry)
        self._config_store = config_store
        self._templates = {t.name: t for t in templates}
        self._logger = logger

        # Dynamic workflows created from templates
        self._dynamic_registry: dict[WorkflowId, WorkflowSpec] = {}
        # Track which workflows came from which template instance
        self._template_instances: dict[WorkflowId, TemplateInstance] = {}

        # Load persisted template instances
        self._load_template_instances()

    def _load_template_instances(self) -> None:
        """Recreate dynamic specs from persisted template instances."""
        if self._config_store is None:
            return

        instances_data = self._config_store.get(_TEMPLATE_INSTANCES_KEY)
        if not instances_data:
            return

        for workflow_id_str, instance_data in instances_data.items():
            try:
                instance = TemplateInstance.model_validate(instance_data)
                template = self._templates.get(instance.template_name)
                if template is None:
                    self._logger.warning(
                        'Template %s not found for persisted instance %s, skipping',
                        instance.template_name,
                        workflow_id_str,
                    )
                    continue

                # Use raw config model - doesn't require timeseries to exist
                config_model = template.get_raw_configuration_model()
                config = config_model.model_validate(instance.config)
                spec = template.create_workflow_spec(config)
                workflow_id = spec.get_id()

                self._dynamic_registry[workflow_id] = spec
                self._template_instances[workflow_id] = instance
                self._logger.info(
                    'Restored template instance %s from %s',
                    workflow_id,
                    instance.template_name,
                )
            except Exception as e:
                self._logger.warning(
                    'Failed to restore template instance %s: %s',
                    workflow_id_str,
                    e,
                )

    def _persist_template_instances(self) -> None:
        """Persist template instances to config store."""
        if self._config_store is None:
            return

        instances_data = {
            str(wid): instance.model_dump(mode='json')
            for wid, instance in self._template_instances.items()
        }
        self._config_store[_TEMPLATE_INSTANCES_KEY] = instances_data

    def register_from_template(
        self, template_name: str, config: dict
    ) -> WorkflowId | None:
        """
        Create and register a workflow spec from a template.

        Template instantiation does not require timeseries data to exist. The
        config is validated with basic type checking (strings for axis names),
        not against available timeseries. Enum validation against available
        timeseries is done by UI widgets, not here.

        Parameters
        ----------
        template_name:
            Name of the template to use.
        config:
            Configuration dict for the template (e.g., {'x_param': 'temperature'}).

        Returns
        -------
        :
            The WorkflowId of the created workflow, or None if registration failed.
        """
        template = self._templates.get(template_name)
        if template is None:
            self._logger.error('Template %s not found', template_name)
            return None

        # Use raw config model for basic validation (no enum check against data)
        config_model = template.get_raw_configuration_model()

        try:
            validated_config = config_model.model_validate(config)
        except pydantic.ValidationError as e:
            self._logger.error('Invalid config for template %s: %s', template_name, e)
            return None

        spec = template.create_workflow_spec(validated_config)
        workflow_id = spec.get_id()

        # Check if already registered
        if (
            workflow_id in self._static_registry
            or workflow_id in self._dynamic_registry
        ):
            self._logger.warning(
                'Workflow %s already registered, not overwriting', workflow_id
            )
            return workflow_id

        self._dynamic_registry[workflow_id] = spec
        self._template_instances[workflow_id] = TemplateInstance(
            template_name=template_name,
            config=config,
        )
        self._persist_template_instances()

        self._logger.info(
            'Registered workflow %s from template %s', workflow_id, template_name
        )
        return workflow_id

    def unregister(self, workflow_id: WorkflowId) -> bool:
        """
        Remove a template-created workflow.

        Parameters
        ----------
        workflow_id:
            The workflow to remove.

        Returns
        -------
        :
            True if the workflow was removed, False if it was a static workflow
            or not found.
        """
        if workflow_id in self._static_registry:
            self._logger.warning('Cannot unregister static workflow %s', workflow_id)
            return False

        if workflow_id not in self._dynamic_registry:
            self._logger.warning(
                'Workflow %s not found in dynamic registry', workflow_id
            )
            return False

        del self._dynamic_registry[workflow_id]
        del self._template_instances[workflow_id]
        self._persist_template_instances()

        self._logger.info('Unregistered dynamic workflow %s', workflow_id)
        return True

    def get_registry(self) -> Mapping[WorkflowId, WorkflowSpec]:
        """
        Get combined view of static + dynamic workflows.

        Returns
        -------
        :
            Combined registry of all workflows.
        """
        return {**self._static_registry, **self._dynamic_registry}

    def is_template_instance(self, workflow_id: WorkflowId) -> bool:
        """
        Check if a workflow was created from a template.

        Parameters
        ----------
        workflow_id:
            The workflow to check.

        Returns
        -------
        :
            True if the workflow was created from a template.
        """
        return workflow_id in self._template_instances

    def get_templates(self) -> Mapping[str, WorkflowTemplate]:
        """
        Get available templates.

        Returns
        -------
        :
            Mapping from template name to template instance.
        """
        return dict(self._templates)

    def get_template_instance(self, workflow_id: WorkflowId) -> TemplateInstance | None:
        """
        Get the template instance info for a dynamically created workflow.

        Parameters
        ----------
        workflow_id:
            The workflow to look up.

        Returns
        -------
        :
            TemplateInstance if the workflow was created from a template,
            None otherwise.
        """
        return self._template_instances.get(workflow_id)


class JobOrchestrator:
    """Orchestrates workflow job lifecycle and state management."""

    def __init__(
        self,
        *,
        command_service: CommandService,
        workflow_config_service: WorkflowConfigService,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        config_store: ConfigStore | None = None,
        templates: Sequence[WorkflowTemplate] | None = None,
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
        templates
            Optional sequence of workflow templates for dynamic workflow creation.
        """
        self._command_service = command_service
        self._workflow_config_service = workflow_config_service
        self._config_store = config_store
        self._logger = logging.getLogger(__name__)

        # Initialize registry manager for combined static + dynamic workflows
        self._registry_manager = WorkflowRegistryManager(
            static_registry=workflow_registry,
            config_store=config_store,
            templates=templates or [],
            logger=self._logger,
        )

        # Workflow state tracking
        self._workflows: dict[WorkflowId, WorkflowState] = {}

        # Workflow subscription tracking
        self._subscriptions: dict[SubscriptionId, Callable[[JobNumber], None]] = {}
        self._workflow_subscriptions: dict[WorkflowId, set[SubscriptionId]] = {}

        # Load persisted configs
        self._load_configs_from_store()

    @property
    def workflow_registry(self) -> Mapping[WorkflowId, WorkflowSpec]:
        """Get the combined workflow registry (static + dynamic)."""
        return self._registry_manager.get_registry()

    def _load_configs_from_store(self) -> None:
        """Initialize all workflows with either loaded configs or defaults from spec."""
        for workflow_id, spec in self.workflow_registry.items():
            # Try to load persisted config directly as dict
            config_data = None
            if self._config_store is not None:
                config_data = self._config_store.get(str(workflow_id))

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
            state = WorkflowState()
            if params:
                for source_name in source_names:
                    state.staged_jobs[source_name] = JobConfig(
                        params=params.copy(),
                        aux_source_names=aux_source_names.copy(),
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

        # Get executor for this workflow type
        executor = self._get_job_executor(workflow_id)

        # Stop old jobs if any
        if state.current is not None:
            self._logger.info(
                'Workflow %s already has active jobs, stopping: %s',
                workflow_id,
                state.current.job_number,
            )
            executor.stop_jobs(state.current.job_ids())
            self._logger.debug('Stopped %d old jobs', len(state.current.jobs))
            # Move current to previous for cleanup once stop commands succeed.
            # Related to #445: Future improvements may wait for stop command
            # success responses before removing old job data.
            state.previous = state.current

        # Start new jobs
        executor.start_jobs(state.staged_jobs, job_set.job_number)

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

        # Notify subscribers immediately that job is active
        self._logger.info(
            'Workflow %s committed with job_number=%s, notifying subscribers',
            workflow_id,
            job_set.job_number,
        )
        self._notify_workflow_available(workflow_id, job_set.job_number)

        # Return JobIds for all created jobs
        return job_set.job_ids()

    def _get_job_executor(self, workflow_id: WorkflowId) -> JobExecutor:
        """
        Get the job executor for a workflow.

        Template-created workflows may provide custom executors for frontend
        execution. Regular backend workflows use BackendJobExecutor.

        Parameters
        ----------
        workflow_id
            The workflow to get an executor for.

        Returns
        -------
        :
            JobExecutor for this workflow type.
        """
        template_instance = self._registry_manager.get_template_instance(workflow_id)
        if template_instance is not None:
            template = self._registry_manager.get_templates().get(
                template_instance.template_name
            )
            if template is not None:
                # Use raw config model - stored config has raw values,
                # not enum display names
                config_model = template.get_raw_configuration_model()
                config = config_model.model_validate(template_instance.config)
                executor = template.create_job_executor(config)
                if executor is not None:
                    return executor

        # Default: backend execution via CommandService
        return BackendJobExecutor(self._command_service, workflow_id)

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
        config_dict = {}
        if state.staged_jobs:
            source_names = list(state.staged_jobs.keys())
            first_job_config = next(iter(state.staged_jobs.values()))
            config_dict = {
                'source_names': source_names,
                'params': first_job_config.params,
                'aux_source_names': first_job_config.aux_source_names,
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
        subscription_id = SubscriptionId(uuid4())
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

    # =========================================================================
    # Template-related methods
    # =========================================================================

    def get_templates(self) -> Mapping[str, WorkflowTemplate]:
        """
        Get available workflow templates.

        Returns
        -------
        :
            Mapping from template name to template instance.
        """
        return self._registry_manager.get_templates()

    def register_from_template(
        self, template_name: str, config: dict
    ) -> WorkflowId | None:
        """
        Create and register a workflow spec from a template.

        After registration, the workflow state is initialized and the workflow
        becomes available for job orchestration.

        Parameters
        ----------
        template_name:
            Name of the template to use.
        config:
            Configuration dict for the template.

        Returns
        -------
        :
            The WorkflowId of the created workflow, or None if registration failed.
        """
        workflow_id = self._registry_manager.register_from_template(
            template_name, config
        )
        if workflow_id is None:
            return None

        # Initialize workflow state for the new workflow
        spec = self.workflow_registry.get(workflow_id)
        if spec is not None:
            self._init_workflow_state(workflow_id, spec)

        return workflow_id

    def _init_workflow_state(self, workflow_id: WorkflowId, spec: WorkflowSpec) -> None:
        """Initialize workflow state for a single workflow."""
        params = {}
        if spec.params is not None:
            try:
                params = spec.params().model_dump(mode='json')
            except pydantic.ValidationError:
                self._workflows[workflow_id] = WorkflowState()
                self._logger.debug(
                    'Initialized workflow %s (params cannot be instantiated)',
                    workflow_id,
                )
                return

        aux_source_names = {}
        if spec.aux_sources is not None:
            aux_source_names = spec.aux_sources().model_dump(mode='json')

        state = WorkflowState()
        if params:
            for source_name in spec.source_names:
                state.staged_jobs[source_name] = JobConfig(
                    params=params.copy(),
                    aux_source_names=aux_source_names.copy(),
                )

        self._workflows[workflow_id] = state
        self._logger.debug(
            'Initialized workflow %s with %d sources',
            workflow_id,
            len(spec.source_names),
        )

    def unregister_workflow(self, workflow_id: WorkflowId) -> bool:
        """
        Remove a template-created workflow.

        This will also clean up any workflow state and subscriptions.
        Static workflows cannot be unregistered.

        Parameters
        ----------
        workflow_id:
            The workflow to remove.

        Returns
        -------
        :
            True if the workflow was removed, False if it was a static workflow
            or not found.
        """
        if not self._registry_manager.unregister(workflow_id):
            return False

        # Clean up workflow state
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]

        # Clean up subscriptions for this workflow
        if workflow_id in self._workflow_subscriptions:
            for sub_id in list(self._workflow_subscriptions[workflow_id]):
                if sub_id in self._subscriptions:
                    del self._subscriptions[sub_id]
            del self._workflow_subscriptions[workflow_id]

        return True

    def is_template_instance(self, workflow_id: WorkflowId) -> bool:
        """
        Check if a workflow was created from a template.

        Parameters
        ----------
        workflow_id:
            The workflow to check.

        Returns
        -------
        :
            True if the workflow was created from a template.
        """
        return self._registry_manager.is_template_instance(workflow_id)

    def get_template_for_workflow(
        self, workflow_id: WorkflowId
    ) -> WorkflowTemplate | None:
        """
        Get the template that created a workflow.

        Parameters
        ----------
        workflow_id:
            The workflow to look up.

        Returns
        -------
        :
            The WorkflowTemplate if the workflow was created from a template,
            None otherwise.
        """
        template_instance = self._registry_manager.get_template_instance(workflow_id)
        if template_instance is None:
            return None
        templates = self._registry_manager.get_templates()
        return templates.get(template_instance.template_name)
