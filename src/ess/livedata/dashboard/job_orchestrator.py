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

import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import TYPE_CHECKING

import pydantic
import structlog
from pydantic import BaseModel, Field

from ess.livedata.config.workflow_spec import (
    JobId,
    JobNumber,
    WorkflowConfig,
    WorkflowId,
    WorkflowSpec,
)
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.core.timestamp import Timestamp

from .active_job_registry import ActiveJobRegistry
from .command_service import CommandService
from .config_store import ConfigStore
from .configuration_adapter import ConfigurationState
from .job_service import JobService
from .notification_queue import NotificationEvent, NotificationQueue, NotificationType
from .pending_command_tracker import PendingCommandTracker
from .workflow_configuration_adapter import WorkflowConfigurationAdapter

if TYPE_CHECKING:
    from ess.livedata.config import Instrument

logger = structlog.get_logger(__name__)

SourceName = str

# A command acknowledgement is a fast round-trip, so a pending command that has
# not been acknowledged within this window is treated as lost (e.g. its send was
# silently dropped by the sink). Deliberately shorter than the 60s
# heartbeat-staleness constants in job_service.py, which answer a different
# question about backend liveness.
PENDING_COMMAND_TIMEOUT_SECONDS = 30.0

# While a job's observed heartbeats contradict the desired state (stopped or
# superseded), reconciliation re-issues the stop command, rate-bounded per
# job_number so the backend gets time to act on the previous stop — including
# the normal case where the just-sent stop of a commit or stop_workflow has
# simply not been processed yet.
STOP_REISSUE_INTERVAL_SECONDS = 30.0


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


def _starts_later(candidate: Timestamp | None, incumbent: Timestamp | None) -> bool:
    """Whether a candidate start time beats the incumbent (None is earliest)."""
    if candidate is None:
        return False
    return incumbent is None or candidate > incumbent


class StoppedReason(StrEnum):
    """Why a workflow was stopped."""

    user = auto()
    backend_shutdown = auto()


@dataclass
class WorkflowState:
    """State for an active workflow, including transitions.

    ``current`` is the desired run-state: the JobSet the user committed (or
    that was adopted from observation). Whether the backend actually runs it
    is observed via heartbeats; the two are compared by reconciliation.

    All run-state changes go through the transition methods below, which keep
    ``current``, ``stopped_reason``, the adoption fields, and ``version``
    consistent. Side effects (commands, generation flips, persistence) are the
    orchestrator's responsibility and must not migrate here.
    """

    current: JobSet | None = None
    staged_jobs: dict[SourceName, JobConfig] = field(default_factory=dict)
    version: int = 0
    stopped_reason: StoppedReason | None = None
    # Set when ``current`` was adopted from heartbeat observation without a
    # persisted record: params are unknown (degraded provenance), and a later
    # observed start_time may replace the adopted job. Cleared on commit/stop.
    adopted_without_record: bool = False
    adopted_start_time: Timestamp | None = None

    def commit(self, job_set: JobSet) -> None:
        """Set a committed JobSet as the desired run-state."""
        self.current = job_set
        self.stopped_reason = None
        self.adopted_without_record = False
        self.adopted_start_time = None
        self.version += 1

    def deactivate(self, reason: StoppedReason) -> None:
        """Clear the desired run-state, recording why it stopped."""
        self.current = None
        self.stopped_reason = reason
        self.adopted_without_record = False
        self.adopted_start_time = None
        self.version += 1

    def adopt(
        self,
        job_number: JobNumber,
        start_time: Timestamp | None,
        source_name: SourceName,
    ) -> None:
        """Take an observed record-less job as the desired run-state."""
        self.current = JobSet(
            job_number=job_number,
            jobs={source_name: JobConfig(params={}, aux_source_names={})},
        )
        self.adopted_without_record = True
        self.adopted_start_time = start_time
        self.stopped_reason = None
        self.version += 1

    def extend_adopted(self, source_name: SourceName) -> None:
        """Add a newly observed source to the adopted JobSet."""
        if self.current is None:
            msg = "Cannot extend adopted job set: no current job set"
            raise RuntimeError(msg)
        if not self.adopted_without_record:
            msg = "Cannot extend adopted job set: not in adopted_without_record state"
            raise RuntimeError(msg)
        self.current.jobs[source_name] = JobConfig(params={}, aux_source_names={})
        self.version += 1


class JobOrchestrator:
    """Orchestrates workflow job lifecycle and state management."""

    def __init__(
        self,
        *,
        command_service: CommandService,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        active_job_registry: ActiveJobRegistry,
        job_service: JobService,
        config_store: ConfigStore | None = None,
        instrument_config: Instrument | None = None,
        notification_queue: NotificationQueue | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """
        Initialize the job orchestrator.

        Parameters
        ----------
        command_service
            Service for sending workflow commands to backend services.
        workflow_registry
            Registry of available workflows and their specifications.
        active_job_registry
            Thread-safe registry for tracking active job numbers and
            cleaning up data on deactivation.
        job_service
            Holder of observed job statuses (heartbeats). The orchestrator
            subscribes to its updates for adoption and reads it during
            reconciliation.
        config_store
            Optional store for persisting workflow configurations across sessions.
            Orchestrator loads configs on init and persists on commit.
        instrument_config
            Optional instrument configuration for source metadata lookup.
        notification_queue
            Optional queue for multi-session notifications. When provided,
            command success/error notifications are pushed here instead of
            using direct callbacks.
        clock
            Callable returning the current time in seconds, used for command
            expiry. Injectable to allow deterministic testing without real waits.
        """
        self._command_service = command_service
        self._workflow_registry = workflow_registry
        self._active_job_registry = active_job_registry
        self._job_service = job_service
        self._config_store = config_store
        self._instrument_config = instrument_config
        self._notification_queue = notification_queue
        self._clock = clock

        # Command acknowledgement tracking
        self._pending_commands = PendingCommandTracker(clock=clock)

        # Workflow state tracking
        self._workflows: dict[WorkflowId, WorkflowState] = {}
        self._job_states: dict[JobId, JobState] = {}
        # When the last stop was issued per job_number, bounding re-issues
        # during reconciliation.
        self._stop_issued_at: dict[JobNumber, float] = {}

        # Transaction state for batching staging operations
        self._transaction_workflow: WorkflowId | None = None
        self._transaction_depth: int = 0

        # Global, in-memory toggle for the disruptive-action confirmation gate.
        # Shared across all sessions; versioned so each session's checkbox
        # resyncs when another session flips it.
        self._gate_enabled: bool = True
        self._gate_version: int = 0

        # Load persisted configs
        self._load_configs_from_store()

        job_service.on_status_updated = self.on_job_status_updated

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
                logger.info(
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
                        logger.debug(
                            'Initialized workflow %s (params cannot be instantiated)',
                            workflow_id,
                        )
                        continue

                aux_source_names = {}
                if spec.aux_sources is not None:
                    aux_source_names = spec.aux_sources.get_defaults()

                if params:
                    for source_name in spec.source_names:
                        state.staged_jobs[source_name] = JobConfig(
                            params=params.copy(),
                            aux_source_names=aux_source_names.copy(),
                        )
                logger.debug(
                    'Initialized workflow %s with defaults: %d sources',
                    workflow_id,
                    len(spec.source_names),
                )

            # The persisted current_job is the generation→config record: it
            # restores the desired run-state (the workflow shows as pending
            # until its job is observed) and labels the observed job with the
            # committed params. It does not touch ActiveJobRegistry — only a
            # heartbeat observation establishes the generation that admits
            # data (ADR 0008).
            if config_data:
                if current_data := config_data.get('current_job'):
                    try:
                        state.current = JobSet.model_validate(current_data)
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(
                            'Failed to restore active job for workflow %s: %s',
                            workflow_id,
                            e,
                        )

            self._workflows[workflow_id] = state

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

        # Generate message_id for command acknowledgement tracking
        message_id = str(uuid.uuid4())

        # Prepare all commands (stop old jobs + start new workflow) in single batch
        commands = []

        # Stop the desired predecessor and any observed job of this workflow
        # still heartbeating (e.g. an orphan whose stop was lost): the new
        # commit supersedes them all.
        old_job_ids = dict.fromkeys(
            (state.current.job_ids() if state.current is not None else [])
            + self._job_service.fresh_running_job_ids(workflow_id)
        )
        if old_job_ids:
            logger.info(
                'Workflow %s has %d old job(s), stopping before start',
                workflow_id,
                len(old_job_ids),
            )
            # Fire-and-forget: each stop gets an auto-generated message_id the
            # dashboard does not register, so the owning worker's ack is dropped.
            # Reusing the start's message_id would inflate its tracked ack count.
            commands.extend(
                JobCommand(job_id=job_id, action=JobAction.stop)
                for job_id in old_job_ids
            )
            now = self._clock()
            for job_id in old_job_ids:
                self._stop_issued_at[job_id.job_number] = now

        # Send workflow configs to all staged sources
        # Note: Currently all jobs use same params, but aux_source_names may
        # differ
        for source_name, job_config in state.staged_jobs.items():
            # Currently WorkflowConfig conflates "configure" and "start"
            # into one command. Both message_id (for ACK) and job_number
            # (for job identity) are sent together. See #445 for plans to
            # split into separate config and start messages. In that design,
            # config's message_id would serve as a "config handle" that
            # start references, enabling multiple jobs to share the same
            # configuration.
            workflow_config = WorkflowConfig.from_params(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_set.job_number),
                params=job_config.params,
                aux_source_names=job_config.aux_source_names,
                message_id=message_id,
            )
            commands.append(workflow_config)

        # Register pending command for acknowledgement tracking
        self._pending_commands.register(
            message_id, workflow_id, "start", expected_count=len(state.staged_jobs)
        )

        # Drop the superseded generation's tracked states. The old jobs have
        # been stopped above; their final heartbeats are no longer in the
        # current set, so on_job_status_updated would ignore them and never
        # clean these up (the same cleanup _deactivate_workflow performs).
        if state.current is not None:
            for job_id in state.current.job_ids():
                self._job_states.pop(job_id, None)

        state.commit(job_set)

        # Flip the generation, clear the workflow's buffers, and notify
        # subscribers atomically under the ingestion guard, *before* sending
        # the commands: new-generation data cannot exist on the wire before
        # the flip, and when MessagePump.update() next runs, the generation
        # filter and the DataService subscribers are consistent. The clear
        # gives the new generation a blank slate — a windowed extractor can
        # never aggregate across a parameter change — and applies equally to
        # a byte-identical recommit (uniform semantics, matching the fresh
        # keys that per-commit job_numbers used to provide).
        with self._active_job_registry.ingestion_guard():
            self._active_job_registry.begin_generation(
                workflow_id, job_set.job_number, config=job_set.jobs
            )

        # A send failure swallowed by the sink (#1037 finding 7) leaves the
        # generation flipped and buffers cleared with no job to fill them:
        # a blank plot until pending-command expiry surfaces it. This is
        # behavior parity with the pre-stable-keys ordering, which also
        # deactivated the old job before sending.
        self._command_service.send_batch(commands)

        logger.info(
            'Started workflow %s with job_number %s on sources %s',
            workflow_id,
            job_set.job_number,
            list(state.staged_jobs.keys()),
        )

        # Persist full state (staged configs + active jobs) to store
        self._persist_state_to_store(workflow_id)

        # Return JobIds for all created jobs
        return job_set.job_ids()

    @property
    def active_job_registry(self) -> ActiveJobRegistry:
        """Registry shared with MessagePump for thread-safe data flow."""
        return self._active_job_registry

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

        # The generation→config record. Not written for a job adopted without
        # a record — its params are unknown and the record must not lie; after
        # a restart such a job is simply re-adopted from observation.
        if state.current is not None and not state.adopted_without_record:
            config_dict['current_job'] = state.current.model_dump(mode='json')

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

    def get_running_workflow_sources(self) -> dict[WorkflowId, set[SourceName]]:
        """Return the source names of every currently-running job, by workflow.

        A workflow's job is "running" when it has a committed (current) JobSet.
        The returned mapping omits workflows with no active job. It is the
        live projection the dashboard intersects with the device contract to
        decide which outputs are currently exposed as NICOS devices.

        Returns
        -------
        :
            Mapping from workflow ID to the set of source names with a running
            job. Workflows without an active job are absent from the mapping.
        """
        return {
            workflow_id: set(state.current.jobs)
            for workflow_id, state in self._workflows.items()
            if state.current is not None
        }

    def get_stopped_reason(self, workflow_id: WorkflowId) -> StoppedReason | None:
        """Get the reason why a workflow was stopped, if any."""
        state = self._workflows.get(workflow_id)
        if state is None:
            return None
        return state.stopped_reason

    def get_workflow_registry(self) -> Mapping[WorkflowId, WorkflowSpec]:
        """
        Get the workflow registry containing all managed workflows.

        Returns
        -------
        :
            Mapping from workflow ID to workflow spec.
        """
        return self._workflow_registry

    def get_source_title(self, source_name: str) -> str:
        """Get display title for a source name.

        Falls back to the source name if no instrument config is available
        or no title is defined for the source.
        """
        if self._instrument_config is not None:
            return self._instrument_config.get_source_title(source_name)
        return source_name

    def create_workflow_adapter(
        self,
        workflow_id: WorkflowId,
        selected_sources: list[str] | None = None,
        *,
        commit: bool = True,
    ) -> WorkflowConfigurationAdapter:
        """
        Create a workflow configuration adapter for the given workflow ID.

        The adapter provides the interface for configuration widgets to display
        workflow parameters and start the workflow.

        Parameters
        ----------
        workflow_id
            The workflow to create an adapter for.
        selected_sources
            Optional list of source names to scope the adapter to. If provided,
            the adapter will use the first selected source's config as reference
            and pre-select these sources in the UI.
        commit
            If True (default), the adapter's start_action will stage configs AND
            commit the workflow. If False, it will only stage configs without
            committing, allowing the user to review staged changes before committing.

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

        # Get reference config from first selected source (or first staged)
        config_state = self._get_reference_config(workflow_id, selected_sources)

        # Determine initial source names: explicit selection > staged sources > all
        initial_source_names = selected_sources
        if initial_source_names is None:
            staged_jobs = self.get_staged_config(workflow_id)
            if staged_jobs:
                initial_source_names = list(staged_jobs.keys())

        def start_callback(
            selected_sources: list[str],
            parameter_values,
            aux_source_names=None,
        ) -> None:
            """Stage configs for selected sources and commit the workflow.

            Only updates the selected sources, preserving configs for other
            sources that may have been configured independently.
            """
            params_dict = parameter_values.model_dump(mode='json')
            aux_dict = aux_source_names or {}

            # Update only the selected sources, preserving other staged configs
            for source_name in selected_sources:
                self.stage_config(
                    workflow_id,
                    source_name=source_name,
                    params=params_dict,
                    aux_source_names=aux_dict,
                )

            if commit:
                self.commit_workflow(workflow_id)

        return WorkflowConfigurationAdapter(
            spec,
            config_state,
            start_callback,
            initial_source_names=initial_source_names,
            instrument_config=self._instrument_config,
        )

    def _get_reference_config(
        self,
        workflow_id: WorkflowId,
        selected_sources: list[str] | None = None,
    ) -> ConfigurationState | None:
        """Get reference config for adapter initialization.

        Parameters
        ----------
        workflow_id
            The workflow to get config for.
        selected_sources
            Optional list of source names. If provided, returns config from
            the first selected source that has a staged config.

        Returns
        -------
        :
            ConfigurationState from the reference source, or None if no staged configs.
        """
        staged_jobs = self.get_staged_config(workflow_id)
        if not staged_jobs:
            return None

        # Pick reference: first selected source with config, or first staged
        if selected_sources:
            for source in selected_sources:
                if source in staged_jobs:
                    job = staged_jobs[source]
                    return ConfigurationState(
                        params=job.params,
                        aux_source_names=job.aux_source_names,
                    )

        # Fallback to first staged
        first_job = next(iter(staged_jobs.values()))
        return ConfigurationState(
            params=first_job.params,
            aux_source_names=first_job.aux_source_names,
        )

    def stop_workflow(self, workflow_id: WorkflowId) -> bool:
        """
        Stop all jobs for a workflow.

        Sends stop commands to the backend and clears the local active job state.
        Covers both the desired JobSet and any observed job of the workflow
        still heartbeating outside it. The workflow configuration remains
        staged for future restarts. If a stop is lost, reconciliation
        re-issues it while heartbeats keep arriving.

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
        desired = state.current
        observed_extra = [
            job_id
            for job_id in self._job_service.fresh_running_job_ids(workflow_id)
            if desired is None or job_id.job_number != desired.job_number
        ]
        if desired is None and not observed_extra:
            logger.debug('No active jobs for workflow %s to stop', workflow_id)
            return False

        if desired is not None:
            self._send_job_commands(workflow_id, JobAction.stop)
            self._stop_issued_at[desired.job_number] = self._clock()
        if observed_extra:
            # Fire-and-forget, like reconciliation's re-issued stops: these
            # jobs are outside the desired set, so there is no per-source ack
            # count to track for user feedback.
            logger.info(
                'Stopping %d observed job(s) of workflow %s outside the desired set',
                len(observed_extra),
                workflow_id,
            )
            self._command_service.send_batch(
                [
                    JobCommand(job_id=job_id, action=JobAction.stop)
                    for job_id in observed_extra
                ]
            )
            now = self._clock()
            for job_id in observed_extra:
                self._stop_issued_at[job_id.job_number] = now

        # Remove from active set immediately, before the backend has processed
        # the stop command. This means any final results the backend publishes
        # between now and actually stopping will be discarded by the ingest
        # filter in MessagePump.forward(). This is intentional: the user has
        # requested a stop, so late-arriving data would be confusing.
        self._deactivate_workflow(workflow_id, StoppedReason.user)
        return True

    def _deactivate_workflow(
        self, workflow_id: WorkflowId, reason: StoppedReason
    ) -> None:
        """Clear active job state, persist, and notify subscribers.

        The just-stopped job's buffered data stays under the workflow's
        stable keys — nothing is evicted on stop — so plots created while
        the workflow is stopped still display its last results; the next
        commit's generation flip clears them.
        """
        state = self._workflows[workflow_id]

        if state.current is not None:
            for job_id in state.current.job_ids():
                self._job_states.pop(job_id, None)
            self._active_job_registry.deactivate(workflow_id)
        state.deactivate(reason)

        self._persist_state_to_store(workflow_id)

    def on_job_status_updated(self, job_status: JobStatus) -> None:
        """React to a job status update from ``JobService``.

        Two responsibilities:

        - Adoption (ADR 0008): a non-stopped status for a known workflow is
          the observation that a job runs as that workflow's generation.
        - Backend shutdown: a worker's final heartbeat marks its jobs
          stopped; once all jobs of the desired set reported stopped, the
          workflow is deactivated.
        """
        workflow_id = job_status.workflow_id
        state = self._workflows.get(workflow_id)
        if state is None:
            return
        if job_status.state != JobState.stopped:
            self._adopt_observed_job(workflow_id, state, job_status)
        if state.current is None:
            return
        job_ids = list(state.current.job_ids())
        if job_status.job_id not in job_ids:
            return
        self._job_states[job_status.job_id] = job_status.state
        if job_status.state != JobState.stopped:
            return
        if not job_ids:
            return
        if all(self._job_states.get(jid) == JobState.stopped for jid in job_ids):
            logger.info(
                'Backend reported all jobs stopped, deactivating workflow %s',
                workflow_id,
            )
            self._deactivate_workflow(workflow_id, StoppedReason.backend_shutdown)

    def _adopt_observed_job(
        self, workflow_id: WorkflowId, state: WorkflowState, job_status: JobStatus
    ) -> None:
        """Derive the workflow's current generation from an observed running job.

        Cases (ADR 0008):

        - The observed job matches the desired JobSet but the registry has no
          generation for it (dashboard restart): re-attach, labelled by the
          persisted record.
        - The observed job is entirely unknown (record miss: crash between
          send and persist, store loss): adopt as running-with-unknown-config;
          among several record-less jobs the latest start time wins.
        - The observed job is a current/last generation or is superseded by a
          committed JobSet: nothing to adopt — reconciliation re-issues its
          stop if it keeps running.
        """
        job_number = job_status.job_id.job_number
        source_name = job_status.job_id.source_name
        if state.current is not None and state.current.job_number == job_number:
            if not self._active_job_registry.is_current(workflow_id, job_number):
                with self._active_job_registry.ingestion_guard():
                    self._active_job_registry.begin_generation(
                        workflow_id, job_number, config=state.current.jobs
                    )
                logger.info(
                    'Adopted running job from heartbeat observation: '
                    'workflow %s, job_number %s',
                    workflow_id,
                    job_number,
                )
            elif state.adopted_without_record and source_name not in state.current.jobs:
                # Another source of the adopted job surfaces: extend the
                # JobSet so stop/status cover it. Params stay unknown.
                state.extend_adopted(source_name)
            return
        if self._active_job_registry.is_known_generation(workflow_id, job_number):
            # A just-replaced or just-stopped generation reporting late
            # states; if it keeps running, reconciliation stops it.
            return
        if state.current is not None and not state.adopted_without_record:
            # Superseded by a committed JobSet; reconciliation stops it.
            return
        if state.current is not None and not _starts_later(
            job_status.start_time, state.adopted_start_time
        ):
            # Among record-less jobs the latest start time wins; this one
            # loses and reconciliation stops it.
            return
        state.adopt(job_number, job_status.start_time, source_name)
        with self._active_job_registry.ingestion_guard():
            self._active_job_registry.begin_generation(
                workflow_id, job_number, config=None
            )
        logger.warning(
            'Adopted running job with unknown config: workflow %s, job_number %s',
            workflow_id,
            job_number,
        )

    def is_known_workflow(self, workflow_id: WorkflowId) -> bool:
        """Whether this workflow is managed here (heartbeat admit filter)."""
        return workflow_id in self._workflows

    def reconcile_observed_jobs(self) -> None:
        """Re-issue stops while observed run-state contradicts desired state.

        Desired "stopped" (or superseded) with fresh non-stopped heartbeats
        re-issues the stop, rate-bounded per job_number and logged: lost
        sends, lost ACKs, and orphans all collapse into this one recovery
        path (ADR 0008).

        Deliberately asymmetric: starts are never re-issued. A stop re-issue
        acts on positive evidence (heartbeats prove the job exists and the
        stop has not taken effect) and a redundant stop is free. A start
        re-issue would act on absence of evidence — "desired running, no
        heartbeats" conflates a lost send (re-issue correct), a slow or down
        backend (the original command sits durably in the commands topic;
        re-issue duplicates), delayed heartbeats from a healthy job (re-issue
        destructive: ``JobManager.schedule_job`` overwrites the running job,
        wiping its accumulated data), and a crashed job (re-issue is silent
        auto-restart — supervision policy, and a crash loop if the params
        crash the job deterministically). That divergence instead surfaces
        through the PENDING display and pending-command expiry; recommit is
        the explicit, safe re-issue. If lost starts turn out to be common in
        the field, the fix is an idempotent backend start (no-op when the
        job_id is already active), after which reconciliation could converge
        in both directions.
        """
        self._job_service.prune_stale()
        now = self._clock()
        to_stop: list[JobId] = []
        fresh_numbers: set[JobNumber] = set()
        for job_id, status in self._job_service.job_statuses.items():
            if status.state == JobState.stopped:
                continue
            if self._job_service.is_status_stale(job_id):
                continue
            fresh_numbers.add(job_id.job_number)
            state = self._workflows.get(status.workflow_id)
            if state is None:
                continue
            if (
                state.current is not None
                and state.current.job_number == job_id.job_number
            ):
                continue
            issued = self._stop_issued_at.get(job_id.job_number)
            if issued is not None and now - issued < STOP_REISSUE_INTERVAL_SECONDS:
                continue
            to_stop.append(job_id)
        # Drop ledger entries no longer backed by fresh observations so the
        # dict does not grow with every job_number ever stopped.
        self._stop_issued_at = {
            number: t
            for number, t in self._stop_issued_at.items()
            if number in fresh_numbers or now - t < STOP_REISSUE_INTERVAL_SECONDS
        }
        if not to_stop:
            return
        for job_id in to_stop:
            self._stop_issued_at[job_id.job_number] = now
        logger.warning(
            'Re-issuing stop for %d job(s) observed running against desired state: %s',
            len(to_stop),
            [str(job_id) for job_id in to_stop],
        )
        self._command_service.send_batch(
            [JobCommand(job_id=job_id, action=JobAction.stop) for job_id in to_stop]
        )

    def reset_workflow(self, workflow_id: WorkflowId) -> bool:
        """
        Reset all jobs for a workflow.

        Sends reset commands to the backend to clear accumulated data without
        stopping the job. The job continues running but starts fresh.

        Parameters
        ----------
        workflow_id
            The workflow to reset.

        Returns
        -------
        :
            True if reset commands were sent, False if no active jobs.
        """
        return self._send_job_commands(workflow_id, JobAction.reset)

    def _send_job_commands(self, workflow_id: WorkflowId, action: JobAction) -> bool:
        """Send a command to all jobs for a workflow.

        Parameters
        ----------
        workflow_id
            The workflow to send commands for.
        action
            The action to perform on all jobs.

        Returns
        -------
        :
            True if commands were sent, False if no active jobs.
        """
        state = self._workflows[workflow_id]
        if state.current is None:
            logger.debug(
                'No active jobs for workflow %s to %s', workflow_id, action.value
            )
            return False

        message_id = str(uuid.uuid4())

        commands = [
            JobCommand(job_id=job_id, action=action, message_id=message_id)
            for job_id in state.current.job_ids()
        ]

        self._pending_commands.register(
            message_id,
            workflow_id,
            action.value,
            expected_count=len(state.current.jobs),
        )

        self._command_service.send_batch(commands)

        logger.info(
            '%s workflow %s (job_number=%s, %d jobs)',
            action.value.capitalize(),
            workflow_id,
            state.current.job_number,
            len(state.current.jobs),
        )

        return True

    def get_workflow_state_version(self, workflow_id: WorkflowId) -> int:
        """
        Get the version counter for a workflow's state.

        The version is incremented on every state mutation (staging changes,
        commits, stops). Widgets can compare this against a cached version
        to detect when a rebuild is needed.

        Parameters
        ----------
        workflow_id
            The workflow to query.

        Returns
        -------
        :
            The current version counter. Returns 0 for unknown workflow IDs.
        """
        state = self._workflows.get(workflow_id)
        if state is None:
            return 0
        return state.version

    @property
    def gate_enabled(self) -> bool:
        """Whether the disruptive-action confirmation gate is active.

        When ``False``, device-bearing stop/reset/reconfigure actions run
        immediately without a confirmation modal. Shared across all sessions.
        """
        return self._gate_enabled

    def set_gate_enabled(self, enabled: bool) -> None:
        """Enable or disable the disruptive-action confirmation gate globally."""
        if enabled == self._gate_enabled:
            return
        self._gate_enabled = enabled
        self._gate_version += 1

    def get_gate_version(self) -> int:
        """Version counter bumped whenever the global gate toggle changes."""
        return self._gate_version

    def _notify_staged_changed(self, workflow_id: WorkflowId) -> None:
        """Increment workflow state version when staging area changes.

        Version increments are deferred if currently in a staging transaction.
        The transaction context manager will call this on exit.
        """
        if self._transaction_workflow is not None:
            return

        self._workflows[workflow_id].version += 1

    def _notify_command_success(
        self, workflow_id: WorkflowId, action: str, success_count: int, total_count: int
    ) -> None:
        """Push command success notification to the queue.

        Notifications are pushed to NotificationQueue for multi-session support.
        Each session polls the queue and shows notifications in its own context.
        """
        if self._notification_queue is None:
            return

        # Get workflow title for user-friendly message
        spec = self._workflow_registry.get(workflow_id)
        title = spec.title if spec else str(workflow_id)

        # Convert action to past tense
        past_tense = {"start": "Started", "stop": "Stopped", "reset": "Reset"}
        verb = past_tense.get(action, action.capitalize())

        message = f"{verb} {success_count}/{total_count} jobs for '{title}'"
        self._notification_queue.push(
            NotificationEvent(
                message=message,
                notification_type=NotificationType.SUCCESS,
            )
        )

    def _notify_command_error(
        self,
        workflow_id: WorkflowId,
        action: str,
        error_count: int,
        total_count: int,
        error_message: str,
    ) -> None:
        """Push command error notification to the queue.

        Notifications are pushed to NotificationQueue for multi-session support.
        Each session polls the queue and shows notifications in its own context.
        """
        if self._notification_queue is None:
            return

        # Get workflow title for user-friendly message
        spec = self._workflow_registry.get(workflow_id)
        title = spec.title if spec else str(workflow_id)

        message = (
            f"Failed to {action} {error_count}/{total_count} jobs "
            f"for '{title}': {error_message}"
        )
        self._notification_queue.push(
            NotificationEvent(
                message=message,
                notification_type=NotificationType.ERROR,
            )
        )

    def process_acknowledgement(
        self, message_id: str, response: str, error_message: str | None = None
    ) -> None:
        """
        Process a command acknowledgement from the backend.

        This method should be called when a CommandAcknowledgement is received
        from the responses topic. It correlates the acknowledgement with the
        original command and notifies widgets when all responses are received.

        For commands targeting multiple sources, responses are accumulated until
        all expected responses arrive. On partial success, both success and error
        callbacks are invoked.

        Parameters
        ----------
        message_id:
            The message_id echoed from the original command.
        response:
            "ACK" for success, "ERR" for failure.
        error_message:
            Error message if response is "ERR".
        """
        result = self._pending_commands.record_response(
            message_id, success=(response == "ACK"), error_message=error_message
        )

        if result is None:
            # Either unknown message_id or still waiting for more responses
            return

        total = result.success_count + result.error_count

        if result.all_succeeded:
            logger.info(
                'Command %s for workflow %s acknowledged (%d/%d)',
                result.action,
                result.workflow_id,
                result.success_count,
                total,
            )
            self._notify_command_success(
                result.workflow_id, result.action, result.success_count, total
            )
        elif result.all_failed:
            combined_errors = "; ".join(result.error_messages) or "Unknown error"
            logger.warning(
                'Command %s for workflow %s failed (%d/%d): %s',
                result.action,
                result.workflow_id,
                result.error_count,
                total,
                combined_errors,
            )
            self._notify_command_error(
                result.workflow_id,
                result.action,
                result.error_count,
                total,
                combined_errors,
            )
        else:
            # Partial success - notify both
            combined_errors = "; ".join(result.error_messages) or "Unknown error"
            logger.warning(
                'Command %s for workflow %s partially succeeded (%d/%d, %d failed): %s',
                result.action,
                result.workflow_id,
                result.success_count,
                total,
                result.error_count,
                combined_errors,
            )
            self._notify_command_success(
                result.workflow_id, result.action, result.success_count, total
            )
            self._notify_command_error(
                result.workflow_id,
                result.action,
                result.error_count,
                total,
                combined_errors,
            )

    def expire_pending_commands(self) -> None:
        """
        Expire commands that were never acknowledged by the backend.

        A command whose acknowledgement never arrives (e.g. because its send was
        silently dropped by the sink) would otherwise leave the tracker entry
        alive forever, with no feedback to the user. Expired commands are logged
        and surfaced as an error notification. The workflow state and status
        badge are left untouched.
        """
        expired = self._pending_commands.expire_stale(PENDING_COMMAND_TIMEOUT_SECONDS)
        for cmd in expired:
            spec = self._workflow_registry.get(cmd.workflow_id)
            title = spec.title if spec else str(cmd.workflow_id)
            logger.warning(
                'No response from backend for %s command on workflow %s; '
                'expired after %.0fs',
                cmd.action,
                title,
                PENDING_COMMAND_TIMEOUT_SECONDS,
            )
            self._notify_command_error(
                cmd.workflow_id,
                cmd.action,
                cmd.expected_count,
                cmd.expected_count,
                f"no response from backend "
                f"(timed out after {PENDING_COMMAND_TIMEOUT_SECONDS:.0f}s)",
            )
