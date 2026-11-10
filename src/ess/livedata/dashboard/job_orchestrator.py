# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
JobOrchestrator - Manages workflow job lifecycle and state transitions.

Coordinates workflow execution across multiple sources, handling:
- Configuration staging and commit (two-phase workflow start)
- Writing the desired state (one record per commit, a tombstone per stop)
- Keeping each workflow's current generation in sync with the desired table
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pydantic
import structlog

from ess.livedata.config.workflow_spec import (
    JobId,
    JobNumber,
    WorkflowId,
    WorkflowSpec,
)
from ess.livedata.core.desired_state import (
    DesiredJob,
    DesiredStateWriteError,
    DesiredStateWriter,
    DesiredUpdate,
    DesiredWorkflow,
)
from ess.livedata.core.job_manager import JobAction, JobCommand

from .active_job_registry import ActiveJobRegistry
from .command_service import CommandService
from .config_store import ConfigStore
from .configuration_adapter import ConfigurationState
from .notification_queue import NotificationEvent, NotificationQueue, NotificationType
from .pending_command_tracker import PendingCommandTracker
from .workflow_configuration_adapter import WorkflowConfigurationAdapter

if TYPE_CHECKING:
    from ess.livedata.config import Instrument

logger = structlog.get_logger(__name__)

SourceName = str

# A reset acknowledgement is a fast round-trip, so a reset that has not been
# acknowledged within this window is treated as lost (e.g. its send was silently
# dropped by the sink). Deliberately shorter than the 60s heartbeat-staleness
# constants in job_service.py, which answer a different question about backend
# liveness.
PENDING_COMMAND_TIMEOUT_SECONDS = 30.0


def desired_job_ids(desired: DesiredWorkflow) -> list[JobId]:
    """The ids of the jobs a desired-state record wants, one per source."""
    return [
        JobId(source_name=source_name, job_number=desired.job_number)
        for source_name in desired.jobs
    ]


@dataclass
class WorkflowState:
    """Desired run-state and staged configs of one registered workflow.

    ``current`` is the workflow's entry in the desired table, ``None`` if the
    table has none. Whether the backend runs it is observed via heartbeats.
    """

    current: DesiredWorkflow | None = None
    staged_jobs: dict[SourceName, DesiredJob] = field(default_factory=dict)
    version: int = 0


def _copy_jobs(jobs: Mapping[SourceName, DesiredJob]) -> dict[SourceName, DesiredJob]:
    return {
        source_name: DesiredJob(
            params=job.params.copy(), aux_source_names=job.aux_source_names.copy()
        )
        for source_name, job in jobs.items()
    }


class JobOrchestrator:
    """Orchestrates workflow job lifecycle and state management.

    The desired table (the desired-state topic, read back through
    :meth:`apply_desired_updates`) is the authority on what should run. A
    commit or stop writes one record and applies it locally once the writer
    confirms it is stored; reading it back is then a no-op. A record from
    another writer is applied the same way. Records for workflows outside the
    registry are kept as *unknown entries* for the UI and never removed
    without a user action.
    """

    def __init__(
        self,
        *,
        command_service: CommandService,
        desired_state: DesiredStateWriter,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        active_job_registry: ActiveJobRegistry,
        config_store: ConfigStore | None = None,
        instrument_config: Instrument | None = None,
        notification_queue: NotificationQueue | None = None,
        clock: Callable[[], float] = time.time,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        """
        Initialize the job orchestrator.

        Parameters
        ----------
        command_service
            Service for sending reset commands to backend services.
        desired_state
            Writer of the desired-state topic. Blocks until a record is stored.
        workflow_registry
            Registry of available workflows and their specifications.
        active_job_registry
            Thread-safe registry for tracking active job numbers and
            cleaning up data on deactivation.
        config_store
            Optional store for persisting staged workflow configurations across
            sessions. Orchestrator loads configs on init and persists on commit.
        instrument_config
            Optional instrument configuration for source metadata lookup.
        notification_queue
            Optional queue for multi-session notifications of failed writes and
            reset acknowledgements.
        clock
            Callable returning the current time in seconds, used for command
            expiry. Injectable to allow deterministic testing without real waits.
        on_change
            Called after UI-driven workflow state-version bumps (staging,
            commit, stop, gate toggle), e.g. to wake sessions so widgets
            refresh without waiting for the next poll. Ingestion-driven bumps
            (records read back from the desired table) need no poke: the
            ingestion loop wakes sessions after each drain pass anyway.
        """
        self._command_service = command_service
        self._desired_state = desired_state
        self._workflow_registry = workflow_registry
        self._active_job_registry = active_job_registry
        self._config_store = config_store
        self._instrument_config = instrument_config
        self._notification_queue = notification_queue
        self._on_change = on_change

        self._pending_resets = PendingCommandTracker(clock=clock)

        self._workflows: dict[WorkflowId, WorkflowState] = {}
        self._unknown: dict[WorkflowId, DesiredWorkflow] = {}
        self._unknown_version = 0

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

    def _load_configs_from_store(self) -> None:
        """Initialize all workflows with either loaded configs or defaults from spec."""
        for workflow_id, spec in self._workflow_registry.items():
            # Try to load persisted config directly as dict
            config_data = None
            if self._config_store is not None:
                config_data = self._config_store.get(str(workflow_id))

            state = WorkflowState()
            available_sources = set(spec.source_names)

            # Load per-source configs from 'jobs' dict. Other keys of a stored
            # entry are ignored.
            if config_data and config_data.get('jobs'):
                jobs_data = config_data['jobs']
                loaded_count = 0
                for source_name, job_data in jobs_data.items():
                    # Filter to sources that exist in the spec
                    if source_name in available_sources:
                        state.staged_jobs[source_name] = DesiredJob(
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
                        state.staged_jobs[source_name] = DesiredJob(
                            params=params.copy(),
                            aux_source_names=aux_source_names.copy(),
                        )
                logger.debug(
                    'Initialized workflow %s with defaults: %d sources',
                    workflow_id,
                    len(spec.source_names),
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
        self._workflows[workflow_id].staged_jobs[source_name] = DesiredJob(
            params=params.copy(), aux_source_names=aux_source_names.copy()
        )
        self._notify_staged_changed(workflow_id)

    def commit_workflow(self, workflow_id: WorkflowId) -> list[JobId]:
        """
        Commit staged configs and start workflow.

        Writes one desired-state record with a new job number. Only once the
        writer confirms it is stored does the workflow switch to it: the
        generation flips and the workflow's buffers are cleared. A failed write
        is reported to the user and changes nothing. Workers stop the
        superseded jobs themselves when they read the record.

        Parameters
        ----------
        workflow_id
            The workflow to start.

        Returns
        -------
        :
            List of JobIds created (one per staged source), empty if the write
            failed.

        Raises
        ------
        ValueError
            If no configs have been staged for this workflow.
        """
        state = self._workflows[workflow_id]
        if not state.staged_jobs:
            msg = f'No staged configs for workflow {workflow_id}'
            raise ValueError(msg)

        desired = DesiredWorkflow(
            workflow_id=workflow_id,
            job_number=uuid.uuid4(),
            jobs=dict(state.staged_jobs),
        )
        if not self._write(workflow_id, desired, action='start'):
            return []
        logger.info(
            'Started workflow %s with job_number %s on sources %s',
            workflow_id,
            desired.job_number,
            list(desired.jobs),
        )
        self._persist_state_to_store(workflow_id)
        return desired_job_ids(desired)

    def stop_workflow(self, workflow_id: WorkflowId) -> bool:
        """
        Stop all jobs for a workflow.

        Writes a tombstone for the workflow and, once it is stored, clears its
        current generation. Data the jobs publish until the workers stop them
        is dropped by the ingest filter. The just-stopped job's buffered data
        stays under the workflow's stable keys, so plots still display its last
        results; the next commit's generation flip clears them. The workflow
        configuration remains staged for future restarts.

        Parameters
        ----------
        workflow_id
            The workflow to stop.

        Returns
        -------
        :
            True if the tombstone was written, False if the workflow had no
            desired jobs or the write failed.
        """
        if self._workflows[workflow_id].current is None:
            logger.debug('No active jobs for workflow %s to stop', workflow_id)
            return False
        return self._write(workflow_id, None, action='stop')

    def remove_unknown_workflow(self, workflow_id: WorkflowId) -> bool:
        """Write the tombstone for an unknown entry of the desired table.

        Returns
        -------
        :
            True if the tombstone was written, False if the write failed.
        """
        return self._write(workflow_id, None, action='remove')

    def _write(
        self, workflow_id: WorkflowId, desired: DesiredWorkflow | None, action: str
    ) -> bool:
        """Write a desired-state record and apply it once it is stored.

        Blocks until the writer reports the outcome. A failed write is reported
        to the user and leaves all state untouched.
        """
        try:
            self._desired_state.write(workflow_id, desired)
        except DesiredStateWriteError as error:
            logger.error('Failed to %s workflow %s: %s', action, workflow_id, error)
            self._notify(
                f"Failed to {action} '{self._title(workflow_id)}': {error}",
                NotificationType.ERROR,
            )
            return False
        self.apply_desired_updates([(workflow_id, desired)])
        self._notify_change()
        return True

    def apply_desired_updates(self, updates: Iterable[DesiredUpdate]) -> None:
        """Apply desired-state records, in order, to the workflows' run-state.

        Called with the records read back from the desired-state topic (at
        startup and on the ingestion thread) and with the dashboard's own
        confirmed writes:

        - a record whose job number is not the workflow's current one becomes
          current, flipping the generation and clearing the workflow's buffers;
        - a tombstone clears the current generation, retaining its data;
        - a record that is already current (a read-back of the dashboard's own
          write) changes nothing.

        Records for workflows outside the registry update the unknown entries.
        """
        with self._active_job_registry.ingestion_guard():
            for workflow_id, desired in updates:
                state = self._workflows.get(workflow_id)
                if state is None:
                    self._apply_unknown(workflow_id, desired)
                    continue
                if desired is None:
                    if state.current is None:
                        continue
                    self._active_job_registry.deactivate(workflow_id)
                elif (
                    state.current is not None
                    and state.current.job_number == desired.job_number
                ):
                    continue
                else:
                    self._active_job_registry.begin_generation(
                        workflow_id, desired.job_number, config=desired.jobs
                    )
                state.current = desired
                state.version += 1

    def _apply_unknown(
        self, workflow_id: WorkflowId, desired: DesiredWorkflow | None
    ) -> None:
        if desired is None:
            if self._unknown.pop(workflow_id, None) is None:
                return
        else:
            self._unknown[workflow_id] = desired
        self._unknown_version += 1

    def get_unknown_workflows(self) -> list[DesiredWorkflow]:
        """Entries of the desired table whose workflow is not in the registry.

        Such an entry is left by a removed workflow or a version bump. No worker
        runs it, and the dashboard removes it only on user request
        (:meth:`remove_unknown_workflow`): a dashboard whose registry differs
        from the workers' must not stop their jobs just by starting.
        """
        # Copy first: the ingestion thread may update the dict meanwhile.
        return sorted(self._unknown.copy().values(), key=lambda d: str(d.workflow_id))

    def get_unknown_workflows_version(self) -> int:
        """Version counter bumped whenever the unknown entries change."""
        return self._unknown_version

    def get_desired_job_ids(self) -> set[JobId]:
        """The ids of all jobs the desired table wants for registered workflows."""
        return {
            job_id
            for state in self._workflows.values()
            if state.current is not None
            for job_id in desired_job_ids(state.current)
        }

    @property
    def active_job_registry(self) -> ActiveJobRegistry:
        """Registry shared with MessagePump for thread-safe data flow."""
        return self._active_job_registry

    def _persist_state_to_store(self, workflow_id: WorkflowId) -> None:
        """Persist the staged configs of a workflow to the config store.

        Returns silently if config_store is None (persistence is optional).
        """
        if self._config_store is None:
            return
        state = self._workflows[workflow_id]
        if state.staged_jobs:
            self._config_store[str(workflow_id)] = {
                'jobs': {
                    source_name: job.model_dump(mode='json')
                    for source_name, job in state.staged_jobs.items()
                }
            }

    def get_staged_config(
        self, workflow_id: WorkflowId
    ) -> dict[SourceName, DesiredJob]:
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
        return _copy_jobs(self._workflows[workflow_id].staged_jobs)

    def get_active_config(
        self, workflow_id: WorkflowId
    ) -> dict[SourceName, DesiredJob]:
        """
        Get active (desired) configuration for a workflow.

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
        current = self._workflows[workflow_id].current
        return {} if current is None else _copy_jobs(current.jobs)

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

        A workflow's job is "running" when the desired table has an entry for
        the workflow.
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

    def is_known_workflow(self, workflow_id: WorkflowId) -> bool:
        """Whether this workflow is managed here (heartbeat admit filter)."""
        return workflow_id in self._workflows

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
        current = self._workflows[workflow_id].current
        if current is None:
            logger.debug('No active jobs for workflow %s to reset', workflow_id)
            return False
        message_id = str(uuid.uuid4())
        self._pending_resets.register(
            message_id, workflow_id, expected_count=len(current.jobs)
        )
        self._command_service.send_batch(
            [
                JobCommand(job_id=job_id, action=JobAction.reset, message_id=message_id)
                for job_id in desired_job_ids(current)
            ]
        )
        logger.info(
            'Reset workflow %s (job_number=%s, %d jobs)',
            workflow_id,
            current.job_number,
            len(current.jobs),
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
        self._notify_change()

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
        self._notify_change()

    def _notify_change(self) -> None:
        """Poke the wake hook after a UI-driven state-version bump."""
        if self._on_change is not None:
            self._on_change()

    def _title(self, workflow_id: WorkflowId) -> str:
        spec = self._workflow_registry.get(workflow_id)
        return spec.title if spec else str(workflow_id)

    def _notify(self, message: str, notification_type: NotificationType) -> None:
        """Push a notification to the queue, which every session polls."""
        if self._notification_queue is not None:
            self._notification_queue.push(
                NotificationEvent(message=message, notification_type=notification_type)
            )

    def process_acknowledgement(
        self, message_id: str, response: str, error_message: str | None = None
    ) -> None:
        """
        Process a reset acknowledgement from the backend.

        This method should be called when a CommandAcknowledgement is received
        from the responses topic. It correlates the acknowledgement with the
        reset it answers and notifies the user once all responses are received.
        Responses for a reset of several jobs are accumulated; on partial
        success, both a success and an error notification are pushed.

        Parameters
        ----------
        message_id:
            The message_id echoed from the original command.
        response:
            "ACK" for success, "ERR" for failure.
        error_message:
            Error message if response is "ERR".
        """
        result = self._pending_resets.record_response(
            message_id, success=(response == "ACK"), error_message=error_message
        )
        if result is None:
            # Either unknown message_id or still waiting for more responses
            return

        total = result.success_count + result.error_count
        title = self._title(result.workflow_id)
        if result.success_count:
            logger.info(
                'Reset of workflow %s acknowledged (%d/%d)',
                result.workflow_id,
                result.success_count,
                total,
            )
            self._notify(
                f"Reset {result.success_count}/{total} jobs for '{title}'",
                NotificationType.SUCCESS,
            )
        if result.error_count:
            combined_errors = "; ".join(result.error_messages) or "Unknown error"
            logger.warning(
                'Reset of workflow %s failed (%d/%d): %s',
                result.workflow_id,
                result.error_count,
                total,
                combined_errors,
            )
            self._notify(
                f"Failed to reset {result.error_count}/{total} jobs "
                f"for '{title}': {combined_errors}",
                NotificationType.ERROR,
            )

    def expire_pending_commands(self) -> None:
        """
        Expire resets that were never acknowledged by the backend.

        A reset whose acknowledgement never arrives (e.g. because its send was
        silently dropped by the sink) would otherwise leave the tracker entry
        alive forever, with no feedback to the user. Expired resets are logged
        and surfaced as an error notification.
        """
        expired = self._pending_resets.expire_stale(PENDING_COMMAND_TIMEOUT_SECONDS)
        for cmd in expired:
            title = self._title(cmd.workflow_id)
            logger.warning(
                'No response from backend for reset of workflow %s; expired after '
                '%.0fs',
                title,
                PENDING_COMMAND_TIMEOUT_SECONDS,
            )
            self._notify(
                f"Failed to reset {cmd.expected_count}/{cmd.expected_count} jobs "
                f"for '{title}': no response from backend "
                f"(timed out after {PENDING_COMMAND_TIMEOUT_SECONDS:.0f}s)",
                NotificationType.ERROR,
            )
