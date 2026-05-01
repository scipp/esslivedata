# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
FOMOrchestrator - Manages stable-alias slot lifecycle for figure-of-merit jobs.

Parallel to :class:`JobOrchestrator`, but slot-keyed instead of workflow-keyed.
Each slot binds a stable alias (e.g. ``fom-0``) to a single ``(job, output)``
pair. The orchestrator owns the FOM job's lifecycle: committing or releasing a
slot composes the explicit Kafka batch (stop previous + unbind + new
WorkflowConfig + bind) so the dashboard does not depend on backend
auto-stop-on-rebind semantics.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, NewType

import pydantic
import structlog

import ess.livedata.config.keys as keys
from ess.livedata.config.models import ConfigKey
from ess.livedata.config.workflow_spec import (
    JobId,
    JobNumber,
    WorkflowConfig,
    WorkflowId,
    WorkflowSpec,
)
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.core.stream_alias import BindStreamAlias, UnbindStreamAlias

from .active_job_registry import ActiveJobRegistry
from .command_service import CommandService
from .config_store import ConfigStore
from .job_service import JobService
from .notification_queue import NotificationEvent, NotificationQueue, NotificationType

logger = structlog.get_logger(__name__)

FOMSlot = NewType('FOMSlot', str)
FOMAction = str  # "commit" | "release" | "reset"

# On-disk schema version for persisted slot state.
# Bump and add migration when changing the persisted layout.
_SLOT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FOMSlotState:
    """In-memory state of a configured FOM slot.

    ``job_number`` is ``None`` when the slot has been stopped but its
    configuration is retained for restart or for prefilling the config wizard.
    """

    workflow_id: WorkflowId
    source_name: str
    output_name: str
    params: dict
    aux_source_names: dict
    job_number: JobNumber | None

    @property
    def is_running(self) -> bool:
        return self.job_number is not None

    @property
    def job_id(self) -> JobId | None:
        if self.job_number is None:
            return None
        return JobId(source_name=self.source_name, job_number=self.job_number)


@dataclass
class _PendingBatch:
    slot: FOMSlot
    action: FOMAction
    timestamp: float
    expected_count: int
    success_count: int = 0
    error_count: int = 0
    error_messages: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.success_count + self.error_count >= self.expected_count


@dataclass(frozen=True)
class FOMCommandResult:
    slot: FOMSlot
    action: FOMAction
    success_count: int
    error_count: int
    error_messages: list[str]

    @property
    def all_succeeded(self) -> bool:
        return self.error_count == 0


class _FOMPendingTracker:
    """Slot-scoped variant of :class:`PendingCommandTracker`."""

    def __init__(self) -> None:
        self._pending: dict[str, _PendingBatch] = {}

    def register(
        self,
        message_id: str,
        slot: FOMSlot,
        action: FOMAction,
        expected_count: int,
    ) -> None:
        self._pending[message_id] = _PendingBatch(
            slot=slot,
            action=action,
            timestamp=time.time(),
            expected_count=expected_count,
        )

    def record_response(
        self, message_id: str, *, success: bool, error_message: str | None = None
    ) -> FOMCommandResult | None:
        pending = self._pending.get(message_id)
        if pending is None:
            return None
        if success:
            pending.success_count += 1
        else:
            pending.error_count += 1
            if error_message:
                pending.error_messages.append(error_message)
        if not pending.is_complete:
            return None
        del self._pending[message_id]
        return FOMCommandResult(
            slot=pending.slot,
            action=pending.action,
            success_count=pending.success_count,
            error_count=pending.error_count,
            error_messages=pending.error_messages,
        )


class FOMOrchestrator:
    """Slot-keyed orchestrator for figure-of-merit jobs."""

    def __init__(
        self,
        *,
        command_service: CommandService,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        active_job_registry: ActiveJobRegistry,
        job_service: JobService,
        notification_queue: NotificationQueue | None = None,
        config_store: ConfigStore | None = None,
        n_slots: int = 2,
        probe_timeout_seconds: float = 10.0,
    ) -> None:
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}")
        self._command_service = command_service
        self._workflow_registry = workflow_registry
        self._active_job_registry = active_job_registry
        self._job_service = job_service
        self._notification_queue = notification_queue
        self._config_store = config_store
        self._slot_names: list[FOMSlot] = [FOMSlot(f"fom-{i}") for i in range(n_slots)]
        self._slots: dict[FOMSlot, FOMSlotState | None] = dict.fromkeys(
            self._slot_names
        )
        self._versions: dict[FOMSlot, int] = dict.fromkeys(self._slot_names, 0)
        self._pending = _FOMPendingTracker()
        self._slot_listeners: list[Callable[[FOMSlot], None]] = []
        # Slots restored from store with a non-None job_number start as
        # "probing": tentatively running until a fresh JobStatus arrives or
        # the probe deadline elapses. Backend services and their alias
        # registries share an in-memory lifetime, so JobStatus presence is
        # sufficient evidence the binding is still live.
        self._probe_pending: set[FOMSlot] = set()
        self._probe_deadline = time.monotonic() + probe_timeout_seconds
        self._load_from_store()

    # ----- introspection ------------------------------------------------------

    @property
    def slot_names(self) -> list[FOMSlot]:
        """Return the list of slot identifiers."""
        return list(self._slot_names)

    def get_slot_state(self, slot: FOMSlot) -> FOMSlotState | None:
        """Return the slot's current state, or ``None`` if unbound."""
        return self._slots.get(slot)

    def get_slot_state_version(self, slot: FOMSlot) -> int:
        """Version counter incremented on every slot mutation."""
        return self._versions.get(slot, 0)

    def get_workflow_registry(self) -> Mapping[WorkflowId, WorkflowSpec]:
        return self._workflow_registry

    def add_slot_changed_listener(self, callback: Callable[[FOMSlot], None]) -> None:
        """Register a callback fired when a slot's state changes.

        Used by widgets that want push-based update rather than polling.
        """
        self._slot_listeners.append(callback)

    # ----- mutations ----------------------------------------------------------

    def commit_slot(
        self,
        slot: FOMSlot,
        *,
        workflow_id: WorkflowId,
        source_name: str,
        output_name: str,
        params: dict,
        aux_source_names: dict | None = None,
    ) -> JobId:
        """Bind ``slot`` to a fresh job, replacing any existing binding.

        Sends ``[Stop(prev), Unbind, WorkflowConfig(new), Bind]`` (or just
        ``[WorkflowConfig(new), Bind]`` if the slot is empty) as a single
        Kafka batch, then updates local state.
        """
        self._require_slot(slot)

        prev = self._slots[slot]
        new_job_number: JobNumber = uuid.uuid4()
        new_job_id = JobId(source_name=source_name, job_number=new_job_number)
        message_id = str(uuid.uuid4())
        aux_dict = aux_source_names or {}

        commands: list[tuple[ConfigKey, object]] = []
        if prev is not None and prev.is_running:
            commands.append(
                (
                    ConfigKey(key=JobCommand.key, source_name=str(prev.job_id)),
                    JobCommand(
                        job_id=prev.job_id,
                        action=JobAction.stop,
                        message_id=message_id,
                    ),
                )
            )
            commands.append(
                (
                    ConfigKey(key=UnbindStreamAlias.key, source_name=slot),
                    UnbindStreamAlias(alias=slot, message_id=message_id),
                )
            )
        commands.append(
            (
                keys.WORKFLOW_CONFIG.create_key(source_name=source_name),
                WorkflowConfig.from_params(
                    workflow_id=workflow_id,
                    params=params,
                    aux_source_names=aux_dict,
                    job_number=new_job_number,
                    message_id=message_id,
                ),
            )
        )
        commands.append(
            (
                ConfigKey(key=BindStreamAlias.key, source_name=slot),
                BindStreamAlias(
                    alias=slot,
                    job_id=new_job_id,
                    output_name=output_name,
                    message_id=message_id,
                ),
            )
        )

        self._pending.register(message_id, slot, "commit", len(commands))
        self._command_service.send_batch(commands)

        # Activate the new job inside the ingestion guard so that data
        # arriving for the new job_number is accepted; deactivate the
        # previous one to clean up its DataService entries.
        with self._active_job_registry.ingestion_guard():
            self._active_job_registry.activate(new_job_number)
        if prev is not None and prev.job_number is not None:
            self._active_job_registry.deactivate(prev.job_number)

        self._slots[slot] = FOMSlotState(
            workflow_id=workflow_id,
            source_name=source_name,
            output_name=output_name,
            params=dict(params),
            aux_source_names=dict(aux_dict),
            job_number=new_job_number,
        )
        self._bump_version(slot)

        logger.info(
            "fom_slot_committed",
            slot=slot,
            workflow_id=str(workflow_id),
            source=source_name,
            output=output_name,
            job_number=str(new_job_number),
        )
        return new_job_id

    def release_slot(self, slot: FOMSlot) -> bool:
        """Stop the bound job; retain the slot's configuration for restart.

        Sends ``[Stop, Unbind]`` and locally clears ``job_number`` while keeping
        the rest of the slot's binding (workflow, source, output, params) so
        that the user can re-launch the same configuration via
        :meth:`start_slot` or tweak it through the config wizard.

        Returns ``False`` if the slot was already stopped or empty (no
        commands sent).
        """
        self._require_slot(slot)
        prev = self._slots[slot]
        if prev is None or not prev.is_running:
            return False

        message_id = str(uuid.uuid4())
        commands: list[tuple[ConfigKey, object]] = [
            (
                ConfigKey(key=JobCommand.key, source_name=str(prev.job_id)),
                JobCommand(
                    job_id=prev.job_id,
                    action=JobAction.stop,
                    message_id=message_id,
                ),
            ),
            (
                ConfigKey(key=UnbindStreamAlias.key, source_name=slot),
                UnbindStreamAlias(alias=slot, message_id=message_id),
            ),
        ]
        self._pending.register(message_id, slot, "release", len(commands))
        self._command_service.send_batch(commands)

        self._active_job_registry.deactivate(prev.job_number)
        self._slots[slot] = FOMSlotState(
            workflow_id=prev.workflow_id,
            source_name=prev.source_name,
            output_name=prev.output_name,
            params=prev.params,
            aux_source_names=prev.aux_source_names,
            job_number=None,
        )
        self._bump_version(slot)

        logger.info("fom_slot_released", slot=slot)
        return True

    def clear_slot(self, slot: FOMSlot) -> bool:
        """Drop a stopped slot's retained configuration.

        Only valid for slots that are stopped (``job_number`` is ``None``).
        Returns ``False`` if the slot is empty or still running.
        """
        self._require_slot(slot)
        prev = self._slots[slot]
        if prev is None or prev.is_running:
            return False
        self._slots[slot] = None
        self._bump_version(slot)
        logger.info("fom_slot_cleared", slot=slot)
        return True

    def start_slot(self, slot: FOMSlot) -> JobId | None:
        """Re-launch a stopped slot using its retained configuration.

        Returns ``None`` if the slot is empty or already running.
        """
        self._require_slot(slot)
        prev = self._slots[slot]
        if prev is None or prev.is_running:
            return None
        return self.commit_slot(
            slot,
            workflow_id=prev.workflow_id,
            source_name=prev.source_name,
            output_name=prev.output_name,
            params=prev.params,
            aux_source_names=prev.aux_source_names,
        )

    def reset_slot(self, slot: FOMSlot) -> bool:
        """Send a reset to the bound job.

        Returns ``False`` if the slot is empty or stopped (no running job).
        """
        self._require_slot(slot)
        prev = self._slots[slot]
        if prev is None or not prev.is_running:
            return False

        message_id = str(uuid.uuid4())
        cmd = (
            ConfigKey(key=JobCommand.key, source_name=str(prev.job_id)),
            JobCommand(
                job_id=prev.job_id,
                action=JobAction.reset,
                message_id=message_id,
            ),
        )
        self._pending.register(message_id, slot, "reset", 1)
        self._command_service.send_batch([cmd])

        logger.info("fom_slot_reset", slot=slot)
        return True

    # ----- ack / status routing -----------------------------------------------

    def process_acknowledgement(
        self,
        message_id: str,
        response: str,
        error_message: str | None = None,
    ) -> None:
        """Record an ACK from the responses topic.

        Ignored if the ``message_id`` is not one we registered.
        """
        result = self._pending.record_response(
            message_id, success=(response == "ACK"), error_message=error_message
        )
        if result is None:
            return
        total = result.success_count + result.error_count
        if result.all_succeeded:
            self._notify_success(
                result.slot, result.action, result.success_count, total
            )
        else:
            combined = "; ".join(result.error_messages) or "Unknown error"
            self._notify_error(
                result.slot, result.action, result.error_count, total, combined
            )

    def on_job_status_updated(self, job_status: JobStatus) -> None:
        """React to a backend-reported job status.

        Any fresh status for a probing slot's job confirms the backend still
        hosts the job: the slot is taken out of ``_probe_pending`` so the
        timeout in :meth:`tick` does not later clear it. A ``stopped`` status
        additionally transitions the slot to STOPPED while retaining its
        configuration.
        """
        for slot, state in list(self._slots.items()):
            if state is None or not state.is_running:
                continue
            if (
                job_status.job_id.source_name != state.source_name
                or job_status.job_id.job_number != state.job_number
            ):
                continue
            self._probe_pending.discard(slot)
            if job_status.state == JobState.stopped:
                logger.info(
                    "fom_slot_stopped_by_backend",
                    slot=slot,
                    job_number=str(state.job_number),
                )
                self._active_job_registry.deactivate(state.job_number)
                self._slots[slot] = FOMSlotState(
                    workflow_id=state.workflow_id,
                    source_name=state.source_name,
                    output_name=state.output_name,
                    params=state.params,
                    aux_source_names=state.aux_source_names,
                    job_number=None,
                )
                self._bump_version(slot)
            return

    def tick(self) -> None:
        """Resolve pending probes that have not seen a fresh JobStatus.

        Called periodically (e.g. from the dashboard's update loop). Until
        the probe deadline, restored slots stay tentatively running so the
        widget shows PENDING. Once the deadline has passed, any slot that
        has not had a fresh ``JobStatus`` arrive is treated as orphaned
        (the backend was restarted while the dashboard was down) and is
        transitioned to STOPPED — config retained, ``job_number`` cleared.
        """
        if not self._probe_pending:
            return
        if time.monotonic() < self._probe_deadline:
            return
        for slot in list(self._probe_pending):
            state = self._slots[slot]
            if state is None or not state.is_running:
                self._probe_pending.discard(slot)
                continue
            job_id = state.job_id
            if (
                job_id is not None
                and self._job_service.job_statuses.get(job_id) is not None
                and not self._job_service.is_status_stale(job_id)
            ):
                # A fresh status arrived between the listener firing and
                # tick(); treat as confirmed.
                self._probe_pending.discard(slot)
                continue
            logger.info(
                "fom_slot_probe_timed_out",
                slot=slot,
                job_number=str(state.job_number),
            )
            self._active_job_registry.deactivate(state.job_number)
            self._slots[slot] = FOMSlotState(
                workflow_id=state.workflow_id,
                source_name=state.source_name,
                output_name=state.output_name,
                params=state.params,
                aux_source_names=state.aux_source_names,
                job_number=None,
            )
            self._probe_pending.discard(slot)
            self._bump_version(slot)

    # ----- internals ----------------------------------------------------------

    def _require_slot(self, slot: FOMSlot) -> None:
        if slot not in self._slots:
            raise KeyError(f"Unknown FOM slot: {slot!r}")

    def _bump_version(self, slot: FOMSlot) -> None:
        # Any explicit mutation supersedes a pending probe — otherwise
        # tick() could later kill a freshly committed job whose status has
        # not yet arrived.
        self._probe_pending.discard(slot)
        self._versions[slot] = self._versions.get(slot, 0) + 1
        self._persist_slot(slot)
        for listener in self._slot_listeners:
            try:
                listener(slot)
            except Exception:
                logger.exception("fom_slot_listener_error", slot=slot)

    def _persist_slot(self, slot: FOMSlot) -> None:
        """Write the slot's state to ``config_store`` (or remove if empty)."""
        if self._config_store is None:
            return
        state = self._slots[slot]
        if state is None:
            try:
                del self._config_store[slot]
            except KeyError:
                pass
            return
        self._config_store[slot] = {
            'version': _SLOT_SCHEMA_VERSION,
            'workflow_id': str(state.workflow_id),
            'source_name': state.source_name,
            'output_name': state.output_name,
            'params': dict(state.params),
            'aux_source_names': dict(state.aux_source_names),
            'job_number': (
                str(state.job_number) if state.job_number is not None else None
            ),
        }

    def _load_from_store(self) -> None:
        """Restore slot configurations from ``config_store``.

        Stale entries (unknown workflow, source, or output; invalid params)
        are dropped with a warning. Slots persisted with a non-None
        ``job_number`` are restored as probing: ``active_job_registry`` is
        primed so incoming results are accepted, and the slot is added to
        ``_probe_pending`` until a fresh ``JobStatus`` confirms the backend
        still hosts the job (or the probe deadline elapses).
        """
        if self._config_store is None:
            return
        for slot in self._slot_names:
            data = self._config_store.get(slot)
            if data is None:
                continue
            state = self._deserialize_slot(slot, data)
            if state is None:
                continue
            self._slots[slot] = state
            if state.job_number is not None:
                self._active_job_registry.restore(state.job_number)
                self._probe_pending.add(slot)

    def _deserialize_slot(
        self, slot: FOMSlot, data: dict[str, Any]
    ) -> FOMSlotState | None:
        version = data.get('version')
        if version != _SLOT_SCHEMA_VERSION:
            logger.warning(
                "fom_slot_load_unknown_schema_version", slot=slot, version=version
            )
            return None
        try:
            workflow_id = WorkflowId.from_string(data['workflow_id'])
        except (KeyError, ValueError):
            logger.warning(
                "fom_slot_load_bad_workflow_id",
                slot=slot,
                raw=data.get('workflow_id'),
            )
            return None
        spec = self._workflow_registry.get(workflow_id)
        if spec is None:
            logger.warning(
                "fom_slot_load_unknown_workflow",
                slot=slot,
                workflow_id=str(workflow_id),
            )
            return None
        source_name = data.get('source_name')
        if source_name not in spec.source_names:
            logger.warning(
                "fom_slot_load_unknown_source", slot=slot, source_name=source_name
            )
            return None
        output_name = data.get('output_name')
        if spec.outputs is None or output_name not in spec.outputs.model_fields:
            logger.warning(
                "fom_slot_load_unknown_output", slot=slot, output_name=output_name
            )
            return None
        params = data.get('params') or {}
        if spec.params is not None:
            try:
                spec.params.model_validate(params)
            except pydantic.ValidationError as e:
                logger.warning("fom_slot_load_invalid_params", slot=slot, error=str(e))
                return None
        aux = data.get('aux_source_names') or {}
        job_number_raw = data.get('job_number')
        try:
            job_number = (
                uuid.UUID(job_number_raw) if job_number_raw is not None else None
            )
        except (TypeError, ValueError):
            logger.warning(
                "fom_slot_load_bad_job_number", slot=slot, raw=job_number_raw
            )
            job_number = None
        return FOMSlotState(
            workflow_id=workflow_id,
            source_name=source_name,
            output_name=output_name,
            params=dict(params),
            aux_source_names=dict(aux),
            job_number=job_number,
        )

    def _notify_success(
        self, slot: FOMSlot, action: FOMAction, success: int, total: int
    ) -> None:
        if self._notification_queue is None:
            return
        verb = {"commit": "Configured", "release": "Released", "reset": "Reset"}.get(
            action, action.capitalize()
        )
        self._notification_queue.push(
            NotificationEvent(
                message=f"{verb} FOM slot '{slot}' ({success}/{total} ACKs)",
                notification_type=NotificationType.SUCCESS,
            )
        )

    def _notify_error(
        self,
        slot: FOMSlot,
        action: FOMAction,
        errors: int,
        total: int,
        message: str,
    ) -> None:
        if self._notification_queue is None:
            return
        self._notification_queue.push(
            NotificationEvent(
                message=(
                    f"FOM slot '{slot}' {action} failed ({errors}/{total}): {message}"
                ),
                notification_type=NotificationType.ERROR,
            )
        )
