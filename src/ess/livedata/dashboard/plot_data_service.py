# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Plot data service for multi-session synchronization.

Provides storage for plot layer state using an explicit state machine.
Each layer transitions through well-defined states: WAITING_FOR_DATA,
READY, STOPPED, and ERROR.

Change notification uses version-based polling - UI components track
last-seen versions and rebuild when versions change.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, NewType
from uuid import UUID

import structlog

if TYPE_CHECKING:
    from .plots import Plotter

logger = structlog.get_logger(__name__)


class LayerState(Enum):
    """Explicit states for a plot layer's lifecycle.

    States are derived from workflow run-state (polled) plus data arrival,
    not from job lifecycle callbacks. State transitions are validated to
    ensure correct sequencing:

    WAITING_FOR_DATA → READY             [data_arrived()]
    WAITING_FOR_DATA → STOPPED           [job_stopped()]
    WAITING_FOR_DATA → ERROR             [error_occurred(msg)]

    READY → STOPPED                      [job_stopped()]
    READY → WAITING_FOR_DATA             [job_started(plotter)]
    READY → ERROR                        [error_occurred(msg)]

    STOPPED → WAITING_FOR_DATA           [job_started(plotter)]
    STOPPED → ERROR                      [error_occurred(msg)]

    ERROR → WAITING_FOR_DATA             [job_started(plotter)]

    A layer in STOPPED may still display retained data: ``data_arrived``
    while STOPPED is a no-op and ``has_displayable_plot`` reports True once
    the plotter holds a computed frame.
    """

    WAITING_FOR_DATA = auto()
    READY = auto()
    STOPPED = auto()
    ERROR = auto()


class LayerStateMachine:
    """Manages state transitions for a single plot layer.

    Encapsulates state, version tracking, and associated data (plotter, error).
    State transitions are validated and increment the version counter.

    Thread-safe: all state modifications happen through atomic operations.

    Viewer gate
    -----------
    Tokens express viewer interest, per (session, layer) — decoupled from
    lifecycle state, which is per-layer. ``has_viewers`` is consulted at
    frame-flush time on the ingestion thread: layers without viewers are
    skipped (no extraction, no compute). ``set_active`` is called from the
    per-session polling thread (Bokeh main); on the 0→1 transition it returns
    True and the orchestrator rebuilds the layer from a fresh DataService
    snapshot, synchronously on the polling thread — deliberately, so the same
    poll pass's component rebuild observes fresh ``has_cached_state``.
    """

    def __init__(self) -> None:
        self._state = LayerState.WAITING_FOR_DATA
        self._version = 0
        self._plotter: Plotter | None = None
        self._error_message: str | None = None
        self._active_tokens: set[int] = set()
        # Tokens we've already attached a weakref finalizer to, so we don't
        # register a second one on a False→True re-acquire.
        self._finalized_keys: set[int] = set()
        self._gate_lock = threading.Lock()

    @property
    def state(self) -> LayerState:
        """Current state of the layer."""
        return self._state

    @property
    def version(self) -> int:
        """Version counter, incremented on every state change.

        UI components compare versions to detect when rebuilds are needed.
        We track version rather than just state because some transitions
        (e.g., plotter replacement while in WAITING_FOR_DATA) don't change
        state but still require UI updates.
        """
        return self._version

    @property
    def plotter(self) -> Plotter | None:
        """Plotter instance, set when job starts."""
        return self._plotter

    @property
    def error_message(self) -> str | None:
        """Error message if in ERROR state."""
        return self._error_message

    def job_started(self, plotter: Plotter) -> None:
        """
        Transition to WAITING_FOR_DATA when a job starts.

        Valid from any state. When called from WAITING_FOR_DATA (workflow
        restarted before data arrived), state doesn't change but version
        still increments because the plotter is replaced and UI needs to
        rebuild with the new one.

        Parameters
        ----------
        plotter:
            The plotter instance for this job.
        """
        self._state = LayerState.WAITING_FOR_DATA
        self._error_message = None
        self._version += 1
        self._plotter = plotter

    def data_arrived(self) -> None:
        """
        Transition to READY when data arrives.

        Valid from: WAITING_FOR_DATA.
        No-op from READY (data continues to arrive after first update) and
        from STOPPED (a layer bound to a stopped workflow's retained data
        renders it but stays STOPPED).
        """
        if self._state in {LayerState.READY, LayerState.STOPPED}:
            return

        if self._state != LayerState.WAITING_FOR_DATA:
            logger.warning(
                "Invalid transition: data_arrived() called in state %s (expected %s)",
                self._state.name,
                LayerState.WAITING_FOR_DATA.name,
            )
            return

        self._state = LayerState.READY
        self._version += 1

    def job_stopped(self) -> None:
        """
        Transition to STOPPED when job is stopped.

        Valid from: WAITING_FOR_DATA, READY.
        Marks presenters dirty so sessions see the change.
        """
        valid_from = {LayerState.WAITING_FOR_DATA, LayerState.READY}
        if self._state not in valid_from:
            logger.warning(
                "Invalid transition: job_stopped() called in state %s (expected %s)",
                self._state.name,
                [s.name for s in valid_from],
            )
            return

        self._state = LayerState.STOPPED
        self._version += 1
        if self._plotter is not None:
            self._plotter.mark_presenters_dirty()

    def error_occurred(self, error_msg: str) -> None:
        """
        Transition to ERROR state.

        Valid from: any state.
        Marks presenters dirty so sessions see the change.

        Parameters
        ----------
        error_msg:
            Error message to display.
        """
        self._state = LayerState.ERROR
        self._error_message = error_msg
        self._version += 1
        if self._plotter is not None:
            self._plotter.mark_presenters_dirty()

    def set_active(self, token: object, active: bool) -> bool:
        """Acquire or release a viewer interest token on this layer.

        Returns True on the 0→1 transition, i.e. when the first token is
        acquired. Frame flushes skipped the layer while no viewer was
        watching (see ``has_viewers``), so the caller must then rebuild the
        layer from current DataService content.

        A ``weakref.finalize`` is attached on first acquire so the token is
        auto-released if the caller is garbage-collected without an explicit
        release. Explicit ``set_active(token, False)`` remains the fast path;
        the finalizer is belt-and-braces against missed cleanup (e.g., a
        session disposed without ``PlotGridTabs.shutdown`` running).
        """
        key = id(token)
        with self._gate_lock:
            was_active = bool(self._active_tokens)
            if active:
                self._active_tokens.add(key)
                if key not in self._finalized_keys:
                    self._finalized_keys.add(key)
                    # Captures ``key`` (an int) and a bound method, not the
                    # token itself — so the finalizer does not keep the token
                    # alive. CPython runs the finalizer before the token's
                    # memory can be reused for a different object, so an
                    # ``id()`` collision cannot race with an active token.
                    weakref.finalize(token, self._release_token, key)
            else:
                self._active_tokens.discard(key)
            return active and not was_active

    @property
    def has_viewers(self) -> bool:
        """Whether any viewer token is held; gates frame-flush compute."""
        with self._gate_lock:
            return bool(self._active_tokens)

    def _release_token(self, key: int) -> None:
        """Drop a token key from the gate (called by the weakref finalizer)."""
        with self._gate_lock:
            self._active_tokens.discard(key)
            self._finalized_keys.discard(key)

    def has_displayable_plot(self) -> bool:
        """
        Check if the layer has a displayable plot.

        Returns True if in READY or STOPPED state with a plotter that has
        cached state.
        """
        if self._state not in {LayerState.READY, LayerState.STOPPED}:
            return False
        return self._plotter is not None and self._plotter.has_cached_state()


LayerId = NewType('LayerId', UUID)


class PlotDataService:
    """
    Manages plot layer state using explicit state machines.

    PlotOrchestrator controls layer lifecycle through state machine transitions:
    - job_started(): Called at layer setup and when a run-state poll observes
      a new generation (with a fresh plotter)
    - data_arrived(): Called when first data arrives for a layer
    - job_stopped(): Called when a run-state poll observes the workflow stopped
    - error_occurred(): Called when an error occurs

    Each transition increments the layer's version counter. UI components poll
    for version changes to detect when they need to rebuild.

    Thread-safe: can be called from background threads and periodic callbacks.
    """

    def __init__(self) -> None:
        self._layers: dict[LayerId, LayerStateMachine] = {}
        self._lock = threading.Lock()
        self._version = 0

    @property
    def version(self) -> int:
        """Counter advanced whenever any layer's version advanced.

        Aggregates the per-layer counters into a single cheap read, so a
        session can gate its poll pass on "did any layer's lifecycle change"
        without scanning every layer. Only transitions that took effect count:
        a no-op ``data_arrived`` on an already-READY layer leaves it alone,
        which is what keeps the gate quiet under steady data flow.
        """
        return self._version

    def _apply(
        self,
        layer_id: LayerId,
        transition: Callable[[LayerStateMachine], None],
        *,
        create: bool = False,
    ) -> None:
        """Run a state transition, advancing ``version`` if it took effect.

        Transitions that the state machine rejects or treats as a no-op leave
        the layer's version untouched and must not advance the aggregate, or
        every data message would arm every session's poll gate.
        """
        with self._lock:
            if create:
                state = self._layers.setdefault(layer_id, LayerStateMachine())
            else:
                state = self._layers.get(layer_id)
                if state is None:
                    return
            before = state.version
            transition(state)
            if state.version != before:
                self._version += 1

    def get(self, layer_id: LayerId) -> LayerStateMachine | None:
        """
        Get current state for a layer.

        Parameters
        ----------
        layer_id:
            Layer ID to retrieve.

        Returns
        -------
        :
            Current layer state, or None if not set.
        """
        with self._lock:
            return self._layers.get(layer_id)

    def job_started(self, layer_id: LayerId, plotter: Any) -> None:
        """
        Transition a layer to WAITING_FOR_DATA when a job starts.

        Creates the layer if it doesn't exist.

        Parameters
        ----------
        layer_id:
            Layer ID to update.
        plotter:
            Plotter instance for per-session presenter creation.
        """
        self._apply(layer_id, lambda state: state.job_started(plotter), create=True)

    def data_arrived(self, layer_id: LayerId) -> None:
        """
        Transition a layer to READY when data arrives.

        Parameters
        ----------
        layer_id:
            Layer ID to update.
        """
        self._apply(layer_id, lambda state: state.data_arrived())

    def job_stopped(self, layer_id: LayerId) -> None:
        """
        Transition a layer to STOPPED when a job ends.

        Parameters
        ----------
        layer_id:
            Layer ID to update.
        """
        self._apply(layer_id, lambda state: state.job_stopped())

    def error_occurred(self, layer_id: LayerId, error_msg: str) -> None:
        """
        Transition a layer to ERROR state.

        Creates the layer if it doesn't exist.

        Parameters
        ----------
        layer_id:
            Layer ID to update.
        error_msg:
            Error message to display.
        """
        self._apply(
            layer_id, lambda state: state.error_occurred(error_msg), create=True
        )

    def remove(self, layer_id: LayerId) -> None:
        """
        Remove state for a layer.

        Parameters
        ----------
        layer_id:
            Layer ID to remove.
        """
        with self._lock:
            self._layers.pop(layer_id, None)

    def clear(self) -> None:
        """Clear all state."""
        with self._lock:
            self._layers.clear()
