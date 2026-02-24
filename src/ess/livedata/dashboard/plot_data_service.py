# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Plot data service for multi-session synchronization.

Provides storage for plot layer state using an explicit state machine.
Each layer transitions through well-defined states: WAITING_FOR_JOB,
WAITING_FOR_DATA, READY, STOPPED, and ERROR.

Change notification uses version-based polling - UI components track
last-seen versions and rebuild when versions change.
"""

from __future__ import annotations

import threading
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, NewType
from uuid import UUID

import structlog

if TYPE_CHECKING:
    from .plots import Plotter

logger = structlog.get_logger(__name__)


class LayerState(Enum):
    """Explicit states for a plot layer's lifecycle.

    State transitions are validated to ensure correct sequencing:

    WAITING_FOR_JOB → WAITING_FOR_DATA   [job_started(plotter)]
    WAITING_FOR_JOB → ERROR              [error_occurred(msg)]

    WAITING_FOR_DATA → READY             [data_arrived()]
    WAITING_FOR_DATA → STOPPED           [job_stopped()]
    WAITING_FOR_DATA → ERROR             [error_occurred(msg)]

    READY → STOPPED                      [job_stopped()]
    READY → WAITING_FOR_DATA             [job_restarted(plotter)]
    READY → ERROR                        [error_occurred(msg)]

    STOPPED → WAITING_FOR_DATA           [job_started(plotter)]
    STOPPED → ERROR                      [error_occurred(msg)]

    ERROR → WAITING_FOR_DATA             [job_started(plotter)]
    """

    WAITING_FOR_JOB = auto()
    WAITING_FOR_DATA = auto()
    READY = auto()
    STOPPED = auto()
    ERROR = auto()


class LayerStateMachine:
    """Manages state transitions for a single plot layer.

    Encapsulates state, version tracking, and associated data (plotter, error).
    State transitions are validated and increment the version counter.

    Thread-safe: all state modifications happen through atomic operations.

    Parameters
    ----------
    initial_state:
        Initial state for the layer. Defaults to WAITING_FOR_JOB.
    """

    def __init__(self, initial_state: LayerState = LayerState.WAITING_FOR_JOB) -> None:
        self._state = initial_state
        self._version = 0
        self._plotter: Plotter | None = None
        self._error_message: str | None = None

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

        Valid from: WAITING_FOR_JOB, WAITING_FOR_DATA, STOPPED, ERROR, READY.

        When called from WAITING_FOR_DATA (workflow restarted before data
        arrived), state doesn't change but version still increments because
        the plotter is replaced and UI needs to rebuild with the new one.

        Parameters
        ----------
        plotter:
            The plotter instance for this job.
        """
        valid_from = {
            LayerState.WAITING_FOR_JOB,
            LayerState.WAITING_FOR_DATA,  # Allow replacing plotter while waiting
            LayerState.STOPPED,
            LayerState.ERROR,
            LayerState.READY,
        }
        if self._state not in valid_from:
            return

        self._state = LayerState.WAITING_FOR_DATA
        self._plotter = plotter
        self._error_message = None
        self._version += 1

    def data_arrived(self) -> None:
        """
        Transition to READY when data arrives.

        Valid from: WAITING_FOR_DATA.
        No-op if already in READY state (data continues to arrive after first update).
        """
        if self._state == LayerState.READY:
            # Already ready - subsequent data arrivals are expected, no-op
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
    - job_started(): Called when a workflow job starts with a plotter
    - data_arrived(): Called when first data arrives for a layer
    - job_stopped(): Called when a workflow job stops
    - error_occurred(): Called when an error occurs

    Each transition increments the layer's version counter. UI components poll
    for version changes to detect when they need to rebuild.

    Thread-safe: can be called from background threads and periodic callbacks.
    """

    def __init__(self) -> None:
        self._layers: dict[LayerId, LayerStateMachine] = {}
        self._lock = threading.Lock()

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

    def layer_added(self, layer_id: LayerId) -> None:
        """
        Register a layer in WAITING_FOR_JOB state.

        Called when a dynamic layer is added to the grid but before any
        workflow job has started for it.

        Parameters
        ----------
        layer_id:
            Layer ID to register.
        """
        with self._lock:
            if layer_id not in self._layers:
                self._layers[layer_id] = LayerStateMachine()

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
        with self._lock:
            state = self._layers.setdefault(layer_id, LayerStateMachine())
            state.job_started(plotter)

    def data_arrived(self, layer_id: LayerId) -> None:
        """
        Transition a layer to READY when data arrives.

        Parameters
        ----------
        layer_id:
            Layer ID to update.
        """
        with self._lock:
            state = self._layers.get(layer_id)
            if state is not None:
                state.data_arrived()

    def job_stopped(self, layer_id: LayerId) -> None:
        """
        Transition a layer to STOPPED when a job ends.

        Parameters
        ----------
        layer_id:
            Layer ID to update.
        """
        with self._lock:
            state = self._layers.get(layer_id)
            if state is not None:
                state.job_stopped()

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
        with self._lock:
            state = self._layers.setdefault(layer_id, LayerStateMachine())
            state.error_occurred(error_msg)

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
