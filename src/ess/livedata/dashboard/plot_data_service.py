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

import logging
import threading
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, NewType
from uuid import UUID

if TYPE_CHECKING:
    from .plots import Plotter

logger = logging.getLogger(__name__)


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
        """Version counter, incremented on every state transition."""
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

        Valid from: WAITING_FOR_JOB, WAITING_FOR_DATA (replace), STOPPED, ERROR,
        READY (restart).

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
            logger.warning("Invalid transition job_started from %s", self._state.name)
            return

        self._state = LayerState.WAITING_FOR_DATA
        self._plotter = plotter
        self._error_message = None
        self._version += 1
        logger.debug(
            "Layer transition: job_started → WAITING_FOR_DATA (version=%d)",
            self._version,
        )

    def data_arrived(self) -> None:
        """
        Transition to READY when data arrives.

        Valid from: WAITING_FOR_DATA.
        """
        if self._state != LayerState.WAITING_FOR_DATA:
            logger.warning("Invalid transition data_arrived from %s", self._state.name)
            return

        self._state = LayerState.READY
        self._version += 1
        logger.debug(
            "Layer transition: data_arrived → READY (version=%d)", self._version
        )

    def job_stopped(self) -> None:
        """
        Transition to STOPPED when job is stopped.

        Valid from: WAITING_FOR_DATA, READY.
        Marks presenters dirty so sessions see the change.
        """
        valid_from = {LayerState.WAITING_FOR_DATA, LayerState.READY}
        if self._state not in valid_from:
            logger.warning("Invalid transition job_stopped from %s", self._state.name)
            return

        self._state = LayerState.STOPPED
        self._version += 1
        if self._plotter is not None:
            self._plotter.mark_presenters_dirty()
        logger.debug(
            "Layer transition: job_stopped → STOPPED (version=%d)", self._version
        )

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
        logger.debug(
            "Layer transition: error_occurred → ERROR (version=%d)", self._version
        )

    def has_displayable_plot(self) -> bool:
        """
        Check if the layer has a displayable plot.

        Returns True if in READY or STOPPED state with a plotter that has
        cached state.
        """
        if self._state not in {LayerState.READY, LayerState.STOPPED}:
            return False
        return self._plotter is not None and self._plotter.has_cached_state()


class PlotLayerState:
    """Facade for LayerStateMachine providing backward-compatible properties.

    This class wraps LayerStateMachine to provide the original interface
    (plotter, error, stopped properties) for code that hasn't been updated
    to use the state machine directly.
    """

    def __init__(self) -> None:
        self._machine = LayerStateMachine()

    @property
    def state_machine(self) -> LayerStateMachine:
        """Access the underlying state machine."""
        return self._machine

    @property
    def state(self) -> LayerState:
        """Current state of the layer."""
        return self._machine.state

    @property
    def version(self) -> int:
        """Version counter for change detection."""
        return self._machine.version

    @property
    def plotter(self) -> Plotter | None:
        """Plotter instance (backward-compatible property)."""
        return self._machine.plotter

    @property
    def error(self) -> str | None:
        """Error message (backward-compatible property)."""
        return self._machine.error_message

    @property
    def stopped(self) -> bool:
        """Whether the layer is stopped (backward-compatible property)."""
        return self._machine.state == LayerState.STOPPED

    def has_displayable_plot(self) -> bool:
        """Check if the layer has a displayable plot."""
        return self._machine.has_displayable_plot()


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
        self._layers: dict[LayerId, PlotLayerState] = {}
        self._lock = threading.Lock()

    def get(self, layer_id: LayerId) -> PlotLayerState | None:
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

    def ensure_layer(self, layer_id: LayerId) -> PlotLayerState:
        """
        Get or create state for a layer.

        Creates a new layer in WAITING_FOR_JOB state if it doesn't exist.

        Parameters
        ----------
        layer_id:
            Layer ID to get or create.

        Returns
        -------
        :
            The layer state (existing or newly created).
        """
        with self._lock:
            if layer_id not in self._layers:
                self._layers[layer_id] = PlotLayerState()
                logger.debug("Created layer %s in WAITING_FOR_JOB", layer_id)
            return self._layers[layer_id]

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
            state = self._layers.setdefault(layer_id, PlotLayerState())
            state.state_machine.job_started(plotter)
            logger.debug("Job started for %s", layer_id)

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
                state.state_machine.data_arrived()
                logger.debug("Data arrived for %s", layer_id)

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
                state.state_machine.job_stopped()
                logger.debug("Job stopped for %s", layer_id)

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
            state = self._layers.setdefault(layer_id, PlotLayerState())
            state.state_machine.error_occurred(error_msg)
            logger.debug("Error for %s: %s", layer_id, error_msg)

    # --- Backward-compatible methods (delegate to state machine) ---

    def set_plotter(self, layer_id: LayerId, plotter: Any) -> None:
        """
        Set the plotter for a layer (backward-compatible).

        Delegates to job_started() for state machine transition.
        """
        self.job_started(layer_id, plotter)

    def set_error(self, layer_id: LayerId, error_msg: str) -> None:
        """
        Set error state for a layer (backward-compatible).

        Delegates to error_occurred() for state machine transition.
        """
        self.error_occurred(layer_id, error_msg)

    def set_stopped(self, layer_id: LayerId) -> None:
        """
        Mark a layer as stopped (backward-compatible).

        Delegates to job_stopped() for state machine transition.
        """
        self.job_stopped(layer_id)

    def remove(self, layer_id: LayerId) -> None:
        """
        Remove state for a layer.

        Parameters
        ----------
        layer_id:
            Layer ID to remove.
        """
        with self._lock:
            if layer_id in self._layers:
                del self._layers[layer_id]
                logger.debug("Removed plot state for %s", layer_id)

    def clear(self) -> None:
        """Clear all state. Mainly useful for testing."""
        with self._lock:
            self._layers.clear()
