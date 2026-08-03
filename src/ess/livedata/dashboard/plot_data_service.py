# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Plot data service for multi-session synchronization.

Provides storage for plot layer state as immutable snapshots. Each layer
transitions through well-defined states: WAITING_FOR_DATA, READY, STOPPED,
and ERROR.

Change notification uses version-based polling - UI components track
last-seen versions and rebuild when versions change.
"""

from __future__ import annotations

import dataclasses
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


@dataclasses.dataclass(frozen=True, slots=True)
class LayerSnapshot:
    """Immutable lifecycle state of a single plot layer.

    Readers obtain a snapshot from :meth:`PlotDataService.get` and may read
    any combination of its fields without further synchronization: the four
    fields always describe one point in the layer's history, so a half-applied
    transition cannot be observed. This is the reader-side half of the
    single-writer versioned-pull model (ADR 0007).

    Transitions are pure: each returns the successor snapshot, or None when
    the transition is a no-op or invalid from the current state. Only
    :class:`PlotDataService` applies them, under its lock.

    ``plotter`` is a mutable object shared with the compute path, so a
    snapshot pins *which* plotter the layer had, not that plotter's contents.
    That is exactly what callers need: a plotter is created fresh per
    ``job_started``, so snapshot identity answers "is my plotter still the
    current one?".

    Parameters
    ----------
    state:
        Current lifecycle state.
    version:
        Counter incremented on every effective transition. UI components
        compare versions to detect when rebuilds are needed. Tracked
        separately from ``state`` because some transitions (e.g. plotter
        replacement while in WAITING_FOR_DATA) don't change state but still
        require UI updates.
    plotter:
        Plotter instance, set when a job starts.
    error_message:
        Error message if in ERROR state.
    """

    state: LayerState = LayerState.WAITING_FOR_DATA
    version: int = 0
    plotter: Plotter | None = None
    error_message: str | None = None

    def job_started(self, plotter: Plotter) -> LayerSnapshot:
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

        Returns
        -------
        :
            The successor snapshot; never None, as this is valid from any state.
        """
        return dataclasses.replace(
            self,
            state=LayerState.WAITING_FOR_DATA,
            version=self.version + 1,
            plotter=plotter,
            error_message=None,
        )

    def data_arrived(self) -> LayerSnapshot | None:
        """
        Transition to READY when data arrives.

        Valid from: WAITING_FOR_DATA.
        No-op from READY (data continues to arrive after first update) and
        from STOPPED (a layer bound to a stopped workflow's retained data
        renders it but stays STOPPED).

        Returns
        -------
        :
            The successor snapshot, or None if this was a no-op or invalid.
        """
        if self.state in {LayerState.READY, LayerState.STOPPED}:
            return None

        if self.state != LayerState.WAITING_FOR_DATA:
            logger.warning(
                "Invalid transition: data_arrived() called in state %s (expected %s)",
                self.state.name,
                LayerState.WAITING_FOR_DATA.name,
            )
            return None

        return dataclasses.replace(
            self, state=LayerState.READY, version=self.version + 1
        )

    def job_stopped(self) -> LayerSnapshot | None:
        """
        Transition to STOPPED when job is stopped.

        Valid from: WAITING_FOR_DATA, READY.

        Returns
        -------
        :
            The successor snapshot, or None if invalid from the current state.
        """
        valid_from = {LayerState.WAITING_FOR_DATA, LayerState.READY}
        if self.state not in valid_from:
            logger.warning(
                "Invalid transition: job_stopped() called in state %s (expected %s)",
                self.state.name,
                [s.name for s in valid_from],
            )
            return None

        return dataclasses.replace(
            self, state=LayerState.STOPPED, version=self.version + 1
        )

    def error_occurred(self, error_msg: str) -> LayerSnapshot:
        """
        Transition to ERROR state.

        Valid from: any state.

        Parameters
        ----------
        error_msg:
            Error message to display.

        Returns
        -------
        :
            The successor snapshot; never None, as this is valid from any state.
        """
        return dataclasses.replace(
            self,
            state=LayerState.ERROR,
            version=self.version + 1,
            error_message=error_msg,
        )

    def has_displayable_plot(self) -> bool:
        """
        Check if the layer has a displayable plot.

        Returns True if in READY or STOPPED state with a plotter that has
        cached state.
        """
        if self.state not in {LayerState.READY, LayerState.STOPPED}:
            return False
        return self.plotter is not None and self.plotter.has_cached_state()


LayerId = NewType('LayerId', UUID)

#: States that change what a cell renders without producing new data, so
#: nothing else will nudge sessions to re-render. Entering one marks the
#: layer's presenters dirty.
_PRESENTATION_CHANGING_STATES = frozenset({LayerState.STOPPED, LayerState.ERROR})


class PlotDataService:
    """
    Manages plot layer state as immutable snapshots.

    PlotOrchestrator controls layer lifecycle through state transitions:
    - job_started(): Called at layer setup and when a run-state poll observes
      a new generation (with a fresh plotter)
    - data_arrived(): Called when first data arrives for a layer
    - job_stopped(): Called when a run-state poll observes the workflow stopped
    - error_occurred(): Called when an error occurs

    Each effective transition replaces the layer's snapshot and increments its
    version counter. UI components poll for version changes to detect when they
    need to rebuild.

    Viewer gate
    -----------
    Tokens express viewer interest, per (session, layer) — decoupled from
    lifecycle state, which is per-layer, hence held separately from the
    snapshots. ``has_viewers`` is consulted at frame-flush time on the
    ingestion thread: layers without viewers are skipped (no extraction, no
    compute). ``set_active`` is called from the per-session polling thread
    (Bokeh main); on the 0→1 transition it returns True and the orchestrator
    rebuilds the layer from a fresh DataService snapshot, synchronously on the
    polling thread — deliberately, so the same poll pass's component rebuild
    observes fresh ``has_cached_state``.

    Thread-safe: can be called from background threads and periodic callbacks.
    Snapshots are replaced wholesale under the lock, so readers never need it.
    """

    def __init__(self) -> None:
        self._layers: dict[LayerId, LayerSnapshot] = {}
        self._viewers: dict[LayerId, set[int]] = {}
        # Token keys we've already attached a weakref finalizer to, so we don't
        # register a second one on a False→True re-acquire.
        self._finalized_keys: set[tuple[LayerId, int]] = set()
        # Reentrant: a token's weakref finalizer calls back into
        # ``_release_token``, and garbage collection can run it on a thread
        # that already holds the lock.
        self._lock = threading.RLock()
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
        transition: Callable[[LayerSnapshot], LayerSnapshot | None],
        *,
        create: bool,
    ) -> None:
        """Run a state transition, advancing ``version`` if it took effect.

        Transitions that the snapshot rejects or treats as a no-op return None
        and must not advance the aggregate, or every data message would arm
        every session's poll gate.

        Parameters
        ----------
        layer_id:
            Layer to transition.
        transition:
            Pure state transition to run under the lock.
        create:
            Whether an unknown ``layer_id`` starts a fresh layer or the
            transition is dropped. Required, since which of the two is correct
            depends on whether the caller can legitimately arrive first.
        """
        with self._lock:
            current = self._layers.get(layer_id)
            if current is None:
                if not create:
                    return
                current = LayerSnapshot()
            updated = transition(current)
            if updated is None:
                return
            self._layers[layer_id] = updated
            self._version += 1
        # Outside the lock: this calls into the plotter, and holding the
        # service lock across foreign code invites lock-order inversions.
        if updated.state in _PRESENTATION_CHANGING_STATES and (
            updated.plotter is not None
        ):
            updated.plotter.mark_presenters_dirty()

    def get(self, layer_id: LayerId) -> LayerSnapshot | None:
        """
        Get the current snapshot for a layer.

        The returned snapshot is immutable, so all of its fields may be read
        without synchronization and always describe one consistent point in
        the layer's history. Snapshot identity doubles as a change check: a
        later ``get`` returning the same object means no transition took
        effect in between.

        Parameters
        ----------
        layer_id:
            Layer ID to retrieve.

        Returns
        -------
        :
            Current layer snapshot, or None if not set.
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
        self._apply(layer_id, lambda state: state.data_arrived(), create=False)

    def job_stopped(self, layer_id: LayerId) -> None:
        """
        Transition a layer to STOPPED when a job ends.

        Parameters
        ----------
        layer_id:
            Layer ID to update.
        """
        self._apply(layer_id, lambda state: state.job_stopped(), create=False)

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

    def set_active(self, layer_id: LayerId, token: object, active: bool) -> bool:
        """Acquire or release a viewer interest token on a layer.

        Returns True on the 0→1 transition, i.e. when the first token is
        acquired. Frame flushes skipped the layer while no viewer was
        watching (see ``has_viewers``), so the caller must then rebuild the
        layer from current DataService content.

        A ``weakref.finalize`` is attached on first acquire so the token is
        auto-released if the caller is garbage-collected without an explicit
        release. Explicit ``set_active(..., False)`` remains the fast path;
        the finalizer is belt-and-braces against missed cleanup (e.g., a
        session disposed without ``PlotGridTabs.shutdown`` running).
        """
        key = id(token)
        with self._lock:
            tokens = self._viewers.setdefault(layer_id, set())
            was_active = bool(tokens)
            if active:
                tokens.add(key)
                if (layer_id, key) not in self._finalized_keys:
                    self._finalized_keys.add((layer_id, key))
                    # Captures ``key`` (an int) and a bound method, not the
                    # token itself — so the finalizer does not keep the token
                    # alive. CPython runs the finalizer before the token's
                    # memory can be reused for a different object, so an
                    # ``id()`` collision cannot race with an active token.
                    weakref.finalize(token, self._release_token, layer_id, key)
            else:
                tokens.discard(key)
            return active and not was_active

    def has_viewers(self, layer_id: LayerId) -> bool:
        """Whether any viewer token is held on a layer; gates frame-flush compute."""
        with self._lock:
            return bool(self._viewers.get(layer_id))

    def _release_token(self, layer_id: LayerId, key: int) -> None:
        """Drop a token key from the gate (called by the weakref finalizer)."""
        with self._lock:
            tokens = self._viewers.get(layer_id)
            if tokens is not None:
                tokens.discard(key)
            self._finalized_keys.discard((layer_id, key))

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
            self._viewers.pop(layer_id, None)
            self._finalized_keys = {
                entry for entry in self._finalized_keys if entry[0] != layer_id
            }

    def clear(self) -> None:
        """Clear all state."""
        with self._lock:
            self._layers.clear()
            self._viewers.clear()
            self._finalized_keys.clear()
