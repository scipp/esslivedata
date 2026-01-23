# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
State stores for multi-session synchronization.

These stores provide polling-based access to shared state, ensuring each
session's periodic callback accesses state in the correct session context.

Note: Widget state synchronization uses direct callbacks (WidgetLifecycleCallbacks)
rather than polling, since the callback mechanism works correctly for widgets.
Only notifications and plot data use polling stores.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, NewType

from .session_registry import SessionId

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications that can be queued."""

    INFO = 'info'
    SUCCESS = 'success'
    WARNING = 'warning'
    ERROR = 'error'


@dataclass
class NotificationEvent:
    """A notification event to be shown to users."""

    message: str
    notification_type: NotificationType = NotificationType.INFO
    duration: int = 3000  # milliseconds


@dataclass
class _SessionCursor:
    """Tracks a session's position in the notification queue."""

    next_index: int = 0


class NotificationQueue:
    """
    Cursor-based notification queue so each session sees all notifications.

    Unlike direct Panel notifications (which go to one random session),
    this queue ensures each browser session receives all notifications
    by tracking per-session cursors.

    Thread-safe: can be called from background threads and periodic callbacks.

    Notifications older than `max_age_events` are automatically purged
    to prevent unbounded memory growth.
    """

    def __init__(self, *, max_age_events: int = 100) -> None:
        """
        Initialize the notification queue.

        Parameters
        ----------
        max_age_events:
            Maximum number of events to retain. Older events are purged.
        """
        self._events: deque[NotificationEvent] = deque(maxlen=max_age_events)
        self._event_offset = 0  # Index of first event in deque
        self._cursors: dict[SessionId, _SessionCursor] = {}
        self._lock = threading.Lock()

    def push(self, event: NotificationEvent) -> None:
        """
        Add a notification event to the queue.

        Parameters
        ----------
        event:
            Notification event to add.
        """
        with self._lock:
            # Check if we're about to overflow (deque at max capacity)
            if len(self._events) == self._events.maxlen:
                # The oldest event will be dropped, so increment offset
                self._event_offset += 1
            self._events.append(event)
            logger.debug(
                "Pushed notification: %s (%s)",
                event.message[:50],
                event.notification_type.value,
            )

    def register_session(self, session_id: SessionId) -> None:
        """
        Register a session to receive notifications.

        The session starts with cursor at the current end of the queue,
        so it only sees new notifications from this point forward.

        Parameters
        ----------
        session_id:
            Session ID to register.
        """
        with self._lock:
            # Start at current end - don't replay old notifications
            current_index = self._event_offset + len(self._events)
            self._cursors[session_id] = _SessionCursor(next_index=current_index)
            logger.debug("Registered session %s at index %d", session_id, current_index)

    def unregister_session(self, session_id: SessionId) -> None:
        """
        Unregister a session.

        Parameters
        ----------
        session_id:
            Session ID to unregister.
        """
        with self._lock:
            if session_id in self._cursors:
                del self._cursors[session_id]
                logger.debug(
                    "Unregistered session %s from notification queue", session_id
                )

    def get_new_events(self, session_id: SessionId) -> list[NotificationEvent]:
        """
        Get new notification events for a session.

        Returns all events the session hasn't seen yet and advances
        the session's cursor.

        Parameters
        ----------
        session_id:
            Session ID to get events for.

        Returns
        -------
        :
            List of new notification events. Empty if session is not
            registered or no new events.
        """
        with self._lock:
            cursor = self._cursors.get(session_id)
            if cursor is None:
                return []

            # Calculate which events are new for this session
            start_deque_index = max(0, cursor.next_index - self._event_offset)
            new_events = list(self._events)[start_deque_index:]

            # Advance cursor to current end
            cursor.next_index = self._event_offset + len(self._events)

            return new_events

    def clear(self) -> None:
        """Clear all events and cursors. Mainly useful for testing."""
        with self._lock:
            self._events.clear()
            self._event_offset = 0
            self._cursors.clear()


@dataclass
class PlotLayerState:
    """State for a single plot layer with version tracking.

    Stores the computed plot state along with references needed for
    per-session component creation:

    - state: The computed HoloViews elements from plotter.compute()
    - plotter: Reference to the Plotter instance for create_presenter()
    """

    state: Any  # The computed HoloViews elements from plotter.compute()
    version: int = 0
    plotter: Any = None  # Reference to Plotter for create_presenter()


LayerId = NewType('LayerId', str)


class PlotDataService:
    """
    Stores computed plot state with version tracking.

    PlotOrchestrator stores results after calling `plotter.compute()`.
    Each session polls for updates in its periodic callback.

    Thread-safe: can be called from background threads and periodic callbacks.
    """

    def __init__(self) -> None:
        self._layers: dict[LayerId, PlotLayerState] = {}
        self._lock = threading.Lock()

    def update(
        self,
        layer_id: LayerId,
        state: Any,
        *,
        plotter: Any = None,
    ) -> None:
        """
        Update state for a layer.

        Parameters
        ----------
        layer_id:
            Layer ID to update.
        state:
            New computed plot state (HoloViews elements from plotter.compute()).
        plotter:
            Optional plotter instance for per-session presenter creation.
            Only needs to be provided on first update.
        """
        with self._lock:
            if layer_id in self._layers:
                current = self._layers[layer_id]
                # Preserve plotter reference if not provided on update
                effective_plotter = plotter if plotter is not None else current.plotter
                self._layers[layer_id] = PlotLayerState(
                    state=state,
                    version=current.version + 1,
                    plotter=effective_plotter,
                )
            else:
                self._layers[layer_id] = PlotLayerState(
                    state=state,
                    version=1,
                    plotter=plotter,
                )
            logger.debug(
                "Updated plot state for %s at version %d",
                layer_id,
                self._layers[layer_id].version,
            )

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

    def get_version(self, layer_id: LayerId) -> int:
        """
        Get version for a specific layer.

        Parameters
        ----------
        layer_id:
            Layer ID to check.

        Returns
        -------
        :
            Version number, or 0 if layer doesn't exist.
        """
        with self._lock:
            if layer_id in self._layers:
                return self._layers[layer_id].version
            return 0

    def get_updates_since(
        self, versions: dict[LayerId, int]
    ) -> dict[LayerId, PlotLayerState]:
        """
        Get layers that have been updated since the given versions.

        Parameters
        ----------
        versions:
            Dictionary mapping layer IDs to last-seen versions.

        Returns
        -------
        :
            Dictionary of layers with newer versions than provided.
        """
        with self._lock:
            updates: dict[LayerId, PlotLayerState] = {}
            for layer_id, layer_state in self._layers.items():
                last_version = versions.get(layer_id, 0)
                if layer_state.version > last_version:
                    updates[layer_id] = layer_state
            return updates

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
