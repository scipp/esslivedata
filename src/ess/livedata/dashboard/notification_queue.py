# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Notification queue for multi-session toast notifications.

Panel's notification system is session-bound, so background threads cannot show
toasts directly. This queue allows services to push notifications that each
session polls and displays in its own context.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum

from .session_registry import SessionId


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
