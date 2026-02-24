# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
SessionUpdater - Per-session component that drives all widget updates.

This module implements the polling-based update model for multi-session support.
Each browser session has its own SessionUpdater that polls for changes from
shared services in the correct session context.
"""

from __future__ import annotations

from collections.abc import Callable

import panel as pn
import structlog

from .notification_queue import (
    NotificationEvent,
    NotificationQueue,
    NotificationType,
)
from .session_registry import SessionId, SessionRegistry

logger = structlog.get_logger(__name__)


class SessionUpdater:
    """
    Per-session component that drives all widget updates.

    Each browser session creates its own SessionUpdater instance. The updater
    polls shared services (NotificationQueue) and runs custom handlers in its
    periodic callback, ensuring all session-bound components are updated
    in the correct session context.

    Plot updates are driven via custom handlers that call
    SessionPlotManager.update_pipes(), which uses the Presenter dirty flag
    mechanism for change detection.

    Note: Widget state synchronization uses direct callbacks (WidgetLifecycleCallbacks)
    rather than polling, since the callback mechanism works correctly for widgets.

    Parameters
    ----------
    session_id:
        Unique identifier for this session.
    session_registry:
        Registry for session heartbeats and tracking.
    notification_queue:
        Shared queue for notifications.
    """

    def __init__(
        self,
        *,
        session_id: SessionId,
        session_registry: SessionRegistry,
        notification_queue: NotificationQueue | None = None,
    ) -> None:
        self._session_id = session_id
        self._session_registry = session_registry
        self._notification_queue = notification_queue

        # Callbacks for custom updates (e.g., SessionPlotManager.update_pipes)
        self._custom_handlers: list[Callable[[], None]] = []

        # Register with notification queue
        if self._notification_queue is not None:
            self._notification_queue.register_session(session_id)

        # Auto-register this session with the registry
        self._session_registry.register(session_id, self)

        logger.debug("SessionUpdater created for session %s", session_id)

    def register_custom_handler(self, handler: Callable[[], None]) -> None:
        """
        Register a custom handler to be called during periodic updates.

        Custom handlers are called in the correct session context during
        the periodic update cycle. Use this for processing pending setups
        or other session-specific work (e.g., SessionPlotManager.update_pipes).

        Parameters
        ----------
        handler:
            Callback to invoke during periodic updates.
        """
        self._custom_handlers.append(handler)

    def unregister_custom_handler(self, handler: Callable[[], None]) -> None:
        """Unregister a custom handler."""
        if handler in self._custom_handlers:
            self._custom_handlers.remove(handler)

    def periodic_update(self) -> None:
        """
        Called from this session's periodic callback.

        Polls shared services for changes and runs custom handlers
        in a single batched UI update.
        """
        # Send heartbeat to registry
        self._session_registry.heartbeat(self._session_id)

        # Poll for notifications
        notifications = self._poll_notifications()

        # Apply all changes in a single batched update to avoid staggered rendering
        with pn.io.hold():
            self._show_notifications(notifications)

            # Run custom handlers (e.g., SessionPlotManager.update_pipes)
            # in correct session context
            for handler in self._custom_handlers:
                try:
                    handler()
                except Exception:
                    logger.exception(
                        "Error in custom handler for session %s", self._session_id
                    )

    def _poll_notifications(self) -> list[NotificationEvent]:
        """Poll NotificationQueue for new events."""
        if self._notification_queue is None:
            return []

        return self._notification_queue.get_new_events(self._session_id)

    def _show_notifications(self, notifications: list[NotificationEvent]) -> None:
        """Show notifications using Panel's notification system."""
        for event in notifications:
            try:
                notification_type = event.notification_type
                if notification_type == NotificationType.SUCCESS:
                    pn.state.notifications.success(
                        event.message, duration=event.duration
                    )
                elif notification_type == NotificationType.WARNING:
                    pn.state.notifications.warning(
                        event.message, duration=event.duration
                    )
                elif notification_type == NotificationType.ERROR:
                    pn.state.notifications.error(event.message, duration=event.duration)
                else:
                    pn.state.notifications.info(event.message, duration=event.duration)
            except Exception:
                # Panel notifications may not be available in all contexts
                logger.debug(
                    "Could not show notification in session %s: %s",
                    self._session_id,
                    event.message,
                )

    def cleanup(self) -> None:
        """Clean up session resources."""
        if self._notification_queue is not None:
            self._notification_queue.unregister_session(self._session_id)

        self._custom_handlers.clear()

        logger.debug("SessionUpdater cleaned up for session %s", self._session_id)

    @property
    def session_id(self) -> SessionId:
        """Get this session's ID."""
        return self._session_id
