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
from .widgets.heartbeat_widget import HeartbeatWidget

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

        # Browser heartbeat mechanism using ReactiveHTML.
        # The widget's JavaScript increments a counter every 5 seconds.
        # We watch for changes to detect browser liveness.
        self._heartbeat_widget = HeartbeatWidget(interval_ms=5000)
        self._last_heartbeat_value = 0

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

        Heartbeats are sent to the registry only when we have evidence that
        the browser is still connected (via the browser heartbeat widget).
        """
        # Check if browser has sent a heartbeat (widget value changed)
        self._check_browser_heartbeat()

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

    def _check_browser_heartbeat(self) -> None:
        """
        Check if the browser has sent a heartbeat via the widget.

        The browser-side JavaScript increments the heartbeat counter value
        every 5 seconds. If the value has changed since our last check,
        the browser is alive and we send a heartbeat to the registry.
        """
        current_value = self._heartbeat_widget.counter
        if current_value != self._last_heartbeat_value:
            self._last_heartbeat_value = current_value
            self._session_registry.heartbeat(self._session_id)
            logger.debug(
                "Browser heartbeat received for session %s (counter=%d)",
                self._session_id,
                current_value,
            )

    @property
    def heartbeat_widget(self) -> HeartbeatWidget:
        """
        Get the heartbeat widget.

        This must be added to the page layout for browser heartbeats to work.
        The widget is invisible but its JavaScript runs to send periodic
        heartbeats from the browser.
        """
        return self._heartbeat_widget

    @property
    def session_id(self) -> SessionId:
        """Get this session's ID."""
        return self._session_id
