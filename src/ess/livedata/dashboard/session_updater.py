# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
SessionUpdater - Per-session component that drives all widget updates.

This module implements the polling-based update model for multi-session support.
Each browser session has its own SessionUpdater that polls for changes from
shared services in the correct session context.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import panel as pn

from .session_registry import SessionId, SessionRegistry
from .state_stores import (
    LayerId,
    NotificationEvent,
    NotificationQueue,
    NotificationType,
    PlotDataService,
    PlotLayerState,
    StateKey,
    WidgetStateStore,
)

logger = logging.getLogger(__name__)


class SessionUpdater:
    """
    Per-session component that drives all widget updates.

    Each browser session creates its own SessionUpdater instance. The updater
    polls shared services (WidgetStateStore, PlotDataService, NotificationQueue)
    in its periodic callback, ensuring all session-bound components are created
    and updated in the correct session context.

    Parameters
    ----------
    session_id:
        Unique identifier for this session.
    session_registry:
        Registry for session heartbeats and tracking.
    widget_state_store:
        Shared store for widget state changes.
    plot_data_service:
        Shared service for plot data with version tracking.
    notification_queue:
        Shared queue for notifications.
    """

    def __init__(
        self,
        *,
        session_id: SessionId,
        session_registry: SessionRegistry,
        widget_state_store: WidgetStateStore | None = None,
        plot_data_service: PlotDataService | None = None,
        notification_queue: NotificationQueue | None = None,
    ) -> None:
        self._session_id = session_id
        self._session_registry = session_registry
        self._widget_state_store = widget_state_store
        self._plot_data_service = plot_data_service
        self._notification_queue = notification_queue

        # Version tracking for polling
        self._last_widget_version = 0
        self._last_plot_versions: dict[LayerId, int] = {}

        # Callbacks for applying updates
        self._widget_change_handlers: dict[StateKey, Callable[[Any], None]] = {}
        self._plot_update_handlers: dict[LayerId, Callable[[PlotLayerState], None]] = {}

        # Register with notification queue
        if self._notification_queue is not None:
            self._notification_queue.register_session(session_id)

        logger.debug("SessionUpdater created for session %s", session_id)

    def register_widget_handler(
        self, key: StateKey, handler: Callable[[Any], None]
    ) -> None:
        """
        Register a handler for widget state changes.

        Parameters
        ----------
        key:
            State key to watch.
        handler:
            Callback to invoke when state changes.
        """
        self._widget_change_handlers[key] = handler

    def unregister_widget_handler(self, key: StateKey) -> None:
        """Unregister a widget state handler."""
        self._widget_change_handlers.pop(key, None)

    def register_plot_handler(
        self, layer_id: LayerId, handler: Callable[[PlotLayerState], None]
    ) -> None:
        """
        Register a handler for plot data updates.

        Parameters
        ----------
        layer_id:
            Layer ID to watch.
        handler:
            Callback to invoke when plot state changes.
        """
        self._plot_update_handlers[layer_id] = handler
        # Initialize version tracking for this layer
        if self._plot_data_service is not None:
            self._last_plot_versions[layer_id] = self._plot_data_service.get_version(
                layer_id
            )

    def unregister_plot_handler(self, layer_id: LayerId) -> None:
        """Unregister a plot data handler."""
        self._plot_update_handlers.pop(layer_id, None)
        self._last_plot_versions.pop(layer_id, None)

    def periodic_update(self) -> None:
        """
        Called from this session's periodic callback.

        Polls all shared services for changes and applies updates
        in a single batched UI update.
        """
        # Send heartbeat to registry
        self._session_registry.heartbeat(self._session_id)

        # Poll for changes
        widget_changes = self._poll_widget_state()
        plot_updates = self._poll_plot_updates()
        notifications = self._poll_notifications()

        # Apply changes in a single batched update
        if widget_changes or plot_updates or notifications:
            with pn.io.hold():
                self._apply_widget_state_changes(widget_changes)
                self._apply_plot_updates(plot_updates)
                self._show_notifications(notifications)

    def _poll_widget_state(self) -> dict[StateKey, Any]:
        """Poll WidgetStateStore for changes since last update."""
        if self._widget_state_store is None:
            return {}

        current_version, changes = self._widget_state_store.get_changes_since(
            self._last_widget_version
        )
        self._last_widget_version = current_version

        # Filter to only keys we have handlers for
        return {k: v for k, v in changes.items() if k in self._widget_change_handlers}

    def _poll_plot_updates(self) -> dict[LayerId, PlotLayerState]:
        """Poll PlotDataService for updated layers."""
        if self._plot_data_service is None:
            return {}

        updates = self._plot_data_service.get_updates_since(self._last_plot_versions)

        # Update version tracking
        for layer_id, state in updates.items():
            self._last_plot_versions[layer_id] = state.version

        # Filter to only layers we have handlers for
        return {k: v for k, v in updates.items() if k in self._plot_update_handlers}

    def _poll_notifications(self) -> list[NotificationEvent]:
        """Poll NotificationQueue for new events."""
        if self._notification_queue is None:
            return []

        return self._notification_queue.get_new_events(self._session_id)

    def _apply_widget_state_changes(self, changes: dict[StateKey, Any]) -> None:
        """Apply widget state changes by invoking registered handlers."""
        for key, value in changes.items():
            handler = self._widget_change_handlers.get(key)
            if handler is not None:
                try:
                    handler(value)
                except Exception:
                    logger.exception(
                        "Error in widget handler for %s in session %s",
                        key,
                        self._session_id,
                    )

    def _apply_plot_updates(self, updates: dict[LayerId, PlotLayerState]) -> None:
        """Apply plot updates by invoking registered handlers."""
        for layer_id, state in updates.items():
            handler = self._plot_update_handlers.get(layer_id)
            if handler is not None:
                try:
                    handler(state)
                except Exception:
                    logger.exception(
                        "Error in plot handler for %s in session %s",
                        layer_id,
                        self._session_id,
                    )

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

        self._widget_change_handlers.clear()
        self._plot_update_handlers.clear()
        self._last_plot_versions.clear()

        logger.debug("SessionUpdater cleaned up for session %s", self._session_id)

    @property
    def session_id(self) -> SessionId:
        """Get this session's ID."""
        return self._session_id
