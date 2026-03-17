# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Combined widget displaying system status: sessions, backend workers, and
notifications."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import panel as pn

from ess.livedata.dashboard.notification_queue import NotificationQueue
from ess.livedata.dashboard.service_registry import ServiceRegistry
from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry

from .backend_status_widget import BackendStatusWidget
from .notification_log_widget import NotificationLogWidget
from .session_status_widget import SessionStatusWidget

if TYPE_CHECKING:
    from ess.livedata.dashboard.session_updater import SessionUpdater


class SystemStatusWidget:
    """
    Combined widget displaying system status.

    Organizes notification log, session status, and backend worker status
    into a unified view with clear section separation.
    """

    def __init__(
        self,
        session_registry: SessionRegistry,
        service_registry: ServiceRegistry,
        current_session_id: SessionId,
        notification_queue: NotificationQueue,
    ) -> None:
        self._notification_widget = NotificationLogWidget(
            notification_queue=notification_queue,
        )
        self._session_widget = SessionStatusWidget(
            session_registry=session_registry,
            current_session_id=current_session_id,
        )
        self._backend_widget = BackendStatusWidget(
            service_registry=service_registry,
        )
        self._is_visible: Callable[[], bool] | None = None

    def register_periodic_refresh(
        self,
        session_updater: SessionUpdater,
        *,
        is_visible: Callable[[], bool] | None = None,
    ) -> None:
        """
        Register for periodic refresh of system status displays.

        Parameters
        ----------
        session_updater:
            The session updater to register the refresh handler with.
        is_visible:
            Optional predicate returning whether this tab is currently visible.
            When provided, refreshes are skipped while the tab is hidden.
        """
        self._is_visible = is_visible
        session_updater.register_custom_handler(self._refresh_all)

    def _refresh_all(self) -> None:
        """Refresh all sub-widgets, gated by visibility predicate."""
        if self._is_visible is not None and not self._is_visible():
            return
        self._notification_widget.refresh()
        self._session_widget.refresh()
        self._backend_widget.refresh()

    def panel(self) -> pn.Column:
        """Get the main panel for this widget."""
        return pn.Column(
            self._notification_widget.panel(),
            self._session_widget.panel(),
            self._backend_widget.panel(),
            sizing_mode="stretch_width",
        )
