# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Combined widget displaying system status: sessions and backend workers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import panel as pn

from ess.livedata.dashboard.service_registry import ServiceRegistry
from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry

from .backend_status_widget import BackendStatusWidget
from .session_status_widget import SessionStatusWidget

if TYPE_CHECKING:
    from ess.livedata.dashboard.session_updater import SessionUpdater


class SystemStatusWidget:
    """
    Combined widget displaying system status.

    Organizes session status and backend worker status into a unified view
    with clear section separation.
    """

    def __init__(
        self,
        session_registry: SessionRegistry,
        service_registry: ServiceRegistry,
        current_session_id: SessionId,
    ) -> None:
        self._session_widget = SessionStatusWidget(
            session_registry=session_registry,
            current_session_id=current_session_id,
        )
        self._backend_widget = BackendStatusWidget(
            service_registry=service_registry,
        )

    def register_periodic_refresh(self, session_updater: SessionUpdater) -> None:
        """
        Register for periodic refresh of session status display.

        This keeps heartbeat times and stale status indicators current.

        Parameters
        ----------
        session_updater:
            The session updater to register the refresh handler with.
        """
        session_updater.register_custom_handler(self._session_widget.refresh)

    def panel(self) -> pn.Column:
        """Get the main panel for this widget."""
        # Add a divider between sections
        divider = pn.pane.HTML(
            '<hr style="margin: 20px 0; border: 0; ' 'border-top: 1px solid #dee2e6;">',
            sizing_mode="stretch_width",
        )

        return pn.Column(
            self._session_widget.panel(),
            divider,
            self._backend_widget.panel(),
            sizing_mode="stretch_width",
        )
