# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Widget to display active dashboard session status."""

from __future__ import annotations

from typing import ClassVar

import panel as pn

from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry


class SessionUIConstants:
    """Constants for session status UI styling and sizing."""

    # Colors
    ACTIVE_COLOR = "#28a745"  # Green
    YOU_COLOR = "#007bff"  # Blue for "this is you"
    DEFAULT_COLOR = "#6c757d"  # Gray

    # Sizes
    SESSION_ID_WIDTH = 200
    STATUS_WIDTH = 120
    HEARTBEAT_WIDTH = 150
    ROW_HEIGHT = 35

    # Margins
    STANDARD_MARGIN: ClassVar[tuple[int, int]] = (5, 5)
    HEADER_MARGIN: ClassVar[tuple[int, int, int, int]] = (10, 10, 5, 10)


def _format_heartbeat_age(seconds: float | None) -> str:
    """Format heartbeat age in human-readable form."""
    if seconds is None:
        return "Unknown"

    if seconds < 1:
        return "Just now"
    elif seconds < 60:
        value = int(seconds)
        unit = "second" if value == 1 else "seconds"
        return f"{value} {unit} ago"
    else:
        value = seconds / 60
        return f"{value:.1f} minutes ago"


class SessionStatusRow:
    """Widget to display the status of a single session.

    Uses stable pane references that are updated in-place to avoid flicker.
    """

    def __init__(
        self,
        session_id: SessionId,
        is_current_session: bool,
        seconds_since_heartbeat: float | None,
    ) -> None:
        # Create stable pane references
        self._session_id_pane = pn.pane.HTML(
            width=SessionUIConstants.SESSION_ID_WIDTH,
            height=SessionUIConstants.ROW_HEIGHT,
            margin=SessionUIConstants.STANDARD_MARGIN,
        )
        self._status_pane = pn.pane.HTML(
            width=SessionUIConstants.STATUS_WIDTH,
            height=SessionUIConstants.ROW_HEIGHT,
            margin=SessionUIConstants.STANDARD_MARGIN,
        )
        self._heartbeat_pane = pn.pane.HTML(
            width=SessionUIConstants.HEARTBEAT_WIDTH,
            height=SessionUIConstants.ROW_HEIGHT,
            margin=SessionUIConstants.STANDARD_MARGIN,
        )

        self._panel = pn.Row(
            self._session_id_pane,
            self._status_pane,
            self._heartbeat_pane,
            styles={"border-bottom": "1px solid #dee2e6"},
            sizing_mode="stretch_width",
        )

        # Set initial content
        self.update(session_id, is_current_session, seconds_since_heartbeat)

    def update(
        self,
        session_id: SessionId,
        is_current_session: bool,
        seconds_since_heartbeat: float | None,
    ) -> None:
        """Update the row content in-place."""
        # Session ID (truncated)
        session_id_short = session_id[:12]
        self._session_id_pane.object = f"<code>{session_id_short}</code>"

        # Status indicator
        if is_current_session:
            color = SessionUIConstants.YOU_COLOR
            status_text = "You"
        else:
            color = SessionUIConstants.ACTIVE_COLOR
            status_text = "Active"

        status_style = (
            f"background-color: {color}; color: white; "
            f"padding: 2px 8px; border-radius: 3px; text-align: center; "
            f"font-weight: bold; font-size: 11px;"
        )
        self._status_pane.object = f'<div style="{status_style}">{status_text}</div>'

        # Last heartbeat
        heartbeat_text = _format_heartbeat_age(seconds_since_heartbeat)
        self._heartbeat_pane.object = f"<span>{heartbeat_text}</span>"

    @property
    def panel(self) -> pn.Row:
        """Get the panel for this widget."""
        return self._panel


class SessionStatusWidget:
    """Widget to display a list of active dashboard sessions."""

    def __init__(
        self,
        session_registry: SessionRegistry,
        current_session_id: SessionId,
    ) -> None:
        self._session_registry = session_registry
        self._current_session_id = current_session_id
        self._session_rows: dict[SessionId, SessionStatusRow] = {}
        self._empty_placeholder: pn.pane.HTML | None = None
        self._setup_layout()

        # Subscribe to session registry updates
        self._session_registry.register_update_subscriber(self._on_status_update)

    def _setup_layout(self) -> None:
        """Set up the main layout."""
        self._header = pn.pane.HTML(
            "<h3>Dashboard Sessions</h3>",
            margin=SessionUIConstants.HEADER_MARGIN,
        )

        # Summary row
        self._summary = pn.pane.HTML(
            self._format_summary(),
            margin=(5, 10),
        )

        # Table header
        header_style = (
            "font-weight: bold; font-size: 12px; "
            "background-color: #f8f9fa; padding: 5px 0;"
        )
        self._table_header = pn.Row(
            pn.pane.HTML(
                f'<span style="{header_style}">Session ID</span>',
                width=SessionUIConstants.SESSION_ID_WIDTH,
                margin=SessionUIConstants.STANDARD_MARGIN,
            ),
            pn.pane.HTML(
                f'<span style="{header_style}">Status</span>',
                width=SessionUIConstants.STATUS_WIDTH,
                margin=SessionUIConstants.STANDARD_MARGIN,
            ),
            pn.pane.HTML(
                f'<span style="{header_style}">Last Heartbeat</span>',
                width=SessionUIConstants.HEARTBEAT_WIDTH,
                margin=SessionUIConstants.STANDARD_MARGIN,
            ),
            styles={"border-bottom": "2px solid #dee2e6"},
            sizing_mode="stretch_width",
            margin=(0, 10),
        )

        self._session_list = pn.Column(sizing_mode="stretch_width", margin=(0, 10))

        # Initialize with current sessions
        self._update_session_list()

    def _format_summary(self) -> str:
        """Format the summary text."""
        total = self._session_registry.session_count
        is_current_active = self._session_registry.is_active(self._current_session_id)

        if total == 0:
            return "<i>No active sessions</i>"

        if total == 1 and is_current_active:
            return "<b>1</b> active session (just you)"

        you_text = " (including you)" if is_current_active else ""
        session_word = "session" if total == 1 else "sessions"
        return f"<b>{total}</b> active {session_word}{you_text}"

    def _on_status_update(self) -> None:
        """Handle session status updates from the registry."""
        with pn.io.hold():
            self._summary.object = self._format_summary()
            self._update_session_list()

    def _update_session_list(self) -> None:
        """Update the session list, reusing existing rows where possible."""
        current_sessions = set(self._session_registry.get_active_sessions())
        existing_keys = set(self._session_rows.keys())

        # Remove rows for sessions that no longer exist
        removed_keys = existing_keys - current_sessions
        for session_id in removed_keys:
            row = self._session_rows.pop(session_id)
            self._session_list.remove(row.panel)

        # Remove empty placeholder if we have sessions
        if current_sessions and self._empty_placeholder is not None:
            self._session_list.remove(self._empty_placeholder)
            self._empty_placeholder = None

        # Sort sessions: current session first, then by ID
        sorted_sessions = sorted(
            current_sessions,
            key=lambda s: (s != self._current_session_id, s),
        )

        # Update existing rows and add new ones
        for session_id in sorted_sessions:
            is_current = session_id == self._current_session_id
            seconds_ago = self._session_registry.get_seconds_since_heartbeat(session_id)

            if session_id in self._session_rows:
                # Update existing row in-place
                self._session_rows[session_id].update(
                    session_id, is_current, seconds_ago
                )
            else:
                # Create new row
                row = SessionStatusRow(session_id, is_current, seconds_ago)
                self._session_rows[session_id] = row
                self._session_list.append(row.panel)

        # Show empty placeholder if no sessions
        if not current_sessions and self._empty_placeholder is None:
            color = SessionUIConstants.DEFAULT_COLOR
            self._empty_placeholder = pn.pane.HTML(
                f'<div style="text-align: center; padding: 20px; color: {color};">'
                "No active sessions"
                "</div>",
                sizing_mode="stretch_width",
            )
            self._session_list.append(self._empty_placeholder)

    def panel(self) -> pn.Column:
        """Get the main panel for this widget."""
        return pn.Column(
            self._header,
            self._summary,
            self._table_header,
            self._session_list,
            sizing_mode="stretch_width",
        )
