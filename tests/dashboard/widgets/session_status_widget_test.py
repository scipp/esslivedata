# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for session status widget."""

import panel as pn
import pytest

from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry
from ess.livedata.dashboard.widgets.session_status_widget import (
    SessionStatusRow,
    SessionStatusWidget,
    _format_heartbeat_age,
)


class TestFormatHeartbeatAge:
    def test_none_returns_unknown(self):
        assert _format_heartbeat_age(None) == "Unknown"

    def test_less_than_one_second(self):
        assert _format_heartbeat_age(0.5) == "Just now"

    def test_one_second(self):
        assert _format_heartbeat_age(1.0) == "1 second ago"

    def test_multiple_seconds(self):
        assert _format_heartbeat_age(30.0) == "30 seconds ago"

    def test_minutes(self):
        assert _format_heartbeat_age(90.0) == "1.5 minutes ago"


class TestSessionStatusRow:
    def test_creates_panel_row(self):
        row = SessionStatusRow(
            session_id=SessionId("test-session-id"),
            is_current_session=False,
            seconds_since_heartbeat=1.0,
        )
        assert isinstance(row.panel, pn.Row)

    def test_current_session_shows_you_badge(self):
        row = SessionStatusRow(
            session_id=SessionId("test-session-id"),
            is_current_session=True,
            seconds_since_heartbeat=1.0,
        )
        # The status pane should contain "You"
        assert "You" in row._status_pane.object

    def test_other_session_shows_active_badge(self):
        row = SessionStatusRow(
            session_id=SessionId("test-session-id"),
            is_current_session=False,
            seconds_since_heartbeat=1.0,
        )
        assert "Active" in row._status_pane.object

    def test_stale_session_shows_stale_badge(self):
        row = SessionStatusRow(
            session_id=SessionId("test-session-id"),
            is_current_session=False,
            seconds_since_heartbeat=20.0,  # > 15 second threshold
        )
        assert "Stale" in row._status_pane.object

    def test_current_session_never_shows_stale(self):
        # Current session should show "You" even with old heartbeat
        row = SessionStatusRow(
            session_id=SessionId("test-session-id"),
            is_current_session=True,
            seconds_since_heartbeat=100.0,
        )
        assert "You" in row._status_pane.object
        assert "Stale" not in row._status_pane.object

    def test_update_changes_content(self):
        row = SessionStatusRow(
            session_id=SessionId("session-1"),
            is_current_session=False,
            seconds_since_heartbeat=1.0,
        )
        assert "Active" in row._status_pane.object

        row.update(
            session_id=SessionId("session-1"),
            is_current_session=True,
            seconds_since_heartbeat=2.0,
        )
        assert "You" in row._status_pane.object

    def test_update_to_stale_status(self):
        row = SessionStatusRow(
            session_id=SessionId("session-1"),
            is_current_session=False,
            seconds_since_heartbeat=1.0,
        )
        assert "Active" in row._status_pane.object

        row.update(
            session_id=SessionId("session-1"),
            is_current_session=False,
            seconds_since_heartbeat=20.0,  # > 15 second threshold
        )
        assert "Stale" in row._status_pane.object


class TestSessionStatusWidget:
    @pytest.fixture
    def registry(self):
        return SessionRegistry()

    def test_creates_panel(self, registry):
        widget = SessionStatusWidget(
            session_registry=registry,
            current_session_id=SessionId("my-session"),
        )
        panel = widget.panel()
        assert isinstance(panel, pn.Column)

    def test_shows_empty_state_initially(self, registry):
        widget = SessionStatusWidget(
            session_registry=registry,
            current_session_id=SessionId("my-session"),
        )
        assert "No active sessions" in widget._summary.object

    def test_shows_session_count_after_register(self, registry):
        current_id = SessionId("my-session")
        registry.register(current_id)
        registry.register(SessionId("other-session"))

        widget = SessionStatusWidget(
            session_registry=registry,
            current_session_id=current_id,
        )
        assert "2" in widget._summary.object
        assert "including you" in widget._summary.object

    def test_shows_just_you_when_alone(self, registry):
        current_id = SessionId("my-session")
        registry.register(current_id)

        widget = SessionStatusWidget(
            session_registry=registry,
            current_session_id=current_id,
        )
        assert "just you" in widget._summary.object

    def test_updates_on_session_register(self, registry):
        current_id = SessionId("my-session")
        widget = SessionStatusWidget(
            session_registry=registry,
            current_session_id=current_id,
        )
        assert "No active sessions" in widget._summary.object

        registry.register(current_id)
        # Subscriber should have updated the widget
        assert "1" in widget._summary.object

    def test_updates_on_session_unregister(self, registry):
        current_id = SessionId("my-session")
        other_id = SessionId("other-session")
        registry.register(current_id)
        registry.register(other_id)

        widget = SessionStatusWidget(
            session_registry=registry,
            current_session_id=current_id,
        )
        assert "2" in widget._summary.object

        registry.unregister(other_id)
        assert "1" in widget._summary.object

    def test_current_session_sorted_first(self, registry):
        current_id = SessionId("zzz-my-session")
        other_id = SessionId("aaa-other-session")
        registry.register(other_id)
        registry.register(current_id)

        widget = SessionStatusWidget(
            session_registry=registry,
            current_session_id=current_id,
        )

        # Check that current session's row is first in the list
        rows = list(widget._session_rows.values())
        first_row = rows[0]
        assert "You" in first_row._status_pane.object

    def test_refresh_updates_display(self, registry):
        current_id = SessionId("my-session")
        registry.register(current_id)

        widget = SessionStatusWidget(
            session_registry=registry,
            current_session_id=current_id,
        )

        # Register another session
        registry.register(SessionId("other-session"))

        # Refresh should update the display
        widget.refresh()
        assert "2" in widget._summary.object

    def test_summary_shows_stale_count(self, registry):
        current_id = SessionId("my-session")
        registry.register(current_id)

        # Directly manipulate the session info to simulate a stale session
        import time

        stale_id = SessionId("stale-session")
        registry.register(stale_id)
        # Make the session appear stale by backdating its heartbeat (> 15s threshold)
        registry._sessions[stale_id].last_heartbeat = time.monotonic() - 20.0

        widget = SessionStatusWidget(
            session_registry=registry,
            current_session_id=current_id,
        )

        # Summary should indicate stale session
        assert "stale" in widget._summary.object.lower()
