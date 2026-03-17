# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for system status widget."""

import panel as pn

from ess.livedata.dashboard.notification_queue import NotificationQueue
from ess.livedata.dashboard.service_registry import ServiceRegistry
from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry
from ess.livedata.dashboard.widgets.system_status_widget import SystemStatusWidget


class FakeSessionUpdater:
    """Fake session updater for testing."""

    def __init__(self):
        self.registered_handlers = []

    def register_custom_handler(self, handler):
        self.registered_handlers.append(handler)


def _make_widget() -> SystemStatusWidget:
    return SystemStatusWidget(
        session_registry=SessionRegistry(),
        service_registry=ServiceRegistry(),
        current_session_id=SessionId("my-session"),
        notification_queue=NotificationQueue(),
    )


class TestSystemStatusWidget:
    def test_creates_combined_panel(self):
        widget = _make_widget()
        panel = widget.panel()
        assert isinstance(panel, pn.Column)

    def test_panel_contains_notification_session_and_backend_sections(self):
        widget = _make_widget()
        panel = widget.panel()
        # Panel should contain 3 items: notification, session, backend
        assert len(panel) == 3

    def test_notification_widget_is_first(self):
        widget = _make_widget()
        panel = widget.panel()
        first_section = panel[0]
        # First section should be the notification log pane (an HTML pane)
        assert isinstance(first_section, pn.pane.HTML)

    def test_register_periodic_refresh(self):
        widget = _make_widget()
        fake_updater = FakeSessionUpdater()
        widget.register_periodic_refresh(fake_updater)

        # Should have registered a single handler for the composite refresh
        assert len(fake_updater.registered_handlers) == 1

    def test_is_visible_false_skips_refresh(self):
        widget = _make_widget()
        fake_updater = FakeSessionUpdater()
        widget.register_periodic_refresh(fake_updater, is_visible=lambda: False)

        # Call the registered handler — it should not raise
        handler = fake_updater.registered_handlers[0]
        handler()

    def test_is_visible_true_runs_refresh(self):
        widget = _make_widget()
        fake_updater = FakeSessionUpdater()
        widget.register_periodic_refresh(fake_updater, is_visible=lambda: True)

        handler = fake_updater.registered_handlers[0]
        handler()  # Should not raise
