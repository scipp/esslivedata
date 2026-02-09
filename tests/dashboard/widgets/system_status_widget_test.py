# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for system status widget."""

import panel as pn

from ess.livedata.dashboard.service_registry import ServiceRegistry
from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry
from ess.livedata.dashboard.widgets.system_status_widget import SystemStatusWidget


class FakeSessionUpdater:
    """Fake session updater for testing."""

    def __init__(self):
        self.registered_handlers = []

    def register_custom_handler(self, handler):
        self.registered_handlers.append(handler)


class TestSystemStatusWidget:
    def test_creates_combined_panel(self):
        session_registry = SessionRegistry()
        service_registry = ServiceRegistry()

        widget = SystemStatusWidget(
            session_registry=session_registry,
            service_registry=service_registry,
            current_session_id=SessionId("my-session"),
        )

        panel = widget.panel()
        assert isinstance(panel, pn.Column)

    def test_panel_contains_session_and_backend_sections(self):
        session_registry = SessionRegistry()
        service_registry = ServiceRegistry()

        widget = SystemStatusWidget(
            session_registry=session_registry,
            service_registry=service_registry,
            current_session_id=SessionId("my-session"),
        )

        panel = widget.panel()
        # Panel should contain 2 items: session widget and backend widget
        assert len(panel) == 2

    def test_session_widget_is_first(self):
        session_registry = SessionRegistry()
        service_registry = ServiceRegistry()

        widget = SystemStatusWidget(
            session_registry=session_registry,
            service_registry=service_registry,
            current_session_id=SessionId("my-session"),
        )

        panel = widget.panel()
        first_section = panel[0]
        # First section should be the session widget panel (a Column)
        assert isinstance(first_section, pn.Column)

    def test_register_periodic_refresh(self):
        session_registry = SessionRegistry()
        service_registry = ServiceRegistry()

        widget = SystemStatusWidget(
            session_registry=session_registry,
            service_registry=service_registry,
            current_session_id=SessionId("my-session"),
        )

        fake_updater = FakeSessionUpdater()
        widget.register_periodic_refresh(fake_updater)

        # Should have registered a handler
        assert len(fake_updater.registered_handlers) == 1

        # The handler should be the session widget's refresh method
        handler = fake_updater.registered_handlers[0]
        assert handler == widget._session_widget.refresh
