# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for system status widget."""

import panel as pn

from ess.livedata.dashboard.service_registry import ServiceRegistry
from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry
from ess.livedata.dashboard.widgets.system_status_widget import SystemStatusWidget


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
        # Panel should contain 3 items: session widget, divider, backend widget
        assert len(panel) == 3

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

    def test_divider_between_sections(self):
        session_registry = SessionRegistry()
        service_registry = ServiceRegistry()

        widget = SystemStatusWidget(
            session_registry=session_registry,
            service_registry=service_registry,
            current_session_id=SessionId("my-session"),
        )

        panel = widget.panel()
        divider = panel[1]
        # Divider is an HTML pane with an <hr> element
        assert isinstance(divider, pn.pane.HTML)
        assert "<hr" in divider.object
