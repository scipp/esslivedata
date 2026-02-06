# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for HeartbeatWidget."""

from ess.livedata.dashboard.widgets.heartbeat_widget import HeartbeatWidget


class TestHeartbeatWidget:
    def test_creates_widget(self):
        widget = HeartbeatWidget()
        assert widget is not None

    def test_default_interval(self):
        widget = HeartbeatWidget()
        assert widget.interval_ms == 5000

    def test_custom_interval(self):
        widget = HeartbeatWidget(interval_ms=10000)
        assert widget.interval_ms == 10000

    def test_counter_starts_at_zero(self):
        widget = HeartbeatWidget()
        assert widget.counter == 0

    def test_widget_is_invisible(self):
        widget = HeartbeatWidget()
        assert widget.visible is False

    def test_widget_has_zero_dimensions(self):
        widget = HeartbeatWidget()
        assert widget.width == 0
        assert widget.height == 0

    def test_has_render_script(self):
        # Verify the JavaScript render script is defined
        assert 'render' in HeartbeatWidget._scripts
        assert 'setInterval' in HeartbeatWidget._scripts['render']

    def test_has_remove_script_for_cleanup(self):
        # Verify cleanup script is defined to prevent memory leaks
        assert 'remove' in HeartbeatWidget._scripts
        assert 'clearInterval' in HeartbeatWidget._scripts['remove']
