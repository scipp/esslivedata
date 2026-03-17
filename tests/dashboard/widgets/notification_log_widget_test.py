# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for NotificationLogWidget."""

from ess.livedata.dashboard.notification_queue import (
    NotificationEvent,
    NotificationQueue,
    NotificationType,
)
from ess.livedata.dashboard.widgets.notification_log_widget import (
    NotificationLogWidget,
)


class TestNotificationLogWidget:
    def test_empty_queue_renders_placeholder(self):
        queue = NotificationQueue()
        widget = NotificationLogWidget(notification_queue=queue)

        widget.refresh()

        html = widget.panel().object
        assert "No notifications" in html

    def test_renders_events(self):
        queue = NotificationQueue()
        queue.push(
            NotificationEvent(
                message="Something failed",
                notification_type=NotificationType.ERROR,
            )
        )
        queue.push(
            NotificationEvent(
                message="All good",
                notification_type=NotificationType.SUCCESS,
            )
        )
        widget = NotificationLogWidget(notification_queue=queue)

        widget.refresh()

        html = widget.panel().object
        assert "Something failed" in html
        assert "All good" in html
        assert "ERROR" in html
        assert "SUCCESS" in html
        assert "2 notifications" in html

    def test_newest_first_ordering(self):
        queue = NotificationQueue()
        queue.push(NotificationEvent(message="First"))
        queue.push(NotificationEvent(message="Second"))
        widget = NotificationLogWidget(notification_queue=queue)

        widget.refresh()

        html = widget.panel().object
        # "Second" should appear before "First" in the rendered HTML
        assert html.index("Second") < html.index("First")

    def test_version_based_skip(self):
        queue = NotificationQueue()
        queue.push(NotificationEvent(message="Event"))
        widget = NotificationLogWidget(notification_queue=queue)

        widget.refresh()
        html_after_first = widget.panel().object

        # Refresh again without new events: content should remain identical
        widget.refresh()
        assert widget.panel().object is html_after_first

    def test_refreshes_when_version_changes(self):
        queue = NotificationQueue()
        widget = NotificationLogWidget(notification_queue=queue)
        widget.refresh()

        queue.push(
            NotificationEvent(
                message="New event",
                notification_type=NotificationType.WARNING,
            )
        )
        widget.refresh()

        html = widget.panel().object
        assert "New event" in html
        assert "WARNING" in html

    def test_renders_mixed_types(self):
        queue = NotificationQueue()
        for ntype in NotificationType:
            queue.push(
                NotificationEvent(message=f"msg-{ntype.value}", notification_type=ntype)
            )
        widget = NotificationLogWidget(notification_queue=queue)

        widget.refresh()

        html = widget.panel().object
        for ntype in NotificationType:
            assert ntype.value.upper() in html
