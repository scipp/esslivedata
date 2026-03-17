# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Widget displaying a scrollable log of notification events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar

import panel as pn

from ess.livedata.dashboard.notification_queue import (
    NotificationQueue,
    NotificationType,
)

from .styles import StatusColors

_BADGE_COLORS: dict[NotificationType, str] = {
    NotificationType.ERROR: StatusColors.ERROR,
    NotificationType.WARNING: StatusColors.WARNING,
    NotificationType.SUCCESS: StatusColors.SUCCESS,
    NotificationType.INFO: StatusColors.INFO,
}


class NotificationLogWidget:
    """Scrollable log of notification events from the shared queue.

    Renders newest events first. Uses version-based change detection to skip
    unnecessary refreshes.
    """

    MAX_DISPLAY_EVENTS: ClassVar[int] = 50

    def __init__(self, notification_queue: NotificationQueue) -> None:
        self._queue = notification_queue
        self._last_version: int | None = None
        self._header = pn.pane.HTML(
            "<h3>Notifications</h3>",
            margin=(10, 10, 5, 10),
        )
        self._content_pane = pn.pane.HTML(
            self._render_empty(),
            sizing_mode="stretch_width",
            margin=(0, 10),
        )

    def refresh(self) -> None:
        """Refresh the log display if the queue has changed."""
        version = self._queue.version
        if version == self._last_version:
            return
        self._last_version = version

        events = self._queue.get_all_events()
        if not events:
            self._content_pane.object = self._render_empty()
            return

        # Newest first, capped
        display_events = list(reversed(events))[: self.MAX_DISPLAY_EVENTS]

        rows = []
        for event in display_events:
            color = _BADGE_COLORS.get(event.notification_type, StatusColors.INFO)
            label = event.notification_type.value.upper()
            ts = datetime.fromtimestamp(event.timestamp, tz=timezone.utc)
            time_str = ts.strftime("%H:%M:%S")

            badge = (
                f'<span style="background-color: {color}; color: white; '
                f'padding: 1px 6px; border-radius: 3px; font-size: 10px; '
                f'font-weight: bold; display: inline-block; min-width: 55px; '
                f'text-align: center;">{label}</span>'
            )
            message_html = (
                f'<pre style="margin: 0; white-space: pre-wrap; '
                f'word-break: break-word; max-height: 150px; overflow-y: auto; '
                f'font-size: 12px; line-height: 1.4;">{event.message}</pre>'
            )
            rows.append(
                f'<div style="padding: 6px 8px; border-bottom: 1px solid #dee2e6;">'
                f'<span style="color: #6c757d; font-size: 11px; '
                f'margin-right: 8px;">{time_str}</span>'
                f'{badge} {message_html}</div>'
            )

        self._content_pane.object = (
            f'<div style="max-height: 400px; overflow-y: auto; '
            f'border: 1px solid #dee2e6; border-radius: 4px;">'
            f'{"".join(rows)}</div>'
        )

    def _render_empty(self) -> str:
        return (
            '<div style="text-align: center; padding: 20px; '
            f'color: {StatusColors.MUTED};">No notifications</div>'
        )

    def panel(self) -> pn.Column:
        """Get the panel for this widget."""
        return pn.Column(
            self._header,
            self._content_pane,
            sizing_mode="stretch_width",
        )
