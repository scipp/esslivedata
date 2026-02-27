# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Session-local notification helpers for Panel dashboards.

Panel's notification system is session-bound, so these helpers must only be
called from within a session context (e.g. Panel callbacks or periodic callbacks).
For notifications from background threads, use NotificationQueue instead.

Display durations per type (milliseconds):
  - ERROR: 0 (persistent until the user dismisses)
  - WARNING: 8000
  - SUCCESS / INFO: 3000
"""

from __future__ import annotations

from collections.abc import Callable

import panel as pn

from .notification_queue import NotificationEvent, NotificationType

_DURATIONS: dict[NotificationType, int] = {
    NotificationType.ERROR: 0,
    NotificationType.WARNING: 8000,
    NotificationType.SUCCESS: 3000,
    NotificationType.INFO: 3000,
}


def show_info(message: str) -> None:
    """Show an info notification."""
    if pn.state.notifications is not None:
        pn.state.notifications.info(message, duration=_DURATIONS[NotificationType.INFO])


def show_success(message: str) -> None:
    """Show a success notification."""
    if pn.state.notifications is not None:
        pn.state.notifications.success(
            message, duration=_DURATIONS[NotificationType.SUCCESS]
        )


def show_warning(message: str) -> None:
    """Show a warning notification."""
    if pn.state.notifications is not None:
        pn.state.notifications.warning(
            message, duration=_DURATIONS[NotificationType.WARNING]
        )


def show_error(message: str) -> None:
    """Show a persistent error notification that the user must dismiss."""
    if pn.state.notifications is not None:
        pn.state.notifications.error(
            message, duration=_DURATIONS[NotificationType.ERROR]
        )


_DISPATCH: dict[NotificationType, Callable[[str], None]] = {
    NotificationType.INFO: show_info,
    NotificationType.SUCCESS: show_success,
    NotificationType.WARNING: show_warning,
    NotificationType.ERROR: show_error,
}


def show_notification(event: NotificationEvent) -> None:
    """Show a notification for a NotificationEvent, dispatching on its type."""
    _DISPATCH[event.notification_type](event.message)
