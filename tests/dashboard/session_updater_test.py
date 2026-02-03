# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for SessionUpdater."""

from ess.livedata.dashboard.notification_queue import (
    NotificationEvent,
    NotificationQueue,
    NotificationType,
)
from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry
from ess.livedata.dashboard.session_updater import SessionUpdater


class TestSessionUpdater:
    def test_periodic_update_runs_without_error(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()

        updater = SessionUpdater(session_id=session_id, session_registry=registry)

        # Session is registered at construction time (not via heartbeat)
        # periodic_update should run without error
        updater.periodic_update()

        # Session should remain active (registered at construction)
        assert registry.is_active(session_id)

    def test_polls_notifications(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()
        queue = NotificationQueue()

        updater = SessionUpdater(
            session_id=session_id,
            session_registry=registry,
            notification_queue=queue,
        )

        # Push notification
        queue.push(
            NotificationEvent(message='Test', notification_type=NotificationType.INFO)
        )

        # Poll picks up notification (even though we can't show it in tests)
        notifications = updater._poll_notifications()
        assert len(notifications) == 1
        assert notifications[0].message == 'Test'

    def test_custom_handler_called_during_periodic_update(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()

        updater = SessionUpdater(session_id=session_id, session_registry=registry)

        # Register custom handler
        calls = []
        updater.register_custom_handler(lambda: calls.append(1))

        # Periodic update should call handler
        updater.periodic_update()

        assert len(calls) == 1

    def test_unregister_custom_handler(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()

        updater = SessionUpdater(session_id=session_id, session_registry=registry)

        calls = []

        def handler() -> None:
            calls.append(1)

        updater.register_custom_handler(handler)

        # Unregister
        updater.unregister_custom_handler(handler)

        # Periodic update should not call handler
        updater.periodic_update()

        assert calls == []

    def test_cleanup_unregisters_from_notification_queue(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()
        queue = NotificationQueue()

        updater = SessionUpdater(
            session_id=session_id,
            session_registry=registry,
            notification_queue=queue,
        )

        # Push notification
        queue.push(NotificationEvent(message='Before cleanup'))

        # Cleanup
        updater.cleanup()

        # After cleanup, session should not receive notifications
        queue.push(NotificationEvent(message='After cleanup'))
        notifications = queue.get_new_events(session_id)
        assert notifications == []

    def test_session_id_property(self):
        session_id = SessionId('test-session')
        registry = SessionRegistry()

        updater = SessionUpdater(session_id=session_id, session_registry=registry)

        assert updater.session_id == session_id

    def test_works_without_optional_services(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()

        # Create with no optional services
        updater = SessionUpdater(session_id=session_id, session_registry=registry)

        # Should not raise
        updater.periodic_update()
