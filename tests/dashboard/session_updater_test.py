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


def _make_updater(
    session_id: SessionId,
    registry: SessionRegistry,
    *,
    notification_queue: NotificationQueue | None = None,
    username: str | None = None,
) -> SessionUpdater:
    return SessionUpdater(
        session_id=session_id,
        session_registry=registry,
        notification_queue=notification_queue or NotificationQueue(),
        username=username,
    )


class TestSessionUpdater:
    def test_periodic_update_runs_without_error(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()

        updater = _make_updater(session_id, registry)

        # Session is registered at construction time (not via heartbeat)
        # periodic_update should run without error
        updater.periodic_update()

        # Session should remain active (registered at construction)
        assert registry.is_active(session_id)

    def test_polls_notifications(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()
        queue = NotificationQueue()

        updater = _make_updater(session_id, registry, notification_queue=queue)

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

        updater = _make_updater(session_id, registry)

        # Register custom handler
        calls = []
        updater.register_custom_handler(lambda: calls.append(1))

        # Periodic update should call handler
        updater.periodic_update()

        assert len(calls) == 1

    def test_unregister_custom_handler(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()

        updater = _make_updater(session_id, registry)

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

        updater = _make_updater(session_id, registry, notification_queue=queue)

        # Push notification
        queue.push(NotificationEvent(message='Before cleanup'))

        # Cleanup
        updater.cleanup()

        # After cleanup, session should not receive notifications
        queue.push(NotificationEvent(message='After cleanup'))
        notifications = queue.get_new_events(session_id)
        assert notifications == []

    def test_cleanup_handler_called_on_cleanup(self):
        registry = SessionRegistry()
        updater = _make_updater(SessionId('session-1'), registry)

        calls = []
        updater.register_cleanup_handler(lambda: calls.append(1))

        updater.cleanup()

        assert calls == [1]

    def test_cleanup_handler_invoked_by_stale_session_reaper(self):
        # The heartbeat-based stale reaper must run cleanup handlers even when
        # Panel's on_session_destroyed never fires. Negative timeout makes the
        # session stale immediately, regardless of elapsed time.
        registry = SessionRegistry(stale_timeout_seconds=-1.0)
        session_id = SessionId('session-1')
        updater = _make_updater(session_id, registry)

        calls = []
        updater.register_cleanup_handler(lambda: calls.append(1))

        cleaned = registry.cleanup_stale_sessions()

        assert session_id in cleaned
        assert calls == [1]

    def test_cleanup_handler_exception_does_not_block_others(self):
        registry = SessionRegistry()
        updater = _make_updater(SessionId('session-1'), registry)

        calls = []

        def boom() -> None:
            raise RuntimeError("boom")

        updater.register_cleanup_handler(boom)
        updater.register_cleanup_handler(lambda: calls.append(1))

        updater.cleanup()  # must not raise

        assert calls == [1]

    def test_session_id_property(self):
        session_id = SessionId('test-session')
        registry = SessionRegistry()

        updater = _make_updater(session_id, registry)

        assert updater.session_id == session_id

    def test_cleanup_stops_periodic_callback(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()

        updater = _make_updater(session_id, registry)

        class FakeCallback:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        callback = FakeCallback()
        updater.set_periodic_callback(callback)

        updater.cleanup()

        assert callback.stopped

    def test_username_forwarded_to_registry(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()

        _make_updater(session_id, registry, username='Simon')

        info = registry.get_session_info(session_id)
        assert info is not None
        assert info.username == 'Simon'
