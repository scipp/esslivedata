# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for NotificationQueue."""

from ess.livedata.dashboard.notification_queue import (
    NotificationEvent,
    NotificationQueue,
    NotificationType,
)
from ess.livedata.dashboard.session_registry import SessionId


class TestNotificationQueue:
    def test_push_and_get_events(self):
        queue = NotificationQueue()
        session_id = SessionId('session-1')
        queue.register_session(session_id)

        event = NotificationEvent(
            message='Test', notification_type=NotificationType.INFO
        )
        queue.push(event)

        events = queue.get_new_events(session_id)

        assert len(events) == 1
        assert events[0].message == 'Test'

    def test_get_events_advances_cursor(self):
        queue = NotificationQueue()
        session_id = SessionId('session-1')
        queue.register_session(session_id)

        queue.push(NotificationEvent(message='Event 1'))
        queue.get_new_events(session_id)
        queue.push(NotificationEvent(message='Event 2'))

        events = queue.get_new_events(session_id)

        assert len(events) == 1
        assert events[0].message == 'Event 2'

    def test_unregistered_session_gets_empty_list(self):
        queue = NotificationQueue()
        queue.push(NotificationEvent(message='Test'))

        events = queue.get_new_events(SessionId('unknown'))

        assert events == []

    def test_new_session_only_sees_future_events(self):
        queue = NotificationQueue()

        queue.push(NotificationEvent(message='Before registration'))

        session_id = SessionId('session-1')
        queue.register_session(session_id)

        queue.push(NotificationEvent(message='After registration'))

        events = queue.get_new_events(session_id)

        assert len(events) == 1
        assert events[0].message == 'After registration'

    def test_multiple_sessions_independent_cursors(self):
        queue = NotificationQueue()
        session1 = SessionId('session-1')
        session2 = SessionId('session-2')
        queue.register_session(session1)
        queue.register_session(session2)

        queue.push(NotificationEvent(message='Event 1'))
        queue.get_new_events(session1)  # session1 advances
        queue.push(NotificationEvent(message='Event 2'))

        events1 = queue.get_new_events(session1)
        events2 = queue.get_new_events(session2)

        assert len(events1) == 1
        assert events1[0].message == 'Event 2'
        assert len(events2) == 2

    def test_unregister_session(self):
        queue = NotificationQueue()
        session_id = SessionId('session-1')
        queue.register_session(session_id)
        queue.push(NotificationEvent(message='Test'))

        queue.unregister_session(session_id)
        events = queue.get_new_events(session_id)

        assert events == []

    def test_max_age_events_purges_old(self):
        queue = NotificationQueue(max_age_events=2)
        session_id = SessionId('session-1')
        queue.register_session(session_id)

        queue.push(NotificationEvent(message='Event 1'))
        queue.push(NotificationEvent(message='Event 2'))
        queue.push(NotificationEvent(message='Event 3'))

        events = queue.get_new_events(session_id)

        # Event 1 should be purged, only Events 2 and 3 remain
        assert len(events) == 2
        assert [e.message for e in events] == ['Event 2', 'Event 3']

    def test_clear(self):
        queue = NotificationQueue()
        session_id = SessionId('session-1')
        queue.register_session(session_id)
        queue.push(NotificationEvent(message='Test'))

        queue.clear()

        assert queue.get_new_events(session_id) == []
