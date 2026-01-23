# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for state stores."""

from ess.livedata.dashboard.session_registry import SessionId
from ess.livedata.dashboard.state_stores import (
    LayerId,
    NotificationEvent,
    NotificationQueue,
    NotificationType,
    PlotDataService,
)


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


class TestPlotDataService:
    def test_update_and_get(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        service.update(layer_id, {'plot': 'data'})
        state = service.get(layer_id)

        assert state is not None
        assert state.state == {'plot': 'data'}
        assert state.version == 1

    def test_get_unknown_layer_returns_none(self):
        service = PlotDataService()
        assert service.get(LayerId('unknown')) is None

    def test_update_increments_version(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        service.update(layer_id, 'v1')
        assert service.get_version(layer_id) == 1

        service.update(layer_id, 'v2')
        assert service.get_version(layer_id) == 2

    def test_get_version_unknown_layer_returns_zero(self):
        service = PlotDataService()
        assert service.get_version(LayerId('unknown')) == 0

    def test_get_updates_since(self):
        service = PlotDataService()
        layer1 = LayerId('layer-1')
        layer2 = LayerId('layer-2')

        service.update(layer1, 'v1')
        checkpoint = {layer1: service.get_version(layer1)}
        service.update(layer2, 'v2')
        service.update(layer1, 'v1-updated')

        updates = service.get_updates_since(checkpoint)

        assert set(updates.keys()) == {layer1, layer2}
        assert updates[layer1].state == 'v1-updated'
        assert updates[layer2].state == 'v2'

    def test_get_updates_since_empty_versions(self):
        service = PlotDataService()
        layer1 = LayerId('layer-1')

        service.update(layer1, 'v1')

        updates = service.get_updates_since({})

        assert layer1 in updates

    def test_remove_layer(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')
        service.update(layer_id, 'v1')

        service.remove(layer_id)

        assert service.get(layer_id) is None

    def test_remove_unknown_layer_is_noop(self):
        service = PlotDataService()
        service.remove(LayerId('unknown'))

    def test_clear(self):
        service = PlotDataService()
        service.update(LayerId('layer-1'), 'v1')

        service.clear()

        assert service.get(LayerId('layer-1')) is None
