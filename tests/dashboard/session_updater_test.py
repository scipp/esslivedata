# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for SessionUpdater."""

from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry
from ess.livedata.dashboard.session_updater import SessionUpdater
from ess.livedata.dashboard.state_stores import (
    LayerId,
    NotificationEvent,
    NotificationQueue,
    NotificationType,
    PlotDataService,
)


class TestSessionUpdater:
    def test_periodic_update_sends_heartbeat(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()
        registry.register(session_id)

        updater = SessionUpdater(session_id=session_id, session_registry=registry)

        # Heartbeat should be sent on periodic update
        updater.periodic_update()

        assert registry.is_active(session_id)

    def test_polls_plot_updates(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()
        service = PlotDataService()

        updater = SessionUpdater(
            session_id=session_id,
            session_registry=registry,
            plot_data_service=service,
        )

        # Register handler
        received_states = []
        layer_id = LayerId('layer-1')
        updater.register_plot_handler(layer_id, lambda s: received_states.append(s))

        # Update service
        service.update(layer_id, {'data': 'test'})

        # Poll should pick up change
        updater.periodic_update()

        assert len(received_states) == 1
        assert received_states[0].state == {'data': 'test'}

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

    def test_unregister_plot_handler(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()
        service = PlotDataService()

        updater = SessionUpdater(
            session_id=session_id,
            session_registry=registry,
            plot_data_service=service,
        )

        received_states = []
        layer_id = LayerId('layer-1')
        updater.register_plot_handler(layer_id, lambda s: received_states.append(s))

        # Unregister
        updater.unregister_plot_handler(layer_id)

        # Update service
        service.update(layer_id, {'data': 'test'})

        # Poll should not invoke handler
        updater.periodic_update()

        assert received_states == []

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
