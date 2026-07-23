# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for SessionUpdater."""

from collections.abc import Callable

from ess.livedata.dashboard.notification_queue import (
    NotificationEvent,
    NotificationQueue,
    NotificationType,
)
from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry
from ess.livedata.dashboard.session_updater import SessionUpdater
from ess.livedata.dashboard.wakeup_hub import WakeupHub


class FakeDocument:
    """Stand-in for a Bokeh document that records scheduled next-tick callbacks.

    The reaper path marshals document-mutating teardown onto the session's
    IOLoop via ``add_next_tick_callback``. This fake captures those callbacks so
    a test can assert they were scheduled (not run on the calling thread) and
    fire them explicitly.
    """

    def __init__(self, *, raise_on_schedule: bool = False) -> None:
        self.next_tick_callbacks: list[Callable[[], None]] = []
        self._raise_on_schedule = raise_on_schedule

    def add_next_tick_callback(self, callback: Callable[[], None]) -> None:
        if self._raise_on_schedule:
            raise RuntimeError("document already destroyed")
        self.next_tick_callbacks.append(callback)

    def run_next_tick_callbacks(self) -> None:
        callbacks = self.next_tick_callbacks[:]
        self.next_tick_callbacks.clear()
        for callback in callbacks:
            callback()


class FakeCallback:
    """Records whether ``stop`` was called (stands in for a PeriodicCallback)."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _make_updater(
    session_id: SessionId,
    registry: SessionRegistry,
    *,
    notification_queue: NotificationQueue | None = None,
    username: str | None = None,
    document: FakeDocument | None = None,
    wakeup_hub: WakeupHub | None = None,
    clock: Callable[[], float] | None = None,
    on_load: Callable[[Callable[[], None]], None] | None = None,
) -> SessionUpdater:
    kwargs = {}
    if clock is not None:
        kwargs['clock'] = clock
    if on_load is not None:
        kwargs['on_load'] = on_load
    return SessionUpdater(
        session_id=session_id,
        session_registry=registry,
        notification_queue=notification_queue or NotificationQueue(),
        username=username,
        document=document,
        wakeup_hub=wakeup_hub,
        **kwargs,
    )


class TestSessionUpdater:
    def test_housekeeping_tick_runs_without_error(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()

        updater = _make_updater(session_id, registry)

        # Session is registered at construction time (not via heartbeat)
        # housekeeping_tick should run without error
        updater.housekeeping_tick()

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

    def test_custom_handler_called_during_housekeeping_tick(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()

        updater = _make_updater(session_id, registry)

        # Register custom handler
        calls = []
        updater.register_custom_handler(lambda: calls.append(1))

        # Periodic update should call handler
        updater.housekeeping_tick()

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
        updater.housekeeping_tick()

        assert calls == []

    def test_wake_runs_handler_when_predicate_fires(self):
        updater = _make_updater(SessionId('s'), SessionRegistry())
        calls: list[int] = []
        updater.register_custom_handler(lambda: calls.append(1), has_work=lambda: True)

        updater.wake()

        assert calls == [1]

    def test_wake_skips_handler_when_predicate_is_false(self):
        updater = _make_updater(SessionId('s'), SessionRegistry())
        calls: list[int] = []
        updater.register_custom_handler(lambda: calls.append(1), has_work=lambda: False)

        updater.wake()

        assert calls == []

    def test_wake_skips_handler_without_predicate(self):
        """Handlers with no cheap change signal run on full passes only."""
        updater = _make_updater(SessionId('s'), SessionRegistry())
        calls: list[int] = []
        updater.register_custom_handler(lambda: calls.append(1))

        updater.wake()
        assert calls == []
        updater.housekeeping_tick()  # first housekeeping tick is a full pass
        assert calls == [1]

    def test_full_pass_runs_handler_despite_false_predicate(self):
        """The full pass is the safety net: predicates never gate it."""
        updater = _make_updater(SessionId('s'), SessionRegistry())
        calls: list[int] = []
        updater.register_custom_handler(lambda: calls.append(1), has_work=lambda: False)

        updater.housekeeping_tick()  # first housekeeping tick is a full pass

        assert calls == [1]

    def test_housekeeping_between_full_passes_is_predicate_gated(self):
        now = 0.0
        updater = _make_updater(SessionId('s'), SessionRegistry(), clock=lambda: now)
        gated: list[int] = []
        ungated: list[int] = []
        updater.register_custom_handler(lambda: gated.append(1), has_work=lambda: False)
        updater.register_custom_handler(lambda: ungated.append(1))

        updater.housekeeping_tick()  # t=0: full pass
        assert (len(gated), len(ungated)) == (1, 1)

        for t in (1.0, 2.0, 3.0, 4.0):
            now = t
            updater.housekeeping_tick()  # gated: false predicate, no run
        assert (len(gated), len(ungated)) == (1, 1)

        now = 5.0
        updater.housekeeping_tick()  # full-pass interval elapsed
        assert (len(gated), len(ungated)) == (2, 2)

    def test_gated_housekeeping_runs_handler_whose_predicate_fires(self):
        now = 0.0
        updater = _make_updater(SessionId('s'), SessionRegistry(), clock=lambda: now)
        calls: list[int] = []
        updater.register_custom_handler(lambda: calls.append(1), has_work=lambda: True)

        updater.housekeeping_tick()  # t=0: full pass
        now = 1.0
        updater.housekeeping_tick()  # gated, but predicate fires

        assert calls == [1, 1]

    def test_wake_runs_handler_when_predicate_raises(self):
        updater = _make_updater(SessionId('s'), SessionRegistry())
        calls: list[int] = []

        def broken_predicate() -> bool:
            raise RuntimeError("boom")

        updater.register_custom_handler(
            lambda: calls.append(1), has_work=broken_predicate
        )

        updater.wake()

        assert calls == [1]

    def test_registers_with_wakeup_hub_and_unregisters_on_cleanup(self):
        hub = WakeupHub()
        document = FakeDocument()
        updater = _make_updater(
            SessionId('s'), SessionRegistry(), document=document, wakeup_hub=hub
        )
        calls: list[int] = []
        updater.register_custom_handler(lambda: calls.append(1), has_work=lambda: True)

        hub.wake_all()
        document.run_next_tick_callbacks()
        assert calls == [1]

        updater.cleanup()
        hub.wake_all()
        assert document.next_tick_callbacks == []

    def test_no_wake_delivery_before_session_loaded(self):
        # A wake tick mutating the document between the initial HTML render
        # and the client's websocket sync builds widgets the client never
        # sees, so wake registration must wait for the session load signal.
        hub = WakeupHub()
        document = FakeDocument()
        on_load_callbacks: list[Callable[[], None]] = []
        updater = _make_updater(
            SessionId('s'),
            SessionRegistry(),
            document=document,
            wakeup_hub=hub,
            on_load=on_load_callbacks.append,
        )
        calls: list[int] = []
        updater.register_custom_handler(lambda: calls.append(1), has_work=lambda: True)

        hub.wake_all()
        assert document.next_tick_callbacks == []

        for callback in on_load_callbacks:
            callback()
        hub.wake_all()
        document.run_next_tick_callbacks()
        assert calls == [1]

    def test_load_after_cleanup_does_not_resurrect_session(self):
        hub = WakeupHub()
        document = FakeDocument()
        on_load_callbacks: list[Callable[[], None]] = []
        updater = _make_updater(
            SessionId('s'),
            SessionRegistry(),
            document=document,
            wakeup_hub=hub,
            on_load=on_load_callbacks.append,
        )

        updater.cleanup()
        for callback in on_load_callbacks:
            callback()
        hub.wake_all()
        assert document.next_tick_callbacks == []

    def test_request_tick_schedules_wake_on_own_document(self):
        document = FakeDocument()
        updater = _make_updater(SessionId('s'), SessionRegistry(), document=document)
        calls: list[int] = []
        updater.register_custom_handler(lambda: calls.append(1), has_work=lambda: True)

        updater.request_tick()
        assert calls == []
        document.run_next_tick_callbacks()
        assert calls == [1]

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

        callback = FakeCallback()
        updater.set_periodic_callback(callback)

        updater.cleanup()

        assert callback.stopped

    def test_reaper_does_not_stop_periodic_callback_on_calling_thread(self):
        # #955: the stale-session reaper runs off the session's IOLoop.
        # PeriodicCallback.stop() mutates the Bokeh document, so it must be
        # scheduled onto the IOLoop, not called on the reaper thread.
        registry = SessionRegistry()
        document = FakeDocument()
        updater = _make_updater(SessionId('session-1'), registry, document=document)

        callback = FakeCallback()
        updater.set_periodic_callback(callback)

        updater.cleanup(defer_document_teardown=True)

        # Not stopped on the calling thread; instead scheduled onto the IOLoop.
        assert not callback.stopped
        assert len(document.next_tick_callbacks) == 1

        # The scheduled tick, running on the IOLoop, performs the stop.
        document.run_next_tick_callbacks()
        assert callback.stopped

    def test_reaper_severs_tier2_even_when_document_tick_never_runs(self):
        # #955 leak fix: tier-2 severing (cleanup handlers, notification-queue
        # release) must happen on the reaper thread regardless of whether the
        # tier-1 document tick ever runs.
        registry = SessionRegistry()
        queue = NotificationQueue()
        document = FakeDocument()
        session_id = SessionId('session-1')
        updater = _make_updater(
            session_id, registry, notification_queue=queue, document=document
        )

        tier2: list[int] = []
        tier1: list[int] = []
        updater.register_cleanup_handler(lambda: tier2.append(1))
        updater.register_document_teardown_handler(lambda: tier1.append(1))

        updater.cleanup(defer_document_teardown=True)

        # Deliberately do NOT run the scheduled tick.
        assert tier2 == [1]
        assert tier1 == []

        # Notification-queue slot released inline (tier 2).
        queue.push(NotificationEvent(message='after cleanup'))
        assert queue.get_new_events(session_id) == []

    def test_document_teardown_handler_runs_when_tick_fires(self):
        registry = SessionRegistry()
        document = FakeDocument()
        updater = _make_updater(SessionId('session-1'), registry, document=document)

        tier1: list[int] = []
        updater.register_document_teardown_handler(lambda: tier1.append(1))

        updater.cleanup(defer_document_teardown=True)
        assert tier1 == []

        document.run_next_tick_callbacks()
        assert tier1 == [1]

    def test_reaper_tolerates_destroyed_document(self):
        # Scheduling into an already-destroyed document raises; tier 2 has
        # already severed everything, so tier 1 is moot and must not propagate.
        registry = SessionRegistry()
        document = FakeDocument(raise_on_schedule=True)
        updater = _make_updater(SessionId('session-1'), registry, document=document)

        callback = FakeCallback()
        updater.set_periodic_callback(callback)
        tier2: list[int] = []
        updater.register_cleanup_handler(lambda: tier2.append(1))

        updater.cleanup(defer_document_teardown=True)  # must not raise

        assert tier2 == [1]
        assert not callback.stopped

    def test_clean_path_runs_document_teardown_inline(self):
        # The clean on_session_destroyed path runs on the IOLoop, so tier-1
        # teardown runs inline even with a document present.
        registry = SessionRegistry()
        document = FakeDocument()
        updater = _make_updater(SessionId('session-1'), registry, document=document)

        callback = FakeCallback()
        updater.set_periodic_callback(callback)
        tier1: list[int] = []
        updater.register_document_teardown_handler(lambda: tier1.append(1))

        updater.cleanup()  # defer_document_teardown=False

        assert callback.stopped
        assert tier1 == [1]
        assert document.next_tick_callbacks == []

    def test_username_forwarded_to_registry(self):
        session_id = SessionId('session-1')
        registry = SessionRegistry()

        _make_updater(session_id, registry, username='Simon')

        info = registry.get_session_info(session_id)
        assert info is not None
        assert info.username == 'Simon'
