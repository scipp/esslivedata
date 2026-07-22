# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for WakeupHub."""

from collections.abc import Callable

from ess.livedata.dashboard.session_registry import SessionId
from ess.livedata.dashboard.wakeup_hub import WakeupHub


class FakeDocument:
    """Records scheduled next-tick callbacks; can simulate a destroyed doc."""

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


class TestWakeupHub:
    def test_wake_all_schedules_tick_per_registered_session(self):
        hub = WakeupHub()
        docs = [FakeDocument(), FakeDocument()]
        ticks: list[str] = []
        for i, doc in enumerate(docs):
            hub.register(SessionId(f's{i}'), doc, lambda i=i: ticks.append(f's{i}'))

        hub.wake_all()
        for doc in docs:
            doc.run_next_tick_callbacks()

        assert sorted(ticks) == ['s0', 's1']

    def test_pending_wake_coalesces_bursts(self):
        hub = WakeupHub()
        doc = FakeDocument()
        ticks: list[int] = []
        hub.register(SessionId('s'), doc, lambda: ticks.append(1))

        hub.wake_all()
        hub.wake_all()
        hub.wake_all()

        assert len(doc.next_tick_callbacks) == 1
        doc.run_next_tick_callbacks()
        assert ticks == [1]

    def test_wake_after_tick_ran_schedules_again(self):
        hub = WakeupHub()
        doc = FakeDocument()
        ticks: list[int] = []
        hub.register(SessionId('s'), doc, lambda: ticks.append(1))

        hub.wake_all()
        doc.run_next_tick_callbacks()
        hub.wake_all()
        doc.run_next_tick_callbacks()

        assert ticks == [1, 1]

    def test_wake_during_tick_schedules_fresh_wake(self):
        """The pending flag clears before the tick body, so a change landing
        mid-tick wakes the session again instead of waiting for the poll."""
        hub = WakeupHub()
        doc = FakeDocument()
        ticks: list[int] = []

        def tick() -> None:
            ticks.append(1)
            if len(ticks) == 1:
                hub.wake_all()

        hub.register(SessionId('s'), doc, tick)

        hub.wake_all()
        doc.run_next_tick_callbacks()
        assert len(doc.next_tick_callbacks) == 1
        doc.run_next_tick_callbacks()

        assert ticks == [1, 1]

    def test_scheduling_failure_drops_session(self):
        hub = WakeupHub()
        dead = FakeDocument(raise_on_schedule=True)
        live = FakeDocument()
        ticks: list[str] = []
        hub.register(SessionId('dead'), dead, lambda: ticks.append('dead'))
        hub.register(SessionId('live'), live, lambda: ticks.append('live'))

        hub.wake_all()
        live.run_next_tick_callbacks()
        hub.wake_all()
        live.run_next_tick_callbacks()

        assert ticks == ['live', 'live']

    def test_unregistered_session_is_not_woken(self):
        hub = WakeupHub()
        doc = FakeDocument()
        ticks: list[int] = []
        hub.register(SessionId('s'), doc, lambda: ticks.append(1))
        hub.unregister(SessionId('s'))

        hub.wake_all()

        assert doc.next_tick_callbacks == []

    def test_unregister_is_idempotent(self):
        hub = WakeupHub()
        hub.unregister(SessionId('never-registered'))

    def test_pending_wake_for_unregistered_session_skips_tick(self):
        hub = WakeupHub()
        doc = FakeDocument()
        ticks: list[int] = []
        hub.register(SessionId('s'), doc, lambda: ticks.append(1))

        hub.wake_all()
        hub.unregister(SessionId('s'))
        doc.run_next_tick_callbacks()

        assert ticks == []

    def test_tick_exception_is_contained(self):
        hub = WakeupHub()
        doc = FakeDocument()

        def tick() -> None:
            raise RuntimeError("boom")

        hub.register(SessionId('s'), doc, tick)

        hub.wake_all()
        doc.run_next_tick_callbacks()

        # Pending flag cleared despite the exception: the next wake schedules.
        hub.wake_all()
        assert len(doc.next_tick_callbacks) == 1
