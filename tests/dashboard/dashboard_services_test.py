# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the dashboard update loop fail-fast behavior."""

import threading

from confluent_kafka import KafkaException

from ess.livedata.dashboard.dashboard_services import DashboardServices


class _FakeRegistry:
    def cleanup_stale_sessions(self) -> None:
        pass


class _FakePlotOrchestrator:
    def __init__(self) -> None:
        self.flush_calls = 0

    def flush_frames(self) -> None:
        self.flush_calls += 1


class _FakeMessagePump:
    def __init__(self, exception: BaseException | None = None) -> None:
        self._exception = exception
        self.update_calls = 0

    def update(self) -> None:
        self.update_calls += 1
        if self._exception is not None:
            raise self._exception


def _make_loop_target(message_pump: _FakeMessagePump) -> DashboardServices:
    """Build a minimal stand-in with the attributes ``_update_loop`` needs."""
    target = DashboardServices.__new__(DashboardServices)
    target.message_pump = message_pump
    target.session_registry = _FakeRegistry()
    target._stop_event = threading.Event()
    target._update_interval = 0.01
    target.plot_orchestrator = _FakePlotOrchestrator()
    target.transport_failure_calls = 0

    def _on_transport_failure() -> None:
        target.transport_failure_calls += 1

    target._on_transport_failure = _on_transport_failure
    return target


def test_update_loop_exits_and_signals_failure_on_kafka_exception() -> None:
    target = _make_loop_target(_FakeMessagePump(KafkaException("auth denied")))

    thread = threading.Thread(target=DashboardServices._update_loop, args=(target,))
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert target.transport_failure_calls == 1
    assert target.message_pump.update_calls == 1


def test_update_loop_continues_on_generic_exception() -> None:
    """Non-Kafka exceptions stay log-and-continue, not fail-fast."""
    raised = threading.Event()

    class FlakyMessagePump:
        def __init__(self) -> None:
            self.update_calls = 0

        def update(self) -> None:
            self.update_calls += 1
            if self.update_calls == 1:
                raised.set()
                raise RuntimeError("transient widget bug")

    message_pump = FlakyMessagePump()
    target = DashboardServices.__new__(DashboardServices)
    target.message_pump = message_pump
    target.session_registry = _FakeRegistry()
    target._stop_event = threading.Event()
    target._update_interval = 0.01
    target.plot_orchestrator = _FakePlotOrchestrator()
    target.transport_failure_calls = 0

    def _on_transport_failure() -> None:
        target.transport_failure_calls += 1

    target._on_transport_failure = _on_transport_failure

    thread = threading.Thread(target=DashboardServices._update_loop, args=(target,))
    thread.start()
    assert raised.wait(timeout=2.0)
    # Let the loop iterate at least once more after the exception.
    while message_pump.update_calls < 2 and thread.is_alive():
        target._stop_event.wait(0.01)
    target._stop_event.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert target.transport_failure_calls == 0
    assert message_pump.update_calls >= 2


def test_on_transport_failure_is_noop_on_main_thread() -> None:
    """Guard prevents the test process from killing itself when invoked directly."""
    target = DashboardServices.__new__(DashboardServices)
    # Should return without raising and without sending a signal.
    DashboardServices._on_transport_failure(target)


class _FakeJobOrchestrator:
    def expire_pending_commands(self) -> None:
        pass

    def reconcile_observed_jobs(self) -> None:
        pass


class _RecordingPlotOrchestrator:
    """Records the calls teardown ordering depends on."""

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def sync_job_states(self) -> None:
        pass

    def flush_frames(self) -> None:
        self._events.append('flush')

    def shutdown(self) -> None:
        self._events.append('shutdown')


class _RecordingTransport:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def stop(self) -> None:
        self._events.append('transport')


def test_stop_shuts_down_the_orchestrator_after_the_update_loop() -> None:
    """Teardown is what writes a layout change the debounced write left pending.

    Ordered after the update thread has been joined: the loop drives the plot
    model, so nothing may still be touching it while it is torn down.
    """
    events: list[str] = []
    target = _make_loop_target(_FakeMessagePump())
    target.job_orchestrator = _FakeJobOrchestrator()
    target.plot_orchestrator = _RecordingPlotOrchestrator(events)
    target._transport = _RecordingTransport(events)
    target._update_thread = None
    target._start_update_thread()

    DashboardServices.stop(target)

    assert 'flush' in events, 'the update loop never ran'
    assert events[events.index('shutdown') :] == ['shutdown', 'transport']
