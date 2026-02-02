# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for session registry."""

import time

from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry


class FakeSessionUpdater:
    """Fake updater for testing cleanup behavior."""

    def __init__(self, *, raise_on_cleanup: bool = False):
        self.cleanup_called = False
        self._raise_on_cleanup = raise_on_cleanup

    def cleanup(self) -> None:
        self.cleanup_called = True
        if self._raise_on_cleanup:
            raise ValueError("Cleanup error")


class TestSessionRegistry:
    def test_register_new_session(self):
        registry = SessionRegistry()
        session_id = SessionId('session-1')

        registry.register(session_id)

        assert registry.is_active(session_id)
        assert registry.session_count == 1

    def test_register_multiple_sessions(self):
        registry = SessionRegistry()

        registry.register(SessionId('session-1'))
        registry.register(SessionId('session-2'))

        assert registry.session_count == 2
        assert set(registry.get_active_sessions()) == {
            SessionId('session-1'),
            SessionId('session-2'),
        }

    def test_unregister_session(self):
        registry = SessionRegistry()
        session_id = SessionId('session-1')
        registry.register(session_id)

        registry.unregister(session_id)

        assert not registry.is_active(session_id)
        assert registry.session_count == 0

    def test_unregister_unknown_session_is_noop(self):
        registry = SessionRegistry()
        registry.unregister(SessionId('unknown'))
        assert registry.session_count == 0

    def test_heartbeat_updates_timestamp(self):
        registry = SessionRegistry(stale_timeout_seconds=0.1)
        session_id = SessionId('session-1')
        registry.register(session_id)

        # Wait for session to become stale
        time.sleep(0.05)

        # Send heartbeat
        registry.heartbeat(session_id)

        # Wait a bit more (would be stale without heartbeat)
        time.sleep(0.07)

        # Session should still be active due to heartbeat
        stale = registry.cleanup_stale_sessions()
        assert stale == []
        assert registry.is_active(session_id)

    def test_heartbeat_registers_unknown_session(self):
        registry = SessionRegistry()
        session_id = SessionId('session-1')

        registry.heartbeat(session_id)

        assert registry.is_active(session_id)

    def test_cleanup_stale_sessions(self):
        registry = SessionRegistry(stale_timeout_seconds=0.05)
        session_id = SessionId('session-1')
        registry.register(session_id)

        # Wait for session to become stale
        time.sleep(0.1)

        stale = registry.cleanup_stale_sessions()

        assert stale == [session_id]
        assert not registry.is_active(session_id)

    def test_cleanup_only_removes_stale_sessions(self):
        registry = SessionRegistry(stale_timeout_seconds=0.1)
        stale_session = SessionId('stale')
        active_session = SessionId('active')

        registry.register(stale_session)
        time.sleep(0.05)
        registry.register(active_session)
        time.sleep(0.06)

        # stale_session is >0.1s old, active_session is ~0.06s old
        stale = registry.cleanup_stale_sessions()

        assert stale == [stale_session]
        assert not registry.is_active(stale_session)
        assert registry.is_active(active_session)

    def test_stale_cleanup_calls_updater_cleanup(self):
        registry = SessionRegistry(stale_timeout_seconds=0.01)
        session_id = SessionId('session-1')
        updater = FakeSessionUpdater()
        registry.register(session_id, updater)

        time.sleep(0.02)
        registry.cleanup_stale_sessions()

        assert updater.cleanup_called

    def test_unregister_calls_updater_cleanup(self):
        registry = SessionRegistry()
        session_id = SessionId('session-1')
        updater = FakeSessionUpdater()
        registry.register(session_id, updater)

        registry.unregister(session_id)

        assert updater.cleanup_called

    def test_cleanup_error_does_not_stop_other_cleanups(self):
        registry = SessionRegistry(stale_timeout_seconds=0.01)
        error_updater = FakeSessionUpdater(raise_on_cleanup=True)
        ok_updater = FakeSessionUpdater()
        registry.register(SessionId('error'), error_updater)
        registry.register(SessionId('ok'), ok_updater)

        time.sleep(0.02)
        stale = registry.cleanup_stale_sessions()

        # Both sessions should be cleaned up from registry
        assert len(stale) == 2
        # Both updaters should have cleanup called (even though one raised)
        assert error_updater.cleanup_called
        assert ok_updater.cleanup_called

    def test_get_active_sessions_returns_copy(self):
        registry = SessionRegistry()
        registry.register(SessionId('session-1'))

        sessions = registry.get_active_sessions()
        sessions.append(SessionId('fake'))

        # Original registry should not be modified
        assert registry.session_count == 1

    def test_get_seconds_since_heartbeat_returns_elapsed_time(self):
        registry = SessionRegistry()
        session_id = SessionId('session-1')
        registry.register(session_id)

        time.sleep(0.05)

        elapsed = registry.get_seconds_since_heartbeat(session_id)
        assert elapsed is not None
        assert elapsed >= 0.05

    def test_get_seconds_since_heartbeat_returns_none_for_unknown(self):
        registry = SessionRegistry()
        assert registry.get_seconds_since_heartbeat(SessionId('unknown')) is None
