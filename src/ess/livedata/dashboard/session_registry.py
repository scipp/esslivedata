# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Track active browser sessions with heartbeat-based cleanup."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NewType

import structlog

if TYPE_CHECKING:
    from .session_updater import SessionUpdater

logger = structlog.get_logger(__name__)

SessionId = NewType('SessionId', str)


@dataclass
class SessionInfo:
    """Information about an active session."""

    session_id: SessionId
    last_heartbeat: float = field(default_factory=time.monotonic)
    updater: SessionUpdater | None = None


class SessionRegistry:
    """
    Track active browser sessions and clean up stale ones.

    Provides defense-in-depth for session cleanup when Panel's
    `pn.state.on_session_destroyed()` fails to fire (e.g., due to
    browser crashes or network disconnects).

    Each session's periodic callback should send heartbeats. Sessions
    that haven't sent a heartbeat within the timeout are considered
    stale and cleaned up.

    Thread-safe: can be called from background threads and periodic callbacks.
    """

    def __init__(
        self,
        *,
        stale_timeout_seconds: float = 60.0,
    ) -> None:
        """
        Initialize the session registry.

        Parameters
        ----------
        stale_timeout_seconds:
            Seconds since last heartbeat before a session is considered stale.
            Default is 60 seconds.
        """
        self._stale_timeout = stale_timeout_seconds
        self._sessions: dict[SessionId, SessionInfo] = {}
        self._lock = threading.Lock()

    def register(
        self, session_id: SessionId, updater: SessionUpdater | None = None
    ) -> None:
        """
        Register a new session with its updater.

        Parameters
        ----------
        session_id:
            Unique identifier for the session (typically from curdoc session).
        updater:
            The SessionUpdater instance for this session.
        """
        with self._lock:
            if session_id in self._sessions:
                logger.debug(
                    "Session %s already registered, updating heartbeat", session_id
                )
                # Update the updater if provided
                if updater is not None:
                    self._sessions[session_id].updater = updater
                self._sessions[session_id].last_heartbeat = time.monotonic()
            else:
                logger.info("Registered new session: %s", session_id)
                self._sessions[session_id] = SessionInfo(
                    session_id=session_id, updater=updater
                )

    def unregister(self, session_id: SessionId) -> None:
        """
        Unregister a session explicitly.

        Called when pn.state.on_session_destroyed() fires. Cleans up the
        session's updater if one was registered.

        Parameters
        ----------
        session_id:
            Session ID to unregister.
        """
        updater = None
        with self._lock:
            if session_id in self._sessions:
                updater = self._sessions[session_id].updater
                del self._sessions[session_id]
                logger.info("Unregistered session: %s", session_id)

        # Clean up updater outside lock to avoid potential deadlocks
        if updater is not None:
            try:
                updater.cleanup()
            except Exception:
                logger.exception("Error cleaning up updater for session %s", session_id)

    def heartbeat(self, session_id: SessionId) -> None:
        """
        Update heartbeat timestamp for a session.

        Called from each session's periodic callback.

        Parameters
        ----------
        session_id:
            Session ID to update.
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].last_heartbeat = time.monotonic()
            else:
                # Session not yet registered - register it now
                logger.debug(
                    "Session %s sent heartbeat before registration, registering",
                    session_id,
                )
                self._sessions[session_id] = SessionInfo(session_id=session_id)

    def cleanup_stale_sessions(self) -> list[SessionId]:
        """
        Remove sessions that haven't sent a heartbeat within the timeout.

        Returns
        -------
        :
            List of session IDs that were cleaned up.
        """
        now = time.monotonic()
        stale_sessions: list[tuple[SessionId, SessionUpdater | None]] = []

        with self._lock:
            for session_id, info in list(self._sessions.items()):
                if now - info.last_heartbeat > self._stale_timeout:
                    stale_sessions.append((session_id, info.updater))
                    del self._sessions[session_id]
                    logger.warning(
                        "Cleaned up stale session: %s (no heartbeat for %.1f seconds)",
                        session_id,
                        now - info.last_heartbeat,
                    )

        # Clean up updaters outside lock to avoid potential deadlocks
        for session_id, updater in stale_sessions:
            if updater is not None:
                try:
                    updater.cleanup()
                except Exception:
                    logger.exception(
                        "Error cleaning up updater for stale session %s", session_id
                    )

        return [session_id for session_id, _ in stale_sessions]

    def get_active_sessions(self) -> list[SessionId]:
        """
        Get list of currently active session IDs.

        Returns
        -------
        :
            List of active session IDs.
        """
        with self._lock:
            return list(self._sessions.keys())

    def is_active(self, session_id: SessionId) -> bool:
        """
        Check if a session is currently active.

        Parameters
        ----------
        session_id:
            Session ID to check.

        Returns
        -------
        :
            True if the session is active, False otherwise.
        """
        with self._lock:
            return session_id in self._sessions

    @property
    def session_count(self) -> int:
        """Number of currently active sessions."""
        with self._lock:
            return len(self._sessions)

    def get_seconds_since_heartbeat(self, session_id: SessionId) -> float | None:
        """
        Get seconds since last heartbeat for a session.

        Parameters
        ----------
        session_id:
            Session ID to check.

        Returns
        -------
        :
            Seconds since last heartbeat, or None if session not found.
        """
        with self._lock:
            if session_id not in self._sessions:
                return None
            return time.monotonic() - self._sessions[session_id].last_heartbeat
