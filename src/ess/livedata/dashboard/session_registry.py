# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Track active browser sessions with heartbeat-based cleanup."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NewType

logger = logging.getLogger(__name__)

SessionId = NewType('SessionId', str)


@dataclass
class SessionInfo:
    """Information about an active session."""

    session_id: SessionId
    last_heartbeat: float = field(default_factory=time.monotonic)


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
        on_session_cleanup: Callable[[SessionId], None] | None = None,
    ) -> None:
        """
        Initialize the session registry.

        Parameters
        ----------
        stale_timeout_seconds:
            Seconds since last heartbeat before a session is considered stale.
            Default is 60 seconds.
        on_session_cleanup:
            Optional callback invoked when a stale session is cleaned up.
            Receives the session ID of the cleaned-up session.
        """
        self._stale_timeout = stale_timeout_seconds
        self._on_session_cleanup = on_session_cleanup
        self._sessions: dict[SessionId, SessionInfo] = {}
        self._lock = threading.Lock()

    def register(self, session_id: SessionId) -> None:
        """
        Register a new session.

        Parameters
        ----------
        session_id:
            Unique identifier for the session (typically from curdoc session).
        """
        with self._lock:
            if session_id in self._sessions:
                logger.debug(
                    "Session %s already registered, updating heartbeat", session_id
                )
            else:
                logger.info("Registered new session: %s", session_id)
            self._sessions[session_id] = SessionInfo(session_id=session_id)

    def unregister(self, session_id: SessionId) -> None:
        """
        Unregister a session explicitly.

        Called when pn.state.on_session_destroyed() fires.

        Parameters
        ----------
        session_id:
            Session ID to unregister.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info("Unregistered session: %s", session_id)

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
        stale_sessions: list[SessionId] = []

        with self._lock:
            for session_id, info in list(self._sessions.items()):
                if now - info.last_heartbeat > self._stale_timeout:
                    stale_sessions.append(session_id)
                    del self._sessions[session_id]
                    logger.warning(
                        "Cleaned up stale session: %s (no heartbeat for %.1f seconds)",
                        session_id,
                        now - info.last_heartbeat,
                    )

        # Invoke cleanup callback outside lock to avoid deadlocks
        if self._on_session_cleanup is not None:
            for session_id in stale_sessions:
                try:
                    self._on_session_cleanup(session_id)
                except Exception:
                    logger.exception(
                        "Error in session cleanup callback for %s", session_id
                    )

        return stale_sessions

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
