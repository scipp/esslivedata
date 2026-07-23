# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Cross-thread, wake-up-only scheduling of session ticks (ADR 0007)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from threading import Lock
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from bokeh.document import Document

    from .session_registry import SessionId

logger = structlog.get_logger(__name__)


@dataclass
class _Entry:
    session_id: SessionId
    document: Document
    tick: Callable[[], None]
    pending: bool = False


class WakeupHub:
    """Wakes registered sessions when shared state may have changed.

    :meth:`wake_all` may be called from any thread — the ingestion loop after
    a drain pass, or a session's UI thread after a shared version bump. A wake
    carries no data: it schedules the session's tick onto that session's
    IOLoop via ``Document.add_next_tick_callback`` (Bokeh's documented pattern
    for cross-thread scheduling), and the tick re-reads shared state in
    session context. Ticks are idempotent and version-gated, so lost or
    duplicate wakes are harmless; the housekeeping poll is the safety net.

    A per-session pending flag coalesces bursts: while a wake is scheduled but
    has not run, further ``wake_all`` calls skip that session. The flag is
    cleared *before* the tick body runs, so a change landing during the tick
    schedules a fresh wake instead of waiting for the housekeeping poll.

    Sessions register only once their browser session has loaded (see the
    registration site in :class:`SessionUpdater`: a wake tick mutating the
    document before the client's initial sync would be invisible to the
    client forever) and unregister on teardown (:meth:`SessionUpdater.cleanup`,
    both the clean and reaper paths); scheduling into a destroyed document
    raises, and the session is then dropped lazily.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[SessionId, _Entry] = {}

    def register(
        self, session_id: SessionId, document: Document, tick: Callable[[], None]
    ) -> None:
        """Register a session's document and wake tick."""
        with self._lock:
            self._sessions[session_id] = _Entry(
                session_id=session_id, document=document, tick=tick
            )

    def unregister(self, session_id: SessionId) -> None:
        """Remove a session; idempotent, safe on any thread."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def wake_all(self) -> None:
        """Schedule a tick for every registered session without one pending."""
        with self._lock:
            due = [e for e in self._sessions.values() if not e.pending]
            for entry in due:
                entry.pending = True
        for entry in due:
            try:
                entry.document.add_next_tick_callback(partial(self._run, entry))
            except Exception:
                # Destroyed/unloaded document that missed regular teardown.
                logger.debug(
                    "Dropping session %s: wake scheduling failed", entry.session_id
                )
                self.unregister(entry.session_id)

    def _run(self, entry: _Entry) -> None:
        """Tick wrapper running on the session's IOLoop.

        Clears the pending flag before the tick body so a concurrent
        ``wake_all`` schedules a fresh wake for changes the tick misses.
        """
        with self._lock:
            entry.pending = False
            if entry.session_id not in self._sessions:
                return
        try:
            entry.tick()
        except Exception:
            logger.exception("Wake tick failed for session %s", entry.session_id)
