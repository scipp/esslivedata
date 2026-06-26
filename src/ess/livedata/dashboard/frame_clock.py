# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Per-key frame counter coordinating session flushes with data bursts."""

from __future__ import annotations

import threading
from collections.abc import Hashable


class FrameClock:
    """Signals completed data-burst frames to per-session poll loops.

    The ingestion thread calls :meth:`mark` with a key (a grid id) whenever a
    visible layer in that group is recomputed, and :meth:`commit` once a burst
    has finished draining, which advances the :meth:`generation` of every marked
    key. Each session's periodic callback reads the generation of the key it is
    showing (its active tab) to decide when a coalesced flush is due, so all
    layers from one burst repaint in a single frame instead of being scattered
    across poll ticks.

    Keying by group rather than a single global counter keeps the signal scoped:
    a session is only woken by bursts in the tab it is actually displaying, not
    by data arriving for some other session's tab.

    Keeping the flush on the session callback (rather than pushing from the
    ingestion thread) preserves the requirement that session-bound objects are
    mutated only in their own document context; mutating them off that context
    corrupts the document and misroutes updates to the wrong tab (see ADR 0005).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation: dict[Hashable, int] = {}
        self._pending: set[Hashable] = set()

    def mark(self, key: Hashable) -> None:
        """Record that a layer in ``key`` was recomputed (a frame is forming)."""
        with self._lock:
            self._pending.add(key)

    def commit(self) -> None:
        """Advance every pending key's generation; called on burst end."""
        with self._lock:
            for key in self._pending:
                self._generation[key] = self._generation.get(key, 0) + 1
            self._pending.clear()

    def generation(self, key: Hashable) -> int:
        """Counter incremented once per completed data-burst frame for ``key``."""
        with self._lock:
            return self._generation.get(key, 0)
