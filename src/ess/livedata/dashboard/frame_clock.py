# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Per-key frame counter coordinating session flushes with data bursts."""

from __future__ import annotations

import threading
from collections.abc import Hashable


class FrameClock:
    """Signals completed data-burst frames to per-session poll loops.

    The ingestion thread calls :meth:`commit` with a key (a grid id) once that
    grid's layers have finished recomputing for a drained burst, advancing the
    grid's :meth:`generation`. Each session's periodic callback reads the
    generation of the key it is showing (its active tab) to decide when a
    coalesced flush is due, so all layers from one burst repaint in a single
    frame instead of being scattered across poll ticks.

    Committing per grid (rather than all grids together at burst end) means a
    session showing tab A sees its frame the moment tab A's layers finish, not
    after every other visible tab's compute. Keying by grid also scopes the
    signal: a session is only woken by bursts in the tab it is actually
    displaying, not by data arriving for some other session's tab.

    Keeping the flush on the session callback (rather than pushing from the
    ingestion thread) preserves the requirement that session-bound objects are
    mutated only in their own document context; mutating them off that context
    corrupts the document and misroutes updates to the wrong tab (see ADR 0005).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation: dict[Hashable, int] = {}

    def commit(self, key: Hashable) -> None:
        """Advance ``key``'s generation; called once its burst frame is done."""
        with self._lock:
            self._generation[key] = self._generation.get(key, 0) + 1

    def generation(self, key: Hashable) -> int:
        """Counter incremented once per completed data-burst frame for ``key``."""
        with self._lock:
            return self._generation.get(key, 0)
