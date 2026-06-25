# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Cross-thread frame counter coordinating session flushes with data bursts."""

from __future__ import annotations

import threading


class FrameClock:
    """Signals completed data-burst frames to per-session poll loops.

    The ingestion thread calls :meth:`mark` whenever a visible layer is
    recomputed and :meth:`commit` once a burst has finished draining, which
    advances :attr:`generation`. Each session's periodic callback reads
    ``generation`` to decide when a coalesced plot flush is due, so all layers
    from one burst repaint in a single frame instead of being scattered across
    poll ticks.

    Keeping the flush on the session callback (rather than pushing from the
    ingestion thread) preserves the requirement that session-bound objects are
    mutated only in their own document context.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._pending = False

    def mark(self) -> None:
        """Record that a visible layer was recomputed (a frame is forming)."""
        with self._lock:
            self._pending = True

    def commit(self) -> None:
        """Advance the generation if a frame is pending; called on burst end."""
        with self._lock:
            if self._pending:
                self._pending = False
                self._generation += 1

    @property
    def generation(self) -> int:
        """Monotonic counter incremented once per completed data-burst frame."""
        with self._lock:
            return self._generation
