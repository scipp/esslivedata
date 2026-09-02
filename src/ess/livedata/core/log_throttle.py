# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Rate limiting for conditions that recur on every cycle."""

from __future__ import annotations

DEFAULT_COOLDOWN_S = 60.0
"""Cooldown used where the caller has no reason to pick another."""


class LogThrottle:
    """Passes the first event, then at most one per cooldown.

    A condition that holds for a whole run recurs on every cycle, so an
    unthrottled warning would arrive at the cycle rate for as long as it holds
    -- loudest exactly when the journal most needs to stay readable. Counting
    what is suppressed keeps the frequency in the record: the next event
    through carries how many it stands for.
    """

    def __init__(self, cooldown: float = DEFAULT_COOLDOWN_S) -> None:
        self._cooldown = cooldown
        self._last: float | None = None
        self._suppressed = 0

    def take(self, now: float) -> int | None:
        """Report this event, or suppress it.

        Returns
        -------
        :
            How many events were suppressed since the last one reported, or
            ``None`` if this event is itself suppressed.
        """
        if self._last is not None and now - self._last < self._cooldown:
            self._suppressed += 1
            return None
        self._last = now
        suppressed, self._suppressed = self._suppressed, 0
        return suppressed
