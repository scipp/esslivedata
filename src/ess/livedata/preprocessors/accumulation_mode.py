# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Accumulation-mode marker types shared across workflows.

These markers parametrize generic Sciline output types so a single provider
serves both the run-cumulative and per-update views of a quantity. The window
mode selected in the dashboard then picks which view to subscribe to.
"""

from __future__ import annotations

from typing import TypeVar


class Current:
    """Marker type for window accumulation (clears after finalize)."""


class Cumulative:
    """Marker type for cumulative accumulation (accumulates forever)."""


AccumulationMode = TypeVar('AccumulationMode', Current, Cumulative)
"""Type variable for accumulation mode, constrained to Current or Cumulative."""
