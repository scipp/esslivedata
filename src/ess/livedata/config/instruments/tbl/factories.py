# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL instrument factory implementations.
"""

from ess.livedata.config import Instrument

from . import specs  # noqa: F401


def setup_factories(instrument: Instrument) -> None:
    """Initialize TBL-specific factories and workflows."""
    # Future workflow factories can be added here
    pass
