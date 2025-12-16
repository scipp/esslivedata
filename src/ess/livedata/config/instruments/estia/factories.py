# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA instrument factory implementations.
"""

from ess.livedata.config import Instrument


def setup_factories(instrument: Instrument) -> None:
    """Initialize ESTIA-specific factories and workflows."""
    # Detector configuration will be loaded from the geometry file.
    # No views or workflows are set up yet.
    pass
