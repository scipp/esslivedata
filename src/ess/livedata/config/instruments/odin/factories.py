# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ODIN instrument factory implementations.
"""

from ess.livedata.config import Instrument

from . import specs  # noqa: F401  # Import to ensure instrument is registered


def setup_factories(instrument: Instrument) -> None:
    """Initialize ODIN-specific factories and workflows."""
    # Configure detector with custom group name
    instrument.configure_detector(
        'timepix3', detector_group_name='event_mode_detectors'
    )
