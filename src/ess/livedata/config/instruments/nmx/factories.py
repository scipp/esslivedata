# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
NMX instrument factory implementations.
"""

import scipp as sc

from ess.livedata.config import Instrument


def setup_factories(instrument: Instrument) -> None:
    """Initialize NMX-specific factories and configure detectors."""
    # Configure detectors with computed arrays
    # TODO Unclear if this is transposed or not. Wait for updated files.
    dim = 'detector_number'
    sizes = {'x': 1280, 'y': 1280}
    for panel in range(3):
        instrument.configure_detector(
            f'detector_panel_{panel}',
            detector_number=sc.arange(
                'detector_number',
                panel * 1280**2 + 1,
                (panel + 1) * 1280**2 + 1,
                unit=None,
            ).fold(dim=dim, sizes=sizes),
        )
