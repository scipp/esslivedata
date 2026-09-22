# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
FREIA instrument factory implementations.
"""

import scipp as sc

from ess.livedata.config import Instrument

from .specs import detector_pixel_range


def setup_factories(instrument: Instrument) -> None:
    """Initialize FREIA-specific factories.

    The logical view factory is attached by ``load_factories``. Supplying
    ``detector_number`` here means no geometry file is needed.
    """
    first, last = detector_pixel_range
    instrument.configure_detector(
        'multiblade_detector',
        detector_number=sc.arange(
            'detector_number', first, last + 1, unit=None, dtype='int32'
        ),
    )
