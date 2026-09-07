# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
NMX instrument factory implementations.
"""

import scipp as sc

from ess.livedata.config import Instrument

from .specs import PANEL_RESOLUTION


def setup_factories(instrument: Instrument) -> None:
    """Initialize NMX-specific factories and configure detectors.

    Each panel enumerates its ids contiguously in row-major order, 1-based and
    one block per panel, which is the layout the reduced-resolution ingest
    configured in ``specs.py`` inverts.

    TODO Unclear if this is transposed or not. Wait for updated files. The
    ingest decomposes an id exactly as this fold does, and the two axes are the
    same length and share a block size, so a transposed fold transposes the
    reduced image the same way it already transposes the full-resolution one.
    The fix, when the files arrive, is to swap the names here.
    """
    dim = 'detector_number'
    sizes = {'x': PANEL_RESOLUTION, 'y': PANEL_RESOLUTION}
    pixels = PANEL_RESOLUTION**2
    for panel in range(3):
        instrument.configure_detector(
            f'detector_panel_{panel}',
            detector_number=sc.arange(
                'detector_number',
                panel * pixels + 1,
                (panel + 1) * pixels + 1,
                unit=None,
            ).fold(dim=dim, sizes=sizes),
        )
