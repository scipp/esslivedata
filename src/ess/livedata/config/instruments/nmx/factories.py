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
    one block per panel, matching ``detector_number`` in the run files exactly.
    That is the layout the reduced-resolution ingest configured in ``specs.py``
    inverts.

    The axis order follows the run files too, where ``x_pixel_offset`` carries
    ``axis=1`` and ``y_pixel_offset`` ``axis=2``, making x the slow dimension.
    Nothing has confirmed that against an installed detector. If it disagrees,
    the image is transposed and swapping the two names here is the whole fix:
    the ingest decomposes an id exactly as this fold does, and the axes are the
    same length and share a block size, so a transposition is all it can be.
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
