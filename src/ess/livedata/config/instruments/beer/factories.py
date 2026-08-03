# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
BEER instrument factory implementations.
"""

import scipp as sc

from ess.livedata.config import Instrument

from .specs import detector_pixel_ranges


def setup_factories(instrument: Instrument) -> None:
    """Configure the BEER detector pixel numbering.

    Detector numbers are supplied explicitly rather than loaded from a geometry
    artifact: the banks in the NeXus baseline carry ``depends_on = '.'``, so the
    file holds no placement for them and the logical views do not need one.

    The bank and panel views are wired via ``add_logical_view`` in ``specs.py``,
    and the beam monitors use the generic monitor workflow, so neither needs a
    factory here.
    """
    for name, (first, last) in detector_pixel_ranges.items():
        instrument.configure_detector(
            name,
            detector_number=sc.arange(
                'detector_number', first, last + 1, unit=None, dtype='int32'
            ),
        )
