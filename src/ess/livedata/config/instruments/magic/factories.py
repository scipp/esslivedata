# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
MAGIC instrument factory implementations.
"""

import scipp as sc

from ess.livedata.config import Instrument

#: Contiguous detector_number ranges (inclusive) per bank, matching the
#: NeXus files. The logical views need only this pixel structure, so we provide
#: it directly instead of loading a geometry file.
_detector_number_ranges: dict[str, tuple[int, int]] = {
    'magic_detector_a': (1, 491520),
    'magic_detector_b': (491521, 622592),
}


def setup_factories(instrument: Instrument) -> None:
    """Initialize MAGIC-specific factories and workflows.

    Logical view factories are attached automatically by ``load_factories``;
    here we only supply the detector_number arrays the views fold over.
    """
    for name, (start, stop) in _detector_number_ranges.items():
        instrument.configure_detector(
            name,
            detector_number=sc.arange(
                'detector_number', start, stop + 1, unit=None, dtype='int32'
            ),
        )
