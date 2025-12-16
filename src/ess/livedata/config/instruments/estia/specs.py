# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA instrument spec registration.
"""

from ess.livedata.config import Instrument, instrument_registry

detector_names = ['multiblade_detector']

instrument = Instrument(
    name='estia',
    detector_names=detector_names,
    monitors=[],
    f144_attribute_registry={},
)

# Register instrument
instrument_registry.register(instrument)
