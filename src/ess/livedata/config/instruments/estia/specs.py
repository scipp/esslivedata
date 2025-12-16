# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA instrument spec registration.
"""

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.handlers.detector_view_specs import (
    register_logical_detector_view_spec,
)

detector_names = ['multiblade_detector']

instrument = Instrument(
    name='estia',
    detector_names=detector_names,
    monitors=[],
    f144_attribute_registry={},
)

instrument_registry.register(instrument)

multiblade_view_handle = register_logical_detector_view_spec(
    instrument=instrument,
    name='estia_multiblade_detector_view',
    title='Multiblade Detector',
    description='Counts folded into strip, blade, and wire dimensions',
    source_names=['multiblade_detector'],
    roi_support=True,
)
