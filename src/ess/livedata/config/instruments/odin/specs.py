# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ODIN instrument spec registration.
"""

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.handlers.detector_view_specs import DetectorViewParams

instrument = Instrument(
    name='odin',
    detector_names=['timepix3'],
    monitors=['monitor1', 'monitor2'],
)

instrument_registry.register(instrument)

# Detector view spec registration

panel_0_view_handle = instrument.register_spec(
    namespace='detector_data',
    name='odin_detector_xy',
    version=1,
    title='Timepix3 XY Detector Counts',
    description='2D view of the Timepix3 detector counts',
    source_names=['timepix3'],
    params=DetectorViewParams,
)
