# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL instrument spec registration.
"""

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.handlers.detector_view_specs import (
    DetectorViewOutputs,
    register_logical_detector_view_spec,
)

detector_names = ['timepix3_detector', 'multiblade_detector']

instrument = Instrument(name='tbl', detector_names=detector_names)

instrument_registry.register(instrument)

# Timepix3 detector view spec (ev44 event detector)
timepix3_view_handle = register_logical_detector_view_spec(
    instrument=instrument,
    name='tbl_detector_timepix3',
    title='Timepix3 Detector Counts',
    description='2D view of the Timepix3 detector counts',
    source_names=['timepix3_detector'],
    roi_support=True,
)

multiblade_view_handle = register_logical_detector_view_spec(
    instrument=instrument,
    name='multiblade_detector_view',
    title='Multiblade Detector Counts',
    description='',
    source_names=['multiblade_detector'],
    roi_support=True,
)

# Orca area detector view spec (ad00 image detector)
orca_view_handle = instrument.register_spec(
    namespace='detector_data',
    name='tbl_area_detector_orca',
    version=1,
    title='Hamamatsu Orca',
    description='Area detector image view for Hamamatsu Orca camera',
    source_names=['orca_detector'],
    params=None,
    outputs=DetectorViewOutputs,
)
