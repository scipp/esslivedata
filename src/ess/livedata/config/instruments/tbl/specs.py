# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL workflow spec registration.
"""

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.handlers.detector_view_specs import DetectorViewOutputs

from .views import fold_image, get_he3_detector_view, get_multiblade_view, identity

detector_names = [
    'timepix3_detector',
    'multiblade_detector',
    'he3_detector_bank0',
    'he3_detector_bank1',
    'ngem_detector',
    # not listing orca since it does not have (and does not need) detector numbers
]

instrument = Instrument(name='tbl', detector_names=detector_names)

instrument_registry.register(instrument)

instrument.add_logical_view(
    name='tbl_detector_timepix3',
    title='Timepix3 Detector',
    description='512x512 image downsampled from full resolution',
    source_names=['timepix3_detector'],
    transform=fold_image,
    reduction_dim=['x_bin', 'y_bin'],
    roi_support=True,
)

instrument.add_logical_view(
    name='multiblade_detector_view',
    title='Multiblade Detector',
    description='Counts folded into blade, wire, and strip dimensions',
    source_names=['multiblade_detector'],
    transform=get_multiblade_view,
    roi_support=True,
    output_ndim=3,
)

instrument.add_logical_view(
    name='he3_detector_view',
    title='He3 Detector',
    description='Combined view of both detector banks with tube and pixel axes',
    source_names=['he3_detector_bank0', 'he3_detector_bank1'],
    transform=get_he3_detector_view,
    roi_support=True,
)

instrument.add_logical_view(
    name='ngem_detector_view',
    title='NGEM Detector',
    description='2D detector counts view',
    source_names=['ngem_detector'],
    transform=identity,
    reduction_dim='dim_0',
    roi_support=True,
)

orca_view_handle = instrument.register_spec(
    namespace='detector_data',
    name='tbl_area_detector_orca',
    version=1,
    title='Orca Detector',
    description='512x512 image downsampled from full resolution',
    source_names=['orca_detector'],
    params=None,
    outputs=DetectorViewOutputs,
)
