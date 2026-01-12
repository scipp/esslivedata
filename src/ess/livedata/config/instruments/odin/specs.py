# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ODIN instrument spec registration.
"""

from ess.livedata.config import Instrument, instrument_registry

from .views import fold_image

instrument = Instrument(
    name='odin',
    detector_names=['timepix3'],
    monitors=['monitor1', 'monitor2'],
)

instrument_registry.register(instrument)

# Detector view spec registration (with ROI support)
instrument.add_logical_view(
    name='odin_detector_xy',
    title='Timepix3 XY Detector Counts',
    description='2D view of the Timepix3 detector counts',
    source_names=['timepix3'],
    transform=fold_image,
    reduction_dim=['x_bin', 'y_bin'],
    roi_support=True,
)
