# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ODIN instrument factory implementations.
"""

import h5py
import scipp as sc

from ess.livedata.config import instrument_registry
from ess.livedata.handlers.detector_data_handler import DetectorProjection
from ess.livedata.handlers.monitor_data_handler import attach_monitor_workflow_factory

from . import specs

# Get instrument from registry (already registered by specs.py)
instrument = instrument_registry['odin']

# Patch the Odin geometry file with:
# 1. Non-zero z (needed for detector xy projection)
# 2. Axes names and mapping to detector number shape, since ScippNexus cannot infer
#    these automatically from the Timepix3 data.
# Note: We do this every time on import. Accessing `instrument.nexus_file` the first
# time will actually fetch the file using pooch, so it reverts this change every time.
with h5py.File(instrument.nexus_file, 'r+') as f:
    det = f['entry/instrument/event_mode_detectors/timepix3']
    trans = det['transformations/translation']
    trans[...] = 1.0
    det.attrs['axes'] = ['x_pixel_offset', 'y_pixel_offset']
    det.attrs['detector_number_indices'] = [0, 1]

# Add detector
instrument.add_detector('timepix3', detector_group_name='event_mode_detectors')

# Create detector projection
_xy_projection = DetectorProjection(
    instrument=instrument,
    projection='xy_plane',
    resolution={'timepix3': {'y': 512, 'x': 512}},
)


def _resize_image(da: sc.DataArray) -> sc.DataArray:
    from ess.imaging.tools import resample

    # 2048*2048 is the actual panel size, and 1024*1024 in the test file,
    # but ess.livedata might not be able to keep up with that
    # so we resample to 128*128 ((1024/8) * (1024/8)) for now.
    return resample(da, sizes={'x': 8, 'y': 8}, method='sum')


# Detector view configuration (currently disabled)
# WARNING: Disabled until fixed
# _panel_0_config = LogicalViewConfig(
#     name='odin_detector_xy',
#     title='Timepix3 XY Detector Counts',
#     description='2D view of the Timepix3 detector counts',
#     source_names=['timepix3'],
#     transform=_resize_image,
# )
# _panel_0_view = DetectorLogicalView(
#     instrument=instrument, config=_panel_0_config
# )
#
# from .specs import panel_0_view_handle
# from ess.livedata.handlers.detector_view_specs import DetectorViewParams
#
# @panel_0_view_handle.attach_factory()
# def _panel_0_view_factory(source_name: str, params: DetectorViewParams):
#     """Factory for timepix3 detector view."""
#     return _panel_0_view.make_view(source_name, params=params)

# Attach monitor workflow factory
attach_monitor_workflow_factory(specs.monitor_workflow_handle)
