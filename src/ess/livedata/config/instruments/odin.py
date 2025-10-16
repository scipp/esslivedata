# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import h5py
import scipp as sc

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.config.env import StreamingEnv
from ess.livedata.handlers.detector_data_handler import (
    DetectorProjection,
    LogicalViewConfig,
)
from ess.livedata.handlers.monitor_data_handler import register_monitor_workflows
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from ._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping

instrument = Instrument(name='odin')
instrument_registry.register(instrument)

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
    orca_det = f['entry/instrument/histogram_mode_detectors/orca']
    orca_trans = orca_det['transformations/translation']
    orca_trans[...] = 1.0


register_monitor_workflows(
    instrument=instrument, source_names=['monitor1', 'monitor2']
)  # Monitor names - in the streaming module

instrument.add_detector('timepix3', detector_group_name='event_mode_detectors')

instrument.add_detector(
    'orca',
    detector_number=sc.arange('yx', 1, 2048**2 + 1, unit=None).fold(
        dim='yx',
        sizes={'y': -1, 'x': 2048},
    ),
    detector_group_name='histogram_mode_detectors',
)

_xy_projection = DetectorProjection(
    instrument=instrument,
    projection='xy_plane',
    resolution={'timepix3': {'y': 512, 'x': 512}, 'orca': {'y': 2048, 'x': 2048}},
)


def _resize_image(da: sc.DataArray) -> sc.DataArray:
    from ess.imaging.tools import resample

    # 2048*2048 is the actual panel size, and 1024*1024 in the test file,
    # but ess.livedata might not be able to keep up with that
    # so we resample to 128*128 ((1024/8) * (1024/8)) for now.
    return resample(da, sizes={'x': 8, 'y': 8}, method='sum')


_panel_0_config = LogicalViewConfig(
    name='odin_detector_xy',
    title='Timepix3 XY Detector Counts',
    description='2D view of the Timepix3 detector counts',
    source_names=['timepix3'],
    # transform allows to scale the view.
    transform=_resize_image,
)
# WARNING: Disabled until fidex
# _panel_0_view = DetectorLogicalView(
#    instrument=instrument, config=_panel_0_config
# )  # Instantiating the DetectorLogicalView itself registers it.


detectors_config = {'fakes': {'timepix3': (1, 4096**2), 'orca': (1, 2048**2)}}


def _make_odin_detectors() -> StreamLUT:
    """
    Odin detector mapping.

    Input keys based on
    https://confluence.ess.eu/display/ECDC/Kafka+Topics+Overview+for+Instruments
    """
    # return {InputStreamKey(topic='odin_detector', source_name='timepix3'): 'timepix3'}
    return {
        InputStreamKey(
            topic='odin_detector_tpx3_empir', source_name='test'
        ): 'timepix3',
        InputStreamKey(
            topic='odin_area_detector_orca', source_name='hama_kfk1'
        ): 'orca',
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'odin', detector_names=list(detectors_config['fakes'])
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='odin'),
        detectors=_make_odin_detectors(),
    ),
}
