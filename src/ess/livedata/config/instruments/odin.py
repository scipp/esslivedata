# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import scipp as sc

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.config.env import StreamingEnv
from ess.livedata.handlers.detector_data_handler import (
    DetectorLogicalView,
    LogicalViewConfig,
)
from ess.livedata.handlers.monitor_data_handler import register_monitor_workflows
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from ._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping

instrument = Instrument(name='odin')
instrument_registry.register(instrument)
register_monitor_workflows(
    instrument=instrument, source_names=['monitor1', 'monitor2']
)  # Monitor names - in the streaming module

instrument.add_detector(
    # Should be consistent with detector config keys,
    # i.e. detector_group name in nexus file
    # Test file has 1024*1024 pixels but real data may have 4096*4096 pixels
    'timepix3',
    detector_number=sc.arange('yx', 1, 1024**2 + 1, unit=None).fold(
        dim='yx', sizes={'y': -1, 'x': 1024}
    ),
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
    # source_names=['timepix3'],
    source_names=['test'],
    # transform allows to scale the view.
    transform=_resize_image,
)
_panel_0_view = DetectorLogicalView(
    instrument=instrument, config=_panel_0_config
)  # Instantiating the DetectorLogicalView itself registers it.


detectors_config = {'fakes': {'timepix3': (1, 1024**2)}}


def _make_odin_detectors() -> StreamLUT:
    """
    Odin detector mapping.

    Input keys based on
    https://confluence.ess.eu/display/ECDC/Kafka+Topics+Overview+for+Instruments
    """
    # return {InputStreamKey(topic='odin_detector', source_name='timepix3'): 'timepix3'}
    return {
        InputStreamKey(topic='odin_detector_tpx3_empir', source_name='test'): 'timepix3'
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
