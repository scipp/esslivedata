# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import scipp as sc

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.config.env import StreamingEnv
from ess.livedata.handlers.detector_data_handler import (
    DetectorLogicalView,
    LogicalViewConfig,
)
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from ._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping

instrument = Instrument(name='odin')
instrument_registry.register(instrument)

instrument.add_detector(
    'orca',
    detector_number=sc.arange('yx', 1, 2048**2 + 1, unit=None).fold(
        dim='yx',
        sizes={'y': -1, 'x': 2048},
    ),
    detector_group_name='histogram_mode_detectors',
)

_odin_detector_config = LogicalViewConfig(
    name='odin_detector_orca',
    title='Odin Orca',
    description='Odin detector counts per pixel.',
    source_names=instrument.detector_names,
)
_odin_detector_view = DetectorLogicalView(
    instrument=instrument, config=_odin_detector_config
)


detectors_config = {'fakes': {'timepix3': (1, 4096**2), 'orca': (1, 2048**2)}}


def _make_odin_detectors() -> StreamLUT:
    """
    Odin detector mapping.

    Input keys based on
    https://confluence.ess.eu/display/ECDC/Kafka+Topics+Overview+for+Instruments
    """
    return {
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
