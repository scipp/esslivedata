# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
LOKI instrument stream mapping configuration.
"""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping

detector_fakes = {
    'loki_detector_0': (1, 802816),
    'loki_detector_1': (802817, 1032192),
    'loki_detector_2': (1032193, 1204224),
    'loki_detector_3': (1204225, 1433600),
    'loki_detector_4': (1433601, 1605632),
    'loki_detector_5': (1605633, 2007040),
    'loki_detector_6': (2007041, 2465792),
    'loki_detector_7': (2465793, 2752512),
    'loki_detector_8': (2752513, 3211264),
}


def _make_loki_detectors() -> StreamLUT:
    """
    Loki detector mapping.

    Input keys based on
    https://confluence.ess.eu/display/ECDC/Kafka+Topics+Overview+for+Instruments
    """
    return {
        InputStreamKey(
            topic=f'loki_detector_bank{bank}', source_name='caen'
        ): f'loki_detector_{bank}'
        for bank in range(9)
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'loki', detector_names=list(detector_fakes)
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='loki'),
        detectors=_make_loki_detectors(),
    ),
}
