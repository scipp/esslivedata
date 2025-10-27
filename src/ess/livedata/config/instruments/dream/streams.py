# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
DREAM instrument stream mapping configuration.
"""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping

detector_fakes = {
    'mantle_detector': (229377, 720896),
    'endcap_backward_detector': (71618, 229376),
    'endcap_forward_detector': (1, 71680),
    'high_resolution_detector': (1122337, 1523680),  # Note: Not consecutive!
    'sans_detector': (0, 0),  # TODO
}


def _make_dream_detectors() -> StreamLUT:
    """
    Dream detector mapping.

    Input keys based on
    https://confluence.ess.eu/display/ECDC/Kafka+Topics+Overview+for+Instruments
    """
    mapping = {
        'bwec': 'endcap_backward',
        'fwec': 'endcap_forward',
        'hr': 'high_resolution',
        'mantle': 'mantle',
        'sans': 'sans',
    }
    return {
        InputStreamKey(
            topic=f'dream_detector_{key}', source_name='dream'
        ): f'{value}_detector'
        for key, value in mapping.items()
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'dream', detector_names=list(detector_fakes)
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='dream'),
        detectors=_make_dream_detectors(),
    ),
}
