# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
NMX instrument stream mapping configuration.
"""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping

detector_fakes = {
    f'detector_panel_{i}': (i * 1280**2 + 1, (i + 1) * 1280**2) for i in range(3)
}


def _make_nmx_detectors() -> StreamLUT:
    """
    NMX detector mapping.

    Input keys based on
    https://confluence.ess.eu/display/ECDC/Kafka+Topics+Overview+for+Instruments
    """
    return {
        InputStreamKey(
            topic=f'nmx_detector_p{panel}', source_name='nmx'
        ): 'nmx_detector'
        for panel in range(3)
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'nmx', detector_names=list(detector_fakes)
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='nmx'),
        detectors=_make_nmx_detectors(),
    ),
}
