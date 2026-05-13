# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ODIN instrument stream mapping configuration.
"""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping
from .specs import instrument

detector_fakes = {'timepix3': (1, 4096**2)}


def _make_odin_detectors() -> StreamLUT:
    """
    Odin detector mapping.

    Input keys based on
    https://confluence.ess.eu/display/ECDC/Kafka+Topics+Overview+for+Instruments
    """
    return {InputStreamKey(topic='odin_detector', source_name='timepix3'): 'timepix3'}
    # The following combination was used during ODIN detector tests,
    # we may need them again.
    # return {
    #     InputStreamKey(topic='odin_detector_tpx3_empir', source_name='test'):
    #     'timepix3'
    # }


def _make_odin_logs() -> StreamLUT:
    """Odin f144 log mapping derived from instrument.streams."""
    return {
        InputStreamKey(topic=s.topic, source_name=s.source): s.stream_name
        for s in instrument.f144_streams.values()
        if s.topic is not None
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'odin',
        detector_names=list(detector_fakes),
        log_names=list(instrument.f144_streams),
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='odin'),
        detectors=_make_odin_detectors(),
        logs=_make_odin_logs(),
    ),
}
