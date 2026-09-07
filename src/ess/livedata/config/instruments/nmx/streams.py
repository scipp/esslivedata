# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
NMX instrument stream mapping configuration.
"""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import (
    make_common_stream_mapping_inputs,
    make_dev_stream_mapping,
    make_f144_log_lut,
)
from .specs import instrument

detector_fakes = {
    f'detector_panel_{i}': (i * 1280**2 + 1, (i + 1) * 1280**2) for i in range(3)
}


def _make_nmx_detectors() -> StreamLUT:
    """
    NMX detector mapping.

    All three panels publish under the same source name and are told apart by
    their topic alone, as confirmed by the ``topic`` and ``source`` attributes
    the file writer records on each panel's ``NXevent_data`` group.
    """
    return {
        InputStreamKey(
            topic=f'nmx_detector_p{panel}', source_name='nmx'
        ): f'detector_panel_{panel}'
        for panel in range(3)
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'nmx',
        detector_names=list(detector_fakes),
        log_names=list(instrument.f144_streams),
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='nmx'),
        detectors=_make_nmx_detectors(),
        logs=make_f144_log_lut(instrument),
    ),
}
