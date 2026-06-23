# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
MAGIC instrument stream mapping configuration.
"""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping
from .specs import detector_names

#: detector_number ranges (start, end) per bank, used to generate fake events.
detector_fakes = {
    'magic_detector_a': (1, 245760),
    'magic_detector_b': (245761, 376832),
}


def _make_magic_detectors() -> StreamLUT:
    """MAGIC production detector mapping.

    Topic name and source-name convention are provisional, pending the real
    Kafka topic configuration for MAGIC.
    """
    return {
        InputStreamKey(topic='magic_detector', source_name=name): name
        for name in detector_names
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'magic',
        detector_names=detector_names,
        monitor_names=[],
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='magic', monitor_names=[]),
        detectors=_make_magic_detectors(),
        logs={},
    ),
}
