# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
MAGIC instrument stream mapping configuration.
"""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import (
    make_common_stream_mapping_inputs,
    make_dev_stream_mapping,
    make_f144_log_lut,
)
from .specs import detector_names, detector_pixel_ranges, instrument

# The pixel-ID ranges of the real banks are also what the fakes generate over.
detector_fakes = detector_pixel_ranges


def _make_magic_detectors() -> StreamLUT:
    """MAGIC detector mapping for event detectors (ev44).

    Each bank has its own topic and they share the source name ``magic``, so the
    topic alone identifies the bank.
    """
    return {
        InputStreamKey(topic=name, source_name='magic'): name for name in detector_names
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'magic',
        detector_names=detector_names,
        monitor_names=instrument.monitors,
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(
            instrument='magic', monitor_names=instrument.monitors
        ),
        detectors=_make_magic_detectors(),
        logs=make_f144_log_lut(instrument),
    ),
}
