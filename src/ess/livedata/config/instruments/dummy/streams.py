# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Dummy instrument stream mapping configuration."""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping

detector_fakes = {'panel_0': (1, 128**2)}


def _make_dummy_detectors() -> StreamLUT:
    """Dummy detector mapping."""
    return {InputStreamKey(topic='dummy_detector', source_name='panel_0'): 'panel_0'}


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'dummy', detector_names=list(detector_fakes)
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='dummy'),
        detectors=_make_dummy_detectors(),
    ),
}
