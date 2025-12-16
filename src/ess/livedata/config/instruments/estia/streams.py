# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""ESTIA instrument stream mapping configuration."""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping
from .specs import instrument

# Fake detector configuration: detector_name -> (first_id, last_id)
# ESTIA multiblade detector has 98,304 pixels with IDs 98305-196608
detector_fakes = {'multiblade_detector': (98305, 196608)}


def _make_estia_detectors() -> StreamLUT:
    """ESTIA detector mapping for event detectors (ev44)."""
    return {
        InputStreamKey(
            topic='estia_detector', source_name='multiblade_detector'
        ): 'multiblade_detector',
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'estia',
        detector_names=list(detector_fakes),
        log_names=list(instrument.f144_attribute_registry.keys()),
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='estia'),
        detectors=_make_estia_detectors(),
    ),
}
