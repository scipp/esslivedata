# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Dummy instrument stream mapping configuration."""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping
from .specs import instrument

detector_fakes = {'panel_0': (1, 128**2)}

# Area detector fakes: detector_name -> (height, width) shape
area_detector_fakes = {'area_panel': (256, 256)}


def _make_dummy_detectors() -> StreamLUT:
    """Dummy detector mapping for event detectors (ev44)."""
    return {
        InputStreamKey(topic='dummy_detector', source_name='panel_0'): 'panel_0',
    }


def _make_dummy_area_detectors() -> StreamLUT:
    """Dummy detector mapping for area detectors (ad00)."""
    return {
        InputStreamKey(
            topic='dummy_area_detector', source_name='area_panel'
        ): 'area_panel',
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'dummy',
        detector_names=list(detector_fakes),
        area_detector_names=list(area_detector_fakes),
        log_names=list(instrument.f144_attribute_registry.keys()),
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='dummy'),
        detectors=_make_dummy_detectors(),
        area_detectors=_make_dummy_area_detectors(),
    ),
}
