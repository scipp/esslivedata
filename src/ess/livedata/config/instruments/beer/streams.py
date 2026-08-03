# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""BEER instrument stream mapping configuration."""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import (
    make_common_stream_mapping_inputs,
    make_dev_stream_mapping,
    make_f144_log_lut,
)
from .specs import detector_pixel_ranges, instrument, monitor_names

# The pixel-ID ranges of the real banks are also what the fakes generate over.
detector_fakes = detector_pixel_ranges


def _make_beer_detectors() -> StreamLUT:
    """BEER detector mapping for event detectors (ev44)."""
    return {
        InputStreamKey(
            topic='beer_detector', source_name='detector_a'
        ): 'beer_detector_s2',
        InputStreamKey(
            topic='beer_detector', source_name='detector_b'
        ): 'beer_detector_n2',
    }


def _make_beer_monitors() -> StreamLUT:
    """BEER beam monitor mapping; internal names are the source names."""
    return {
        InputStreamKey(topic='beer_beam_monitor', source_name=name): name
        for name in monitor_names
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'beer',
        detector_names=list(detector_fakes),
        monitor_names=monitor_names,
        log_names=list(instrument.f144_streams),
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(
            instrument='beer', monitors=_make_beer_monitors()
        ),
        detectors=_make_beer_detectors(),
        logs=make_f144_log_lut(instrument),
    ),
}
