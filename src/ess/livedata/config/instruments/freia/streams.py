# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""FREIA instrument stream mapping configuration."""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import (
    make_common_stream_mapping_inputs,
    make_dev_stream_mapping,
    make_f144_log_lut,
)
from .specs import detector_pixel_range, instrument

detector_fakes = {'multiblade_detector': detector_pixel_range}


def _make_freia_detectors() -> StreamLUT:
    """FREIA detector mapping for event detectors (ev44)."""
    return {
        InputStreamKey(
            topic='freia_detector', source_name='multiblade'
        ): 'multiblade_detector',
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'freia',
        detector_names=list(detector_fakes),
        monitor_names=instrument.monitors,
        log_names=list(instrument.f144_streams),
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(
            instrument='freia',
            monitor_names=instrument.monitors,
            cbm_start=1,
        ),
        detectors=_make_freia_detectors(),
        logs=make_f144_log_lut(instrument),
    ),
}
