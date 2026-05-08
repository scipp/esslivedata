# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
LOKI instrument stream mapping configuration.
"""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.handlers.wavelength_lut_workflow_specs import (
    make_chopper_log_topic_for_stream,
    make_chopper_stream_lut,
)
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping
from .specs import _LOKI_CHOPPERS, f144_log_streams

#: Production chopper PV prefixes from ``coda_loki_999999_00026352.hdf``.
#: Suffixes (`:TotDly`, `:Spd_S`) are an ECDC-wide convention and live in the
#: shared chopper-stream helpers.
_LOKI_CHOPPER_PV_PREFIXES: dict[str, str] = {
    'bw_chopper1': 'LOKI-ChpSy1:Chop-BWC-101',
    'bw_chopper2': 'LOKI-ChpSy1:Chop-BWC-102',
    'fo_chopper1': 'LOKI-ChpSy3:Chop-SFOC-101',
    'fo_chopper2': 'LOKI-ChpSy3:Chop-SFOC-102',
}

detector_fakes = {
    'loki_detector_0': (1, 802816),
    'loki_detector_1': (802817, 1032192),
    'loki_detector_2': (1032193, 1204224),
    'loki_detector_3': (1204225, 1433600),
    'loki_detector_4': (1433601, 1605632),
    'loki_detector_5': (1605633, 2007040),
    'loki_detector_6': (2007041, 2465792),
    'loki_detector_7': (2465793, 2752512),
    'loki_detector_8': (2752513, 3211264),
}


# Monitor names use 0-based indices matching the NeXus beam monitor groups
# (beam_monitor_mN). Kafka source names are 1-based (cbm1..5).
# Ref: ``coda_loki_999999_00026352.hdf``
monitor_names = [
    'beam_monitor_m0',
    'beam_monitor_m1',
    'beam_monitor_m2',
    'beam_monitor_m3',
    'beam_monitor_m4',
]


def _make_loki_logs() -> StreamLUT:
    """LOKI log data mapping (f144 streams).

    Motion streams are declared in ``specs.f144_log_streams``; chopper
    streams are generated from PV prefixes via the shared helper.
    """
    return {
        InputStreamKey(topic=info['topic'], source_name=info['source']): internal_name
        for internal_name, info in f144_log_streams.items()
    } | make_chopper_stream_lut('loki', _LOKI_CHOPPER_PV_PREFIXES)


def _make_loki_detectors() -> StreamLUT:
    """
    Loki detector mapping.

    Input keys based on
    https://confluence.ess.eu/display/ECDC/Kafka+Topics+Overview+for+Instruments
    """
    return {
        InputStreamKey(
            topic=f'loki_detector_bank{bank}', source_name='caen'
        ): f'loki_detector_{bank}'
        for bank in range(9)
    }


_common_prod = make_common_stream_mapping_inputs(
    instrument='loki', monitor_names=monitor_names
)
_common_prod['detectors'] = _make_loki_detectors()

_chopper_log_topics = make_chopper_log_topic_for_stream('loki', _LOKI_CHOPPERS)

stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'loki',
        detector_names=list(detector_fakes),
        monitor_names=monitor_names,
        log_names=[*f144_log_streams, *_chopper_log_topics],
        log_topic_for_stream={
            **{name: info['topic'] for name, info in f144_log_streams.items()},
            **_chopper_log_topics,
        },
    ),
    StreamingEnv.PROD: StreamMapping(**_common_prod, logs=_make_loki_logs()),
}
