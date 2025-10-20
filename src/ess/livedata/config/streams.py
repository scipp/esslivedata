# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Mappings from (topic, source_name) to internal stream identifier.

Raw data is received from Kafka on a variety of topics. Each message has a source name.
The latter is not unique (not event within an instrument), only the combination of topic
and source name is. To isolate ESSlivedata code from the details of the Kafka topics, we
use this mapping to assign a unique stream name to each (topic, source name) pair.
"""

from __future__ import annotations

from ess.livedata import StreamKind
from ess.livedata.kafka import StreamMapping

from .env import StreamingEnv
from .instruments import get_config


def stream_kind_to_topic(instrument: str, kind: StreamKind) -> str:
    """
    Convert a StreamKind to a topic name.

    Used for constructing the topic name from the StreamKind when publishing to Kafka.
    The non-livedata topics are thus only used when using our fake data generators.
    """
    match kind:
        case StreamKind.MONITOR_COUNTS:
            return f'{instrument}_beam_monitor'
        case StreamKind.MONITOR_EVENTS:
            return f'{instrument}_beam_monitor'
        case StreamKind.DETECTOR_EVENTS:
            return f'{instrument}_detector'
        case StreamKind.LOG:
            return f'{instrument}_motion'
        case StreamKind.LIVEDATA_DATA:
            return f'{instrument}_livedata_data'
        case StreamKind.LIVEDATA_ROI:
            return f'{instrument}_livedata_roi'
        case StreamKind.LIVEDATA_CONFIG:
            return f'{instrument}_livedata_commands'
        case StreamKind.LIVEDATA_STATUS:
            return f'{instrument}_livedata_heartbeat'  # Expected by Nicos
        case _:
            raise ValueError(f'Unknown stream kind: {kind}')


def get_stream_mapping(*, instrument: str, dev: bool) -> StreamMapping:
    """
    Returns the stream mapping for the given instrument.
    """
    config = get_config(instrument=instrument)
    env = StreamingEnv.DEV if dev else StreamingEnv.PROD
    return config.stream_mapping[env]
