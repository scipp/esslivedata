# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.config import streams
from ess.livedata.config.instruments import available_instruments
from ess.livedata.kafka import InputStreamKey, StreamMapping


@pytest.mark.parametrize('instrument', available_instruments())
def test_get_stream_mapping_dev(instrument: str) -> None:
    stream_mapping = streams.get_stream_mapping(instrument=instrument, dev=True)
    assert stream_mapping is not None
    assert isinstance(stream_mapping, streams.StreamMapping)


@pytest.mark.parametrize('instrument', available_instruments())
def test_get_stream_mapping_production(instrument: str) -> None:
    stream_mapping = streams.get_stream_mapping(instrument=instrument, dev=False)
    assert stream_mapping is not None
    assert isinstance(stream_mapping, streams.StreamMapping)


class TestStreamMappingLogTopics:
    def test_log_topics_returns_empty_set_when_logs_is_none(self) -> None:
        mapping = StreamMapping(
            instrument='test',
            detectors={},
            monitors={},
            logs=None,
            livedata_commands_topic='test_commands',
            livedata_data_topic='test_data',
            livedata_responses_topic='test_responses',
            livedata_roi_topic='test_roi',
            livedata_status_topic='test_status',
        )
        assert mapping.log_topics == set()
        assert mapping.logs is None

    def test_log_topics_returns_topics_from_logs_lut(self) -> None:
        logs = {
            InputStreamKey(topic='motion', source_name='motor1'): 'detector_rotation',
            InputStreamKey(topic='motion', source_name='motor2'): 'sample_rotation',
            InputStreamKey(topic='sensors', source_name='temp1'): 'sample_temperature',
        }
        mapping = StreamMapping(
            instrument='test',
            detectors={},
            monitors={},
            logs=logs,
            livedata_commands_topic='test_commands',
            livedata_data_topic='test_data',
            livedata_responses_topic='test_responses',
            livedata_roi_topic='test_roi',
            livedata_status_topic='test_status',
        )
        assert mapping.log_topics == {'motion', 'sensors'}
        assert mapping.logs == logs
