# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for StreamKind and stream_kind_to_topic mapping."""

import pytest

from ess.livedata.config import streams
from ess.livedata.config.instruments import available_instruments
from ess.livedata.core.message import StreamKind


@pytest.mark.parametrize('instrument', available_instruments())
@pytest.mark.parametrize(
    ('stream_kind', 'expected_suffix'),
    [
        (StreamKind.MONITOR_COUNTS, 'beam_monitor'),
        (StreamKind.MONITOR_EVENTS, 'beam_monitor'),
        (StreamKind.DETECTOR_EVENTS, 'detector'),
        (StreamKind.LOG, 'motion'),
        (StreamKind.LIVEDATA_DATA, 'livedata_data'),
        (StreamKind.LIVEDATA_ROI, 'livedata_roi'),
        (StreamKind.LIVEDATA_COMMANDS, 'livedata_commands'),
        (StreamKind.LIVEDATA_STATUS, 'livedata_heartbeat'),
    ],
)
def test_stream_kind_to_topic_mapping(
    instrument: str, stream_kind: StreamKind, expected_suffix: str
) -> None:
    """Verify all StreamKind values map to correct topic names."""
    topic = streams.stream_kind_to_topic(instrument=instrument, kind=stream_kind)
    expected_topic = f'{instrument}_{expected_suffix}'
    assert topic == expected_topic


def test_stream_kind_to_topic_unknown_raises() -> None:
    """Verify UNKNOWN StreamKind raises ValueError."""
    with pytest.raises(ValueError, match='Unknown stream kind'):
        streams.stream_kind_to_topic(instrument='dummy', kind=StreamKind.UNKNOWN)


@pytest.mark.parametrize('instrument', available_instruments())
def test_stream_mapping_has_all_livedata_topics(instrument: str) -> None:
    """Verify StreamMapping has all livedata topic properties."""
    mapping = streams.get_stream_mapping(instrument=instrument, dev=True)

    # Check all livedata topic properties exist
    assert hasattr(mapping, 'livedata_commands_topic')
    assert hasattr(mapping, 'livedata_data_topic')
    assert hasattr(mapping, 'livedata_roi_topic')
    assert hasattr(mapping, 'livedata_status_topic')

    # Check they return valid topic strings
    assert mapping.livedata_commands_topic
    assert mapping.livedata_data_topic
    assert mapping.livedata_roi_topic
    assert mapping.livedata_status_topic

    # Check instrument name is in the topics
    assert instrument in mapping.livedata_commands_topic
    assert instrument in mapping.livedata_data_topic
    assert instrument in mapping.livedata_roi_topic
    assert instrument in mapping.livedata_status_topic
