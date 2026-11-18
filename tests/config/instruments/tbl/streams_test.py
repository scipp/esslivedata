# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Tests for the TBL production stream mapping."""

import pytest
from streaming_data_types import eventdata_ev44

from ess.livedata.config.env import StreamingEnv
from ess.livedata.config.instruments.tbl.streams import stream_mapping
from ess.livedata.core.message import StreamKind
from ess.livedata.kafka.message_adapter import FakeKafkaMessage, KafkaToEv44Adapter


@pytest.mark.parametrize('source_name', ['tbl-ngem', 'ngem'])
def test_both_ngem_source_names_resolve_to_ngem(source_name: str) -> None:
    payload = eventdata_ev44.serialise_ev44(
        source_name=source_name,
        message_id=0,
        reference_time=[1234],
        reference_time_index=[0],
        time_of_flight=[1],
        pixel_id=[1],
    )
    adapter = KafkaToEv44Adapter(
        stream_kind=StreamKind.DETECTOR_EVENTS,
        stream_lut=stream_mapping[StreamingEnv.PROD].detectors,
    )
    result = adapter.adapt(FakeKafkaMessage(value=payload, topic='tbl_detector_ngem'))
    assert result.stream.name == 'ngem'
