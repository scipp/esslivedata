# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
from streaming_data_types import run_start_pl72, run_stop_6s4t
from streaming_data_types.exceptions import WrongSchemaException

from ess.livedata.core.message import (
    RUN_CONTROL_STREAM_ID,
    RunStart,
    RunStop,
)
from ess.livedata.kafka.message_adapter import (
    FakeKafkaMessage,
    RunControlAdapter,
)

_MS_TO_NS = 1_000_000


def _make_run_start_message(
    *,
    run_name: str = 'run_42',
    start_time: int = 1000,
    stop_time: int = 0,
    topic: str = 'inst_filewriter',
) -> FakeKafkaMessage:
    buf = run_start_pl72.serialise_pl72(
        job_id='fw-job-1',
        filename='/data/run_42.nxs',
        start_time=start_time,
        stop_time=stop_time,
        run_name=run_name,
    )
    return FakeKafkaMessage(value=buf, topic=topic, timestamp=99)


def _make_run_stop_message(
    *, run_name: str = 'run_42', stop_time: int = 2000, topic: str = 'inst_filewriter'
) -> FakeKafkaMessage:
    buf = run_stop_6s4t.serialise_6s4t(
        job_id='fw-job-1',
        run_name=run_name,
        stop_time=stop_time,
    )
    return FakeKafkaMessage(value=buf, topic=topic, timestamp=99)


class TestRunControlAdapter:
    def test_adapts_run_start_to_RunStart(self):
        adapter = RunControlAdapter()
        msg = adapter.adapt(
            _make_run_start_message(run_name='test_run', start_time=500)
        )
        assert isinstance(msg.value, RunStart)
        assert msg.value.run_name == 'test_run'
        assert msg.value.start_time == 500 * _MS_TO_NS

    def test_adapts_run_stop_to_RunStop(self):
        adapter = RunControlAdapter()
        msg = adapter.adapt(_make_run_stop_message(run_name='test_run', stop_time=900))
        assert isinstance(msg.value, RunStop)
        assert msg.value.run_name == 'test_run'
        assert msg.value.stop_time == 900 * _MS_TO_NS

    def test_run_start_with_stop_time(self):
        adapter = RunControlAdapter()
        msg = adapter.adapt(_make_run_start_message(start_time=1000, stop_time=5000))
        assert isinstance(msg.value, RunStart)
        assert msg.value.start_time == 1000 * _MS_TO_NS
        assert msg.value.stop_time == 5000 * _MS_TO_NS

    def test_run_start_without_stop_time_is_none(self):
        adapter = RunControlAdapter()
        msg = adapter.adapt(_make_run_start_message(start_time=1000, stop_time=0))
        assert isinstance(msg.value, RunStart)
        assert msg.value.stop_time is None

    def test_stream_id_is_run_control(self):
        adapter = RunControlAdapter()
        start_msg = adapter.adapt(_make_run_start_message())
        stop_msg = adapter.adapt(_make_run_stop_message())
        assert start_msg.stream == RUN_CONTROL_STREAM_ID
        assert stop_msg.stream == RUN_CONTROL_STREAM_ID

    def test_raises_on_unknown_schema(self):
        adapter = RunControlAdapter()
        from streaming_data_types import eventdata_ev44

        buf = eventdata_ev44.serialise_ev44(
            source_name='det',
            message_id=0,
            reference_time=[],
            reference_time_index=[],
            time_of_flight=[],
            pixel_id=[],
        )
        msg = FakeKafkaMessage(value=buf, topic='inst_filewriter')
        with pytest.raises(WrongSchemaException, match="Unexpected schema"):
            adapter.adapt(msg)
