# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.kafka.stream_mapping import InputStreamKey, StreamMapping


@pytest.fixture
def full_mapping() -> StreamMapping:
    return StreamMapping(
        instrument="test",
        detectors={
            InputStreamKey(topic="det_topic_a", source_name="src"): "det_a",
            InputStreamKey(topic="det_topic_b", source_name="src"): "det_b",
            InputStreamKey(topic="det_topic_c", source_name="src"): "det_c",
        },
        monitors={
            InputStreamKey(topic="mon_topic", source_name="mon1"): "mon_1",
            InputStreamKey(topic="mon_topic", source_name="mon2"): "mon_2",
        },
        area_detectors={
            InputStreamKey(topic="area_topic", source_name="cam"): "camera",
        },
        logs={
            InputStreamKey(topic="motion", source_name="m1"): "motor_x",
            InputStreamKey(topic="motion", source_name="m2"): "motor_y",
        },
        livedata_commands_topic="cmd",
        livedata_data_topic="data",
        livedata_responses_topic="resp",
        livedata_roi_topic="roi",
        livedata_status_topic="status",
    )


class TestStreamMappingFiltered:
    def test_filters_detectors(self, full_mapping: StreamMapping) -> None:
        f = full_mapping.filtered({"det_a", "det_c"})
        assert set(f.detectors.values()) == {"det_a", "det_c"}
        assert f.detector_topics == {"det_topic_a", "det_topic_c"}

    def test_filters_monitors(self, full_mapping: StreamMapping) -> None:
        f = full_mapping.filtered({"mon_1"})
        assert set(f.monitors.values()) == {"mon_1"}

    def test_filters_area_detectors(self, full_mapping: StreamMapping) -> None:
        f = full_mapping.filtered({"camera"})
        assert set(f.area_detectors.values()) == {"camera"}

    def test_filters_logs(self, full_mapping: StreamMapping) -> None:
        f = full_mapping.filtered({"motor_x"})
        assert f.logs is not None
        assert set(f.logs.values()) == {"motor_x"}
        assert f.log_topics == {"motion"}

    def test_empty_result(self, full_mapping: StreamMapping) -> None:
        f = full_mapping.filtered({"nonexistent"})
        assert f.detectors == {}
        assert f.monitors == {}
        assert f.area_detectors == {}
        assert f.logs == {}
        assert f.detector_topics == set()

    def test_preserves_infrastructure_topics(self, full_mapping: StreamMapping) -> None:
        f = full_mapping.filtered(set())
        assert f.topics.livedata_commands == "cmd"
        assert f.topics.livedata_data == "data"
        assert f.topics.livedata_responses == "resp"
        assert f.topics.livedata_roi == "roi"
        assert f.topics.livedata_status == "status"
        assert f.instrument == "test"

    def test_logs_none_preserved(self, infra_kwargs: dict) -> None:
        m = StreamMapping(
            instrument="test",
            detectors={},
            monitors={},
            logs=None,
            **infra_kwargs,
        )
        f = m.filtered({"anything"})
        assert f.logs is None

    def test_mixed_filter(self, full_mapping: StreamMapping) -> None:
        f = full_mapping.filtered({"det_b", "mon_2", "motor_y"})
        assert set(f.detectors.values()) == {"det_b"}
        assert set(f.monitors.values()) == {"mon_2"}
        assert f.area_detectors == {}
        assert set(f.logs.values()) == {"motor_y"}
