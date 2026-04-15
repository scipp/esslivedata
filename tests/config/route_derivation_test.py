# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.config.instrument import Instrument
from ess.livedata.config.route_derivation import (
    gather_source_names,
    get_source_subset,
    resolve_stream_names,
)
from ess.livedata.config.workflow_spec import AuxInput, AuxSources, DefaultOutputs
from ess.livedata.kafka.stream_mapping import InputStreamKey, StreamMapping


def _make_instrument_with_specs() -> Instrument:
    """Create an instrument with specs across multiple namespaces."""
    instrument = Instrument(
        name="test",
        detector_names=["det_a", "det_b", "det_c"],
        monitors=["mon_1", "mon_2"],
    )
    instrument.register_spec(
        namespace="detector_data",
        name="projection",
        version=1,
        title="Projection",
        source_names=["det_a", "det_b", "det_c"],
        aux_sources=AuxSources(
            {
                "motion": AuxInput(choices=("motor_x", "motor_y"), default="motor_x"),
            }
        ),
        outputs=DefaultOutputs,
    )
    instrument.register_spec(
        namespace="detector_data",
        name="special_view",
        version=1,
        title="Special",
        source_names=["det_a"],
        outputs=DefaultOutputs,
    )
    instrument.register_spec(
        namespace="data_reduction",
        name="reduction",
        version=1,
        title="Reduction",
        source_names=["det_a", "det_b"],
        aux_sources=AuxSources(
            {
                "incident_monitor": AuxInput(
                    choices=("mon_1", "mon_2"), default="mon_1"
                ),
            }
        ),
        outputs=DefaultOutputs,
    )
    instrument.register_spec(
        namespace="monitor_data",
        name="monitor_workflow",
        version=1,
        title="Monitor",
        source_names=["mon_1", "mon_2"],
        outputs=DefaultOutputs,
    )
    return instrument


class TestGatherSourceNames:
    def test_returns_empty_for_unknown_namespace(self) -> None:
        instrument = _make_instrument_with_specs()
        result = gather_source_names(instrument, "nonexistent")
        assert result == set()

    def test_detector_data_namespace(self) -> None:
        instrument = _make_instrument_with_specs()
        result = gather_source_names(instrument, "detector_data")
        assert result == {"det_a", "det_b", "det_c", "motor_x", "motor_y"}

    def test_data_reduction_namespace(self) -> None:
        instrument = _make_instrument_with_specs()
        result = gather_source_names(instrument, "data_reduction")
        assert result == {"det_a", "det_b", "mon_1", "mon_2"}

    def test_monitor_data_namespace(self) -> None:
        instrument = _make_instrument_with_specs()
        result = gather_source_names(instrument, "monitor_data")
        assert result == {"mon_1", "mon_2"}

    def test_source_subset_filters_by_primary_sources(self) -> None:
        instrument = _make_instrument_with_specs()
        result = gather_source_names(
            instrument, "detector_data", source_subset={"det_a"}
        )
        # projection spec matches (det_a in source_names) -> adds all its sources + aux
        # special_view spec matches (det_a in source_names) -> adds det_a
        assert result == {"det_a", "det_b", "det_c", "motor_x", "motor_y"}

    def test_source_subset_excludes_unrelated_specs(self) -> None:
        instrument = _make_instrument_with_specs()
        result = gather_source_names(
            instrument, "data_reduction", source_subset={"det_a"}
        )
        # reduction spec matches (det_a in source_names) -> adds det_a, det_b, mon_1,
        # mon_2
        assert result == {"det_a", "det_b", "mon_1", "mon_2"}

    def test_source_subset_no_match(self) -> None:
        instrument = _make_instrument_with_specs()
        result = gather_source_names(
            instrument, "detector_data", source_subset={"nonexistent"}
        )
        assert result == set()

    def test_no_specs(self) -> None:
        instrument = Instrument(name="empty")
        result = gather_source_names(instrument, "detector_data")
        assert result == set()


class TestGetSourceSubset:
    def test_single_shard(self) -> None:
        names = ["a", "b", "c", "d", "e"]
        result = get_source_subset(names, num_shards=1, shard=0)
        assert result == {"a", "b", "c", "d", "e"}

    def test_two_shards(self) -> None:
        names = ["c", "a", "b", "d"]
        shard_0 = get_source_subset(names, num_shards=2, shard=0)
        shard_1 = get_source_subset(names, num_shards=2, shard=1)
        # sorted: a, b, c, d -> shard 0 gets a, c; shard 1 gets b, d
        assert shard_0 == {"a", "c"}
        assert shard_1 == {"b", "d"}

    def test_shards_are_disjoint_and_complete(self) -> None:
        names = ["e", "d", "c", "b", "a"]
        all_shards = [get_source_subset(names, num_shards=3, shard=i) for i in range(3)]
        union = set().union(*all_shards)
        assert union == set(names)
        for i in range(3):
            for j in range(i + 1, 3):
                assert all_shards[i].isdisjoint(all_shards[j])

    def test_shard_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="shard=2 must be less than num_shards=2"):
            get_source_subset(["a", "b"], num_shards=2, shard=2)

    def test_more_shards_than_sources(self) -> None:
        names = ["a", "b"]
        shard_0 = get_source_subset(names, num_shards=5, shard=0)
        shard_1 = get_source_subset(names, num_shards=5, shard=1)
        shard_2 = get_source_subset(names, num_shards=5, shard=2)
        assert shard_0 == {"a"}
        assert shard_1 == {"b"}
        assert shard_2 == set()


def _infra_kwargs() -> dict:
    return {
        "livedata_commands_topic": "cmd",
        "livedata_data_topic": "data",
        "livedata_responses_topic": "resp",
        "livedata_roi_topic": "roi",
        "livedata_status_topic": "status",
    }


class TestResolveStreamNames:
    def test_known_names_pass_through(self) -> None:
        instrument = Instrument(
            name="test", detector_names=["det_a"], monitors=["mon_1"]
        )
        mapping = StreamMapping(
            instrument="test",
            detectors={InputStreamKey(topic="t", source_name="s"): "det_a"},
            monitors={InputStreamKey(topic="m", source_name="s"): "mon_1"},
            **_infra_kwargs(),
        )
        result = resolve_stream_names({"det_a", "mon_1"}, instrument, mapping)
        assert result == {"det_a", "mon_1"}

    def test_unknown_detector_name_expands_to_all_detectors(self) -> None:
        """Bifrost-like case: logical name not in LUT, but in detector_names."""
        instrument = Instrument(name="test", detector_names=["unified"], monitors=[])
        mapping = StreamMapping(
            instrument="test",
            detectors={
                InputStreamKey(topic="t1", source_name="s"): "phys_0",
                InputStreamKey(topic="t2", source_name="s"): "phys_1",
            },
            monitors={},
            **_infra_kwargs(),
        )
        result = resolve_stream_names({"unified"}, instrument, mapping)
        assert result == {"phys_0", "phys_1"}

    def test_unknown_monitor_name_expands_to_all_monitors(self) -> None:
        instrument = Instrument(
            name="test", detector_names=[], monitors=["logical_mon"]
        )
        mapping = StreamMapping(
            instrument="test",
            detectors={},
            monitors={
                InputStreamKey(topic="m1", source_name="s"): "phys_mon_0",
                InputStreamKey(topic="m2", source_name="s"): "phys_mon_1",
            },
            **_infra_kwargs(),
        )
        result = resolve_stream_names({"logical_mon"}, instrument, mapping)
        assert result == {"phys_mon_0", "phys_mon_1"}

    def test_mix_of_known_and_unknown(self) -> None:
        instrument = Instrument(name="test", detector_names=["unified"], monitors=[])
        mapping = StreamMapping(
            instrument="test",
            detectors={
                InputStreamKey(topic="t", source_name="s"): "phys_0",
            },
            monitors={},
            logs={InputStreamKey(topic="l", source_name="s"): "motor_x"},
            **_infra_kwargs(),
        )
        result = resolve_stream_names({"unified", "motor_x"}, instrument, mapping)
        assert result == {"phys_0", "motor_x"}

    def test_empty_needed(self) -> None:
        instrument = Instrument(name="test")
        mapping = StreamMapping(
            instrument="test", detectors={}, monitors={}, **_infra_kwargs()
        )
        result = resolve_stream_names(set(), instrument, mapping)
        assert result == set()
