# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.config.instrument import Instrument
from ess.livedata.config.route_derivation import (
    gather_source_names,
    resolve_stream_names,
)
from ess.livedata.config.workflow_spec import (
    DETECTORS,
    MONITORS,
    REDUCTION,
    AuxInput,
    AuxSources,
    DefaultOutputs,
)
from ess.livedata.kafka.stream_mapping import InputStreamKey, StreamMapping


@pytest.fixture
def instrument_with_specs() -> Instrument:
    """Instrument with specs across multiple namespaces."""
    instrument = Instrument(
        name="test",
        detector_names=["det_a", "det_b", "det_c"],
        monitors=["mon_1", "mon_2"],
    )
    instrument.register_spec(
        group=DETECTORS,
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
        group=DETECTORS,
        name="special_view",
        version=1,
        title="Special",
        source_names=["det_a"],
        outputs=DefaultOutputs,
    )
    instrument.register_spec(
        group=REDUCTION,
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
        group=MONITORS,
        name="monitor_workflow",
        version=1,
        title="Monitor",
        source_names=["mon_1", "mon_2"],
        outputs=DefaultOutputs,
    )
    return instrument


class TestGatherSourceNames:
    def test_returns_empty_for_unknown_namespace(
        self, instrument_with_specs: Instrument
    ) -> None:
        result = gather_source_names(instrument_with_specs, "nonexistent")
        assert result == set()

    def test_detector_data_namespace(self, instrument_with_specs: Instrument) -> None:
        result = gather_source_names(instrument_with_specs, "detector_data")
        assert result == {"det_a", "det_b", "det_c", "motor_x", "motor_y"}

    def test_data_reduction_namespace(self, instrument_with_specs: Instrument) -> None:
        result = gather_source_names(instrument_with_specs, "data_reduction")
        assert result == {"det_a", "det_b", "mon_1", "mon_2"}

    def test_monitor_data_namespace(self, instrument_with_specs: Instrument) -> None:
        result = gather_source_names(instrument_with_specs, "monitor_data")
        assert result == {"mon_1", "mon_2"}

    def test_no_specs(self) -> None:
        instrument = Instrument(name="empty")
        result = gather_source_names(instrument, "detector_data")
        assert result == set()


class TestResolveStreamNames:
    def test_known_names_pass_through(self, infra_kwargs: dict) -> None:
        instrument = Instrument(
            name="test", detector_names=["det_a"], monitors=["mon_1"]
        )
        mapping = StreamMapping(
            instrument="test",
            detectors={InputStreamKey(topic="t", source_name="s"): "det_a"},
            monitors={InputStreamKey(topic="m", source_name="s"): "mon_1"},
            **infra_kwargs,
        )
        result = resolve_stream_names({"det_a", "mon_1"}, instrument, mapping)
        assert result == {"det_a", "mon_1"}

    def test_unknown_detector_name_expands_to_all_detectors(
        self, infra_kwargs: dict
    ) -> None:
        """Bifrost-like case: logical name not in LUT, but in detector_names."""
        instrument = Instrument(name="test", detector_names=["unified"], monitors=[])
        mapping = StreamMapping(
            instrument="test",
            detectors={
                InputStreamKey(topic="t1", source_name="s"): "phys_0",
                InputStreamKey(topic="t2", source_name="s"): "phys_1",
            },
            monitors={},
            **infra_kwargs,
        )
        result = resolve_stream_names({"unified"}, instrument, mapping)
        assert result == {"phys_0", "phys_1"}

    def test_unknown_monitor_name_expands_to_all_monitors(
        self, infra_kwargs: dict
    ) -> None:
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
            **infra_kwargs,
        )
        result = resolve_stream_names({"logical_mon"}, instrument, mapping)
        assert result == {"phys_mon_0", "phys_mon_1"}

    def test_mix_of_known_and_unknown(self, infra_kwargs: dict) -> None:
        instrument = Instrument(name="test", detector_names=["unified"], monitors=[])
        mapping = StreamMapping(
            instrument="test",
            detectors={
                InputStreamKey(topic="t", source_name="s"): "phys_0",
            },
            monitors={},
            logs={InputStreamKey(topic="l", source_name="s"): "motor_x"},
            **infra_kwargs,
        )
        result = resolve_stream_names({"unified", "motor_x"}, instrument, mapping)
        assert result == {"phys_0", "motor_x"}

    def test_empty_needed(self, infra_kwargs: dict) -> None:
        instrument = Instrument(name="test")
        mapping = StreamMapping(
            instrument="test", detectors={}, monitors={}, **infra_kwargs
        )
        result = resolve_stream_names(set(), instrument, mapping)
        assert result == set()
