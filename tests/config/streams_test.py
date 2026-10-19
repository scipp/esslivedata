# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.config import instrument_registry, streams
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.kafka import InputStreamKey, StreamMapping

#: Instruments that merge many physical streams into one logical detector, and
#: for which a declared detector therefore has no LUT entry of its own. Bifrost
#: merges 45 triplets into ``unified_detector``; see ``resolve_stream_names``.
MERGING_INSTRUMENTS = {'bifrost'}


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


@pytest.mark.parametrize('dev', [True, False], ids=['dev', 'production'])
@pytest.mark.parametrize(
    'instrument', sorted(set(available_instruments()) - MERGING_INSTRUMENTS)
)
def test_every_declared_detector_is_the_target_of_an_input_stream(
    instrument: str, dev: bool
) -> None:
    """A detector no LUT entry names receives nothing, silently.

    ``resolve_stream_names`` subscribes to every detector topic when a spec's
    source name matches no LUT value, which is what lets Bifrost's merge work.
    Everywhere else that fallback masks the mistake instead of surfacing it:
    the service subscribes, the messages arrive tagged with a stream name no
    job consumes, and the detector looks idle rather than misconfigured.
    """
    get_config(instrument)  # Register
    declared = set(instrument_registry[instrument].detector_names)
    mapping = streams.get_stream_mapping(instrument=instrument, dev=dev)
    served = set(mapping.detectors.values()) | set(mapping.area_detectors.values())

    assert declared <= served


def test_nmx_panels_are_told_apart_by_topic_alone() -> None:
    # The file writer records source='nmx' and topic='nmx_detector_p{i}' on
    # every panel's NXevent_data group, so the topic carries the whole
    # distinction and the panel index has to line up with it.
    mapping = streams.get_stream_mapping(instrument='nmx', dev=False)

    assert mapping.detectors == {
        InputStreamKey(topic=f'nmx_detector_p{panel}', source_name='nmx'): (
            f'detector_panel_{panel}'
        )
        for panel in range(3)
    }


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
            livedata_context_topic='test_context',
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
            livedata_context_topic='test_context',
            livedata_roi_topic='test_roi',
            livedata_status_topic='test_status',
        )
        assert mapping.log_topics == {'motion', 'sensors'}
        assert mapping.logs == logs


def test_log_source_prefix_is_expected_before_every_log_source() -> None:
    # A simulation serves its process variables under a prefix, so that they
    # cannot be mistaken for the real instrument's, and its logs carry it.
    plain = streams.get_stream_mapping(instrument='bifrost', dev=False)
    prefixed = streams.get_stream_mapping(
        instrument='bifrost', dev=False, log_source_prefix='mcstas:'
    )

    assert prefixed.logs == {
        InputStreamKey(topic=k.topic, source_name=f'mcstas:{k.source_name}'): v
        for k, v in plain.logs.items()
    }
    assert (
        InputStreamKey(
            topic='bifrost_motion', source_name='mcstas:BIFRO-SpRot:MC-RotZ-01:Mtr.RBV'
        )
        in prefixed.logs
    )
    assert prefixed.detectors == plain.detectors
    assert prefixed.monitors == plain.monitors
    assert prefixed.topics == plain.topics


def test_no_log_source_prefix_leaves_the_mapping_as_it_is() -> None:
    plain = streams.get_stream_mapping(instrument='bifrost', dev=False)
    assert plain.with_log_source_prefix('') is plain


def test_monitor_sources_by_name_expects_each_monitor_under_its_own_name() -> None:
    # A simulation names its monitor streams after the monitors' NeXus groups.
    plain = streams.get_stream_mapping(instrument='bifrost', dev=False)
    named = streams.get_stream_mapping(
        instrument='bifrost', dev=False, monitor_sources_by_name=True
    )

    assert named.monitors == {
        InputStreamKey(topic=k.topic, source_name=v): v
        for k, v in plain.monitors.items()
    }
    assert (
        InputStreamKey(topic='bifrost_beam_monitor', source_name='psc_monitor')
        in named.monitors
    )
    assert named.detectors == plain.detectors
    assert named.logs == plain.logs
    assert named.topics == plain.topics
