# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for :mod:`ess.livedata.config.stream`."""

from __future__ import annotations

import pytest

from ess.livedata.config import F144Stream, name_streams
from ess.livedata.config.stream import suggest_names


def _parsed(path: str, *, source: str = 'src') -> F144Stream:
    return F144Stream(nexus_path=path, source=source, topic='topic', units='mm')


class TestSuggestNames:
    def test_includes_value_leaf(self) -> None:
        names = suggest_names(['/entry/instrument/rotation_stage/value'])
        assert names == {
            '/entry/instrument/rotation_stage/value': 'rotation_stage_value'
        }

    def test_includes_value_log_leaf(self) -> None:
        names = suggest_names(['/entry/sample/sample_environment/HTR1/value_log'])
        assert names == {
            '/entry/sample/sample_environment/HTR1/value_log': 'HTR1_value_log'
        }

    def test_includes_aux_leaf(self) -> None:
        names = suggest_names(['/entry/instrument/rotation_stage/idle_flag'])
        assert names == {
            '/entry/instrument/rotation_stage/idle_flag': 'rotation_stage_idle_flag'
        }

    def test_filters_generic_containers(self) -> None:
        names = suggest_names(['/entry/instrument/wfm1/transformations/translation1'])
        assert names == {
            '/entry/instrument/wfm1/transformations/translation1': 'wfm1_translation1'
        }

    def test_disambiguates_via_parent_when_leaf_collides(self) -> None:
        names = suggest_names(
            [
                '/entry/instrument/005_PulseShapingChopper/phase',
                '/entry/instrument/006_FrameOverlapChopper/phase',
            ]
        )
        assert names == {
            '/entry/instrument/005_PulseShapingChopper/phase': (
                '005_PulseShapingChopper_phase'
            ),
            '/entry/instrument/006_FrameOverlapChopper/phase': (
                '006_FrameOverlapChopper_phase'
            ),
        }

    def test_single_meaningful_component_returned_bare(self) -> None:
        names = suggest_names(['/entry/sample/sample_environment/SETP_S1'])
        assert names == {'/entry/sample/sample_environment/SETP_S1': 'SETP_S1'}

    def test_distinguishes_primary_readback_and_aux(self) -> None:
        names = suggest_names(
            [
                '/entry/instrument/motor/value',
                '/entry/instrument/motor/idle_flag',
                '/entry/instrument/motor/target_value',
            ]
        )
        assert names == {
            '/entry/instrument/motor/value': 'motor_value',
            '/entry/instrument/motor/idle_flag': 'motor_idle_flag',
            '/entry/instrument/motor/target_value': 'motor_target_value',
        }

    def test_disambiguates_two_paths_collapsing_to_same_tail(self) -> None:
        # Two NXlogs in different ancestors with the same parent + leaf.
        names = suggest_names(
            [
                '/entry/instrument/foo/temperature/value',
                '/entry/instrument/bar/temperature/value',
            ]
        )
        assert names == {
            '/entry/instrument/foo/temperature/value': 'foo_temperature_value',
            '/entry/instrument/bar/temperature/value': 'bar_temperature_value',
        }

    def test_falls_back_to_unfiltered_path_when_only_generic_ancestors_differ(
        self,
    ) -> None:
        # The filtered tails collide (``foo_value``); resolving requires
        # putting a generic ancestor back to disambiguate.
        names = suggest_names(
            [
                '/entry/instrument/foo/value',
                '/entry/sample/foo/value',
            ]
        )
        assert names == {
            '/entry/instrument/foo/value': 'instrument_foo_value',
            '/entry/sample/foo/value': 'sample_foo_value',
        }


class TestNameStreams:
    def test_assigns_suggested_names_by_default(self) -> None:
        parsed = {
            '/entry/instrument/motor/value': _parsed('/entry/instrument/motor/value')
        }
        result = name_streams(parsed)
        assert list(result) == ['motor_value']
        assert result['motor_value'].nexus_path == '/entry/instrument/motor/value'

    def test_rename_overrides_suggestion(self) -> None:
        path = '/entry/instrument/detector_arm/detector_rotation/value'
        parsed = {path: _parsed(path)}
        result = name_streams(parsed, rename={path: 'detector_rotation'})
        assert list(result) == ['detector_rotation']

    def test_unknown_rename_key_raises(self) -> None:
        parsed = {'/entry/foo/value': _parsed('/entry/foo/value')}
        with pytest.raises(ValueError, match='not in parsed'):
            name_streams(parsed, rename={'/missing': 'x'})

    def test_collision_after_rename_raises(self) -> None:
        parsed = {
            '/entry/instrument/foo/value': _parsed('/entry/instrument/foo/value'),
            '/entry/instrument/bar/value': _parsed('/entry/instrument/bar/value'),
        }
        with pytest.raises(ValueError, match='collides'):
            name_streams(parsed, rename={'/entry/instrument/bar/value': 'foo_value'})

    def test_sibling_leaves_get_distinct_names(self) -> None:
        parsed = {
            '/entry/instrument/motor/value': _parsed('/entry/instrument/motor/value'),
            '/entry/instrument/motor/idle_flag': _parsed(
                '/entry/instrument/motor/idle_flag'
            ),
        }
        result = name_streams(parsed)
        assert set(result) == {'motor_value', 'motor_idle_flag'}
