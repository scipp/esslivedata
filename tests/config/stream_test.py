# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for :mod:`ess.livedata.config.stream`."""

from __future__ import annotations

import pytest

from ess.livedata.config import Device, F144Stream, name_streams
from ess.livedata.config.stream import Stream, suggest_names


def _parsed(path: str, *, source: str = 'src') -> F144Stream:
    return F144Stream(nexus_path=path, source=source, topic='topic', units='mm')


class TestSuggestNames:
    def test_includes_value_leaf(self) -> None:
        names = suggest_names(['/entry/instrument/rotation_stage/value'])
        assert names == {
            '/entry/instrument/rotation_stage/value': 'rotation_stage/value'
        }

    def test_includes_value_log_leaf(self) -> None:
        names = suggest_names(['/entry/sample/sample_environment/HTR1/value_log'])
        assert names == {
            '/entry/sample/sample_environment/HTR1/value_log': 'HTR1/value_log'
        }

    def test_includes_aux_leaf(self) -> None:
        names = suggest_names(['/entry/instrument/rotation_stage/idle_flag'])
        assert names == {
            '/entry/instrument/rotation_stage/idle_flag': 'rotation_stage/idle_flag'
        }

    def test_filters_generic_containers(self) -> None:
        names = suggest_names(['/entry/instrument/wfm1/transformations/translation1'])
        assert names == {
            '/entry/instrument/wfm1/transformations/translation1': 'wfm1/translation1'
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
                '005_PulseShapingChopper/phase'
            ),
            '/entry/instrument/006_FrameOverlapChopper/phase': (
                '006_FrameOverlapChopper/phase'
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
            '/entry/instrument/motor/value': 'motor/value',
            '/entry/instrument/motor/idle_flag': 'motor/idle_flag',
            '/entry/instrument/motor/target_value': 'motor/target_value',
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
            '/entry/instrument/foo/temperature/value': 'foo/temperature/value',
            '/entry/instrument/bar/temperature/value': 'bar/temperature/value',
        }

    def test_falls_back_to_unfiltered_path_when_only_generic_ancestors_differ(
        self,
    ) -> None:
        # The filtered tails collide (``foo/value``); resolving requires
        # putting a generic ancestor back to disambiguate.
        names = suggest_names(
            [
                '/entry/instrument/foo/value',
                '/entry/sample/foo/value',
            ]
        )
        assert names == {
            '/entry/instrument/foo/value': 'instrument/foo/value',
            '/entry/sample/foo/value': 'sample/foo/value',
        }

    def test_min_depth_one_returns_bare_unique_leaf(self) -> None:
        names = suggest_names(
            ['/entry/instrument/114_sample_stack/rotation_stage'], min_depth=1
        )
        assert names == {
            '/entry/instrument/114_sample_stack/rotation_stage': 'rotation_stage'
        }

    def test_min_depth_one_collapses_doubled_prefix(self) -> None:
        # ``transformations`` filters out; depth=2 would emit
        # ``detector_tank_angle/detector_tank_angle_r0``.
        names = suggest_names(
            [
                '/entry/instrument/detector_tank_angle/transformations/'
                'detector_tank_angle_r0'
            ],
            min_depth=1,
        )
        assert names == {
            '/entry/instrument/detector_tank_angle/transformations/'
            'detector_tank_angle_r0': 'detector_tank_angle_r0'
        }

    def test_min_depth_one_extends_on_collision(self) -> None:
        names = suggest_names(
            [
                '/entry/instrument/wfm1/translation1',
                '/entry/instrument/wfm2/translation1',
            ],
            min_depth=1,
        )
        assert names == {
            '/entry/instrument/wfm1/translation1': 'wfm1/translation1',
            '/entry/instrument/wfm2/translation1': 'wfm2/translation1',
        }

    def test_forbidden_extends_to_longer_tail(self) -> None:
        names = suggest_names(
            ['/entry/instrument/parent/rotation_stage'],
            min_depth=1,
            forbidden={'rotation_stage'},
        )
        assert names == {
            '/entry/instrument/parent/rotation_stage': 'parent/rotation_stage'
        }


class TestNameStreams:
    def test_assigns_suggested_names_by_default(self) -> None:
        parsed = {
            '/entry/instrument/motor/value': _parsed('/entry/instrument/motor/value')
        }
        result = name_streams(parsed)
        assert list(result) == ['motor/value']
        assert result['motor/value'].nexus_path == '/entry/instrument/motor/value'

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
            name_streams(parsed, rename={'/entry/instrument/bar/value': 'foo/value'})

    def test_sibling_leaves_get_distinct_names(self) -> None:
        # RBV alone is just an F144Stream; pair with an idle_flag to also
        # auto-detect a Device named after the parent group.
        parsed = {
            '/entry/instrument/motor/value': _parsed(
                '/entry/instrument/motor/value', source='X.RBV'
            ),
            '/entry/instrument/motor/idle_flag': _parsed(
                '/entry/instrument/motor/idle_flag', source='X.DMOV'
            ),
        }
        result = name_streams(parsed)
        assert set(result) == {'motor/value', 'motor/idle_flag', 'motor'}


class TestDeviceDetection:
    def _motor(self, source: str, units: str = 'mm') -> F144Stream:
        return F144Stream(source=source, topic='topic', units=units)

    def test_emits_device_for_rbv_plus_val_plus_dmov(self) -> None:
        parsed = {
            '/entry/instrument/m/value': self._motor('X.RBV'),
            '/entry/instrument/m/target_value': self._motor('X.VAL'),
            '/entry/instrument/m/idle_flag': self._motor(
                'X.DMOV', units='dimensionless'
            ),
        }
        result = name_streams(parsed)
        assert 'm' in result
        device = result['m']
        assert isinstance(device, Device)
        assert device.value == 'm/value'
        assert device.target == 'm/target_value'
        assert device.settled == 'm/idle_flag'
        assert device.units == 'mm'

    def test_emits_device_for_rbv_plus_val_only(self) -> None:
        parsed = {
            '/entry/instrument/m/value': self._motor('X.RBV'),
            '/entry/instrument/m/target_value': self._motor('X.VAL'),
        }
        result = name_streams(parsed)
        device = result['m']
        assert isinstance(device, Device)
        assert device.target == 'm/target_value'
        assert device.settled is None

    def test_emits_device_for_rbv_plus_dmov_only(self) -> None:
        parsed = {
            '/entry/instrument/m/value': self._motor('X.RBV'),
            '/entry/instrument/m/idle_flag': self._motor(
                'X.DMOV', units='dimensionless'
            ),
        }
        result = name_streams(parsed)
        device = result['m']
        assert isinstance(device, Device)
        assert device.target is None
        assert device.settled == 'm/idle_flag'

    def test_no_device_for_lone_rbv(self) -> None:
        parsed = {
            '/entry/instrument/m/value': self._motor('X.RBV'),
        }
        result = name_streams(parsed)
        assert all(not isinstance(s, Device) for s in result.values())

    def test_classifies_by_source_suffix_regardless_of_child_name(self) -> None:
        # tbl-style: non-canonical NeXus child names but canonical EPICS
        # source suffixes. Classification is by source, so the child name is
        # incidental.
        parsed = {
            '/entry/instrument/m/position_readback': self._motor(
                'TBL-AttChg:MC-LinY-01:Mtr.RBV'
            ),
            '/entry/instrument/m/position_setpoint': self._motor(
                'TBL-AttChg:MC-LinY-01:Mtr.VAL'
            ),
        }
        result = name_streams(parsed)
        device = result['m']
        assert isinstance(device, Device)
        assert device.value == 'm/position_readback'
        assert device.target == 'm/position_setpoint'

    def test_ignores_unclassifiable_source_suffix(self) -> None:
        # Piezo potentiometer readback (-PosReadback) sits alongside a real
        # motor on the bifrost slit blades; it is not a motor RBV and must
        # not participate in device synthesis. It remains as a plain stream.
        parsed = {
            '/entry/instrument/blade/value': self._motor('BIFRO-SpSl1:MC.RBV'),
            '/entry/instrument/blade/target_value': self._motor('BIFRO-SpSl1:MC.VAL'),
            '/entry/instrument/blade/potentiometer_value': self._motor(
                'BIFRO-SpSl1:MC-SlZm-01:PzMtr-PosReadback'
            ),
        }
        result = name_streams(parsed)
        device = result['blade']
        assert isinstance(device, Device)
        assert device.value == 'blade/value'
        assert device.target == 'blade/target_value'
        assert isinstance(result['blade/potentiometer_value'], F144Stream)

    def test_ignores_substream_with_no_source(self) -> None:
        # Synthesised in-process F144Stream entries have source=None and are
        # unclassifiable; they do not contribute to device detection.
        parsed: dict[str, Stream] = {
            '/entry/instrument/m/value': F144Stream(units='mm'),
            '/entry/instrument/m/target_value': self._motor('X.VAL'),
        }
        result = name_streams(parsed)
        assert all(not isinstance(s, Device) for s in result.values())

    def test_duplicate_role_in_parent_raises(self) -> None:
        # Two children classify as 'value' under the same parent: ambiguous,
        # must fail loudly rather than silently dropping one.
        parsed = {
            '/entry/instrument/m/value': self._motor('X.RBV'),
            '/entry/instrument/m/value_alt': self._motor('Y.RBV'),
            '/entry/instrument/m/target_value': self._motor('X.VAL'),
        }
        with pytest.raises(ValueError, match='two children classify as'):
            name_streams(parsed)

    def test_unit_mismatch_raises(self) -> None:
        parsed = {
            '/entry/instrument/m/value': self._motor('X.RBV', units='mm'),
            '/entry/instrument/m/target_value': self._motor('X.VAL', units='deg'),
        }
        with pytest.raises(ValueError, match='units must match'):
            name_streams(parsed)

    def test_rename_device_parent_path(self) -> None:
        parsed = {
            '/entry/instrument/big_name/m/value': self._motor('X.RBV'),
            '/entry/instrument/big_name/m/idle_flag': self._motor(
                'X.DMOV', units='dimensionless'
            ),
        }
        result = name_streams(parsed, rename={'/entry/instrument/big_name/m': 'short'})
        assert 'short' in result
        assert isinstance(result['short'], Device)
        # Substream pointers reference auto-suggested names.
        assert result['short'].value == 'm/value'

    def test_invalid_rename_path_raises(self) -> None:
        parsed = {
            '/entry/instrument/m/value': self._motor('X.RBV'),
            '/entry/instrument/m/idle_flag': self._motor('X.DMOV'),
        }
        with pytest.raises(ValueError, match='not in parsed'):
            name_streams(parsed, rename={'/no/such/path': 'foo'})

    def test_device_name_drops_redundant_parent_prefix(self) -> None:
        # Parent group's leaf already encodes the entity name; min_depth=1
        # in the device pass keeps the name short instead of doubling.
        parsed = {
            '/entry/instrument/114_sample_stack/rotation_stage/value': self._motor(
                'X.RBV', units='deg'
            ),
            '/entry/instrument/114_sample_stack/rotation_stage/idle_flag': self._motor(
                'X.DMOV', units='dimensionless'
            ),
        }
        result = name_streams(parsed)
        assert 'rotation_stage' in result
        assert isinstance(result['rotation_stage'], Device)

    def test_device_name_collides_with_substream_extends(self) -> None:
        # Two sibling devices with the same parent-leaf name: substream pass
        # uses parent-prefixed names; device pass starts at depth=1, both
        # collide, extends to depth=2 like substreams.
        parsed = {
            '/entry/instrument/a/motor/value': self._motor('X.RBV'),
            '/entry/instrument/a/motor/idle_flag': self._motor(
                'X.DMOV', units='dimensionless'
            ),
            '/entry/instrument/b/motor/value': self._motor('Y.RBV'),
            '/entry/instrument/b/motor/idle_flag': self._motor(
                'Y.DMOV', units='dimensionless'
            ),
        }
        result = name_streams(parsed)
        assert isinstance(result['a/motor'], Device)
        assert isinstance(result['b/motor'], Device)
