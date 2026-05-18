# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for :mod:`ess.livedata.config.stream`."""

from __future__ import annotations

import pytest

from ess.livedata.config import F144Stream, name_streams


def _parsed(path: str, *, source: str = 'src') -> F144Stream:
    return F144Stream(nexus_path=path, source=source, topic='topic', units='mm')


class TestNameStreams:
    def test_assigns_suggested_names_by_default(self) -> None:
        parsed = {
            '/entry/instrument/motor/value': _parsed('/entry/instrument/motor/value')
        }
        result = name_streams(parsed)
        assert list(result) == ['motor']
        assert result['motor'].nexus_path == '/entry/instrument/motor/value'

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
            name_streams(parsed, rename={'/entry/instrument/bar/value': 'foo'})

    def test_auxiliary_siblings_get_distinct_names(self) -> None:
        # idle_flag is not a primary-readback suffix, so it is preserved
        # and the parent NXlog is prepended for context.
        parsed = {
            '/entry/instrument/motor/value': _parsed('/entry/instrument/motor/value'),
            '/entry/instrument/motor/idle_flag': _parsed(
                '/entry/instrument/motor/idle_flag'
            ),
        }
        result = name_streams(parsed)
        assert set(result) == {'motor', 'motor_idle_flag'}
