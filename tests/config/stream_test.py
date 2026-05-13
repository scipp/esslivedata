# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for ``ess.livedata.config.stream``."""

from __future__ import annotations

import pytest

from ess.livedata.config import F144Stream, build_streams


def _parsed(name: str, *, path: str | None = None) -> F144Stream:
    return F144Stream(
        stream_name=name,
        nexus_path=path or f'/entry/{name}/value',
        source=f'{name}:SOURCE',
        topic='topic',
        units='mm',
    )


class TestBuildStreams:
    def test_passes_parsed_entries_through(self) -> None:
        a = _parsed('a')
        b = _parsed('b')
        result = build_streams([a, b])
        assert result == {'a': a, 'b': b}

    def test_override_by_stream_name_changes_field(self) -> None:
        result = build_streams([_parsed('a')], overrides={'a': {'units': 'K'}})
        assert result['a'].units == 'K'
        # Other fields preserved
        assert result['a'].source == 'a:SOURCE'

    def test_override_by_nexus_path_changes_field(self) -> None:
        result = build_streams(
            [_parsed('a')], overrides={'/entry/a/value': {'units': 'K'}}
        )
        assert result['a'].units == 'K'

    def test_override_renaming_rekeys_dict(self) -> None:
        result = build_streams(
            [_parsed('cryptic_name')],
            overrides={'cryptic_name': {'stream_name': 'temperature'}},
        )
        assert 'cryptic_name' not in result
        assert result['temperature'].stream_name == 'temperature'
        # Path is preserved
        assert result['temperature'].nexus_path == '/entry/cryptic_name/value'

    def test_override_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match='matches no parsed entry'):
            build_streams([_parsed('a')], overrides={'missing': {'units': 'K'}})

    def test_override_rename_collision_raises(self) -> None:
        with pytest.raises(ValueError, match='already exists'):
            build_streams(
                [_parsed('a'), _parsed('b')],
                overrides={'a': {'stream_name': 'b'}},
            )

    def test_synthetics_merged_in(self) -> None:
        parsed = _parsed('a')
        synth = F144Stream(stream_name='synthetic', units='K')
        result = build_streams([parsed], synthetics=[synth])
        assert result == {'a': parsed, 'synthetic': synth}

    def test_synthetic_name_collision_raises(self) -> None:
        synth = F144Stream(stream_name='a', units='K')
        with pytest.raises(ValueError, match='collides with a parsed entry'):
            build_streams([_parsed('a')], synthetics=[synth])

    def test_duplicate_parsed_names_raise(self) -> None:
        with pytest.raises(ValueError, match='Duplicate stream_name'):
            build_streams([_parsed('a'), _parsed('a', path='/entry/other/value')])

    def test_synthetic_has_no_topic_or_source(self) -> None:
        synth = F144Stream(stream_name='synth', units='K')
        assert synth.topic is None
        assert synth.source is None
        assert synth.nexus_path is None
