# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Drift checks: regenerate each ``streams_parsed.py`` from its geometry file
and compare to the checked-in module.

If a geometry file is updated upstream, this test fails until the
corresponding ``streams_parsed.py`` is regenerated. Run

    python -m ess.livedata.nexus_helpers <geometry.nxs> --generate \
        --exclude <pattern> --output <path>

to refresh.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ess.livedata.handlers.detector_data_handler import get_nexus_geometry_filename
from ess.livedata.nexus_helpers import (
    extract_stream_info,
    filter_f144_streams,
    generate_streams_parsed_module,
)

_LOKI_PARSED = (
    Path(__file__).parent.parent.parent
    / 'src'
    / 'ess'
    / 'livedata'
    / 'config'
    / 'instruments'
    / 'loki'
    / 'streams_parsed.py'
)


def test_loki_streams_parsed_matches_geometry_file() -> None:
    try:
        geometry = get_nexus_geometry_filename('loki')
    except Exception as e:
        pytest.skip(f'LOKI geometry file unavailable: {e}')

    infos = filter_f144_streams(
        extract_stream_info(geometry), exclude_patterns=['beam_monitor']
    )
    expected = generate_streams_parsed_module(infos, source_file=str(geometry))
    actual = _LOKI_PARSED.read_text()

    if expected != actual:
        pytest.fail(
            'streams_parsed.py for LOKI is stale; regenerate with\n'
            f'  python -m ess.livedata.nexus_helpers {geometry} \\\n'
            "      --generate --exclude beam_monitor \\\n"
            f'      --output {_LOKI_PARSED}'
        )
