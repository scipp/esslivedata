# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Per-instrument ``streams_parsed.py`` sanity + drift checks.

Each instrument's ``streams_parsed.py`` is auto-generated from a coda HDF5
file checked out under ``/workspace/esslivedata/`` in this devcontainer.
The coda files are not in pooch (the production geometry files lag behind
upstream), so the drift check below runs only when the local copies are
available; it skips otherwise.

To refresh after a coda file update:

    python -m ess.livedata.nexus_helpers <coda.hdf> --generate \
        --output src/ess/livedata/config/instruments/<inst>/streams_parsed.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ess.livedata.nexus_helpers import (
    extract_stream_info,
    filter_f144_streams,
    generate_streams_parsed_module,
)

_REPO_ROOT = Path(__file__).parent.parent.parent

#: Mapping ``instrument -> coda file basename``. The basename is matched
#: against ``/workspace/esslivedata/`` so devcontainer copies are picked up.
_CODA_SOURCES: dict[str, str] = {
    'bifrost': 'coda_bifrost_999999_00002625.hdf',
    'estia': 'coda_estia_999999_00027641.hdf',
    'loki': 'coda_loki_999999_00002438.hdf',
    'nmx': 'coda_nmx_999999_00002449.hdf',
    'odin': 'coda_odin_999999_00000800.hdf',
    'tbl': 'coda_tbl_999999_00023688.hdf',
}


def _parsed_path(instrument: str) -> Path:
    return (
        _REPO_ROOT
        / 'src'
        / 'ess'
        / 'livedata'
        / 'config'
        / 'instruments'
        / instrument
        / 'streams_parsed.py'
    )


@pytest.mark.parametrize('instrument', sorted(_CODA_SOURCES))
def test_streams_parsed_present(instrument: str) -> None:
    """Every instrument with a registered coda source ships a parsed module."""
    assert _parsed_path(instrument).is_file()


@pytest.mark.parametrize('instrument', sorted(_CODA_SOURCES))
def test_streams_parsed_matches_coda_source(instrument: str) -> None:
    """Drift check: regenerate from the coda file and compare bytes-for-bytes.

    Skipped when the coda file isn't present (e.g. in CI without the
    devcontainer's local copy).
    """
    coda = _REPO_ROOT / _CODA_SOURCES[instrument]
    if not coda.is_file():
        pytest.skip(f'coda file unavailable: {coda}')

    infos = filter_f144_streams(extract_stream_info(coda))
    expected = generate_streams_parsed_module(infos, source_file=str(coda))
    actual = _parsed_path(instrument).read_text()

    if expected != actual:
        pytest.fail(
            f'streams_parsed.py for {instrument} is stale; regenerate with\n'
            f'  python -m ess.livedata.nexus_helpers {coda} \\\n'
            f'      --generate --output {_parsed_path(instrument)}'
        )
