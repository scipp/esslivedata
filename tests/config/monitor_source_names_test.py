# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Tests for monitor source name indexing.

These tests verify that monitor source names (cbm0, cbm1, cbm2, ...) are
correctly mapped to internal monitor names.

See https://confluence.ess.eu/display/ECDC/Kafka+Topics+Overview+for+Instruments
"""

import pytest

from ess.livedata.config import streams
from ess.livedata.config.instruments import available_instruments

# LOKI uses 0-based cbm source names (cbm0, cbm1, ...), verified against
# NeXus data file ``coda_loki_999999_00019778.hdf``.
_INSTRUMENTS_WITH_CBM0 = {'loki'}


@pytest.mark.parametrize(
    'instrument',
    [i for i in available_instruments() if i not in _INSTRUMENTS_WITH_CBM0],
)
def test_production_monitors_do_not_use_cbm0(instrument: str) -> None:
    """Most instruments use 1-based cbm source names (cbm1, cbm2, ...).

    LOKI is excluded because it uses 0-based indexing.
    """
    stream_mapping = streams.get_stream_mapping(instrument=instrument, dev=False)
    cbm_source_names = [
        key.source_name
        for key in stream_mapping.monitors
        if key.source_name.startswith('cbm')
    ]
    assert 'cbm0' not in cbm_source_names, (
        f"Monitor mapping for {instrument} includes cbm0, but this instrument "
        "is expected to use 1-indexed cbm source names (cbm1, cbm2, ...)"
    )


def test_loki_monitors_correctly_mapped() -> None:
    """LOKI monitors use 0-based cbm source names matching NeXus group indices.

    Verified against NeXus data file ``coda_loki_999999_00019778.hdf``:
    cbm0 -> beam_monitor_0, ..., cbm4 -> beam_monitor_4.
    """
    stream_mapping = streams.get_stream_mapping(instrument='loki', dev=False)
    actual_mapping = {
        key.source_name: value
        for key, value in stream_mapping.monitors.items()
        if key.source_name.startswith('cbm')
    }
    expected_mapping = {
        'cbm0': 'beam_monitor_0',
        'cbm1': 'beam_monitor_1',
        'cbm2': 'beam_monitor_2',
        'cbm3': 'beam_monitor_3',
        'cbm4': 'beam_monitor_4',
    }
    assert actual_mapping == expected_mapping


def test_bifrost_monitors_correctly_mapped() -> None:
    """Bifrost has specific named monitors that must map to correct cbm indices.

    The monitor list order must match the cbm indices in production:
    - cbm1 -> 090_frame_1 (first monitor)
    - cbm2 -> 097_frame_2 (second monitor)
    - etc.
    """
    stream_mapping = streams.get_stream_mapping(instrument='bifrost', dev=False)

    # Expected mapping based on Bifrost's monitor list in specs.py
    expected_mapping = {
        'cbm1': '090_frame_1',
        'cbm2': '097_frame_2',
        'cbm3': '110_frame_3',
        'cbm4': '111_psd0',
        'cbm5': 'bragg_peak_monitor',
    }

    actual_mapping = {
        key.source_name: value
        for key, value in stream_mapping.monitors.items()
        if key.source_name.startswith('cbm')
    }

    assert actual_mapping == expected_mapping, (
        f"Bifrost monitor mapping mismatch.\n"
        f"Expected: {expected_mapping}\n"
        f"Actual: {actual_mapping}"
    )


def test_dream_monitors_use_correct_cbm_indices() -> None:
    """DREAM monitors use NeXus group names (monitor_bunker, monitor_cave).

    These should map to cbm1, cbm2 with consistent 1-based indexing.
    """
    stream_mapping = streams.get_stream_mapping(instrument='dream', dev=False)

    cbm_to_monitor = {
        key.source_name: value
        for key, value in stream_mapping.monitors.items()
        if key.source_name.startswith('cbm')
    }

    # First monitor should be cbm1 -> monitor_bunker (consistent 1-based indexing)
    assert cbm_to_monitor.get('cbm1') == 'monitor_bunker', (
        "DREAM's first monitor (monitor_bunker) should be mapped to cbm1"
    )
    # cbm0 should not exist
    assert 'cbm0' not in cbm_to_monitor, "cbm0 should not exist in DREAM's mapping"
