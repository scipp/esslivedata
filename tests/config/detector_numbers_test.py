# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Every declared detector of every instrument has a ``detector_number``."""

import pytest

from ess.livedata.config.instrument import instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config

#: Detectors that publish images (ad00) rather than events have no pixel ids.
_IMAGE_DETECTORS = {('dummy', 'area_panel')}


def _detectors() -> list[tuple[str, str]]:
    params = []
    for instrument_name in available_instruments():
        get_config(instrument_name)
        params.extend(
            (instrument_name, detector)
            for detector in instrument_registry[instrument_name].detector_names
            if (instrument_name, detector) not in _IMAGE_DETECTORS
        )
    return params


@pytest.mark.parametrize(('instrument_name', 'detector'), _detectors())
def test_detector_number_resolves(instrument_name: str, detector: str) -> None:
    """``load_factories`` swallows a detector missing from the geometry file, so
    a renamed group would only show up when the detector is first used."""
    instrument = instrument_registry[instrument_name]
    instrument.load_factories()
    assert instrument.get_detector_number(detector).ndim >= 1
