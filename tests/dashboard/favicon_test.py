# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from urllib.parse import unquote

import pytest

from ess.livedata.config.instruments import available_instruments
from ess.livedata.dashboard.favicon import instrument_favicon


@pytest.mark.parametrize('instrument', available_instruments())
def test_every_instrument_has_a_favicon(instrument: str) -> None:
    uri = instrument_favicon(instrument)
    assert uri.startswith('data:image/svg+xml,')
    assert f'>{instrument[:2].capitalize()}</text>' in unquote(uri)


def test_unknown_instrument_raises() -> None:
    with pytest.raises(KeyError):
        instrument_favicon('nonexistent')
