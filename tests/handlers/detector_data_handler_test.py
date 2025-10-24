# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess.livedata import StreamKind
from ess.livedata.config import instrument_registry
from ess.livedata.config.instrument import Instrument
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.core.handler import StreamId
from ess.livedata.handlers.detector_data_handler import (
    DetectorHandlerFactory,
    get_nexus_geometry_filename,
)


def get_instrument(instrument_name: str) -> Instrument:
    _ = get_config(instrument_name)  # Load the module to register the instrument
    instrument = instrument_registry[instrument_name]
    instrument.load_factories()
    return instrument


@pytest.mark.parametrize('instrument', ['dream', 'loki'])
def test_get_nexus_filename_returns_file_for_given_date(instrument: str) -> None:
    filename = get_nexus_geometry_filename(
        instrument, date=sc.datetime('2025-01-02T00:00:00')
    )
    assert str(filename).endswith(f'geometry-{instrument}-2025-01-01.nxs')


def test_get_nexus_filename_uses_current_date_by_default() -> None:
    auto = get_nexus_geometry_filename('dream')
    explicit = get_nexus_geometry_filename('dream', date=sc.datetime('now'))
    assert auto == explicit


def test_get_nexus_filename_raises_if_instrument_unknown() -> None:
    with pytest.raises(ValueError, match='No geometry files found for instrument'):
        get_nexus_geometry_filename('abcde', date=sc.datetime('2025-01-01T00:00:00'))


def test_get_nexus_filename_raises_if_datetime_out_of_range() -> None:
    with pytest.raises(ValueError, match='No geometry file found for given date'):
        get_nexus_geometry_filename('dream', date=sc.datetime('2020-01-01T00:00:00'))


@pytest.mark.parametrize('instrument_name', available_instruments())
def test_factory_can_create_preprocessor(instrument_name: str) -> None:
    instrument = get_instrument(instrument_name)
    factory = DetectorHandlerFactory(instrument=instrument)
    for name in instrument.detector_names:
        _ = factory.make_preprocessor(
            StreamId(kind=StreamKind.DETECTOR_EVENTS, name=name)
        )
