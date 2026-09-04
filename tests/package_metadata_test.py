# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import tomllib
from pathlib import Path

from ess.livedata.config.instruments import available_instruments


def test_all_instruments_extra_composes_every_instrument_extra():
    with (Path(__file__).parents[1] / 'pyproject.toml').open('rb') as file:
        project = tomllib.load(file)['project']

    extras = project['optional-dependencies']
    instruments = available_instruments()
    assert set(instruments) <= extras.keys()

    (self_reference,) = extras['all-instruments']
    project_name, opening_bracket, referenced_extras = self_reference.partition('[')

    assert project_name == project['name']
    assert opening_bracket == '['
    assert referenced_extras.endswith(']')
    assert set(referenced_extras[:-1].split(',')) == set(instruments)
