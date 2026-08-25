# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for LOKI workflow spec registration."""

import pytest

from ess.livedata.config.instruments.loki.specs import instrument, loki_aux_sources


@pytest.mark.parametrize('input_name', ['incident_monitor', 'transmission_monitor'])
def test_monitor_choices_are_registered_monitors(input_name: str) -> None:
    """Guards against drift of the ``beam_monitor_mN`` names in NeXus.

    A choice naming a monitor the instrument does not stream would be selectable
    in the UI but never receive data.
    """
    aux_input = loki_aux_sources.inputs[input_name]

    assert set(aux_input.choices) <= set(instrument.monitors)
    assert aux_input.default in aux_input.choices
