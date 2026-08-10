# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""The dev-mode slider configs must name streams the instrument actually declares.

The widget publishes f144 under the stream name straight from the JSON. A name
the instrument does not declare is dropped as unmapped on the consumer side --
no error anywhere, just workflows that never leave ``pending_context``.
"""

import json
from pathlib import Path

import pytest

from ess.livedata.config import instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.stream import Device

_CONFIG_DIR = Path(__file__).resolve().parents[3] / 'configs'
_STREAM_KEYS = ('stream_name', 'value_stream', 'target_stream', 'idle_stream')


def _instrument(name: str):
    get_config(name)
    config = instrument_registry[name]
    config.load_factories()
    return config


def _configured_instruments() -> list[str]:
    return sorted(
        p.stem[len('log_producer_') :] for p in _CONFIG_DIR.glob('log_producer_*.json')
    )


def _sliders(name: str) -> list[dict]:
    with open(_CONFIG_DIR / f'log_producer_{name}.json') as f:
        return json.load(f)['sliders']


@pytest.fixture(params=_configured_instruments())
def instrument_name(request: pytest.FixtureRequest) -> str:
    return request.param


def test_slider_streams_are_declared(instrument_name: str) -> None:
    declared = set(_instrument(instrument_name).streams)
    for slider in _sliders(instrument_name):
        for key in _STREAM_KEYS:
            if (stream := slider.get(key)) is not None:
                assert stream in declared, (
                    f"{instrument_name} slider {slider['label']!r} names undeclared "
                    f"stream {stream!r} for {key}"
                )


def test_gating_devices_are_drivable(instrument_name: str) -> None:
    """Every device a workflow is gated on must have sliders for its substreams.

    An uncovered device leaves its workflows unstartable in dev: the synthesizer
    emits nothing until all substreams have been seen at least once.
    """
    config = _instrument(instrument_name)
    driven = {
        slider[key]
        for slider in _sliders(instrument_name)
        for key in _STREAM_KEYS
        if key in slider
    }
    for binding in config.context_bindings:
        stream = config.streams[binding.stream_name]
        if not isinstance(stream, Device):
            continue
        missing = set(stream.substream_names) - driven
        assert not missing, (
            f"{instrument_name} gates on device {binding.stream_name!r} but its "
            f"log producer config drives no {sorted(missing)}"
        )
