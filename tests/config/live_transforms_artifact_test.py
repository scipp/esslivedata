# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Check live transforms against every registered geometry artifact.

A job finds the placeholders on the chains it reads when it is created, and needs
a stream for each. These tests find a placeholder without a stream, or a stream
no service would subscribe to, without having to create every job.
"""

from __future__ import annotations

import pytest
import scippnexus as snx
from scippnexus.field import DependsOn
from scippnexus.nxtransformations import parse_depends_on_chain

from ess.livedata.config.instrument import Instrument, instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename


def _placeholders(component: snx.Group) -> set[str]:
    """Paths on the component's chain whose value is a length-0 NXlog."""
    if 'depends_on' not in component:
        return set()
    depends_on = component['depends_on'][()]
    if not isinstance(depends_on, DependsOn):
        depends_on = DependsOn(parent=component.name, value=depends_on)
    chain = parse_depends_on_chain(component, depends_on)
    return {
        path
        for path, transform in chain.transformations.items()
        if getattr(transform.value, 'sizes', {}).get('time') == 0
    }


def _chain_placeholders(artifact: str) -> set[str]:
    """Placeholders on the chains of all detectors, monitors and the sample."""
    with snx.File(artifact, 'r') as f:
        entry = f['entry']
        components = [
            *entry['instrument'][snx.NXdetector].values(),
            *entry['instrument'][snx.NXmonitor].values(),
            *entry[snx.NXsample].values(),
        ]
        return {path for c in components for path in _placeholders(c)}


def _has_geometry_artifact(name: str) -> bool:
    try:
        get_nexus_geometry_filename(name)
    except ValueError:
        return False
    return True


@pytest.fixture(
    scope='module',
    params=[n for n in available_instruments() if _has_geometry_artifact(n)],
)
def instrument(request: pytest.FixtureRequest) -> Instrument:
    get_config(request.param)
    instrument = instrument_registry[request.param]
    instrument.load_factories()
    return instrument


def test_every_placeholder_has_a_stream(instrument: Instrument) -> None:
    artifact = str(get_nexus_geometry_filename(instrument.name))

    missing = _chain_placeholders(artifact) - set(instrument.transform_streams)

    assert not missing, (
        f"Placeholders in {artifact} with no stream of {instrument.name}: "
        f"{sorted(missing)}. A job reading such a chain cannot be created."
    )


def test_services_subscribe_to_every_placeholder(instrument: Instrument) -> None:
    artifact = str(get_nexus_geometry_filename(instrument.name))
    streams = instrument.transform_streams

    needed = {streams[path] for path in _chain_placeholders(artifact)}

    assert needed <= instrument.live_transform_streams
