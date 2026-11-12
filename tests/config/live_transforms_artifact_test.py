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
from ess.livedata.nexus_helpers import live_transform_paths
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename
from ess.livedata.workflows.dynamic_transforms import placeholders


def _chain_placeholders(artifact: str) -> set[str]:
    """Placeholders as a job finds them: on the chains of the components it reads."""
    found = set()
    with snx.File(artifact, 'r') as f:
        entry = f['entry']
        for component in (
            *entry['instrument'][snx.NXdetector].values(),
            *entry['instrument'][snx.NXmonitor].values(),
            *entry[snx.NXsample].values(),
        ):
            if 'depends_on' in component:
                depends_on = component['depends_on'][()]
                if not isinstance(depends_on, DependsOn):
                    depends_on = DependsOn(parent=component.name, value=depends_on)
                found |= placeholders(parse_depends_on_chain(component, depends_on))
    return found


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

    missing = live_transform_paths(artifact) - set(instrument.transform_streams)

    assert not missing, (
        f"Placeholders in {artifact} with no stream of {instrument.name}: "
        f"{sorted(missing)}. A job reading such a chain cannot be created."
    )


def test_services_subscribe_to_what_jobs_find(instrument: Instrument) -> None:
    """Services scan the file and jobs read chains: the two must agree, or a job would
    wait for a stream no service subscribes to.
    """
    artifact = str(get_nexus_geometry_filename(instrument.name))

    assert _chain_placeholders(artifact) <= set(instrument.transform_streams)
