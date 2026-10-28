# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Audit motion bindings against the registered geometry artifacts.

Check transform paths, stream identities, and coverage of every empty NXlog.
ESTIA's sample chains and aliases require explicit discovery; known exceptions
are recorded in ``_KNOWN_ORPHAN_NXLOGS``.
"""

from __future__ import annotations

import h5py
import pytest
import scippnexus as snx
from scippnexus.field import DependsOn
from scippnexus.nxtransformations import TransformationChain, parse_depends_on_chain

from ess.livedata.config.instrument import Instrument, instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.config.stream import ContextBinding, Device
from ess.livedata.config.value_log import ValueLog
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename


def _is_chain_patch(ci: ContextBinding) -> bool:
    key = ci.workflow_key
    return isinstance(key, type) and issubclass(key, ValueLog)


def _chain_patch_inputs(instrument: Instrument) -> list[ContextBinding]:
    return [b for b in instrument.context_bindings if _is_chain_patch(b)]


def _load_chain(artifact: str, source_name: str) -> TransformationChain | None:
    """Load a chain by source name or absolute path; None if unplaced."""
    parent_path = (
        source_name
        if source_name.startswith('/')
        else f'/entry/instrument/{source_name}'
    )
    with snx.File(artifact, 'r') as f:
        comp = f[parent_path]
        try:
            depends_on = comp['depends_on'][()]
        except KeyError:
            return None
        if not isinstance(depends_on, DependsOn):
            depends_on = DependsOn(parent=parent_path, value=depends_on)
        return parse_depends_on_chain(comp, depends_on)


def _chain_paths(artifact: str, source_name: str) -> list[str]:
    chain = _load_chain(artifact, source_name)
    return list(chain.transformations) if chain is not None else []


def _empty_nxlogs(artifact: str, source_name: str) -> set[str]:
    """All paths along the chain whose values are length-0 NXlogs."""
    chain = _load_chain(artifact, source_name)
    if chain is None:
        return set()
    return {
        path
        for path, transform in chain.transformations.items()
        if getattr(transform.value, 'sizes', {}).get('time') == 0
    }


def _instruments_with_chain_patch_bindings() -> list[str]:
    cases = []
    for name in available_instruments():
        get_config(name)
        inst = instrument_registry[name]
        inst.load_factories()
        if _chain_patch_inputs(inst):
            cases.append(name)
    return cases


@pytest.fixture(scope='module', params=_instruments_with_chain_patch_bindings())
def instrument(request) -> Instrument:
    name = request.param
    get_config(name)
    inst = instrument_registry[name]
    inst.load_factories()
    return inst


def test_transform_paths_match_artifact(instrument: Instrument) -> None:
    """Each patch must target a transform in its consumer's chain."""
    artifact = str(get_nexus_geometry_filename(instrument.name))
    for binding in _chain_patch_inputs(instrument):
        path = instrument.chain_patch_path(binding)
        for source_name in binding.dependent_sources:
            chain = _chain_paths(artifact, source_name)
            assert path in chain, (
                f"Binding {binding.stream_name!r} resolves transform path "
                f"{path!r} in consumers of {source_name!r}, but it does not "
                f"appear on the depends_on chain resolved from the artifact "
                f"({artifact}). Walked: {chain}"
            )


def test_consumers_subset_of_registered_sources(instrument: Instrument) -> None:
    """Each consumer must be a registered source on the instrument."""
    valid = set(instrument.detector_names) | set(instrument.monitors)
    for binding in _chain_patch_inputs(instrument):
        unknown = binding.dependent_sources - valid
        assert not unknown, (
            f"Binding {binding.stream_name!r} declares unknown consumers "
            f"{unknown}; valid sources: {sorted(valid)}"
        )


# Unbound placeholders tracked for follow-up: (instrument, source) -> path.
_KNOWN_ORPHAN_NXLOGS: dict[tuple[str, str], str] = {
    # See loki/specs.py: m4 trans_20 needs either a make_geometry_nexus.py
    # change to share the carriage NXlog, or a separate f144 stream
    # registration. Tracked as follow-up to issue #922.
    (
        'loki',
        'beam_monitor_m4',
    ): '/entry/instrument/beam_monitor_m4/transformations/trans_20',
}


def test_no_orphan_empty_nxlogs(instrument: Instrument) -> None:
    """Every empty transform must have a binding for this source."""
    artifact = str(get_nexus_geometry_filename(instrument.name))
    sources = list(instrument.detector_names) + list(instrument.monitors)
    for source_name in sources:
        covered = {
            instrument.chain_patch_path(binding)
            for binding in _chain_patch_inputs(instrument)
            if source_name in binding.dependent_sources
        }
        unbound = _empty_nxlogs(artifact, source_name) - covered
        if known := _KNOWN_ORPHAN_NXLOGS.get((instrument.name, source_name)):
            unbound.discard(known)
        assert not unbound, (
            f'{instrument.name}/{source_name} has unbound transforms: {sorted(unbound)}'
        )


@pytest.fixture(scope='module')
def estia_motion():
    from ess.livedata.config.instruments.estia.live_geometry import load_live_geometry

    get_config('estia')
    instrument = instrument_registry['estia']
    instrument.load_factories()
    artifact = str(get_nexus_geometry_filename('estia'))
    return instrument, artifact, load_live_geometry(artifact, instrument)


@pytest.mark.parametrize(
    ('component_type', 'component_path'),
    [
        (snx.NXsample, '/entry/sample'),
        (snx.NXdetector, '/entry/instrument/multiblade_detector'),
    ],
)
def test_estia_discovered_motion_bindings_match_artifact(
    estia_motion, component_type, component_path
):
    instrument, artifact, geometry = estia_motion
    bindings = geometry.bindings[component_type]
    covered = {path for path, _ in bindings}
    assert covered == _empty_nxlogs(artifact, component_path)

    names_by_key = {key: name for name, key in geometry.context_keys.items()}
    with h5py.File(artifact) as f:
        for path, key in bindings:
            name = names_by_key[key]
            assert instrument.offered_context_streams[key] == name
            stream = instrument.streams[name]
            if isinstance(stream, Device):
                stream = instrument.streams[stream.value]
            assert f[path].attrs['source'] == stream.source
            assert f[path].attrs['topic'] == stream.topic
