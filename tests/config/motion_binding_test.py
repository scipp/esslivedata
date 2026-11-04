# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Validate chain-patch :class:`ContextBinding` bindings against the
currently-registered geometry artifact.

For every chain-patch :class:`ContextBinding` (one whose ``workflow_key``
is a :class:`ValueLog` subclass), walks the ``depends_on`` chain of
every declared consumer (a source, or the sample at ``/entry/sample``) in the
artifact and confirms the path appears.
Catches typos and orphaned bindings before runtime.

Also flags, for every instrument with a geometry artifact, any empty NXlog
placeholder on a registered source or the sample that no binding covers --
except those in :data:`_KNOWN_ORPHAN_NXLOGS`, a deliberate ledger of
placeholders consciously left for follow-up work.
"""

from __future__ import annotations

import pytest
import scippnexus as snx
from scippnexus.field import DependsOn
from scippnexus.nxtransformations import TransformationChain, parse_depends_on_chain

from ess.livedata.config.instrument import Instrument, instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.config.stream import SAMPLE_COMPONENT, ContextBinding
from ess.livedata.config.value_log import ValueLog
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename


def _is_chain_patch(ci: ContextBinding) -> bool:
    key = ci.workflow_key
    return isinstance(key, type) and issubclass(key, ValueLog)


def _chain_patch_inputs(instrument: Instrument) -> list[ContextBinding]:
    return [b for b in instrument.context_bindings if _is_chain_patch(b)]


def _component_path(component: str) -> str:
    if component == SAMPLE_COMPONENT:
        return '/entry/sample'
    return f'/entry/instrument/{component}'


def _load_chain(artifact: str, component: str) -> TransformationChain | None:
    """Walk a component's depends_on chain via scippnexus.

    ``component`` is a source name or :data:`SAMPLE_COMPONENT`. Returns ``None``
    for static components with no ``depends_on`` field and for components the
    artifact does not contain (a missing binding consumer then fails
    :func:`test_transform_paths_match_artifact` on its empty chain).
    """
    parent_path = _component_path(component)
    with snx.File(artifact, 'r') as f:
        if parent_path not in f:
            return None
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


def _empty_nxlogs(artifact: str, component: str) -> set[str]:
    """Paths along the chain whose value is a length-0 NXlog."""
    chain = _load_chain(artifact, component)
    if chain is None:
        return set()
    return {
        path
        for path, t in chain.transformations.items()
        if getattr(t.value, 'sizes', {}).get('time', None) == 0
    }


def _loaded(name: str) -> Instrument:
    get_config(name)
    inst = instrument_registry[name]
    inst.load_factories()
    return inst


def _has_geometry_artifact(name: str) -> bool:
    try:
        get_nexus_geometry_filename(name)
    except ValueError:
        return False
    return True


@pytest.fixture(
    scope='module',
    params=[n for n in available_instruments() if _chain_patch_inputs(_loaded(n))],
)
def instrument(request) -> Instrument:
    return _loaded(request.param)


@pytest.fixture(
    scope='module',
    params=[n for n in available_instruments() if _has_geometry_artifact(n)],
)
def instrument_with_artifact(request) -> Instrument:
    return _loaded(request.param)


def test_transform_paths_match_artifact(instrument: Instrument) -> None:
    """Each chain-patch binding's resolved transform path must appear on the
    depends_on chain of every declared consumer in the geometry artifact."""
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
    """Each consumer must be a registered source on the instrument or the sample."""
    valid = set(instrument.detector_names) | set(instrument.monitors)
    valid.add(SAMPLE_COMPONENT)
    for binding in _chain_patch_inputs(instrument):
        unknown = binding.dependent_sources - valid
        assert not unknown, (
            f"Binding {binding.stream_name!r} declares unknown consumers "
            f"{unknown}; valid sources: {sorted(valid)}"
        )


# Known orphan placeholders not yet covered by a binding, as
# ``(instrument_name, component) -> transform paths``. Keep the test strict
# (no orphans) but record the ones consciously left for a follow-up. Removing
# an entry is a checklist item for the follow-up PR.
_KNOWN_ORPHAN_NXLOGS: dict[tuple[str, str], frozenset[str]] = {
    # ESTIA motion bindings: #1430.
    ('estia', 'multiblade_detector'): frozenset(
        {'/entry/instrument/detector_arm_rotation/value'}
    ),
    # BIFROST tank and sample angles: #1427.
    ('bifrost', 'elastic_monitor'): frozenset(
        {'/entry/instrument/detector_tank_angle/value'}
    ),
    ('bifrost', 'sample'): frozenset(
        {
            '/entry/instrument/goniometer_lower/value',
            '/entry/instrument/goniometer_upper/value',
            '/entry/instrument/sample_rotation/value',
        }
    ),
}


def test_no_orphan_empty_nxlogs(instrument_with_artifact: Instrument) -> None:
    """Every empty NXlog reachable from a registered source or the sample must
    be covered by a binding (or listed in :data:`_KNOWN_ORPHAN_NXLOGS`).
    Otherwise, workflows loading that component trip essreduce's
    ``reject_time_dependent_transform`` at compute time."""
    instrument = instrument_with_artifact
    artifact = str(get_nexus_geometry_filename(instrument.name))
    covered = {instrument.chain_patch_path(b) for b in _chain_patch_inputs(instrument)}
    components = [
        *instrument.detector_names,
        *instrument.monitors,
        SAMPLE_COMPONENT,
    ]
    orphans = {}
    for component in components:
        known = _KNOWN_ORPHAN_NXLOGS.get((instrument.name, component), frozenset())
        if uncovered := _empty_nxlogs(artifact, component) - covered - known:
            orphans[component] = sorted(uncovered)
    assert not orphans, (
        f"Empty NXlog placeholders not covered by any binding in "
        f"{instrument.name}: {orphans}. Add a chain-patch ContextBinding, fix "
        f"the geometry artifact, or list them in _KNOWN_ORPHAN_NXLOGS with a "
        f"follow-up reference."
    )
