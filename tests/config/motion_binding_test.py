# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Validate chain-patch :class:`ContextInput` bindings against the
currently-registered geometry artifact.

For every :class:`ContextInput` declared on an instrument with
``transform_path`` set, walks the ``depends_on`` chain of every declared
consumer in the artifact and confirms the path appears. Catches typos
and orphaned bindings before runtime.

Also flags any empty NXlog placeholder on a registered source that no
binding covers — except those in :data:`_KNOWN_ORPHAN_NXLOGS`, a
deliberate ledger of placeholders consciously left for follow-up work.
"""

from __future__ import annotations

import pytest
import scippnexus as snx
from scippnexus.field import DependsOn
from scippnexus.nxtransformations import TransformationChain, parse_depends_on_chain

from ess.livedata.config.instrument import Instrument, instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.config.stream import ContextInput
from ess.livedata.handlers.detector_data_handler import get_nexus_geometry_filename


def _chain_patch_inputs(instrument: Instrument) -> list[ContextInput]:
    return [b for b in instrument.context_inputs if b.transform_path is not None]


def _load_chain(artifact: str, source_name: str) -> TransformationChain | None:
    """Walk a source's depends_on chain via scippnexus.

    Returns ``None`` for static components with no ``depends_on`` field.
    """
    parent_path = f'/entry/instrument/{source_name}'
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


def _empty_nxlog(artifact: str, source_name: str) -> str | None:
    """First path along the chain whose value is a length-0 NXlog, or None."""
    chain = _load_chain(artifact, source_name)
    if chain is None:
        return None
    for path, t in chain.transformations.items():
        sizes = getattr(t.value, 'sizes', None)
        if sizes is not None and sizes.get('time', None) == 0:
            return path
    return None


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
    """Each chain-patch binding's ``transform_path`` must appear on the
    depends_on chain of every declared consumer in the geometry artifact."""
    artifact = str(get_nexus_geometry_filename(instrument.name))
    for binding in _chain_patch_inputs(instrument):
        for source_name in binding.dependent_sources:
            chain = _chain_paths(artifact, source_name)
            assert binding.transform_path in chain, (
                f"Binding {binding.stream_name!r} declares transform_path "
                f"{binding.transform_path!r} in consumers of {source_name!r}, "
                f"but it does not appear on the depends_on chain "
                f"resolved from the artifact ({artifact}). "
                f"Walked: {chain}"
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


# Known orphan placeholders not yet covered by a binding. Each entry is
# ``(instrument_name, source_name) -> transform_path``. Keep the test
# strict (no orphans) but record the ones consciously left for a
# follow-up. Removing an entry is a checklist item for the follow-up PR.
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
    """Every empty NXlog reachable from a registered source must be
    covered by a binding (or listed in :data:`_KNOWN_ORPHAN_NXLOGS`).
    Otherwise, workflows loading that source trip essreduce's
    ``reject_time_dependent_transform`` at compute time."""
    artifact = str(get_nexus_geometry_filename(instrument.name))
    covered = {b.transform_path for b in _chain_patch_inputs(instrument)}
    sources = list(instrument.detector_names) + list(instrument.monitors)
    for source_name in sources:
        empty = _empty_nxlog(artifact, source_name)
        if empty is None or empty in covered:
            continue
        if _KNOWN_ORPHAN_NXLOGS.get((instrument.name, source_name)) == empty:
            continue
        pytest.fail(
            f"Source {source_name!r} has an empty NXlog placeholder at "
            f"{empty!r} not covered by any binding. Add a chain-patch "
            f"ContextInput to {instrument.name} or fix the geometry "
            f"artifact (or list it in _KNOWN_ORPHAN_NXLOGS with a "
            f"follow-up reference)."
        )
