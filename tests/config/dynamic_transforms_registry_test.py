# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Validate each instrument's ``dynamic_transforms`` registry against the
currently-registered geometry artifact.

For every binding declared on an instrument, walks the depends_on chain
of every declared consumer in the artifact and confirms the binding's
``nxlog_path`` appears on every chain. Catches typos and orphaned
bindings before runtime.

Also rejects duplicate ``log_key``s — Sciline collapses two parameters
of the same key, silently merging two bindings.
"""

from __future__ import annotations

import pytest

from ess.livedata.config.instrument import instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.handlers.detector_data_handler import get_nexus_geometry_filename
from ess.livedata.handlers.dynamic_transforms import load_depends_on_chain


def _chain_paths(artifact: str, source_name: str) -> list[str]:
    chain = load_depends_on_chain(artifact, source_name)
    return list(chain.transformations) if chain is not None else []


def _empty_nxlog(artifact: str, source_name: str) -> str | None:
    """First path along the chain whose value is a length-0 NXlog, or None."""
    chain = load_depends_on_chain(artifact, source_name)
    if chain is None:
        return None
    for path, t in chain.transformations.items():
        sizes = getattr(t.value, 'sizes', None)
        if sizes is not None and sizes.get('time', None) == 0:
            return path
    return None


def _instruments_with_dynamic_transforms() -> list[str]:
    cases = []
    for name in available_instruments():
        get_config(name)
        inst = instrument_registry[name]
        if inst.dynamic_transforms:
            cases.append(name)
    return cases


@pytest.fixture(scope='module', params=_instruments_with_dynamic_transforms())
def instrument(request):
    name = request.param
    get_config(name)
    return instrument_registry[name]


def test_registry_log_keys_are_unique(instrument) -> None:
    keys = [b.log_key for b in instrument.dynamic_transforms]
    assert len(keys) == len(set(keys)), (
        f"Duplicate log_key in {instrument.name}.dynamic_transforms: {keys}"
    )


def test_registry_paths_match_artifact(instrument) -> None:
    artifact = str(get_nexus_geometry_filename(instrument.name))
    for binding in instrument.dynamic_transforms:
        for source_name in binding.consumers:
            chain = _chain_paths(artifact, source_name)
            assert binding.nxlog_path in chain, (
                f"Binding {binding.stream_name!r} declares nxlog_path "
                f"{binding.nxlog_path!r} in consumers of {source_name!r}, "
                f"but it does not appear on the depends_on chain "
                f"resolved from the artifact ({artifact}). "
                f"Walked: {chain}"
            )


# Known orphan placeholders that have not yet been bound. Each entry is
# (instrument_name, source_name) -> nxlog_path. This is a deliberate
# ledger: keep the test strict (no orphans) but record the ones we are
# consciously leaving for a follow-up. Removing an entry is a checklist
# item for the follow-up PR.
_KNOWN_ORPHAN_NXLOGS: dict[tuple[str, str], str] = {
    # See loki/specs.py: m4 trans_20 needs either a make_geometry_nexus.py
    # change to share the carriage NXlog, or a separate f144 stream
    # registration. Tracked as follow-up to issue #922.
    (
        'loki',
        'beam_monitor_m4',
    ): '/entry/instrument/beam_monitor_m4/transformations/trans_20',
}


def test_no_orphan_empty_nxlogs(instrument) -> None:
    """Every empty NXlog reachable from any source on a registered spec
    must be covered by a binding (or in the known-orphan ledger).
    Otherwise, workflows loading that source will trip
    ``apply_dynamic_transforms`` at construction time."""
    artifact = str(get_nexus_geometry_filename(instrument.name))
    covered = {b.nxlog_path for b in instrument.dynamic_transforms}
    sources = list(instrument.detector_names) + list(instrument.monitors)
    for source_name in sources:
        empty = _empty_nxlog(artifact, source_name)
        if empty is None or empty in covered:
            continue
        known = _KNOWN_ORPHAN_NXLOGS.get((instrument.name, source_name))
        if known == empty:
            continue
        pytest.fail(
            f"Source {source_name!r} has an empty NXlog placeholder at "
            f"{empty!r} not covered by any binding. Add a "
            f"DynamicTransformBinding to {instrument.name}.dynamic_transforms "
            f"or fix the geometry artifact (or list it in "
            f"_KNOWN_ORPHAN_NXLOGS with a follow-up reference)."
        )


def test_consumers_subset_of_registered_sources(instrument) -> None:
    """Each consumer must be a registered source on the instrument."""
    valid = set(instrument.detector_names) | set(instrument.monitors)
    for binding in instrument.dynamic_transforms:
        unknown = binding.consumers - valid
        assert not unknown, (
            f"Binding {binding.stream_name!r} declares unknown consumers "
            f"{unknown}; valid sources: {sorted(valid)}"
        )


def test_stream_in_f144_attribute_registry(instrument) -> None:
    """The binding's ``stream_name`` must be a registered f144 stream so
    the routing layer subscribes to it."""
    for binding in instrument.dynamic_transforms:
        assert binding.stream_name in instrument.f144_attribute_registry, (
            f"Binding {binding.stream_name!r} is not declared in "
            f"{instrument.name}.f144_attribute_registry; the routing layer "
            f"will not deliver f144 messages to the workflow."
        )
