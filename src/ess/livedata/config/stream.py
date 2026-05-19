# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Canonical records describing streaming data declarations.

A :class:`Stream` describes one streaming group in a NeXus file (or a
synthesised in-process stream) at the wire level — what it is, not what an
instrument chooses to call it. The instrument-facing name is the key into
:attr:`Instrument.streams`, supplied per-instrument in ``specs.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class Stream:
    """Any streaming group in NeXus (or synthesised in-process).

    Synthesised streams have ``topic``, ``source``, and ``nexus_path`` all None
    (they never traverse Kafka and have no file representation). For streams
    that *do* arrive over Kafka, ``topic`` and ``source`` must be set;
    ``nexus_path`` may still be None for hand-coded entries that have not been
    cross-referenced with a NeXus geometry file.
    """

    writer_module: str
    nexus_path: str | None = None
    topic: str | None = None
    source: str | None = None
    nx_class: str = ''

    def __post_init__(self) -> None:
        # Synthesised streams have no Kafka identity. The converse is not
        # enforced: a real stream may temporarily lack a nexus_path until the
        # codegen path is in place.
        if self.topic is None and self.source is not None:
            raise ValueError(
                f"Stream {self.nexus_path!r}: source set but topic is None"
            )
        if self.source is None and self.topic is not None:
            raise ValueError(
                f"Stream {self.nexus_path!r}: topic set but source is None"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class F144Stream(Stream):
    """f144 NXlog stream — value/time payloads."""

    units: str | None = None
    writer_module: str = 'f144'
    nx_class: str = 'NXlog'


@dataclass(frozen=True, slots=True, kw_only=True)
class Device(Stream):
    """Synthesised stream merging RBV/VAL/DMOV substreams of a motorised device.

    A ``Device`` is not transported over Kafka; it is materialised in-process
    by :class:`DeviceSynthesizer` from the substreams referenced by
    :attr:`value`, :attr:`target`, and :attr:`settled` (each is a key into
    :attr:`Instrument.streams`).

    ``value`` is required and points to the RBV substream. ``target`` and
    ``settled`` are optional: only RBV is required for the synthesizer to
    emit a sample.
    """

    value: str
    target: str | None = None
    settled: str | None = None
    units: str | None = None
    writer_module: str = 'device'
    nx_class: str = 'NXpositioner'


@dataclass(frozen=True, slots=True, kw_only=True)
class LogContextBinding:
    """One f144 stream feeding a Sciline workflow key, scoped to dependents.

    The binding declares that the value of an f144 stream
    (:attr:`stream_name`, an entry in :attr:`Instrument.streams`) should be
    routed to a Sciline pipeline as the value of ``workflow_key``, but only
    when wiring workflows for the spec source names listed in
    :attr:`dependent_sources`.

    ``workflow_key`` is typed as :class:`Any` because Sciline keys are
    parameterised generics (e.g. ``InstrumentAngle[SampleRun]``) and Python's
    type system cannot describe "type of any Sciline key" precisely.
    """

    stream_name: str
    workflow_key: Any
    dependent_sources: frozenset[str]


#: NeXus container groups that carry no entity-level meaning. Removed from the
#: path before constructing an internal name so that e.g.
#: ``entry/instrument/wfm1/transformations/translation1`` becomes
#: ``wfm1_translation1`` and not ``transformations_translation1``.
_GENERIC_GROUPS: frozenset[str] = frozenset(
    {'entry', 'instrument', 'sample', 'sample_environment', 'transformations'}
)


def suggest_names(
    paths: Iterable[str],
    *,
    min_depth: int = 2,
    forbidden: Iterable[str] | None = None,
) -> dict[str, str]:
    """Suggest a unique internal name for each NeXus group path.

    Generic NeXus container groups (``entry``, ``instrument``, ``sample``,
    ``sample_environment``, ``transformations``) are dropped — they add no
    meaning and only inflate names. The name is then the shortest tail of
    the remaining path (minimum ``min_depth`` components, when available)
    that is unique across the input set and not in ``forbidden``. Duplicates
    and forbidden names extend to the next-longer tail until uniqueness is
    reached.

    ``min_depth=2`` (the default) suits leaf-keyed paths where the leaf
    alone (``value``, ``target_value``, …) is a generic role label. Callers
    naming *parent* groups (where the leaf is itself an entity name) should
    pass ``min_depth=1`` and supply already-assigned names as ``forbidden``
    to keep the two namespaces disjoint by construction.

    Paths that still collide after exhausting the filtered tail (i.e. they
    differ only in filtered-out generic ancestors) fall back to the full
    unfiltered path. HDF5 path uniqueness guarantees this resolves.

    The returned dict is keyed by path. Since paths are unique in HDF5 and
    each path produces at most one name, no two paths share a name.
    """
    paths = list(paths)
    forbidden_set = frozenset() if forbidden is None else frozenset(forbidden)
    full: dict[str, list[str]] = {p: p.strip('/').split('/') for p in paths}
    filtered: dict[str, list[str]] = {
        p: [c for c in full[p] if c not in _GENERIC_GROUPS] or full[p] for p in paths
    }

    result: dict[str, str] = {}
    pending = set(paths)
    for parts in (filtered, full):
        max_d = max((len(parts[p]) for p in pending), default=1)
        depth = min_depth
        while pending and depth <= max(max_d, min_depth):
            candidates = {
                p: '_'.join(parts[p][-min(depth, len(parts[p])) :]) for p in pending
            }
            counts: dict[str, int] = {}
            for name in candidates.values():
                counts[name] = counts.get(name, 0) + 1
            next_pending: set[str] = set()
            for path, name in candidates.items():
                if counts[name] == 1 and name not in forbidden_set:
                    result[path] = name
                else:
                    next_pending.add(path)
            pending = next_pending
            depth += 1
    return result


#: EPICS source-attribute suffix identifying each device-substream role.
#: From the EPICS motor record: ``.RBV`` is the readback, ``.VAL`` the setpoint,
#: ``.DMOV`` the "done moving" flag. These are fixed by the EPICS motor-record
#: convention and stable across the facility, while NeXus child names
#: (``value`` vs ``position_readback`` etc.) drift across instruments.
_ROLE_BY_SUFFIX: dict[str, str] = {
    '.RBV': 'value',
    '.VAL': 'target',
    '.DMOV': 'settled',
}


def _classify_source(source: str | None) -> str | None:
    """Map an f144 ``source`` attribute to a device role, or ``None``."""
    if source is None:
        return None
    for suffix, role in _ROLE_BY_SUFFIX.items():
        if source.endswith(suffix):
            return role
    return None


@dataclass(frozen=True, slots=True)
class _DetectedDevice:
    """Internal record for a detected device group during ``name_streams``."""

    value: str
    target: str | None
    settled: str | None
    units: str | None


def _detect_devices(parsed: dict[str, Stream]) -> dict[str, _DetectedDevice]:
    """Detect device groups by EPICS source-suffix classification.

    Each f144 substream is classified by the suffix of its ``source`` attribute
    (``.RBV`` → value, ``.VAL`` → target, ``.DMOV`` → settled). Substreams
    co-located under one NeXus parent group form a Device if classified RBV is
    present and at least one of classified VAL / DMOV is present. Substreams
    with ``source=None`` or an unrecognised suffix are not classifiable and
    are ignored.

    Returns a dict ``parent_path -> _DetectedDevice``. Raises
    :class:`ValueError` if RBV and VAL units disagree, or if two children of
    one parent classify as the same role.
    """
    by_parent: dict[str, dict[str, str]] = {}
    for path, stream in parsed.items():
        if not isinstance(stream, F144Stream):
            continue
        role = _classify_source(stream.source)
        if role is None:
            continue
        parent, _, _ = path.rpartition('/')
        roles = by_parent.setdefault(parent, {})
        if role in roles:
            raise ValueError(
                f"Device at {parent!r}: two children classify as {role!r} "
                f"({roles[role]!r} and {path!r}); only one expected per parent"
            )
        roles[role] = path

    devices: dict[str, _DetectedDevice] = {}
    for parent, roles in by_parent.items():
        if 'value' not in roles:
            continue
        if 'target' not in roles and 'settled' not in roles:
            continue
        rbv = parsed[roles['value']]
        if not isinstance(rbv, F144Stream):
            raise ValueError(
                f"Device at {parent!r}: RBV substream {roles['value']!r} "
                f"is not an F144Stream (got {type(rbv).__name__})"
            )
        units = rbv.units
        if 'target' in roles:
            val = parsed[roles['target']]
            if not isinstance(val, F144Stream):
                raise ValueError(
                    f"Device at {parent!r}: VAL substream {roles['target']!r} "
                    f"is not an F144Stream (got {type(val).__name__})"
                )
            if val.units != units:
                raise ValueError(
                    f"Device at {parent!r}: RBV units {units!r} differ from "
                    f"VAL units {val.units!r}; units must match"
                )
        devices[parent] = _DetectedDevice(
            value=roles['value'],
            target=roles.get('target'),
            settled=roles.get('settled'),
            units=units,
        )
    return devices


def name_streams(
    parsed: dict[str, Stream],
    *,
    rename: dict[str, str] | None = None,
) -> dict[str, Stream]:
    """Build a name-keyed :attr:`Instrument.streams` dict from a parsed dict.

    The typical caller is an instrument's ``specs.py`` that imports a
    path-keyed ``PARSED_STREAMS`` from the auto-generated ``streams_parsed``
    module. Unrenamed entries get auto-suggested names from
    :func:`suggest_names`; entries in ``rename`` (keyed by ``nexus_path``)
    override those suggestions.

    Motorised devices are detected by structural child-name pattern (RBV +
    at least one of VAL/DMOV) and emitted as :class:`Device` entries with
    auto-populated substream-name pointers. Substreams and device parents
    are named in two passes: substreams with ``min_depth=2`` (current
    behaviour), then devices with ``min_depth=1`` and the substream names
    as ``forbidden``. The two namespaces are disjoint by construction.

    Raises ``ValueError`` if a rename key matches no parsed entry, if any
    rename or device name collides, or if RBV/VAL units disagree on a
    detected device.
    """
    rename = rename or {}
    devices = _detect_devices(parsed)
    valid_paths = set(parsed) | set(devices)
    missing = set(rename) - valid_paths
    if missing:
        raise ValueError(
            f"rename keys not in parsed or detected device parents: "
            f"{sorted(missing)}; known nexus_paths: {sorted(parsed)}"
        )
    substream_names = suggest_names(parsed.keys())
    device_names = suggest_names(
        devices.keys(),
        min_depth=1,
        forbidden=substream_names.values(),
    )
    suggested = {**substream_names, **device_names}

    def resolve(path: str) -> str:
        return rename.get(path, suggested[path])

    result: dict[str, Stream] = {}
    for path, stream in parsed.items():
        name = resolve(path)
        if name in result:
            raise ValueError(
                f"name {name!r} produced for {path!r} collides with an "
                f"earlier entry; check rename map"
            )
        result[name] = stream
    for parent_path, info in devices.items():
        device_name = resolve(parent_path)
        if device_name in result:
            raise ValueError(
                f"device name {device_name!r} for parent {parent_path!r} "
                f"collides with an existing stream; check rename map"
            )
        result[device_name] = Device(
            nexus_path=parent_path,
            value=resolve(info.value),
            target=resolve(info.target) if info.target is not None else None,
            settled=resolve(info.settled) if info.settled is not None else None,
            units=info.units,
        )
    return result
