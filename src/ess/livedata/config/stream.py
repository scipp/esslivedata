# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Canonical records describing streaming data declarations.

A :class:`Stream` describes one streaming group in a NeXus file (or a
synthesised in-process stream) at the wire level — what it is, not what an
instrument chooses to call it. The instrument-facing name is the key into
:attr:`Instrument.streams`, supplied per-instrument in ``specs.py``.
"""

from __future__ import annotations

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
    parent_nx_class: str | None = None

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


def name_streams(
    parsed: dict[str, Stream],
    *,
    rename: dict[str, str] | None = None,
) -> dict[str, Stream]:
    """Build a name-keyed :attr:`Instrument.streams` dict from a parsed dict.

    The typical caller is an instrument's ``specs.py`` that imports a
    path-keyed ``PARSED_STREAMS`` from the auto-generated ``streams_parsed``
    module. Unrenamed entries get auto-suggested names from
    :func:`ess.livedata.nexus_helpers.suggest_names`; entries in ``rename``
    (keyed by ``nexus_path``) override those suggestions.

    Raises ``ValueError`` if a rename key matches no parsed entry, or if
    the resulting names are not unique.
    """
    from ess.livedata.nexus_helpers import suggest_names

    rename = rename or {}
    missing = set(rename) - set(parsed)
    if missing:
        raise ValueError(
            f"rename keys not in parsed: {sorted(missing)}; "
            f"known nexus_paths: {sorted(parsed)}"
        )
    suggested = suggest_names(parsed)
    result: dict[str, Stream] = {}
    for path, stream in parsed.items():
        name = rename.get(path, suggested[path])
        if name in result:
            raise ValueError(
                f"name {name!r} produced for {path!r} collides with an "
                f"earlier entry; check rename map"
            )
        result[name] = stream
    return result
