# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Canonical records describing streaming data declarations.

A :class:`Stream` is the unified description of one streaming group in a NeXus
file (or a synthesised in-process stream). :class:`F144Stream` adds f144-specific
fields. Per-instrument configuration holds these records in
``Instrument.streams: dict[str, Stream]``; downstream consumers derive everything
else (StreamLUT, NXlog attrs, timeseries spec registration, ...) from this single
source of truth.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
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

    stream_name: str
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
                f"Stream {self.stream_name!r}: source set but topic is None"
            )
        if self.source is None and self.topic is not None:
            raise ValueError(
                f"Stream {self.stream_name!r}: topic set but source is None"
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


def build_streams(
    parsed: Iterable[F144Stream],
    *,
    overrides: dict[str, dict[str, Any]] | None = None,
    synthetics: Iterable[F144Stream] | None = None,
) -> dict[str, Stream]:
    """Compose a final stream registry from parsed + overrides + synthetics.

    The typical caller is an instrument's ``streams.py`` that imports a
    checked-in ``streams_parsed.py`` (auto-generated from a NeXus geometry
    file) and applies hand-edited fixes:

    * ``overrides`` is keyed by either ``stream_name`` (the suggested
      internal name in the parsed list) or ``nexus_path`` (the stable NeXus
      identity). Values are ``dict``-of-field-overrides applied via
      :func:`dataclasses.replace`. Renaming an entry is just
      ``{key: dict(stream_name='new_name')}``.
    * ``synthetics`` declares streams that do not exist on the wire (e.g.
      in-process synthesisers). They are merged in alongside parsed entries.

    Raises ``ValueError`` if:

    * an override key matches no parsed entry,
    * two parsed entries collide on ``stream_name`` (the parser cannot
      currently produce this; only an after-rename override can),
    * a synthetic collides with a parsed entry on ``stream_name``.
    """
    parsed_list = list(parsed)
    registry: dict[str, F144Stream] = {}
    path_index: dict[str, str] = {}

    for stream in parsed_list:
        if stream.stream_name in registry:
            raise ValueError(
                f"Duplicate stream_name {stream.stream_name!r} in parsed entries"
            )
        registry[stream.stream_name] = stream
        if stream.nexus_path is not None:
            path_index[stream.nexus_path] = stream.stream_name

    for override_key, fields in (overrides or {}).items():
        if override_key in registry:
            current_name = override_key
        elif override_key in path_index:
            current_name = path_index[override_key]
        else:
            raise ValueError(
                f"Override key {override_key!r} matches no parsed entry; "
                f"known stream_names: {sorted(registry)}, "
                f"known nexus_paths: {sorted(path_index)}"
            )
        old = registry.pop(current_name)
        if old.nexus_path is not None:
            path_index.pop(old.nexus_path, None)
        new = replace(old, **fields)
        if new.stream_name in registry:
            raise ValueError(
                f"Override on {override_key!r} renames to {new.stream_name!r}, "
                f"which already exists in the registry"
            )
        registry[new.stream_name] = new
        if new.nexus_path is not None:
            path_index[new.nexus_path] = new.stream_name

    final: dict[str, Stream] = dict(registry)
    for synthetic in synthetics or ():
        if synthetic.stream_name in final:
            raise ValueError(
                f"Synthetic stream {synthetic.stream_name!r} collides with a "
                f"parsed entry"
            )
        final[synthetic.stream_name] = synthetic
    return final
