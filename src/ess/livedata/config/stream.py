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

from dataclasses import dataclass


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
