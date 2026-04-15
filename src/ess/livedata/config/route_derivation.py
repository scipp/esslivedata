# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Derive needed stream names from workflow specs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..kafka.stream_mapping import StreamMapping
    from .instrument import Instrument


def gather_source_names(
    instrument: Instrument,
    namespace: str,
    source_subset: set[str] | None = None,
) -> set[str]:
    """Collect internal stream names that specs in ``namespace`` depend on.

    Parameters
    ----------
    instrument:
        Instrument whose workflow specs to inspect.
    namespace:
        Only consider specs in this namespace (e.g. ``'detector_data'``).
    source_subset:
        If given, only include specs whose primary sources overlap with this
        set. Their full dependencies (including aux sources) are still added.
        Use this for per-source service splits (sharding).
    """
    names: set[str] = set()
    for spec in instrument.workflow_factory.values():
        if spec.namespace != namespace:
            continue
        spec_sources = set(spec.source_names)
        if source_subset is not None and not (spec_sources & source_subset):
            continue
        names |= spec_sources
        if spec.aux_sources:
            for aux_input in spec.aux_sources.inputs.values():
                names.update(aux_input.choices)
    return names


def resolve_stream_names(
    needed: set[str],
    instrument: Instrument,
    stream_mapping: StreamMapping,
) -> set[str]:
    """Expand logical source names to physical stream names for filtering.

    Workflow specs use logical names (e.g. ``'unified_detector'``), which may
    not appear in the ``StreamMapping`` LUTs directly. For example, Bifrost
    merges 45 physical streams (``arc0_triplet0``, ...) into one logical
    detector. When a logical detector/monitor name is not found in any LUT,
    all physical names from that category are included.

    Parameters
    ----------
    needed:
        Set of names from :func:`gather_source_names`.
    instrument:
        Instrument configuration.
    stream_mapping:
        The full (unfiltered) stream mapping.
    """
    known = stream_mapping.all_stream_names
    resolved = needed & known
    unknown = needed - known
    if not unknown:
        return resolved
    # Bifrost compatibility: its specs use logical name 'unified_detector', but the
    # StreamMapping LUT contains 45 physical stream names (arc0_triplet0, ...) that
    # get merged at the route level (merge_detectors=True). When a logical name from
    # instrument.detector_names is not found in any LUT, we include all physical
    # names from that category. This is safe because Bifrost is the only instrument
    # with this many-physical-to-one-logical pattern; all others have a 1:1 mapping
    # between spec source_names and StreamMapping LUT values.
    logical_detectors = set(instrument.detector_names)
    logical_monitors = set(instrument.monitors)
    if unknown & logical_detectors:
        resolved |= set(stream_mapping.detectors.values())
    if unknown & logical_monitors:
        resolved |= set(stream_mapping.monitors.values())
    return resolved


def get_source_subset(
    source_names: list[str],
    num_shards: int,
    shard: int,
) -> set[str]:
    """Compute the source subset for a given shard.

    Parameters
    ----------
    source_names:
        Ordered list of primary source names (e.g. detector names).
    num_shards:
        Total number of shards.
    shard:
        Zero-based shard index.
    """
    if shard >= num_shards:
        raise ValueError(f"shard={shard} must be less than num_shards={num_shards}")
    return set(sorted(source_names)[shard::num_shards])
