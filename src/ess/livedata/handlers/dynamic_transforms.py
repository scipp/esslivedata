# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Drive NeXus ``depends_on`` chains from live f144 streams.

The geometry artifact represents dynamic geometry as length-0 NXlog
placeholders along ``depends_on`` chains. Workflows that load components
walking through such a placeholder must replace its (empty) value with
the latest sample from a live f144 stream — otherwise essreduce's
``reject_time_dependent_transform`` filter raises at workflow
construction time.

Per-instrument bindings are declared as :class:`DynamicTransformBinding`
on :class:`Instrument`. The method :meth:`Instrument.apply_dynamic_transforms`
replaces essreduce's ``NeXusTransformationChain[T, SampleRun]`` provider
with a synthesised one that consumes the matching :class:`ValueLog`
subclasses and patches the chain in place.

The generic f144-NXlog → Sciline parameter machinery lives in
:mod:`.log_context` and :mod:`.stream_processor_workflow`; this module
is the transform-specific consumer of those primitives.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ess.reduce.nexus.types import (
    NeXusComponent,
    NeXusTransformationChain,
    SampleRun,
)
from ess.reduce.nexus.workflow import get_transformation_chain

from .log_context import LogContextBinding
from .stream_processor_workflow import ValueLog, synthesise_provider


@dataclass(frozen=True, slots=True, kw_only=True)
class DynamicTransformBinding(LogContextBinding):
    """Binds an NXlog placeholder in a ``depends_on`` chain to an f144 stream.

    Parameters
    ----------
    nxlog_path:
        Absolute NeXus path of the placeholder NXlog node along a
        ``depends_on`` chain (e.g. ``/entry/instrument/detector_carriage/value``).
    """

    nxlog_path: str


def build_patched_chain_provider(
    component_type: type, matched: list[DynamicTransformBinding]
) -> Any:
    """Build a per-component-type provider that patches NXlog placeholders.

    The returned provider replaces essreduce's ``get_transformation_chain``
    specialised to ``component_type``: it consumes
    ``NeXusComponent[component_type, SampleRun]`` plus one parameter per
    matched binding (annotated with that binding's ``log_key``), and produces
    ``NeXusTransformationChain[component_type, SampleRun]``. It cannot
    instead consume the chain as input — that would self-cycle on its own
    return type.
    """
    bindings_local = list(matched)

    def _impl(component: Any, *containers: ValueLog | None) -> Any:
        chain = get_transformation_chain(component)
        patched = deepcopy(chain)
        for binding, container in zip(bindings_local, containers, strict=True):
            if binding.nxlog_path not in patched.transformations:
                continue
            if (
                container is None
                or container.values is None
                or container.values.sizes.get('time', 0) == 0
            ):
                raise ValueError(
                    f"No samples yet for {binding.stream_name!r} "
                    f"(transform {binding.nxlog_path!r})"
                )
            log = container.values
            patched.transformations[binding.nxlog_path].value = log['time', -1].data
        return patched

    annotations: dict[str, Any] = {
        'component': NeXusComponent[component_type, SampleRun],
        **{f'log_{i}': b.log_key for i, b in enumerate(matched)},
        'return': NeXusTransformationChain[component_type, SampleRun],
    }
    return synthesise_provider(
        name=f'_patched_chain__{component_type.__name__}',
        impl=_impl,
        annotations=annotations,
    )
