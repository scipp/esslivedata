# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Centralised wiring for dynamic NeXus transformations driven by f144 streams.

The geometry artifact represents dynamic geometry as length-0 NXlog
placeholders along ``depends_on`` chains. Workflows that load components
walking through such a placeholder must replace its (empty) value with
the latest sample from a live f144 stream — otherwise essreduce's
``reject_time_dependent_transform`` filter raises at workflow
construction time.

Per-instrument bindings are declared on :class:`Instrument`; the helper
:func:`apply_dynamic_transforms` patches the relevant providers at
factory time.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import scipp as sc
from ess.reduce.nexus.types import (
    NeXusComponent,
    NeXusTransformationChain,
    SampleRun,
)
from ess.reduce.nexus.workflow import get_transformation_chain

from ess.livedata.config.workflow_spec import (
    AuxSources,
    CombinedAuxSources,
    JobId,
)

if TYPE_CHECKING:
    from ess.livedata.config.instrument import Instrument


@dataclass(frozen=True, slots=True)
class TransformLog:
    """Latest NXlog samples for a dynamic-transform binding.

    Subclass to create a distinct Sciline key per binding. ``log`` is
    ``None`` before the first ``set_context`` call (essreduce's
    ``StreamProcessor`` pre-sets every context key to ``None``);
    otherwise it is the NXlog produced by ``ToNXlog`` — possibly still
    empty if no f144 message has arrived yet.
    """

    log: sc.DataArray | None = None


@dataclass(frozen=True, slots=True)
class DynamicTransformBinding:
    """Binding between an NXlog placeholder and the f144 stream that drives it.

    Parameters
    ----------
    nxlog_path:
        Absolute NeXus path of the placeholder NXlog node along a
        depends_on chain (e.g. ``/entry/instrument/detector_carriage/value``).
    stream_name:
        Name of the f144 aux stream that supplies live samples. Must
        appear in :attr:`Instrument.f144_attribute_registry`.
    log_key:
        :class:`TransformLog` subclass used as the Sciline key for this
        binding. Each binding declares its own subclass so the
        per-binding log appears as a distinct, grep-able Sciline node.
    consumers:
        Source names whose ``depends_on`` chain walks through
        ``nxlog_path``. Used to scope aux-source routing per spec —
        only specs covering at least one listed consumer carry the
        stream, and only matching consumers receive it at render time.
    """

    nxlog_path: str
    stream_name: str
    log_key: type[TransformLog]
    consumers: frozenset[str]


def _build_patched_chain_provider(
    component_type: type, matched: list[DynamicTransformBinding]
) -> Any:
    """Build a replacement for essreduce's ``get_transformation_chain``.

    The returned closure has the same signature as ``get_transformation_chain``
    specialised to one ``component_type``: it consumes
    ``NeXusComponent[component_type, SampleRun]`` (and the matched bindings'
    ``log_key`` parameters) and produces
    ``NeXusTransformationChain[component_type, SampleRun]``. It reproduces
    the upstream provider's body (``get_transformation_chain(component)``)
    and patches each matched ``nxlog_path`` with the latest f144 sample.
    It cannot instead consume the chain as input — that would self-cycle on
    its own return type.

    Sciline resolves dependencies by annotation name, so the closure must
    use explicit positional parameters (not ``*logs``) and have its
    ``__annotations__`` set with the per-binding ``log_key`` types.
    """
    n = len(matched)
    arg_names = [f'log_{i}' for i in range(n)]
    arg_list = ', '.join(arg_names)
    bindings_local = list(matched)
    src = (
        f"def _patched_chain(component, {arg_list}):\n"
        f"    return _impl(component, ({arg_list}{',' if n == 1 else ''}))\n"
    )
    namespace: dict[str, Any] = {}

    def _impl(component: Any, containers: tuple[TransformLog | None, ...]) -> Any:
        chain = get_transformation_chain(component)
        patched = deepcopy(chain)
        for binding, container in zip(bindings_local, containers, strict=True):
            if binding.nxlog_path not in patched.transformations:
                continue
            if (
                container is None
                or container.log is None
                or container.log.sizes.get('time', 0) == 0
            ):
                raise ValueError(
                    f"No samples yet for {binding.stream_name!r} "
                    f"(transform {binding.nxlog_path!r})"
                )
            log = container.log
            patched.transformations[binding.nxlog_path].value = log['time', -1].data
        return patched

    namespace['_impl'] = _impl
    exec(src, namespace)  # noqa: S102
    fn = namespace['_patched_chain']
    fn.__annotations__ = {
        'component': NeXusComponent[component_type, SampleRun],
        **{name: b.log_key for name, b in zip(arg_names, matched, strict=True)},
        'return': NeXusTransformationChain[component_type, SampleRun],
    }
    fn.__name__ = f'_patched_chain__{component_type.__name__}'
    return fn


class _DynamicTransformAuxSources(AuxSources):
    """Aux sources covering an instrument's dynamic-transform bindings.

    Inputs include every binding whose ``consumers`` set intersects the
    spec's ``source_names``. ``render`` returns only the streams whose
    binding includes ``job_id.source_name`` in its consumers, rendered
    un-prefixed (these are global f144 streams shared across jobs).
    """

    def __init__(self, instrument: Instrument, source_names: list[str]) -> None:
        self._instrument = instrument
        selected = set(source_names)
        inputs: dict[str, str] = {
            b.stream_name: b.stream_name
            for b in instrument.dynamic_transforms
            if b.consumers & selected
        }
        super().__init__(dict(inputs))

    def render(
        self,
        job_id: JobId,
        selections: dict[str, str] | None = None,
    ) -> dict[str, str]:
        return {
            b.stream_name: b.stream_name
            for b in self._instrument.dynamic_transforms
            if job_id.source_name in b.consumers
        }


def compose_aux_sources(
    instrument: Instrument,
    source_names: list[str],
    caller_aux: AuxSources | None,
) -> AuxSources | None:
    """Merge caller-supplied aux sources with auto-derived dynamic-transform
    aux sources for the given source set."""
    components: list[AuxSources] = []
    if caller_aux is not None:
        components.append(caller_aux)
    if instrument.dynamic_transforms:
        dyn = _DynamicTransformAuxSources(instrument, source_names)
        if dyn.inputs:
            components.append(dyn)
    if not components:
        return None
    if len(components) == 1:
        return components[0]
    return CombinedAuxSources(components)
