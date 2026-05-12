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

from collections.abc import Callable
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


def _synthesise_provider(
    name: str,
    impl: Callable[..., Any],
    annotations: dict[str, Any],
) -> Any:
    """Synthesise a Sciline provider with explicit named positional parameters.

    Returns a function ``name(p1, p2, ...)`` whose ``__annotations__`` come
    from ``annotations`` (with ``'return'`` consumed as the return type) and
    whose body delegates to ``impl(p1, p2, ...)``.

    Why code synthesis is unavoidable: Sciline introspects providers via
    ``inspect.getfullargspec()``, which reads the underlying ``__code__``
    object and ignores ``__signature__``. ``__annotations__`` only attaches
    types to parameters that already exist in the code object — it cannot
    invent them. Producing N named typed positional parameters therefore
    requires building a real function via ``exec``/``compile``. This is the
    same technique ``dataclasses``, ``attrs``, ``pydantic``, and
    ``collections.namedtuple`` use to generate ``__init__`` /
    ``__new__``. The template inputs here are closed: every interpolated
    name is a key of ``annotations``, constructed in-module from
    ``f'log_{i}'``; no external string reaches the template.
    """
    params = [n for n in annotations if n != 'return']
    arg_list = ', '.join(params)
    src = f"def {name}({arg_list}):\n    return _impl({arg_list})\n"
    ns: dict[str, Any] = {'_impl': impl}
    exec(src, ns)  # noqa: S102
    fn = ns[name]
    fn.__annotations__ = dict(annotations)
    return fn


def _build_patched_chain_provider(
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

    def _impl(component: Any, *containers: TransformLog | None) -> Any:
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

    annotations: dict[str, Any] = {
        'component': NeXusComponent[component_type, SampleRun],
        **{f'log_{i}': b.log_key for i, b in enumerate(matched)},
        'return': NeXusTransformationChain[component_type, SampleRun],
    }
    return _synthesise_provider(
        name=f'_patched_chain__{component_type.__name__}',
        impl=_impl,
        annotations=annotations,
    )


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
