# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Patch f144-driven dynamic transforms into a workflow's depends_on chain.

The geometry artifact represents dynamic geometry as length-0 NXlog
placeholders along ``depends_on`` chains. Workflows that load components
walking through such a placeholder must replace the empty value with the
latest sample from a live f144 stream — otherwise essreduce's
``reject_time_dependent_transform`` filter raises at workflow construction.

:func:`build_patched_chain_provider` synthesises a per-component-type provider
that consumes one :class:`ValueLog` parameter per binding and writes the
latest sample into the corresponding transformation along the chain.

:func:`wire_dynamic_transforms` is the routing-layer entry point: it derives
the ``source -> component_type`` map from a workflow's own ``dynamic_keys``,
groups the matching :class:`ChainPatchBinding` records by component type, and
inserts one fused provider per type. The bindings are self-contained data (the
instrument resolves them via :attr:`Instrument.chain_patch_bindings`), so this
module needs no reference to the instrument or its stream topology.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from typing import Any, Protocol, get_args, get_origin, runtime_checkable

import sciline
from ess.reduce.nexus.types import (
    NeXusComponent,
    NeXusData,
    NeXusTransformationChain,
    SampleRun,
)
from ess.reduce.nexus.workflow import get_transformation_chain

from ess.livedata.config.stream import ChainPatchBinding
from ess.livedata.config.value_log import ValueLog

_PY_IDENTIFIER = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


@runtime_checkable
class SupportsDynamicTransforms(Protocol):
    """A workflow whose NeXus inputs can be wired for dynamic transforms.

    Exposes the typed ``dynamic_keys`` (so the routing layer can read each
    input's NeXus component type) and the not-yet-built ``base_pipeline`` to
    patch. Used by :func:`wire_dynamic_transforms` to drive f144-fed NXlog
    placeholders without the factory restating the component mapping.
    """

    @property
    def dynamic_keys(self) -> Mapping[str, Any]: ...
    @property
    def base_pipeline(self) -> sciline.Pipeline: ...


def synthesise_provider(
    name: str,
    impl: Callable[..., Any],
    annotations: dict[str, Any],
) -> Any:
    """Synthesise a Sciline provider with explicit named positional parameters.

    Returns a function ``name(p1, p2, ...)`` whose ``__annotations__`` come
    from ``annotations`` (with ``'return'`` consumed as the return type) and
    whose body delegates to ``impl(p1, p2, ...)``.

    Sciline introspects providers via ``inspect.getfullargspec``, which
    reads the underlying ``__code__`` and ignores ``__signature__``;
    producing N named typed positional parameters therefore requires
    building a real function via ``exec``/``compile``. Same technique
    ``dataclasses`` and ``namedtuple`` use to generate ``__init__``.

    ``name`` and every parameter name (annotation keys other than
    ``'return'``) must be valid Python identifiers; raises
    :class:`ValueError` otherwise. This guards the ``exec`` template
    against injection from caller-supplied strings.
    """
    params = [n for n in annotations if n != 'return']
    for ident in (name, *params):
        if not _PY_IDENTIFIER.fullmatch(ident):
            raise ValueError(
                f"synthesise_provider: {ident!r} is not a valid Python "
                "identifier; refusing to splice into exec template"
            )
    arg_list = ', '.join(params)
    src = f"def {name}({arg_list}):\n    return _impl({arg_list})\n"
    ns: dict[str, Any] = {'_impl': impl}
    exec(src, ns)  # noqa: S102
    fn = ns[name]
    fn.__annotations__ = dict(annotations)
    return fn


def build_patched_chain_provider(
    component_type: type,
    bindings: list[tuple[str, type[ValueLog]]],
) -> Any:
    """Build the fused chain-patch provider for ``component_type``.

    The returned provider replaces essreduce's ``get_transformation_chain``
    specialised to ``component_type``: it consumes
    ``NeXusComponent[component_type, SampleRun]`` plus one parameter per
    binding (annotated with that binding's :class:`ValueLog` subclass) and
    produces ``NeXusTransformationChain[component_type, SampleRun]``. It
    cannot instead consume the chain as input — that would self-cycle on
    its own return type.

    Parameters
    ----------
    component_type:
        The NeXus component type whose transformation chain is being patched
        (e.g. ``snx.NXdetector``).
    bindings:
        ``(transform_path, value_log_cls)`` pairs. The provider takes one
        positional parameter per pair, annotated with the corresponding
        :class:`ValueLog` subclass; at evaluation time the latest sample of
        each container is written into ``transformations[transform_path]``.
    """
    bindings_local = list(bindings)

    def _impl(component: Any, *containers: ValueLog) -> Any:
        chain = get_transformation_chain(component)
        patched = deepcopy(chain)
        for (path, _key), container in zip(bindings_local, containers, strict=True):
            if path not in patched.transformations:
                raise KeyError(
                    f"Transformation entry {path!r} not found in chain. "
                    f"Available entries: {sorted(patched.transformations.keys())}"
                )
            patched.transformations[path].value = container.values['time', -1].data
        return patched

    annotations: dict[str, Any] = {
        'component': NeXusComponent[component_type, SampleRun],
        **{f'log_{i}': key for i, (_, key) in enumerate(bindings_local)},
        'return': NeXusTransformationChain[component_type, SampleRun],
    }
    return synthesise_provider(
        name=f'_patched_chain__{component_type.__name__}',
        impl=_impl,
        annotations=annotations,
    )


def wire_dynamic_transforms(
    workflow: object,
    bindings: Iterable[ChainPatchBinding],
) -> None:
    """Wire f144-driven dynamic transforms into a workflow before its build.

    Derives the ``{source_name: component_type}`` map from the workflow's own
    ``dynamic_keys`` — each ``NeXusData[Component, Run]`` key carries the
    component type as its first type-arg — then, for every binding whose
    ``dependent_sources`` includes a derived source, groups the
    ``(transform_path, workflow_key)`` pairs by component type and inserts one
    fused provider per type (built by :func:`build_patched_chain_provider`).

    Deriving the map from ``dynamic_keys`` avoids restating it in the factory
    (which would duplicate ``dynamic_keys`` and is a recurring source of
    factory-author error). ``dynamic_keys`` is keyed by on-disk stream name
    (aux roles are resolved at workflow construction), so each wire name matches
    directly against ``dependent_sources``.

    Grouping by ``stream_name`` dedups two ways: repeat instrument bindings that
    resolve to the same stream, and a single binding whose ``dependent_sources``
    spans several sources of the same component type — either would otherwise
    yield a provider with duplicate-typed parameters, which Sciline rejects.

    Non-``NeXusData`` dynamic keys are ignored, as are workflows that do not
    expose a typed pipeline.

    Parameters
    ----------
    workflow:
        The not-yet-built workflow whose pipeline is patched in place.
    bindings:
        Pre-resolved chain-patch bindings for the owning instrument.
    """
    if not isinstance(workflow, SupportsDynamicTransforms):
        return
    bindings = list(bindings)
    components = {
        source: get_args(key)[0]
        for source, key in workflow.dynamic_keys.items()
        if get_origin(key) is NeXusData
    }
    by_type: dict[type, dict[str, tuple[str, type]]] = {}
    for source_name, component_type in components.items():
        for binding in bindings:
            if source_name not in binding.dependent_sources:
                continue
            by_type.setdefault(component_type, {})[binding.stream_name] = (
                binding.transform_path,
                binding.workflow_key,
            )
    for component_type, by_stream in by_type.items():
        workflow.base_pipeline.insert(
            build_patched_chain_provider(component_type, list(by_stream.values()))
        )
