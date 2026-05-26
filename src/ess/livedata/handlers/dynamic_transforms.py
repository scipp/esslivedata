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
:func:`add_dynamic_transforms` inserts the provider into the workflow.
Callers usually go through :meth:`Instrument.apply_dynamic_transforms`,
which derives the binding list from the instrument's
:attr:`Instrument.context_inputs` registry.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import sciline
from ess.reduce.nexus.types import NeXusComponent, NeXusTransformationChain, SampleRun
from ess.reduce.nexus.workflow import get_transformation_chain

from ess.livedata.config.value_log import ValueLog

_PY_IDENTIFIER = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


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
    binding (annotated with that binding's ``log_key``) and produces
    ``NeXusTransformationChain[component_type, SampleRun]``. It cannot
    instead consume the chain as input — that would self-cycle on its own
    return type.

    Parameters
    ----------
    component_type:
        The NeXus component type whose transformation chain is being patched
        (e.g. ``snx.NXdetector``).
    bindings:
        ``(transform_path, log_key)`` pairs. The provider takes one positional
        parameter per pair, annotated with the corresponding ``log_key``; at
        evaluation time the latest sample of each container is written into
        ``transformations[transform_path]``.
    """
    bindings_local = list(bindings)

    def _impl(component: Any, *containers: ValueLog | None) -> Any:
        chain = get_transformation_chain(component)
        patched = deepcopy(chain)
        for (path, _key), container in zip(bindings_local, containers, strict=True):
            if path not in patched.transformations:
                raise KeyError(
                    f"Transformation entry {path!r} not found in chain. "
                    f"Available entries: {sorted(patched.transformations.keys())}"
                )
            if (
                container is None
                or container.values is None
                or container.values.sizes.get('time', 0) == 0
            ):
                raise ValueError(
                    f"No samples yet for transformation {path!r}: "
                    "f144 stream has not produced a value."
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


def add_dynamic_transforms(
    workflow: sciline.Pipeline,
    *,
    component_type: type,
    bindings: list[tuple[str, type[ValueLog]]],
) -> None:
    """Patch ``workflow`` to drive matching NXlog placeholders from f144 streams.

    Inserts a fused single-step provider built by
    :func:`build_patched_chain_provider` that consumes one :class:`ValueLog`
    parameter per binding and writes the latest sample into the corresponding
    transformation along the chain. No-op when ``bindings`` is empty.
    """
    if not bindings:
        return
    workflow.insert(build_patched_chain_provider(component_type, bindings))
