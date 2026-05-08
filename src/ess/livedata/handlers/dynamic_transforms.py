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

See ``docs/developer/plans/dynamic-nexus-transforms.md`` for the design.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import h5py
import sciline
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import (
    Filename,
    NeXusComponent,
    NeXusName,
    NeXusTransformationChain,
    SampleRun,
)
from ess.reduce.nexus.workflow import get_transformation_chain

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


def _decode(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, 'decode'):
        value = value.decode()
    if isinstance(value, bytes):
        value = value.decode()
    return value if isinstance(value, str) else None


def _walk_depends_on(f: h5py.File, start_path: str) -> list[str]:
    """Walk a depends_on chain starting from a transformation path.

    Each node in the chain is identified by its absolute NeXus path. The
    next link is found in the node's ``depends_on`` *attribute* — applies
    uniformly to NXlog groups (placeholders) and to plain transformation
    Datasets. The terminal value is the literal string ``.``.
    """
    chain: list[str] = []
    current = start_path
    while current and current != '.':
        path = current.lstrip('/')
        if path not in f:
            break
        obj = f[path]
        chain.append(current)
        next_target = _decode(obj.attrs.get('depends_on'))
        if next_target is None:
            break
        current = next_target
    return chain


def _component_chain_paths(geometry_filename: str, component_path: str) -> list[str]:
    """Resolve all NeXus paths along a component's depends_on chain.

    Returns paths starting from the component's ``depends_on`` target.
    The component's ``depends_on`` is a Dataset (string) at
    ``<component>/depends_on`` — only the entry point uses Dataset
    storage; subsequent links are attributes (handled by
    ``_walk_depends_on``).
    """
    with h5py.File(geometry_filename, 'r') as f:
        depends_on_path = f'{component_path}/depends_on'.lstrip('/')
        if depends_on_path not in f:
            return []
        target = _decode(f[depends_on_path][()])
        if target is None:
            return []
        return _walk_depends_on(f, target)


_NX_CLASS_BY_TYPE: dict[type, str] = {
    snx.NXdetector: 'detector',
    snx.NXmonitor: 'monitor',
}


def _component_group_path(component_type: type, source_name: str) -> str:
    """NeXus group path for a component of given type.

    Components are at ``/entry/instrument/<source_name>``; this helper
    centralises the convention.
    """
    return f'/entry/instrument/{source_name}'


def _scan_for_empty_nxlog(geometry_filename: str, component_path: str) -> str | None:
    """Walk depends_on from a component and return the first empty NXlog
    path encountered, or None if none."""
    paths = _component_chain_paths(geometry_filename, component_path)
    with h5py.File(geometry_filename, 'r') as f:
        for path in paths:
            obj = f[path.lstrip('/')]
            if isinstance(obj, h5py.Group) and obj.attrs.get('NX_class') in (
                b'NXlog',
                'NXlog',
            ):
                value = obj.get('value')
                if value is None or value.shape == (0,):
                    return path
    return None


def _build_patched_chain_provider(
    component_type: type, matched: list[DynamicTransformBinding]
) -> Any:
    """Build the per-component-type closure that patches the chain.

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


def apply_dynamic_transforms(
    workflow: sciline.Pipeline,
    *,
    instrument: Instrument,
    component_types: Iterable[type],
) -> dict[str, type[TransformLog]]:
    """Patch a workflow to drive matching NXlog placeholders from f144 streams.

    For each requested component type, reads the corresponding
    ``NeXusName[T]`` value set on ``workflow``, walks the depends_on
    chain in the geometry artifact (resolved via ``Filename[SampleRun]``
    set on the workflow), and intersects the walked paths with
    ``instrument.dynamic_transforms``. For each component type with at
    least one match, replaces the ``NeXusTransformationChain[T, SampleRun]``
    provider with a closure that consumes the matched bindings'
    ``log_key`` parameters and writes the latest sample into the chain.

    Raises ``ValueError`` if a chain walk encounters an empty NXlog that
    is not matched by any registry binding — this catches the failure
    mode described in issue #922 where a workflow walks an unmanaged
    placeholder and would otherwise hit
    ``reject_time_dependent_transform`` with no actionable hint.

    Parameters
    ----------
    workflow:
        The Sciline pipeline to patch in place. Must have
        ``Filename[SampleRun]`` set; for each component type T, must
        have ``NeXusName[T]`` set to the component's source name.
    instrument:
        The instrument whose ``dynamic_transforms`` registry is
        consulted.
    component_types:
        Component types loaded by the workflow (e.g. ``(NXdetector,)``,
        ``(NXdetector, NXmonitor)``).

    Returns
    -------
    :
        Mapping ``stream_name -> log_key`` for every binding that
        actually matched. The factory passes this as ``context_keys``
        to :class:`StreamProcessorWorkflow` so SPW's wrapping rule
        delivers each f144 NXlog to the right Sciline parameter.
    """
    filename = workflow.compute(Filename[SampleRun])
    context_keys: dict[str, type[TransformLog]] = {}
    for component_type in component_types:
        try:
            source_name = workflow.compute(NeXusName[component_type])
        except Exception:  # noqa: S112 Component type declared but not loaded
            continue
        component_path = _component_group_path(component_type, source_name)
        chain_paths = _component_chain_paths(str(filename), component_path)
        chain_set = set(chain_paths)
        matched = [
            b for b in instrument.dynamic_transforms if b.nxlog_path in chain_set
        ]
        if not matched:
            empty = _scan_for_empty_nxlog(str(filename), component_path)
            if empty is not None:
                raise ValueError(
                    f"Component {source_name!r} (type {component_type.__name__}) "
                    f"depends on an empty NXlog placeholder at {empty!r} that "
                    f"is not declared in instrument.dynamic_transforms. "
                    f"Add a DynamicTransformBinding for it, or fix the "
                    f"geometry artifact."
                )
            continue
        provider = _build_patched_chain_provider(component_type, matched)
        workflow.insert(provider)
        for binding in matched:
            context_keys[binding.stream_name] = binding.log_key
    return context_keys


def dynamic_transform_aux_inputs(
    instrument: Instrument, source_names: Iterable[str]
) -> dict[str, str]:
    """Aux-source ``inputs`` mapping covering bindings whose consumers
    intersect ``source_names``.

    Returned shape matches the dict accepted by
    :class:`AuxSources.__init__`. Used by the spec-side merge in
    ``register_detector_view_spec`` / ``register_monitor_workflow_specs``.
    """
    selected = set(source_names)
    return {
        b.stream_name: b.stream_name
        for b in instrument.dynamic_transforms
        if b.consumers & selected
    }


def dynamic_transform_routes(
    instrument: Instrument, source_name: str
) -> dict[str, str]:
    """Stream names to route to a job, scoped by ``source_name``.

    Returns the bindings whose ``consumers`` set includes ``source_name``,
    rendered un-prefixed (these are global f144 streams, not job-specific).
    """
    return {
        b.stream_name: b.stream_name
        for b in instrument.dynamic_transforms
        if source_name in b.consumers
    }
