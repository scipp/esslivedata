# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import dataclasses
import inspect
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ess.livedata.config.stream import ChainPatchBinding, ContextBinding
from ess.livedata.config.workflow_spec import (
    WorkflowConfig,
    WorkflowId,
    WorkflowSpec,
)
from ess.livedata.core.timestamp import Timestamp


class Workflow(Protocol):
    """
    A workflow that can process streams of data. Instances are run by a :py:class:`Job`.

    Wraps ess.reduce.streaming.StreamProcessor for data-reduction jobs;
    other implementations exist for non-data-reduction jobs.
    """

    def accumulate(
        self, data: dict[str, Any], *, start_time: Timestamp, end_time: Timestamp
    ) -> None: ...
    def finalize(self) -> dict[str, Any]: ...
    def clear(self) -> None: ...


@runtime_checkable
class SupportsContext(Protocol):
    """A :class:`Workflow` realized after creation with routing-resolved bindings.

    Workflows wrapping ``ess.reduce.streaming.StreamProcessor`` defer building
    their graph so the routing layer (:meth:`WorkflowFactory.create`) can inject
    per-job bindings the factory does not know: context streams (motion/geometry/
    ROI) and f144-driven chain-patch transforms. :meth:`build` takes both and
    materializes the workflow in one call, so the caller need not sequence
    separate configuration steps and factories need not thread these through
    their signature. (Aux-role resolution is not among these: factories key
    ``dynamic_keys`` by canonical stream name directly, via the
    ``aux_source_names`` map passed to the factory.)
    """

    def build(
        self,
        *,
        context_keys: Mapping[str, Any] | None = None,
        chain_patch_bindings: Iterable[ChainPatchBinding] = (),
    ) -> None: ...


@dataclass(frozen=True)
class WorkflowRegistration:
    """Per-workflow record held by :class:`WorkflowFactory`.

    Bundles the declarative :class:`WorkflowSpec` with implementation-side
    data: the factory callable, the owning service, spec-scope
    :class:`ContextBinding` declarations, and the opt-out flag for
    instrument-scope context bindings. Keeping these alongside the factory
    (rather than on :class:`WorkflowSpec`) preserves the spec as a purely
    declarative model and keeps the workflow-key imports out of
    :mod:`config.workflow_spec`.
    """

    spec: WorkflowSpec
    service: str
    factory: Callable[..., Workflow] | None = None
    context_bindings: tuple[ContextBinding, ...] = ()
    skip_instrument_contexts: bool = False


@dataclass(frozen=True)
class SpecHandle:
    """
    Handle for attaching factories to registered specs.

    Returned by WorkflowFactory.register_spec(), this handle is used to attach
    factory implementations to a previously registered spec via the attach_factory()
    decorator.
    """

    workflow_id: WorkflowId
    _factory: WorkflowFactory

    def attach_factory(
        self,
    ) -> Callable[[Callable[..., Workflow]], Callable[..., Workflow]]:
        """Decorator to attach factory implementation to this spec."""
        return self._factory.attach_factory(self.workflow_id)

    def add_context_binding(
        self,
        *,
        stream_name: str,
        workflow_key: Any,
        dependent_sources: Iterable[str] | None = None,
    ) -> None:
        """Append a spec-scope :class:`ContextBinding` to the registration.

        Late-bound from ``factories.py`` to keep workflow-key imports out of
        ``specs.py``. When ``dependent_sources`` is None, defaults to the
        spec's ``source_names`` — the binding applies uniformly across the
        spec.

        Spec scope is for context streams that are a property of *this
        workflow* rather than of the source — e.g. a sample-temperature
        sensor feeding one reduction. The stream value is delivered to
        ``workflow_key`` via ``set_context`` and the job gates on it (see
        ADR 0002); the wire name equals ``stream_name`` (no per-job
        suffixing) and there is no cold-start seed, so the gate stays
        closed until the producer publishes — the correct behaviour for a
        context with no safe default.

        Chain-patch contexts (``workflow_key`` is a
        :class:`~ess.livedata.config.value_log.ValueLog` subclass) must be
        declared at instrument scope via
        :meth:`Instrument.add_context_binding`:
        :attr:`Instrument.chain_patch_bindings` reads only instrument-scope
        records, so a spec-scope chain-patch context would route the f144
        value to a Sciline parameter that no provider consumes — silent-wrong.
        """
        self._factory._add_context_binding(
            self.workflow_id,
            stream_name=stream_name,
            workflow_key=workflow_key,
            dependent_sources=dependent_sources,
        )

    def skip_instrument_contexts(self) -> None:
        """Opt this spec out of all instrument-scope context-stream bindings.

        Use when this spec consumes a source whose instrument declaration
        carries context (motion, geometry, etc.) but this spec does not need
        the value — e.g. a counts-only ratemeter on a moving detector. The
        spec is removed from the gate and from the resolved context for those
        streams. Spec-scope bindings (declared via :meth:`add_context_binding`)
        are unaffected.

        Call this from ``factories.py``, co-located with the
        :meth:`Instrument.add_context_binding` it negates: the opt-out has no
        meaning without the instrument-scope binding it cancels, and the
        rationale (does this workflow consume the geometry value?) is
        implementation knowledge that lives with the factory. The call needs
        no workflow-key import, so the ``specs.py`` import constraint does not
        apply.
        """
        self._factory._set_skip_instrument_contexts(self.workflow_id)


class WorkflowFactory(Mapping[WorkflowId, WorkflowSpec]):
    """Registry mapping :class:`WorkflowId` to :class:`WorkflowRegistration`.

    Implements ``Mapping[WorkflowId, WorkflowSpec]`` for read access: the
    dashboard and other spec-only readers see this as a mapping to specs.
    Use :meth:`registration` / :meth:`registrations` to access the full
    record (factory callable, context bindings, etc.).
    """

    def __init__(self) -> None:
        self._registrations: dict[WorkflowId, WorkflowRegistration] = {}

    def __getitem__(self, key: WorkflowId) -> WorkflowSpec:
        return self._registrations[key].spec

    def __iter__(self) -> Iterator[WorkflowId]:
        return iter(self._registrations)

    def __len__(self) -> int:
        return len(self._registrations)

    def register_spec(
        self, spec: WorkflowSpec, *, service: str | None = None
    ) -> SpecHandle:
        """
        Register workflow spec, return handle for later factory attachment.

        This is the first phase of two-phase registration. The spec is stored
        and a handle is returned that can be used later to attach the factory
        implementation.

        Parameters
        ----------
        spec:
            Workflow specification to register.
        service:
            Name of the backend service responsible for running this workflow.
            Used by ``JobFactory`` to reject workflows not belonging to the
            current service. Defaults to ``spec.group.name``.

        Returns
        -------
        Handle for attaching factory later.
        """
        spec_id = spec.get_id()
        if spec_id in self._registrations:
            raise ValueError(f"Workflow spec '{spec_id}' already registered.")
        self._registrations[spec_id] = WorkflowRegistration(
            spec=spec,
            service=service if service is not None else spec.group.name,
        )
        return SpecHandle(workflow_id=spec_id, _factory=self)

    def registration(self, workflow_id: WorkflowId) -> WorkflowRegistration | None:
        """Return the full registration record for ``workflow_id``, or None."""
        return self._registrations.get(workflow_id)

    def registrations(self) -> Iterable[WorkflowRegistration]:
        """Iterate over all registrations."""
        return self._registrations.values()

    def get_service(self, workflow_id: WorkflowId) -> str | None:
        """Return the backend service name that runs the given workflow."""
        reg = self._registrations.get(workflow_id)
        return reg.service if reg is not None else None

    def attach_factory(
        self, workflow_id: WorkflowId
    ) -> Callable[[Callable[..., Workflow]], Callable[..., Workflow]]:
        """
        Decorator to attach factory to a previously registered spec.

        This is the second phase of two-phase registration. The factory's
        params type hint is validated against the spec.params using object
        identity (is not).

        Parameters
        ----------
        workflow_id:
            ID of the previously registered spec.

        Returns
        -------
        Decorator function that validates and attaches the factory.
        """
        if workflow_id not in self._registrations:
            raise ValueError(
                f"Spec '{workflow_id}' not registered. Call register_spec() first."
            )

        spec = self._registrations[workflow_id].spec

        def decorator(factory: Callable[..., Workflow]) -> Callable[..., Workflow]:
            # Validate params type hint matches spec
            # Use get_type_hints to resolve forward references, in case we used
            # `from __future__ import annotations`.
            type_hints = typing.get_type_hints(factory, globalns=factory.__globals__)
            inferred_params = type_hints.get('params', None)

            if spec.params is not None and inferred_params is not None:
                # Spec params may be a subclass of the factory's declared type
                # (e.g. dynamic subclasses produced by ``make_detector_view_params``).
                if not issubclass(spec.params, inferred_params):
                    raise TypeError(f"Params type mismatch for {workflow_id}")
            elif spec.params is None and inferred_params is not None:
                raise TypeError(
                    f"Factory has params but spec has none for {workflow_id}"
                )
            elif spec.params is not None and inferred_params is None:
                raise TypeError(
                    f"Spec has params but factory has none for {workflow_id}"
                )

            self._registrations[workflow_id] = dataclasses.replace(
                self._registrations[workflow_id], factory=factory
            )
            return factory

        return decorator

    def _add_context_binding(
        self,
        workflow_id: WorkflowId,
        *,
        stream_name: str,
        workflow_key: Any,
        dependent_sources: Iterable[str] | None,
    ) -> None:
        # Chain-patch contexts (ValueLog-typed workflow_key) at spec scope
        # would be silent-wrong: Instrument.chain_patch_bindings reads
        # only instrument-scope records, so the f144 value would route to a
        # Sciline parameter no provider consumes.
        from ess.livedata.config.value_log import ValueLog

        if isinstance(workflow_key, type) and issubclass(workflow_key, ValueLog):
            raise ValueError(
                f"workflow_key {workflow_key.__name__!r} is a ValueLog subclass; "
                "chain-patch contexts must be declared at instrument scope via "
                "Instrument.add_context_binding, not at spec scope"
            )
        reg = self._registrations[workflow_id]
        if dependent_sources is None:
            sources = frozenset(reg.spec.source_names)
        else:
            sources = frozenset(dependent_sources)
        new_input = ContextBinding(
            stream_name=stream_name,
            workflow_key=workflow_key,
            dependent_sources=sources,
        )
        self._registrations[workflow_id] = dataclasses.replace(
            reg, context_bindings=(*reg.context_bindings, new_input)
        )

    def _set_skip_instrument_contexts(self, workflow_id: WorkflowId) -> None:
        reg = self._registrations[workflow_id]
        self._registrations[workflow_id] = dataclasses.replace(
            reg, skip_instrument_contexts=True
        )

    def create(
        self,
        *,
        source_name: str,
        config: WorkflowConfig,
        aux_source_names: dict[str, str] | None = None,
        context_keys: dict[str, Any] | None = None,
        chain_patch_bindings: Iterable[ChainPatchBinding] = (),
    ) -> Workflow:
        """
        Create a workflow instance using the registered factory.

        Parameters
        ----------
        source_name:
            Name of the data source.
        config:
            Configuration for the workflow, including the identifier and parameters.
        aux_source_names:
            Rendered auxiliary source names (already resolved by JobFactory).
        context_keys:
            Resolved ``ContextBinding`` mapping (stream_name → workflow_key).
            Passed to :meth:`SupportsContext.build` *after* the factory returns
            the workflow, so factories do not declare ``context_keys`` in their
            signature.
        chain_patch_bindings:
            Pre-resolved instrument-scope chain-patch bindings (see
            :attr:`Instrument.chain_patch_bindings`). Passed to
            :meth:`SupportsContext.build`, which wires them as f144-driven
            dynamic transforms while materializing the graph. Only applied to
            workflows that defer their build (:class:`SupportsContext`).
        """
        workflow_id = config.identifier
        if workflow_id not in self._registrations:
            raise KeyError(f"Unknown workflow ID: {workflow_id}")

        reg = self._registrations[workflow_id]
        workflow_spec = reg.spec
        if (model_cls := workflow_spec.params) is None:
            if config.params:
                raise ValueError(
                    f"Workflow '{workflow_id}' does not require parameters, "
                    f"but received: {config.params}"
                )
            workflow_params = None
        else:
            workflow_params = model_cls.model_validate(config.params)

        # Validate aux_sources configuration
        if workflow_spec.aux_sources is None:
            if config.aux_source_names:
                raise ValueError(
                    f"Workflow '{workflow_id}' does not require auxiliary sources, "
                    f"but received: {config.aux_source_names}"
                )

        if workflow_spec.source_names and source_name not in workflow_spec.source_names:
            allowed_sources = ", ".join(workflow_spec.source_names)
            raise ValueError(
                f"Source '{source_name}' is not allowed for workflow "
                f"'{workflow_spec.name}'. "
                f"Allowed sources: {allowed_sources}"
            )

        factory = reg.factory
        if factory is None:
            raise ValueError(
                f"Workflow '{workflow_id}' has no factory attached; "
                "call attach_factory() before create()."
            )
        sig = inspect.signature(factory)

        # Prepare arguments based on the factory signature. Factories opt in to
        # each argument by declaring it in their signature. For example, factories
        # that need to configure NeXus name keys from aux source selections declare
        # `aux_source_names: dict[str, str]`; factories whose aux sources are
        # routed as raw values (e.g., via context bindings) can omit it.
        kwargs = {}
        if 'source_name' in sig.parameters:
            kwargs['source_name'] = source_name
        if workflow_params and 'params' in sig.parameters:
            kwargs['params'] = workflow_params
        if 'aux_source_names' in sig.parameters:
            kwargs['aux_source_names'] = aux_source_names or {}

        workflow = factory(**kwargs) if kwargs else factory()

        # Context bindings are injected after creation rather than threaded
        # through the factory signature: whether a workflow consumes context is
        # a property of the produced workflow, not of the factory. Eagerly
        # ``build()`` here so graph validation and precompute happen at job
        # creation (we pay the startup cost now).
        if isinstance(workflow, SupportsContext):
            workflow.build(
                context_keys=context_keys,
                chain_patch_bindings=chain_patch_bindings,
            )
        elif context_keys:
            raise TypeError(
                f"Workflow '{workflow_id}' resolved context bindings "
                f"{sorted(context_keys)} but the produced workflow "
                f"{type(workflow).__name__} does not consume context."
            )
        return workflow
