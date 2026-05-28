# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import dataclasses
import inspect
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ess.livedata.config.stream import ContextBinding
from ess.livedata.config.workflow_spec import (
    JobId,
    WorkflowConfig,
    WorkflowId,
    WorkflowSpec,
)
from ess.livedata.core.message import Message
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


@dataclass(frozen=True, slots=True, kw_only=True)
class SpecContextBinding(ContextBinding):
    """Spec-scope context binding with per-job runtime callables.

    Lives next to :class:`WorkflowFactory` because the optional callables
    reference :class:`JobId` and :class:`Message` — runtime concepts that
    the declarative :mod:`config.workflow_spec` layer must not depend on.

    :attr:`stream_resolver`, when set, maps ``(job_id, stream_name)`` to
    the wire stream name used by routing and the gate. Resolvers are
    assumed to be pure name-suffixing operations on ``stream_name``; the
    registration-time collision check relies on this purity. Leaving it
    unset means the wire name equals :attr:`stream_name`.

    :attr:`seed_factory`, when set, produces the cold-start
    :class:`Message` fired at ``schedule_job`` time so the accumulator
    exists before any external producer publishes. Used for spec-level
    inputs with a meaningful "no message yet" default (currently ROI).
    """

    stream_resolver: Callable[[JobId, str], str] | None = field(default=None)
    seed_factory: Callable[[JobId], Message] | None = field(default=None)


@dataclass(frozen=True)
class WorkflowRegistration:
    """Per-workflow record held by :class:`WorkflowFactory`.

    Bundles the declarative :class:`WorkflowSpec` with implementation-side
    data: the factory callable, the owning service, spec-scope
    :class:`ContextBinding` declarations, and the opt-out flag for
    instrument-scope context bindings. Keeping these alongside the factory
    (rather than on :class:`WorkflowSpec`) preserves the spec as a purely
    declarative model and keeps runtime-coupled types
    (:class:`JobId`, :class:`Message`) out of :mod:`config.workflow_spec`.
    """

    spec: WorkflowSpec
    service: str
    factory: Callable[..., Workflow] | None = None
    context_bindings: tuple[SpecContextBinding, ...] = ()
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
        stream_resolver: Callable[[JobId, str], str] | None = None,
        seed_factory: Callable[[JobId], Message] | None = None,
    ) -> None:
        """Append a spec-level :class:`SpecContextBinding` to the registration.

        Late-bound from ``factories.py`` to keep workflow-key imports out of
        ``specs.py``. When ``dependent_sources`` is None, defaults to the
        spec's ``source_names`` — the binding applies uniformly across the
        spec.

        Chain-patch contexts (``workflow_key`` is a
        :class:`~ess.livedata.config.value_log.ValueLog` subclass) must be
        declared at instrument scope via
        :meth:`Instrument.add_context_binding`:
        :meth:`Instrument.apply_dynamic_transforms` reads only
        instrument-scope records, so a spec-scope chain-patch context would
        route the f144 value to a Sciline parameter that no provider
        consumes — silent-wrong.
        """
        self._factory._add_context_binding(
            self.workflow_id,
            stream_name=stream_name,
            workflow_key=workflow_key,
            dependent_sources=dependent_sources,
            stream_resolver=stream_resolver,
            seed_factory=seed_factory,
        )

    def skip_instrument_contexts(self) -> None:
        """Opt this spec out of all instrument-scope context-stream bindings.

        Use from ``specs.py`` when this spec consumes a source whose
        instrument declaration carries context (motion, geometry, etc.) but
        this spec does not need the value — e.g. a counts-only ratemeter on
        a moving detector. The spec is removed from the gate and from the
        resolved context for those streams. Spec-scope bindings (declared
        via :meth:`add_context_binding`) are unaffected.
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
        stream_resolver: Callable[[JobId, str], str] | None,
        seed_factory: Callable[[JobId], Message] | None,
    ) -> None:
        # Chain-patch contexts (ValueLog-typed workflow_key) at spec scope
        # would be silent-wrong: Instrument.apply_dynamic_transforms reads
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
        new_input = SpecContextBinding(
            stream_name=stream_name,
            workflow_key=workflow_key,
            dependent_sources=sources,
            stream_resolver=stream_resolver,
            seed_factory=seed_factory,
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
            Forwarded to factories that opt in by declaring ``context_keys``
            in their signature.
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
        # routed as raw values (e.g., via context_keys) can omit it.
        kwargs = {}
        if 'source_name' in sig.parameters:
            kwargs['source_name'] = source_name
        if workflow_params and 'params' in sig.parameters:
            kwargs['params'] = workflow_params
        if 'aux_source_names' in sig.parameters:
            kwargs['aux_source_names'] = aux_source_names or {}
        if 'context_keys' in sig.parameters:
            kwargs['context_keys'] = context_keys or {}

        # Call factory with appropriate arguments
        if kwargs:
            return factory(**kwargs)
        else:
            return factory()
