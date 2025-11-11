# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import inspect
import typing
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ess.livedata.config.workflow_spec import WorkflowConfig, WorkflowId, WorkflowSpec


class Workflow(Protocol):
    """
    A workflow that can process streams of data. Instances are run by a :py:class:`Job`.

    This protocol matches ess.reduce.streaming.StreamProcessor. There are other
    implementations, in particular for non-data-reduction jobs.
    """

    def accumulate(
        self, data: dict[str, Any], *, start_time: int, end_time: int
    ) -> None: ...
    def finalize(self) -> dict[str, Any]: ...
    def clear(self) -> None: ...


@dataclass(frozen=True)
class SpecHandle:
    """
    Handle for attaching factories to registered specs.

    Returned by WorkflowFactory.register_spec(), this handle is used to attach
    factory implementations to a previously registered spec via the attach_factory()
    decorator.
    """

    workflow_id: WorkflowId
    _factory: 'WorkflowFactory'

    def attach_factory(
        self,
    ) -> Callable[[Callable[..., Workflow]], Callable[..., Workflow]]:
        """Decorator to attach factory implementation to this spec."""
        return self._factory.attach_factory(self.workflow_id)


class WorkflowFactory(Mapping[WorkflowId, WorkflowSpec]):
    def __init__(self) -> None:
        self._factories: dict[WorkflowId, Callable[[], Workflow]] = {}
        self._workflow_specs: dict[WorkflowId, WorkflowSpec] = {}

    def __getitem__(self, key: WorkflowId) -> WorkflowSpec:
        return self._workflow_specs[key]

    def __iter__(self) -> Iterator[WorkflowId]:
        return iter(self._workflow_specs)

    def __len__(self) -> int:
        return len(self._workflow_specs)

    @property
    def source_names(self) -> set[str]:
        """
        Get all source names that have associated workflows.

        Returns
        -------
        Set of source names.
        """
        return {
            source_name
            for spec in self._workflow_specs.values()
            for source_name in spec.source_names
        }

    def register_spec(self, spec: WorkflowSpec) -> SpecHandle:
        """
        Register workflow spec, return handle for later factory attachment.

        This is the first phase of two-phase registration. The spec is stored
        and a handle is returned that can be used later to attach the factory
        implementation.

        Parameters
        ----------
        spec:
            Workflow specification to register.

        Returns
        -------
        Handle for attaching factory later.
        """
        spec_id = spec.get_id()
        if spec_id in self._workflow_specs:
            raise ValueError(f"Workflow spec '{spec_id}' already registered.")
        self._workflow_specs[spec_id] = spec
        return SpecHandle(workflow_id=spec_id, _factory=self)

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
        if workflow_id not in self._workflow_specs:
            raise ValueError(
                f"Spec '{workflow_id}' not registered. Call register_spec() first."
            )

        spec = self._workflow_specs[workflow_id]

        def decorator(factory: Callable[..., Workflow]) -> Callable[..., Workflow]:
            # Validate params type hint matches spec
            # Use get_type_hints to resolve forward references, in case we used
            # `from __future__ import annotations`.
            type_hints = typing.get_type_hints(factory, globalns=factory.__globals__)
            inferred_params = type_hints.get('params', None)

            if spec.params is not None and inferred_params is not None:
                if spec.params is not inferred_params:  # Use `is not`
                    raise TypeError(f"Params type mismatch for {workflow_id}")
            elif spec.params is None and inferred_params is not None:
                raise TypeError(
                    f"Factory has params but spec has none for {workflow_id}"
                )
            elif spec.params is not None and inferred_params is None:
                raise TypeError(
                    f"Spec has params but factory has none for {workflow_id}"
                )

            self._factories[workflow_id] = factory
            return factory

        return decorator

    def create(self, *, source_name: str, config: WorkflowConfig) -> Workflow:
        """
        Create a workflow instance using the registered factory.

        Parameters
        ----------
        source_name:
            Name of the data source.
        config:
            Configuration for the workflow, including the identifier and parameters.
        """
        workflow_id = config.identifier
        if workflow_id not in self._workflow_specs:
            raise KeyError(f"Unknown workflow ID: {workflow_id}")

        workflow_spec = self._workflow_specs[workflow_id]
        if (model_cls := workflow_spec.params) is None:
            if config.params:
                raise ValueError(
                    f"Workflow '{workflow_id}' does not require parameters, "
                    f"but received: {config.params}"
                )
            workflow_params = None
        else:
            workflow_params = model_cls.model_validate(config.params)

        # Validate aux_sources configuration (but don't pass to workflow)
        if (aux_model_cls := workflow_spec.aux_sources) is None:
            if config.aux_source_names:
                raise ValueError(
                    f"Workflow '{workflow_id}' does not require auxiliary sources, "
                    f"but received: {config.aux_source_names}"
                )
        else:
            # Validate that aux_source_names conform to the model
            aux_model_cls.model_validate(config.aux_source_names)

        if workflow_spec.source_names and source_name not in workflow_spec.source_names:
            allowed_sources = ", ".join(workflow_spec.source_names)
            raise ValueError(
                f"Source '{source_name}' is not allowed for workflow "
                f"'{workflow_spec.name}'. "
                f"Allowed sources: {allowed_sources}"
            )

        factory = self._factories[workflow_id]
        sig = inspect.signature(factory)

        # Prepare arguments based on the factory signature
        kwargs = {}
        if 'source_name' in sig.parameters:
            kwargs['source_name'] = source_name
        if workflow_params and 'params' in sig.parameters:
            kwargs['params'] = workflow_params

        # Call factory with appropriate arguments
        if kwargs:
            return factory(**kwargs)
        else:
            return factory()
