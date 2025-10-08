# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import inspect
import typing
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Protocol

from ess.livedata.config.workflow_spec import WorkflowConfig, WorkflowId, WorkflowSpec


class Workflow(Protocol):
    """
    A workflow that can process streams of data. Instances are run by a :py:class:`Job`.

    This protocol matches ess.reduce.streaming.StreamProcessor. There are other
    implementations, in particular for non-data-reduction jobs.
    """

    def accumulate(self, data: dict[str, Any]) -> None: ...
    def finalize(self) -> dict[str, Any]: ...
    def clear(self) -> None: ...


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

    def register(
        self, spec: WorkflowSpec
    ) -> Callable[[Callable[..., Workflow]], Callable[..., Workflow]]:
        """
        Decorator to register a factory function for creating workflow instances.

        Parameters
        ----------
        spec:
            Workflow specification that describes the workflow to register.

        Returns
        -------
        Decorator function that registers the factory and returns it unchanged.
        """
        spec_id = spec.get_id()
        if spec_id in self._factories:
            raise ValueError(f"Workflow ID '{spec_id}' is already registered.")

        def decorator(
            factory: Callable[..., Workflow],
        ) -> Callable[..., Workflow]:
            # Try to get the type hints of the 'params' and 'aux_sources' arguments
            # if they exist. Use get_type_hints to resolve forward references, in case
            # we used `from __future__ import annotations`.
            type_hints = typing.get_type_hints(factory, globalns=factory.__globals__)
            if (params_type := type_hints.get('params', None)) is not None:
                spec.params = params_type
            if (aux_sources_type := type_hints.get('aux_sources', None)) is not None:
                spec.aux_sources = aux_sources_type
            self._factories[spec_id] = factory
            self._workflow_specs[spec_id] = spec
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

        if (aux_model_cls := workflow_spec.aux_sources) is None:
            if config.aux_source_names:
                raise ValueError(
                    f"Workflow '{workflow_id}' does not require auxiliary sources, "
                    f"but received: {config.aux_source_names}"
                )
            workflow_aux_sources = None
        else:
            workflow_aux_sources = aux_model_cls.model_validate(config.aux_source_names)

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
        if workflow_aux_sources and 'aux_sources' in sig.parameters:
            kwargs['aux_sources'] = workflow_aux_sources

        # Call factory with appropriate arguments
        if kwargs:
            return factory(**kwargs)
        else:
            return factory()
