# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

Model = TypeVar('Model')


class ConfigurationState(BaseModel):
    """
    Persisted state for ConfigurationAdapter implementations.

    This model captures the user's configuration choices (sources, params,
    aux sources) that should be restored when reopening the dashboard.
    Used by both workflow and plotter configurations.

    Schema Limitation
    -----------------
    This schema currently assumes all sources share the same `params` configuration,
    with only `aux_source_names` varying per source. In reality, JobOrchestrator's
    internal state (`staged_jobs`) allows different params per source via
    `dict[SourceName, JobConfig]`.

    For now, we expand on load: the single `params` dict is applied to all sources
    in `source_names`, and `aux_source_names` is expanded per-source as needed.
    This works because the current UI (WorkflowController.start_workflow) stages
    the same params for all sources in a single operation.

    Future work: If we support per-source params in the UI (e.g., "stage source1
    with configA, stage source2 with configB"), this schema should be extended to:
    `jobs: dict[str, JobConfigState]` where `JobConfigState` contains both params
    and aux_source_names per source.
    """

    source_names: list[str] = Field(
        default_factory=list,
        description="Selected source names for this workflow or plotter",
    )
    aux_source_names: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Selected auxiliary source names as field name to stream name mapping"
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the workflow, as JSON-serialized Pydantic model",
    )


class ConfigurationAdapter(ABC, Generic[Model]):
    """
    Abstract adapter for providing configuration data to generic widgets.

    Subclasses should call `super().__init__(config_state=...)` to provide
    persistent configuration that will be used by the default implementations
    of `initial_source_names`, `initial_aux_source_names`, and
    `initial_parameter_values`.
    """

    def __init__(self, config_state: ConfigurationState | None = None) -> None:
        """
        Initialize the configuration adapter.

        Parameters
        ----------
        config_state
            Persistent configuration state to restore, or None for default values.
        """
        self._config_state = config_state

    @property
    @abstractmethod
    def title(self) -> str:
        """Configuration title."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Configuration description."""

    @property
    def aux_sources(self) -> type[BaseModel] | None:
        """
        Pydantic model class for auxiliary sources.

        Returns None if the workflow does not use auxiliary sources.
        Field names define the aux source identifiers, and field types (typically
        Literal or Enum) define the available stream choices.
        """
        return None

    @property
    def initial_aux_source_names(self) -> dict[str, str]:
        """
        Initially selected auxiliary source names.

        Returns a mapping from field name (as defined in aux_sources model) to
        the selected stream name. Default implementation filters persisted aux
        sources to only include valid field names from the current aux_sources model.
        """
        if not self._config_state:
            return {}
        if not self.aux_sources:
            return {}
        # Filter to only include valid field names
        valid_fields = set(self.aux_sources.model_fields.keys())
        return {
            k: v
            for k, v in self._config_state.aux_source_names.items()
            if k in valid_fields
        }

    def set_aux_sources(self, aux_source_names: BaseModel | None) -> type[Model] | None:
        """
        Set auxiliary sources and return the parameter model class.

        This method stores the aux sources internally and returns the model class
        for parameters. Implementations can access the stored aux sources via
        self._cached_aux_sources.

        Parameters
        ----------
        aux_source_names
            Selected auxiliary sources as a Pydantic model instance, or None if no
            aux sources are selected.

        Returns
        -------
        :
            Pydantic model class for parameters, or None if no parameters.
        """
        self._cached_aux_sources = aux_source_names
        return self.model_class()

    @abstractmethod
    def model_class(self) -> type[Model] | None:
        """
        Pydantic model class for parameters.

        Implementations can access cached aux sources via self._cached_aux_sources
        if needed to create dynamic parameter models.
        """

    @property
    @abstractmethod
    def source_names(self) -> list[str]:
        """Available source names."""

    @property
    def initial_source_names(self) -> list[str]:
        """
        Initially selected source names.

        Default implementation filters persisted source names to only include
        currently available sources. If no valid persisted sources remain,
        defaults to all available sources.
        """
        if not self._config_state:
            return self.source_names
        filtered = [
            name
            for name in self._config_state.source_names
            if name in self.source_names
        ]
        return filtered if filtered else self.source_names

    @property
    def initial_parameter_values(self) -> dict[str, Any]:
        """
        Initial parameter values.

        Default implementation returns persisted parameter values if available
        and compatible with the current model, otherwise returns empty dict to
        trigger default values.

        If stored params have no field overlap with the current model (indicating
        complete incompatibility, e.g., from a different workflow version), returns
        empty dict to fall back to defaults rather than propagating invalid data.
        """
        if not self._config_state:
            return {}

        # Check compatibility with current model
        model_class = self.model_class()
        if model_class is None:
            # No model defined, return params as-is
            return self._config_state.params

        # Check if stored params have ANY overlap with current model fields
        # If no field names match, the config is from an incompatible version
        stored_keys = set(self._config_state.params.keys())
        model_fields = set(model_class.model_fields.keys())

        if stored_keys and not stored_keys.intersection(model_fields):
            # Complete incompatibility: no field overlap, fall back to defaults
            return {}

        # Partial or full compatibility: let Pydantic handle defaults/validation
        # Note: Pydantic ignores extra fields and uses defaults for missing ones
        return self._config_state.params

    @abstractmethod
    def start_action(
        self,
        selected_sources: list[str],
        parameter_values: Model,
    ) -> None:
        """
        Execute the start action with selected sources and parameters.

        Implementations can access cached aux sources via self._cached_aux_sources
        if needed.

        Parameters
        ----------
        selected_sources
            Selected source names
        parameter_values
            Parameter values as a validated Pydantic model instance

        Raises
        ------
        Exception
            May raise exceptions if the action fails. Callers should handle exceptions
            appropriately (e.g., log and display error messages to users).
        """
