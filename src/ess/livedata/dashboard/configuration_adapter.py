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

        Default implementation returns persisted parameter values if available,
        otherwise returns empty dict.
        """
        if not self._config_state:
            return {}
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
