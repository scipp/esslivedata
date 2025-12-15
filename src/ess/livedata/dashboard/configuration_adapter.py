# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

Model = TypeVar('Model')


class JobConfigState(BaseModel):
    """Per-source configuration state."""

    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the workflow, as JSON-serialized Pydantic model",
    )
    aux_source_names: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Selected auxiliary source names as field name to stream name mapping"
        ),
    )


class ConfigurationState(BaseModel):
    """
    Persisted state for ConfigurationAdapter implementations.

    This model captures the user's configuration choices (sources, params,
    aux sources) that should be restored when reopening the dashboard.
    Used by both workflow and plotter configurations.

    Each source has its own configuration (params and aux_source_names),
    allowing different sources to be configured independently.
    """

    jobs: dict[str, JobConfigState] = Field(
        default_factory=dict,
        description="Per-source configuration, keyed by source name",
    )

    @property
    def source_names(self) -> list[str]:
        """Get source names from jobs dict keys."""
        return list(self.jobs.keys())


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
        self._selected_sources: list[str] | None = None

    def set_selected_sources(self, source_names: list[str]) -> None:
        """
        Scope the adapter to a subset of sources.

        When set, `initial_source_names` will return only these sources (filtered
        to available sources), and `initial_parameter_values`/`initial_aux_source_names`
        will be taken from the first of these sources.

        Parameters
        ----------
        source_names
            Source names to scope to.
        """
        self._selected_sources = source_names

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

    def _get_reference_job_config(self) -> JobConfigState | None:
        """Get the job config to use as reference for params/aux_sources.

        Returns the config for the first selected source (if scoped) or
        the first source in the config state.
        """
        if not self._config_state or not self._config_state.jobs:
            return None

        # If scoped to specific sources, use first scoped source that exists
        if self._selected_sources:
            for source in self._selected_sources:
                if source in self._config_state.jobs:
                    return self._config_state.jobs[source]

        # Otherwise use first available source
        return next(iter(self._config_state.jobs.values()))

    @property
    def initial_aux_source_names(self) -> dict[str, str]:
        """
        Initially selected auxiliary source names.

        Returns a mapping from field name (as defined in aux_sources model) to
        the selected stream name. Default implementation filters persisted aux
        sources to only include valid field names from the current aux_sources model.
        """
        if not self.aux_sources:
            return {}
        job_config = self._get_reference_job_config()
        if not job_config:
            return {}
        # Filter to only include valid field names
        valid_fields = set(self.aux_sources.model_fields.keys())
        return {
            k: v for k, v in job_config.aux_source_names.items() if k in valid_fields
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

        If scoped via set_selected_sources, returns only those sources (filtered
        to available sources). Otherwise, returns persisted source names filtered
        to available sources. Falls back to all available sources if no valid
        persisted sources remain.
        """
        available = set(self.source_names)

        # If scoped to specific sources, return those (filtered to available)
        if self._selected_sources:
            filtered = [s for s in self._selected_sources if s in available]
            return filtered if filtered else self.source_names

        # Otherwise use persisted sources from config state
        if not self._config_state:
            return self.source_names
        filtered = [
            name for name in self._config_state.source_names if name in available
        ]
        return filtered if filtered else self.source_names

    @property
    def initial_parameter_values(self) -> dict[str, Any]:
        """
        Initial parameter values.

        Returns persisted parameter values from the reference job config (first
        selected source if scoped, otherwise first source in config state).

        If stored params have no field overlap with the current model (indicating
        complete incompatibility, e.g., from a different workflow version), returns
        empty dict to fall back to defaults rather than propagating invalid data.
        """
        job_config = self._get_reference_job_config()
        if not job_config:
            return {}

        # Check compatibility with current model
        model_class = self.model_class()
        if model_class is None:
            # No model defined, return params as-is
            return job_config.params

        # Check if stored params have ANY overlap with current model fields
        # If no field names match, the config is from an incompatible version
        stored_keys = set(job_config.params.keys())
        model_fields = set(model_class.model_fields.keys())

        if stored_keys and not stored_keys.intersection(model_fields):
            # Complete incompatibility: no field overlap, fall back to defaults
            return {}

        # Partial or full compatibility: let Pydantic handle defaults/validation
        # Note: Pydantic ignores extra fields and uses defaults for missing ones
        return job_config.params

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
