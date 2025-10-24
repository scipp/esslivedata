# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

Model = TypeVar('Model')


class ConfigurationAdapter(ABC, Generic[Model]):
    """Abstract adapter for providing configuration data to generic widgets."""

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
        the selected stream name.
        """
        return {}

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
    @abstractmethod
    def initial_source_names(self) -> list[str]:
        """Initially selected source names."""

    @property
    @abstractmethod
    def initial_parameter_values(self) -> dict[str, Any]:
        """Initial parameter values."""

    @abstractmethod
    def start_action(
        self,
        selected_sources: list[str],
        parameter_values: Model,
    ) -> Any:
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

        Returns
        -------
        :
            Result of the action (implementation-specific), or None if no result

        Raises
        ------
        Exception
            May raise exceptions if the action fails. Callers should handle exceptions
            appropriately (e.g., log and display error messages to users).
        """
