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

    @abstractmethod
    def model_class(self, aux_source_names: BaseModel | None) -> type[Model] | None:
        """
        Pydantic model class for parameters.

        Parameters
        ----------
        aux_source_names
            Selected auxiliary sources as a Pydantic model instance, or None if no
            aux sources are selected. The adapter can serialize this using
            model_dump() or model_dump(mode='json') as needed.
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
        aux_source_names: BaseModel | None = None,
    ) -> bool:
        """
        Execute the start action with selected sources and parameters.

        Parameters
        ----------
        selected_sources
            Selected source names
        parameter_values
            Parameter values as a validated Pydantic model instance
        aux_source_names
            Selected auxiliary sources as a Pydantic model instance, or None if no
            aux sources are selected

        Returns
        -------
        bool
            True if successful, False otherwise
        """
