# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

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
    def aux_source_names(self) -> dict[str, list[str]]:
        """Available auxiliary source names grouped by category."""
        return {}

    @abstractmethod
    def model_class(self, aux_source_names: dict[str, str]) -> type[Model] | None:
        """
        Pydantic model class for parameters.

        Parameters
        ----------
        aux_source_names
            Selected auxiliary source names by category
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
        self, selected_sources: list[str], parameter_values: Model
    ) -> bool:
        """
        Execute the start action with selected sources and parameters.

        Returns
        -------
        bool
            True if successful, False otherwise
        """
