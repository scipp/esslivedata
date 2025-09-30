# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Any

import pydantic

from ess.livedata.dashboard.workflow_controller import BoundWorkflowController

from .configuration_widget import ConfigurationAdapter


class WorkflowConfigurationAdapter(ConfigurationAdapter):
    """Adapter for workflow configuration using BoundWorkflowController."""

    def __init__(self, controller: BoundWorkflowController) -> None:
        """Initialize adapter with workflow controller."""
        self._controller = controller

    @property
    def title(self) -> str:
        """Get workflow title."""
        return self._controller.spec.title

    @property
    def description(self) -> str:
        """Get workflow description."""
        return self._controller.spec.description

    @property
    def aux_source_names(self) -> dict[str, list[str]]:
        """Get auxiliary source names with unique options."""
        return {key: [key] for key in self._controller.spec.aux_source_names}

    def model_class(
        self, aux_source_names: dict[str, str]
    ) -> type[pydantic.BaseModel] | None:
        """Get workflow parameters model class."""
        return self._controller.params_model_class

    @property
    def source_names(self) -> list[str]:
        """Get available source names."""
        return self._controller.spec.source_names

    @property
    def initial_source_names(self) -> list[str]:
        """Get initial source names."""
        persistent_config = self._controller.get_persistent_config()
        return persistent_config.source_names if persistent_config else []

    @property
    def initial_parameter_values(self) -> dict[str, Any]:
        """Get initial parameter values."""
        persistent_config = self._controller.get_persistent_config()
        if not persistent_config:
            return {}
        return persistent_config.config.params

    def start_action(self, selected_sources: list[str], parameter_values: Any) -> bool:
        """Start the workflow with given sources and parameters."""
        return self._controller.start_workflow(selected_sources, parameter_values)
