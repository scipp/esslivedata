# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Any

import pydantic

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.workflow_controller import WorkflowController

from .configuration_adapter import ConfigurationAdapter


class WorkflowConfigurationAdapter(ConfigurationAdapter):
    """Adapter for workflow configuration using WorkflowController."""

    def __init__(self, controller: WorkflowController, workflow_id: WorkflowId) -> None:
        """Initialize adapter with workflow controller and ID."""
        self._controller = controller
        self._workflow_id = workflow_id
        spec = controller.get_workflow_spec(workflow_id)
        if spec is None:
            raise ValueError(f'Workflow {workflow_id} not found')
        self._spec = spec

    @property
    def title(self) -> str:
        """Get workflow title."""
        return self._spec.title

    @property
    def description(self) -> str:
        """Get workflow description."""
        return self._spec.description

    @property
    def aux_source_names(self) -> dict[str, list[str]]:
        """Get auxiliary source names with unique options."""
        return {key: [key] for key in self._spec.aux_source_names}

    def model_class(
        self, aux_source_names: dict[str, str]
    ) -> type[pydantic.BaseModel] | None:
        """Get workflow parameters model class."""
        return self._spec.params

    @property
    def source_names(self) -> list[str]:
        """Get available source names."""
        return self._spec.source_names

    @property
    def initial_source_names(self) -> list[str]:
        """Get initial source names."""
        persistent_config = self._controller.get_workflow_config(self._workflow_id)
        return persistent_config.source_names if persistent_config else []

    @property
    def initial_parameter_values(self) -> dict[str, Any]:
        """Get initial parameter values."""
        persistent_config = self._controller.get_workflow_config(self._workflow_id)
        if not persistent_config:
            return {}
        return persistent_config.config.params

    def start_action(self, selected_sources: list[str], parameter_values: Any) -> bool:
        """Start the workflow with given sources and parameters."""
        return self._controller.start_workflow(
            self._workflow_id, selected_sources, parameter_values
        )
