# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pydantic

from ess.livedata.config.workflow_spec import PersistentWorkflowConfig, WorkflowSpec

from .configuration_adapter import ConfigurationAdapter


class WorkflowConfigurationAdapter(ConfigurationAdapter[pydantic.BaseModel]):
    """Adapter for workflow configuration using WorkflowSpec and persistent config."""

    def __init__(
        self,
        spec: WorkflowSpec,
        persistent_config: PersistentWorkflowConfig | None,
        start_callback: Callable[
            [list[str], pydantic.BaseModel, pydantic.BaseModel | None], bool
        ],
    ) -> None:
        """Initialize adapter with workflow spec, config, and start callback."""
        self._spec = spec
        self._persistent_config = persistent_config
        self._start_callback = start_callback
        self._cached_aux_sources: pydantic.BaseModel | None = None

    @property
    def title(self) -> str:
        """Get workflow title."""
        return self._spec.title

    @property
    def description(self) -> str:
        """Get workflow description."""
        return self._spec.description

    @property
    def aux_sources(self) -> type[pydantic.BaseModel] | None:
        """Get auxiliary sources Pydantic model."""
        return self._spec.aux_sources

    @property
    def initial_aux_source_names(self) -> dict[str, str]:
        """Get initial auxiliary source names from persistent config."""
        if not self._persistent_config:
            return {}
        return self._persistent_config.config.aux_source_names

    def model_class(self) -> type[pydantic.BaseModel] | None:
        """Get workflow parameters model class."""
        return self._spec.params

    @property
    def source_names(self) -> list[str]:
        """Get available source names."""
        return self._spec.source_names

    @property
    def initial_source_names(self) -> list[str]:
        """Get initial source names."""
        return self._persistent_config.source_names if self._persistent_config else []

    @property
    def initial_parameter_values(self) -> dict[str, Any]:
        """Get initial parameter values."""
        if not self._persistent_config:
            return {}
        return self._persistent_config.config.params

    def start_action(
        self,
        selected_sources: list[str],
        parameter_values: pydantic.BaseModel,
    ) -> bool:
        """Start the workflow with given sources and parameters."""
        return self._start_callback(
            selected_sources, parameter_values, self._cached_aux_sources
        )
