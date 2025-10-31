# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable

import pydantic

from ess.livedata.config.workflow_spec import WorkflowSpec

from .configuration_adapter import ConfigurationAdapter, ConfigurationState


class WorkflowConfigurationAdapter(ConfigurationAdapter[pydantic.BaseModel]):
    """Adapter for workflow configuration using WorkflowSpec and persistent config."""

    def __init__(
        self,
        spec: WorkflowSpec,
        config_state: ConfigurationState | None,
        start_callback: Callable[
            [list[str], pydantic.BaseModel, pydantic.BaseModel | None], None
        ],
    ) -> None:
        """Initialize adapter with workflow spec, config, and start callback."""
        super().__init__(config_state=config_state)
        self._spec = spec
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

    def model_class(self) -> type[pydantic.BaseModel] | None:
        """Get workflow parameters model class."""
        return self._spec.params

    @property
    def source_names(self) -> list[str]:
        """Get available source names."""
        return self._spec.source_names

    def start_action(
        self,
        selected_sources: list[str],
        parameter_values: pydantic.BaseModel,
    ) -> None:
        """Start the workflow with given sources and parameters."""
        self._start_callback(
            selected_sources, parameter_values, self._cached_aux_sources
        )
