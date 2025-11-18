# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Configuration adapter for spec-based plot configuration (without job data)."""

from __future__ import annotations

from typing import Any

import pydantic

from ess.livedata.config.workflow_spec import WorkflowSpec
from ess.livedata.dashboard.configuration_adapter import ConfigurationAdapter
from ess.livedata.dashboard.plotting import PlotterSpec


class SpecBasedPlotConfigurationAdapter(ConfigurationAdapter):
    """
    Adapter for spec-based plot configuration modal.

    Unlike PlotConfigurationAdapter, this adapter works with workflow specs
    rather than job data, enabling plot configuration before data exists.
    It collects the configuration but doesn't create the plot.
    """

    def __init__(
        self,
        workflow_spec: WorkflowSpec,
        plot_spec: PlotterSpec,
        success_callback,
    ):
        """
        Initialize spec-based plot configuration adapter.

        Parameters
        ----------
        workflow_spec:
            Workflow specification containing source names.
        plot_spec:
            Plotter specification containing parameters model.
        success_callback:
            Called with (selected_sources, params) when configuration is complete.
        """
        # No persistent config for spec-based mode (no job yet)
        super().__init__(config_state=None)
        self._workflow_spec = workflow_spec
        self._plot_spec = plot_spec
        self._success_callback = success_callback

    @property
    def title(self) -> str:
        """Title for the configuration panel."""
        return f"Configure {self._plot_spec.title}"

    @property
    def description(self) -> str:
        """Description for the configuration panel."""
        return self._plot_spec.description

    def model_class(self) -> type[pydantic.BaseModel] | None:
        """Get the pydantic model class for plotter parameters."""
        return self._plot_spec.params

    @property
    def source_names(self) -> list[str]:
        """Get available source names from workflow spec."""
        return self._workflow_spec.source_names

    def start_action(
        self,
        selected_sources: list[str],
        parameter_values: Any,
    ) -> None:
        """
        Collect configuration and call success callback.

        Unlike PlotConfigurationAdapter, this doesn't create a plot - it just
        collects the configuration for later use when data becomes available.

        Parameters
        ----------
        selected_sources:
            List of selected source names.
        parameter_values:
            Validated plotter parameters (Pydantic model or dict).
        """
        self._success_callback(selected_sources, parameter_values)
