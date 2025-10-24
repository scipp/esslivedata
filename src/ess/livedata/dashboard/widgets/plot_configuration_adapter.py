# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Any

import pydantic

from ess.livedata.config.workflow_spec import JobNumber
from ess.livedata.dashboard.configuration_adapter import ConfigurationAdapter
from ess.livedata.dashboard.plotting import PlotterSpec
from ess.livedata.dashboard.plotting_controller import PlottingController


class PlotConfigurationAdapter(ConfigurationAdapter):
    """Adapter for plot configuration modal."""

    def __init__(
        self,
        job_number: JobNumber,
        output_name: str | None,
        plot_spec: PlotterSpec,
        available_sources: list[str],
        plotting_controller: PlottingController,
    ):
        self._job_number = job_number
        self._output_name = output_name
        self._plot_spec = plot_spec
        self._available_sources = available_sources
        self._plotting_controller = plotting_controller

        self._persisted_config = (
            self._plotting_controller.get_persistent_plotter_config(
                job_number=self._job_number,
                output_name=self._output_name,
                plot_name=self._plot_spec.name,
            )
        )

    @property
    def title(self) -> str:
        return f"Configure {self._plot_spec.title}"

    @property
    def description(self) -> str:
        return self._plot_spec.description

    def model_class(self) -> type[pydantic.BaseModel] | None:
        return self._plot_spec.params

    @property
    def source_names(self) -> list[str]:
        return self._available_sources

    @property
    def initial_source_names(self) -> list[str]:
        if self._persisted_config is not None:
            # Filter persisted source names to only include those still available
            persisted_sources = [
                name
                for name in self._persisted_config.source_names
                if name in self._available_sources
            ]
            return persisted_sources if persisted_sources else self._available_sources
        return self._available_sources

    @property
    def initial_parameter_values(self) -> dict[str, Any]:
        if self._persisted_config is not None:
            return self._persisted_config.config.params
        return {}

    def start_action(
        self,
        selected_sources: list[str],
        parameter_values: Any,
    ) -> tuple[Any, list[str]]:
        """
        Create and return the plot.

        Returns
        -------
        :
            Tuple of (plot, selected_sources)
        """
        plot = self._plotting_controller.create_plot(
            job_number=self._job_number,
            source_names=selected_sources,
            output_name=self._output_name,
            plot_name=self._plot_spec.name,
            params=parameter_values,
        )
        return plot, selected_sources
