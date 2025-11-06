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
        success_callback,
    ):
        config_state = plotting_controller.get_persistent_plotter_config(
            job_number=job_number,
            output_name=output_name,
            plot_name=plot_spec.name,
        )
        super().__init__(config_state=config_state)
        self._job_number = job_number
        self._output_name = output_name
        self._plot_spec = plot_spec
        self._available_sources = available_sources
        self._plotting_controller = plotting_controller
        self._success_callback = success_callback

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

    def start_action(
        self,
        selected_sources: list[str],
        parameter_values: Any,
    ) -> None:
        """Create the plot and call the success callback with the result."""
        plot = self._plotting_controller.create_plot(
            job_number=self._job_number,
            source_names=selected_sources,
            output_name=self._output_name,
            plot_name=self._plot_spec.name,
            params=parameter_values,
        )
        self._success_callback(plot, selected_sources)
