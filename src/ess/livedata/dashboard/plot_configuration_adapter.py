# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Configuration adapter for plot configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pydantic

from ess.livedata.dashboard.configuration_adapter import (
    ConfigurationAdapter,
    ConfigurationState,
)
from ess.livedata.dashboard.plotting import PlotterSpec

if TYPE_CHECKING:
    from ess.livedata.config import Instrument


class PlotConfigurationAdapter(ConfigurationAdapter):
    """
    Adapter for plot configuration modal.

    This adapter works with both workflow specs and job data, collecting
    the configuration and passing it to a success callback. The callback
    determines whether to create a plot immediately or save the configuration
    for later use.
    """

    def __init__(
        self,
        plot_spec: PlotterSpec,
        source_names: list[str],
        success_callback,
        config_state: ConfigurationState | None = None,
        initial_source_names: list[str] | None = None,
        instrument_config: Instrument | None = None,
    ):
        """
        Initialize plot configuration adapter.

        Parameters
        ----------
        plot_spec:
            Plotter specification containing parameters model.
        source_names:
            Available source names for selection.
        success_callback:
            Called with (selected_sources, params) when configuration is complete.
        config_state:
            Optional reference configuration state (from a single source) to restore.
        initial_source_names:
            Source names to pre-select in the UI. None to select all available.
        instrument_config:
            Optional instrument configuration for source metadata lookup.
        """
        super().__init__(
            config_state=config_state, initial_source_names=initial_source_names
        )
        self._plot_spec = plot_spec
        self._source_names = source_names
        self._success_callback = success_callback
        self._instrument_config = instrument_config

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
        """Get available source names."""
        return self._source_names

    def get_source_title(self, source_name: str) -> str:
        """Get display title for a source name."""
        if self._instrument_config is not None:
            return self._instrument_config.get_source_title(source_name)
        return source_name

    def start_action(
        self,
        selected_sources: list[str],
        parameter_values: Any,
    ) -> None:
        """
        Collect configuration and call success callback.

        Parameters
        ----------
        selected_sources:
            List of selected source names.
        parameter_values:
            Validated plotter parameters (Pydantic model or dict).
        """
        self._success_callback(selected_sources, parameter_values)
