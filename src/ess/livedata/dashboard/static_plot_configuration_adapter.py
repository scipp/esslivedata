# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Configuration adapter for static plot overlays."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pydantic

from ess.livedata.dashboard.configuration_adapter import (
    ConfigurationAdapter,
    ConfigurationState,
)
from ess.livedata.dashboard.plotting import PlotterSpec


class StaticPlotConfigurationAdapter(ConfigurationAdapter):
    """
    Adapter for static plot overlay configuration.

    Static overlays don't have data sources, so this adapter only handles
    parameter configuration without source selection.
    """

    def __init__(
        self,
        plot_spec: PlotterSpec,
        success_callback: Callable[[pydantic.BaseModel | dict[str, Any]], None],
        config_state: ConfigurationState | None = None,
    ):
        """
        Initialize static plot configuration adapter.

        Parameters
        ----------
        plot_spec:
            Plotter specification containing parameters model.
        success_callback:
            Called with (params) when configuration is complete.
            Unlike regular adapters, static overlays don't have source selection.
        config_state:
            Optional reference configuration state to restore.
        """
        super().__init__(config_state=config_state, initial_source_names=[])
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
        """Get available source names (empty for static overlays)."""
        return []

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
            Ignored for static overlays (always empty).
        parameter_values:
            Validated plotter parameters (Pydantic model or dict).
        """
        # Static overlays only pass params to callback
        self._success_callback(parameter_values)
