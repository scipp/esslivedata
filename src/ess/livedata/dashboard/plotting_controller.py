# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import TypeVar

import holoviews as hv
import pydantic

import ess.livedata.config.keys as keys
from ess.livedata.config.workflow_spec import (
    JobId,
    JobNumber,
    PersistentWorkflowConfig,
    PersistentWorkflowConfigs,
    ResultKey,
    WorkflowConfig,
    WorkflowId,
)

from .config_service import ConfigService
from .job_service import JobService
from .plotting import PlotterSpec, plotter_registry
from .stream_manager import StreamManager

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


class PlottingController:
    _plotter_config_key = keys.PERSISTENT_PLOTTING_CONFIGS.create_key()

    def __init__(
        self,
        job_service: JobService,
        stream_manager: StreamManager,
        config_service: ConfigService | None = None,
        logger: logging.Logger | None = None,
        max_persistent_configs: int = 100,
        cleanup_fraction: float = 0.2,
    ) -> None:
        self._job_service = job_service
        self._stream_manager = stream_manager
        self._config_service = config_service
        self._logger = logger or logging.getLogger(__name__)
        self._max_persistent_configs = max_persistent_configs
        self._cleanup_fraction = cleanup_fraction

    def get_available_plotters(
        self, job_number: JobNumber, output_name: str | None
    ) -> dict[str, PlotterSpec]:
        """Get all available plotters for a given job and output."""
        job_data = self._job_service.job_data[job_number]
        data = {k: v[output_name] for k, v in job_data.items()}
        return plotter_registry.get_compatible_plotters(data)

    def get_spec(self, plot_name: str) -> PlotterSpec:
        """Get the parameter model for a given plotter name."""
        return plotter_registry.get_spec(plot_name)

    def get_result_key(
        self, job_number: JobNumber, source_name: str, output_name: str | None
    ) -> ResultKey:
        """Get the ResultKey for a given job number and source name."""
        workflow_id = self._job_service.job_info[job_number]
        return ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(job_number=job_number, source_name=source_name),
            output_name=output_name,
        )

    def get_persistent_plotter_config(
        self, job_number: JobNumber, output_name: str | None, plot_name: str
    ) -> PersistentWorkflowConfig | None:
        """Get persistent plotter configuration for a given job, output, and plot."""
        if self._config_service is None:
            return None

        workflow_id = self._job_service.job_info[job_number]
        all_configs = self._config_service.get_config(
            self._plotter_config_key, PersistentWorkflowConfigs()
        )
        plotter_id = self._create_plotter_id(workflow_id, output_name, plot_name)
        return all_configs.configs.get(plotter_id)

    def _create_plotter_id(
        self, workflow_id: WorkflowId, output_name: str | None, plot_name: str
    ) -> WorkflowId:
        """Create a plotting-specific WorkflowId based on the data workflow."""
        suffix_parts = [plot_name]
        if output_name is not None:
            suffix_parts.insert(0, output_name)
        suffix = "_".join(suffix_parts)

        return WorkflowId(
            instrument=workflow_id.instrument,
            namespace="plotting",
            name=f"{workflow_id.name}_{suffix}",
            version=workflow_id.version,
        )

    def _cleanup_old_configs(self, configs: PersistentWorkflowConfigs) -> None:
        """
        Remove oldest configs when limit is exceeded.

        In the case of workflows we simply remove workflows that do not exist anymore.
        This approach would be more difficult here, since for every workflow there can
        be multiple outputs, and for every output multiple applicable plotters, each of
        which should have its config saved. Hence we simply remove the oldest ones.
        """
        if len(configs.configs) <= self._max_persistent_configs:
            return

        num_to_remove = int(len(configs.configs) * self._cleanup_fraction)
        if num_to_remove == 0:
            num_to_remove = 1

        # Remove oldest configs (dict maintains insertion order, and this should work
        # even across serialized/deserialized states)
        oldest_keys = list(configs.configs.keys())[:num_to_remove]
        for key in oldest_keys:
            del configs.configs[key]

        self._logger.info(
            'Cleaned up %d old plotting configs, %d remaining',
            num_to_remove,
            len(configs.configs),
        )

    def _save_plotting_config(
        self,
        workflow_id: WorkflowId,
        source_names: list[str],
        output_name: str | None,
        plot_name: str,
        params: pydantic.BaseModel,
    ) -> None:
        """Save plotting configuration for persistence."""
        if self._config_service is None:
            return

        plotter_id = self._create_plotter_id(workflow_id, output_name, plot_name)
        plot_config = WorkflowConfig(identifier=plotter_id, params=params.model_dump())

        current_configs = self._config_service.get_config(
            self._plotter_config_key, PersistentWorkflowConfigs()
        )
        current_configs.configs[plotter_id] = PersistentWorkflowConfig(
            source_names=source_names, config=plot_config
        )

        self._cleanup_old_configs(current_configs)
        self._config_service.update_config(self._plotter_config_key, current_configs)

    def create_plot(
        self,
        job_number: JobNumber,
        source_names: list[str],
        output_name: str | None,
        plot_name: str,
        params: pydantic.BaseModel,
    ) -> hv.DynamicMap:
        self._save_plotting_config(
            workflow_id=self._job_service.job_info[job_number],
            source_names=source_names,
            output_name=output_name,
            plot_name=plot_name,
            params=params,
        )
        items = {
            self.get_result_key(
                job_number=job_number, source_name=source_name, output_name=output_name
            ): self._job_service.job_data[job_number][source_name][output_name]
            for source_name in source_names
        }
        pipe = self._stream_manager.make_merging_stream(items)
        plotter = plotter_registry.create_plotter(plot_name, params=params)
        return hv.DynamicMap(plotter, streams=[pipe], cache_size=1).opts(
            shared_axes=False
        )
