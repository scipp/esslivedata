# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import holoviews as hv
import panel as pn

from ess.livedata.config.workflow_spec import JobNumber
from ess.livedata.dashboard.job_controller import JobController
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plotting_controller import PlottingController

from .configuration_widget import ConfigurationModal
from .job_plotter_selection_modal import JobPlotterSelectionModal
from .plot_configuration_adapter import PlotConfigurationAdapter
from .plot_grid import PlotGrid


class PlotGridTab:
    """Tab widget that orchestrates PlotGrid with modal workflow for plot creation."""

    def __init__(
        self,
        *,
        job_service: JobService,
        job_controller: JobController,
        plotting_controller: PlottingController,
    ) -> None:
        """
        Initialize PlotGridTab.

        Parameters
        ----------
        job_service:
            Service for accessing job data
        job_controller:
            Controller for job operations
        plotting_controller:
            Controller for creating plotters
        """
        self._job_service = job_service
        self._job_controller = job_controller
        self._plotting_controller = plotting_controller

        # Create PlotGrid (3x3 fixed)
        self._plot_grid = PlotGrid(
            nrows=3, ncols=3, plot_request_callback=self._on_plot_requested
        )

        # Modal container for lifecycle management
        self._modal_container = pn.Column()

        # State for tracking current workflow
        self._current_job_plotter_modal: JobPlotterSelectionModal | None = None
        self._current_config_modal: ConfigurationModal | None = None

        # Create main widget
        self._widget = pn.Column(
            self._plot_grid.panel, self._modal_container, sizing_mode='stretch_both'
        )

    def _on_plot_requested(self) -> None:
        """Handle plot request from PlotGrid (user completed region selection)."""
        # Create and show JobPlotterSelectionModal
        self._current_job_plotter_modal = JobPlotterSelectionModal(
            job_service=self._job_service,
            plotting_controller=self._plotting_controller,
            success_callback=self._on_job_plotter_selected,
            cancel_callback=self._on_modal_cancelled,
        )

        # Clear modal container and add new modal
        self._modal_container.clear()
        self._modal_container.append(self._current_job_plotter_modal.modal)
        self._current_job_plotter_modal.show()

    def _on_job_plotter_selected(
        self, job_number: JobNumber, output_name: str | None, plot_name: str
    ) -> None:
        """Handle successful job/plotter selection from first modal."""
        # Get available sources for selected job
        job_data = self._job_service.job_data.get(job_number, {})
        available_sources = list(job_data.keys())

        if not available_sources:
            self._show_error('No sources available for selected job')
            self._on_modal_cancelled()
            return

        # Get plot spec
        try:
            plot_spec = self._plotting_controller.get_spec(plot_name)
        except Exception as e:
            self._show_error(f'Error getting plot spec: {e}')
            self._on_modal_cancelled()
            return

        # Create PlotConfigurationAdapter
        config = PlotConfigurationAdapter(
            job_number=job_number,
            output_name=output_name,
            plot_spec=plot_spec,
            available_sources=available_sources,
            plotting_controller=self._plotting_controller,
            success_callback=self._on_plot_created,
        )

        # Create and show ConfigurationModal
        self._current_config_modal = ConfigurationModal(
            config=config, start_button_text="Create Plot"
        )

        # Clear modal container and add new modal
        self._modal_container.clear()
        self._modal_container.append(self._current_config_modal.modal)

        # Watch for modal close to handle cancellation
        self._current_config_modal.modal.param.watch(
            self._on_config_modal_closed, 'open'
        )

        self._current_config_modal.show()

    def _on_plot_created(
        self, plot: hv.DynamicMap, selected_sources: list[str]
    ) -> None:
        """Handle successful plot creation from configuration modal."""
        # Insert plot into grid using deferred insertion
        self._plot_grid.insert_plot_deferred(plot)

        # Clear references
        self._current_job_plotter_modal = None
        self._current_config_modal = None

    def _on_modal_cancelled(self) -> None:
        """Handle modal cancellation (from JobPlotterSelectionModal)."""
        # Cancel pending selection in PlotGrid
        self._plot_grid.cancel_pending_selection()

        # Clear references
        self._current_job_plotter_modal = None
        self._current_config_modal = None

    def _on_config_modal_closed(self, event) -> None:
        """Handle ConfigurationModal being closed via X button or ESC."""
        if not event.new:  # Modal was closed
            # Only cancel if the plot wasn't already created
            if self._current_config_modal is not None:
                self._on_modal_cancelled()

    def _show_error(self, message: str) -> None:
        """Display an error notification."""
        if pn.state.notifications is not None:
            pn.state.notifications.error(message, duration=3000)

    def refresh(self) -> None:
        """Refresh the widget with current job data."""
        # Refresh job plotter modal if it's open
        if self._current_job_plotter_modal is not None:
            self._current_job_plotter_modal.refresh()

    @property
    def widget(self) -> pn.Column:
        """Get the Panel widget."""
        return self._widget
