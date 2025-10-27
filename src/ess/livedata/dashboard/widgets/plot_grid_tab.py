# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import holoviews as hv
import panel as pn

from ess.livedata.dashboard.job_controller import JobController
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plotting_controller import PlottingController

from .job_plotter_selection_modal import JobPlotterSelectionModal
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

        # Modal container for lifecycle management.
        # Using pn.Row with height=0 ensures the modal is part of the component tree
        # (required for rendering) but doesn't compete with the grid for vertical space.
        # The modal itself renders as an overlay when opened.
        self._modal_container = pn.Row(height=0, sizing_mode='stretch_width')

        # State for tracking current workflow
        self._current_modal: JobPlotterSelectionModal | None = None

        # Create main widget - grid with zero-height modal container
        self._widget = pn.Column(
            self._plot_grid.panel,
            self._modal_container,
            sizing_mode='stretch_both',
        )

    def _on_plot_requested(self) -> None:
        """Handle plot request from PlotGrid (user completed region selection)."""
        # Create and show JobPlotterSelectionModal (now includes all 3 steps)
        self._current_modal = JobPlotterSelectionModal(
            job_service=self._job_service,
            plotting_controller=self._plotting_controller,
            success_callback=self._on_plot_created,
            cancel_callback=self._on_modal_cancelled,
        )

        # Add modal to zero-height container so it renders but doesn't affect layout
        self._modal_container.clear()
        self._modal_container.append(self._current_modal.modal)
        self._current_modal.show()

    def _on_plot_created(
        self, plot: hv.DynamicMap, selected_sources: list[str]
    ) -> None:
        """Handle successful plot creation from configuration modal."""
        # Clear references BEFORE inserting plot to prevent cancellation on modal close
        self._current_modal = None

        # Insert plot into grid using deferred insertion
        self._plot_grid.insert_plot_deferred(plot)

    def _on_modal_cancelled(self) -> None:
        """Handle modal cancellation."""
        # Cancel pending selection in PlotGrid
        self._plot_grid.cancel_pending_selection()

        # Clear references
        self._current_modal = None

    @property
    def widget(self) -> pn.Column:
        """Get the Panel widget."""
        return self._widget
