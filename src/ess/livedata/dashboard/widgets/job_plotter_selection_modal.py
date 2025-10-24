# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum, auto

import pandas as pd
import panel as pn

from ess.livedata.config.workflow_spec import JobNumber
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plotting_controller import PlottingController

from .configuration_widget import ConfigurationPanel
from .plot_configuration_adapter import PlotConfigurationAdapter


class WizardState(Enum):
    """State of the wizard workflow."""

    ACTIVE = auto()
    COMPLETED = auto()
    CANCELLED = auto()


class JobPlotterSelectionModal:
    """
    Three-step wizard modal for selecting job/output, plotter type, and configuration.

    The modal guides the user through:
    1. Job and output selection from available data
    2. Plotter type selection based on compatibility with selected job/output
    3. Plotter configuration (source selection and parameters)

    Parameters
    ----------
    job_service:
        Service for accessing job data and information
    plotting_controller:
        Controller for determining available plotters
    success_callback:
        Called with (plot, selected_sources) when user completes configuration
    cancel_callback:
        Called when modal is closed or cancelled
    """

    def __init__(
        self,
        job_service: JobService,
        plotting_controller: PlottingController,
        success_callback: Callable,
        cancel_callback: Callable[[], None],
    ) -> None:
        self._job_service = job_service
        self._plotting_controller = plotting_controller
        self._success_callback = success_callback
        self._cancel_callback = cancel_callback
        self._logger = logging.getLogger(__name__)

        # State tracking
        self._current_step = 1
        self._selected_job: JobNumber | None = None
        self._selected_output: str | None = None
        self._selected_plot: str | None = None
        self._config_panel: ConfigurationPanel | None = None
        self._state = WizardState.ACTIVE

        # UI components
        self._job_output_table = self._create_job_output_table()
        self._plotter_buttons_container = pn.Column(sizing_mode='stretch_width')
        self._config_panel_container = pn.Column(sizing_mode='stretch_width')

        self._back_button = pn.widgets.Button(
            name="Back",
            button_type="light",
            sizing_mode='fixed',
            width=100,
        )
        self._back_button.on_click(self._on_back_clicked)

        self._next_button = pn.widgets.Button(
            name="Next",
            button_type="primary",
            disabled=True,
            sizing_mode='fixed',
            width=120,
        )
        self._next_button.on_click(self._on_next_clicked)

        self._cancel_button = pn.widgets.Button(
            name="Cancel",
            button_type="light",
            sizing_mode='fixed',
            width=100,
        )
        self._cancel_button.on_click(self._on_cancel_clicked)

        # Content container
        self._content = pn.Column(sizing_mode='stretch_width')

        # Create modal
        self._modal = pn.Modal(
            self._content,
            name="Select Job and Plotter",
            margin=20,
            width=900,
            height=700,
        )

        # Watch for modal close events (X button or ESC key)
        self._modal.param.watch(self._on_modal_closed, 'open')

        # Set up watchers
        self._job_output_table.param.watch(
            self._on_job_output_selection_change, 'selection'
        )

        # Initialize with step 1
        self._update_content()
        self._update_job_output_table()

    def _create_job_output_table(self) -> pn.widgets.Tabulator:
        """Create job and output selection table with grouping."""
        return pn.widgets.Tabulator(
            name="Available Jobs and Outputs",
            pagination='remote',
            page_size=15,
            sizing_mode='stretch_width',
            selectable=1,
            disabled=True,
            height=400,
            groupby=['workflow_name', 'job_number'],
            configuration={
                'columns': [
                    {'title': 'Job Number', 'field': 'job_number', 'width': 100},
                    {'title': 'Workflow', 'field': 'workflow_name', 'width': 100},
                    {'title': 'Output Name', 'field': 'output_name', 'width': 200},
                    {'title': 'Source Names', 'field': 'source_names', 'width': 500},
                ],
            },
        )

    def _create_plotter_buttons(
        self, available_plots: dict[str, tuple[str, object]]
    ) -> list[pn.widgets.Button]:
        """Create buttons for each available plotter.

        Parameters
        ----------
        available_plots:
            Dictionary mapping plot names to (title, spec) tuples.

        Returns
        -------
        :
            List of buttons for selecting plotters.
        """
        buttons = []
        for plot_name, (title, _spec) in available_plots.items():
            button = pn.widgets.Button(
                name=title,
                button_type="primary",
                sizing_mode='stretch_width',
                min_width=200,
            )
            # Capture plot_name in closure
            button.on_click(lambda event, pn=plot_name: self._on_plotter_selected(pn))
            buttons.append(button)
        return buttons

    def _update_job_output_table(self) -> None:
        """Update the job and output table with current job data."""
        job_output_data = []
        for job_number, workflow_id in self._job_service.job_info.items():
            job_data = self._job_service.job_data.get(job_number, {})
            sources = list(job_data.keys())

            # Get output names from any source (they all have the same outputs per
            # backend guarantee)
            output_names = set()
            for source_data in job_data.values():
                if isinstance(source_data, dict):
                    output_names.update(source_data.keys())
                    break  # Since all sources have same outputs, we only check one

            # If no outputs found, create a row with empty output name
            if not output_names:
                job_output_data.append(
                    {
                        'output_name': '',
                        'source_names': ', '.join(sources),
                        'workflow_name': workflow_id.name,
                        'job_number': job_number.hex,
                    }
                )
            else:
                # Create one row per output name
                job_output_data.extend(
                    [
                        {
                            'output_name': output_name,
                            'source_names': ', '.join(sources),
                            'workflow_name': workflow_id.name,
                            'job_number': job_number.hex,
                        }
                        for output_name in sorted(output_names)
                    ]
                )

        if job_output_data:
            df = pd.DataFrame(job_output_data)
        else:
            df = pd.DataFrame(
                columns=['job_number', 'workflow_name', 'output_name', 'source_names']
            )
        self._job_output_table.value = df

    def _on_job_output_selection_change(self, event) -> None:
        """Handle job and output selection change."""
        selection = event.new
        if len(selection) != 1:
            self._selected_job = None
            self._selected_output = None
            self._next_button.disabled = True
            return

        # Get selected job number and output name from index
        selected_row = selection[0]
        job_number_str = self._job_output_table.value['job_number'].iloc[selected_row]
        output_name = self._job_output_table.value['output_name'].iloc[selected_row]

        self._selected_job = JobNumber(job_number_str)
        self._selected_output = output_name if output_name else None

        # Enable next button
        self._next_button.disabled = False

    def _on_plotter_selected(self, plot_name: str) -> None:
        """Handle plotter button click.

        Parameters
        ----------
        plot_name:
            Name of the selected plotter.
        """
        if self._selected_job is not None:
            self._selected_plot = plot_name
            self._current_step = 3
            self._update_content()

    def _update_content(self) -> None:
        """Update modal content based on current step."""
        if self._current_step == 1:
            self._show_step_1()
        elif self._current_step == 2:
            self._show_step_2()
        else:
            self._show_step_3()

    def _show_step_1(self) -> None:
        """Show step 1: job and output selection."""
        self._content.clear()
        self._content.extend(
            [
                pn.pane.HTML(
                    "<h3>Step 1: Select Job and Output</h3>"
                    "<p>Choose the job and output you want to visualize.</p>"
                ),
                self._job_output_table,
                pn.Row(
                    pn.Spacer(),
                    self._cancel_button,
                    self._next_button,
                    margin=(10, 0),
                ),
            ]
        )

    def _show_step_2(self) -> None:
        """Show step 2: plotter selection."""
        # Update plotter buttons with available plotters
        self._update_plotter_buttons()

        self._content.clear()
        self._content.extend(
            [
                pn.pane.HTML(
                    "<h3>Step 2: Select Plotter Type</h3>"
                    "<p>Choose the type of plot you want to create.</p>"
                ),
                self._plotter_buttons_container,
                pn.Row(
                    pn.Spacer(),
                    self._back_button,
                    self._cancel_button,
                    margin=(10, 0),
                ),
            ]
        )

    def _show_step_3(self) -> None:
        """Show step 3: plotter configuration."""
        # Create configuration panel if needed
        if self._config_panel is None and self._selected_job and self._selected_plot:
            # Get available sources for selected job
            job_data = self._job_service.job_data.get(self._selected_job, {})
            available_sources = list(job_data.keys())

            if not available_sources:
                self._show_error('No sources available for selected job')
                self._on_cancel_clicked(None)
                return

            # Get plot spec
            try:
                plot_spec = self._plotting_controller.get_spec(self._selected_plot)
            except Exception as e:
                self._show_error(f'Error getting plot spec: {e}')
                self._on_cancel_clicked(None)
                return

            # Create PlotConfigurationAdapter
            # The adapter's success_callback is called from start_action()
            # with (plot, sources)
            config_adapter = PlotConfigurationAdapter(
                job_number=self._selected_job,
                output_name=self._selected_output,
                plot_spec=plot_spec,
                available_sources=available_sources,
                plotting_controller=self._plotting_controller,
                success_callback=self._on_plot_config_complete,
            )

            # Create ConfigurationPanel without cancel button
            # The panel's success_callback is called after execute_action()
            # succeeds (no args)
            self._config_panel = ConfigurationPanel(
                config=config_adapter,
                start_button_text="Create Plot",
                show_cancel_button=False,
                success_callback=self._on_panel_action_success,
            )

        self._content.clear()
        if self._config_panel:
            self._content.extend(
                [
                    pn.pane.HTML("<h3>Step 3: Configure Plot</h3>"),
                    self._config_panel.panel,
                    pn.Row(
                        pn.Spacer(),
                        self._back_button,
                        self._cancel_button,
                        margin=(10, 0),
                    ),
                ]
            )

    def _update_plotter_buttons(self) -> None:
        """Update plotter buttons based on job and output selection."""
        self._plotter_buttons_container.clear()

        if self._selected_job is None:
            self._plotter_buttons_container.append(
                pn.pane.Markdown("*No job selected*")
            )
            return

        try:
            available_plots = self._plotting_controller.get_available_plotters(
                self._selected_job, self._selected_output
            )
            if available_plots:
                # Create dictionary mapping plot names to (title, spec) tuples
                plot_data = {
                    name: (spec.title, spec) for name, spec in available_plots.items()
                }
                buttons = self._create_plotter_buttons(plot_data)
                self._plotter_buttons_container.extend(buttons)
            else:
                self._plotter_buttons_container.append(
                    pn.pane.Markdown("*No plotters available for this selection*")
                )
        except Exception as e:
            self._logger.exception(
                "Error loading plotters for job %s", self._selected_job
            )
            self._plotter_buttons_container.append(
                pn.pane.Markdown(f"*Error loading plotters: {e}*")
            )

    def _on_next_clicked(self, event) -> None:
        """Handle next button click."""
        self._current_step = 2
        self._update_content()

    def _on_back_clicked(self, event) -> None:
        """Handle back button click."""
        if self._current_step > 1:
            self._current_step -= 1
            # Clear config panel when going back from step 3
            if self._current_step == 2:
                self._config_panel = None
            self._update_content()

    def _on_cancel_clicked(self, event) -> None:
        """Handle cancel button click."""
        self._state = WizardState.CANCELLED
        self._modal.open = False
        self._cancel_callback()

    def _on_plot_config_complete(self, plot, selected_sources: list[str]) -> None:
        """
        Handle plot creation completion from PlotConfigurationAdapter.

        This is called by the adapter's start_action() with the created plot.
        """
        # Store for potential use, then trigger parent callback
        self._success_callback(plot, selected_sources)

    def _on_panel_action_success(self) -> None:
        """
        Handle successful action from ConfigurationPanel.

        This is called after execute_action() completes successfully.
        Closes the modal and marks workflow as completed.
        """
        self._state = WizardState.COMPLETED
        self._modal.open = False

    def _show_error(self, message: str) -> None:
        """Display an error notification."""
        if pn.state.notifications is not None:
            pn.state.notifications.error(message, duration=3000)

    def _on_modal_closed(self, event) -> None:
        """Handle modal being closed via X button or ESC key."""
        if not event.new:  # Modal was closed
            # Only call cancel callback if workflow wasn't completed
            if self._state == WizardState.ACTIVE:
                self._state = WizardState.CANCELLED
                self._cancel_callback()

            # Remove modal from its parent container after a short delay
            # to allow the close animation to complete.
            # This uses Panel's private API as there's no public cleanup method.
            def cleanup():
                try:
                    if hasattr(self._modal, '_parent') and self._modal._parent:
                        self._modal._parent.remove(self._modal)
                except Exception as e:
                    # This is expected to fail sometimes due to Panel's lifecycle
                    self._logger.debug("Modal cleanup warning (expected): %s", e)

            pn.state.add_periodic_callback(cleanup, period=100, count=1)

    def show(self) -> None:
        """Show the modal dialog."""
        # Reset state
        self._current_step = 1
        self._selected_job = None
        self._selected_output = None
        self._selected_plot = None
        self._config_panel = None
        self._next_button.disabled = True
        self._state = WizardState.ACTIVE

        # Refresh data and show
        self._update_job_output_table()
        self._update_content()
        self._modal.open = True

    @property
    def modal(self) -> pn.Modal:
        """Get the modal widget."""
        return self._modal
