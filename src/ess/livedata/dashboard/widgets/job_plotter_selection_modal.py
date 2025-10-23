# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import panel as pn

from ess.livedata.config.workflow_spec import JobNumber
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plotting_controller import PlottingController


class JobPlotterSelectionModal:
    """
    Two-step wizard modal for selecting job/output and plotter type.

    The modal guides the user through:
    1. Job and output selection from available data
    2. Plotter type selection based on compatibility with selected job/output

    Parameters
    ----------
    job_service:
        Service for accessing job data and information
    plotting_controller:
        Controller for determining available plotters
    success_callback:
        Called with (job_number, output_name, plot_name) when user completes selection
    cancel_callback:
        Called when modal is closed or cancelled
    """

    def __init__(
        self,
        job_service: JobService,
        plotting_controller: PlottingController,
        success_callback: Callable[[JobNumber, str | None, str], None],
        cancel_callback: Callable[[], None],
    ) -> None:
        self._job_service = job_service
        self._plotting_controller = plotting_controller
        self._success_callback = success_callback
        self._cancel_callback = cancel_callback

        # State tracking
        self._current_step = 1
        self._selected_job: JobNumber | None = None
        self._selected_output: str | None = None
        self._selected_plot: str | None = None
        self._success_callback_invoked = False

        # UI components
        self._job_output_table = self._create_job_output_table()
        self._plot_selector = self._create_plot_selector()
        self._next_button = pn.widgets.Button(
            name="Next",
            button_type="primary",
            disabled=True,
            sizing_mode='fixed',
            width=120,
        )
        self._next_button.on_click(self._on_next_clicked)

        self._configure_button = pn.widgets.Button(
            name="Configure Plot",
            button_type="primary",
            disabled=True,
            sizing_mode='fixed',
            width=150,
        )
        self._configure_button.on_click(self._on_configure_clicked)

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
        self._plot_selector.param.watch(self._on_plot_selection_change, 'value')

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

    def _create_plot_selector(self) -> pn.widgets.Select:
        """Create plot type selection widget."""
        return pn.widgets.Select(
            name="Plot Type",
            options=[],
            value=None,
            sizing_mode='stretch_width',
            disabled=True,
        )

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

    def _on_plot_selection_change(self, event) -> None:
        """Handle plot selection change."""
        self._selected_plot = event.new
        self._configure_button.disabled = self._selected_plot is None

    def _update_content(self) -> None:
        """Update modal content based on current step."""
        if self._current_step == 1:
            self._show_step_1()
        else:
            self._show_step_2()

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
        # Update plot selector with available plotters
        self._update_plot_selector()

        self._content.clear()
        self._content.extend(
            [
                pn.pane.HTML(
                    "<h3>Step 2: Select Plotter Type</h3>"
                    "<p>Choose the type of plot to create.</p>"
                ),
                self._plot_selector,
                pn.Row(
                    pn.Spacer(),
                    self._cancel_button,
                    self._configure_button,
                    margin=(10, 0),
                ),
            ]
        )

    def _update_plot_selector(self) -> None:
        """Update plot selector based on job and output selection."""
        if self._selected_job is None:
            self._plot_selector.options = []
            self._plot_selector.value = None
            self._plot_selector.disabled = True
            self._configure_button.disabled = True
            return

        try:
            available_plots = self._plotting_controller.get_available_plotters(
                self._selected_job, self._selected_output
            )
            if available_plots:
                # Create options with plot class names
                options = {spec.title: name for name, spec in available_plots.items()}
                self._plot_selector.options = options
                self._plot_selector.value = next(iter(options)) if options else None
                self._plot_selector.disabled = False
                self._configure_button.disabled = False
            else:
                self._plot_selector.options = []
                self._plot_selector.value = None
                self._plot_selector.disabled = True
                self._configure_button.disabled = True
        except Exception:
            self._plot_selector.options = []
            self._plot_selector.value = None
            self._plot_selector.disabled = True
            self._configure_button.disabled = True

    def _on_next_clicked(self, event) -> None:
        """Handle next button click."""
        self._current_step = 2
        self._update_content()

    def _on_configure_clicked(self, event) -> None:
        """Handle configure button click."""
        if self._selected_job is not None and self._selected_plot is not None:
            # Mark success callback as invoked BEFORE closing modal
            # to prevent _on_modal_closed from calling cancel callback
            self._success_callback_invoked = True
            self._modal.open = False
            self._success_callback(
                self._selected_job, self._selected_output, self._selected_plot
            )

    def _on_cancel_clicked(self, event) -> None:
        """Handle cancel button click."""
        self._modal.open = False
        self._cancel_callback()

    def _on_modal_closed(self, event) -> None:
        """Handle modal being closed via X button or ESC key."""
        if not event.new:  # Modal was closed
            # Only call cancel callback if success callback wasn't invoked
            if not self._success_callback_invoked:
                self._cancel_callback()

            # Remove modal from its parent container after a short delay
            # to allow the close animation to complete
            def cleanup():
                try:
                    if hasattr(self._modal, '_parent') and self._modal._parent:
                        self._modal._parent.remove(self._modal)
                except Exception:  # noqa: S110
                    pass  # Ignore cleanup errors

            pn.state.add_periodic_callback(cleanup, period=100, count=1)

    def show(self) -> None:
        """Show the modal dialog."""
        # Reset state
        self._current_step = 1
        self._selected_job = None
        self._selected_output = None
        self._selected_plot = None
        self._next_button.disabled = True
        self._configure_button.disabled = True

        # Refresh data and show
        self._update_job_output_table()
        self._update_content()
        self._modal.open = True

    @property
    def modal(self) -> pn.Modal:
        """Get the modal widget."""
        return self._modal
