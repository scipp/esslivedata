# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import panel as pn

from ess.livedata.config.workflow_spec import JobNumber
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plotting_controller import PlottingController

from .configuration_widget import ConfigurationPanel
from .plot_configuration_adapter import PlotConfigurationAdapter
from .wizard import Wizard, WizardState, WizardStep


@dataclass
class PlotterSelectionContext:
    """Data accumulated through wizard steps."""

    job: JobNumber | None = None
    output: str | None = None
    plot_name: str | None = None
    created_plot: Any | None = None
    selected_sources: list[str] | None = None


class JobOutputSelectionStep(WizardStep):
    """Step 1: Job and output selection."""

    def __init__(
        self,
        context: PlotterSelectionContext,
        job_service: JobService,
    ) -> None:
        """
        Initialize job/output selection step.

        Parameters
        ----------
        context:
            Shared wizard context
        job_service:
            Service for accessing job data
        """
        super().__init__()
        self._context = context
        self._job_service = job_service
        self._table = self._create_job_output_table()

        # Set up selection watcher
        self._table.param.watch(self._on_table_selection_change, 'selection')

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
        self._table.value = df

    def _on_table_selection_change(self, event) -> None:
        """Handle job and output selection change."""
        selection = event.new
        if len(selection) != 1:
            self._context.job = None
            self._context.output = None
            self._notify_ready_changed(False)
            return

        # Get selected job number and output name from index
        selected_row = selection[0]
        job_number_str = self._table.value['job_number'].iloc[selected_row]
        output_name = self._table.value['output_name'].iloc[selected_row]

        self._context.job = JobNumber(job_number_str)
        self._context.output = output_name if output_name else None
        self._notify_ready_changed(True)

    def is_valid(self) -> bool:
        """Whether a valid job/output selection has been made."""
        return self._context.job is not None

    def render(self) -> pn.Column:
        """Render job/output selection table."""
        return pn.Column(
            pn.pane.HTML(
                "<h3>Step 1: Select Job and Output</h3>"
                "<p>Choose the job and output you want to visualize.</p>"
            ),
            self._table,
            sizing_mode='stretch_width',
        )

    def on_enter(self) -> None:
        """Update table data when step becomes active."""
        self._update_job_output_table()


class PlotterSelectionStep(WizardStep):
    """Step 2: Plotter type selection."""

    def __init__(
        self,
        context: PlotterSelectionContext,
        plotting_controller: PlottingController,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize plotter selection step.

        Parameters
        ----------
        context:
            Shared wizard context
        plotting_controller:
            Controller for determining available plotters
        logger:
            Logger instance for error reporting
        """
        super().__init__()
        self._context = context
        self._plotting_controller = plotting_controller
        self._wizard: Wizard | None = None
        self._logger = logger
        self._buttons_container = pn.Column(sizing_mode='stretch_width')

    def set_wizard(self, wizard: Wizard) -> None:
        """Set wizard reference after construction."""
        self._wizard = wizard

    def is_valid(self) -> bool:
        """Step 2 doesn't use Next button, so always return False."""
        return False

    def render(self) -> pn.Column:
        """Render plotter selection buttons."""
        return pn.Column(
            pn.pane.HTML(
                "<h3>Step 2: Select Plotter Type</h3>"
                "<p>Choose the type of plot you want to create.</p>"
            ),
            self._buttons_container,
            sizing_mode='stretch_width',
        )

    def on_enter(self) -> None:
        """Update available plotters when step becomes active."""
        self._update_plotter_buttons()

    def _update_plotter_buttons(self) -> None:
        """Update plotter buttons based on job and output selection."""
        self._buttons_container.clear()

        if self._context.job is None:
            self._buttons_container.append(pn.pane.Markdown("*No job selected*"))
            return

        try:
            available_plots = self._plotting_controller.get_available_plotters(
                self._context.job, self._context.output
            )
            if available_plots:
                plot_data = {
                    name: (spec.title, spec) for name, spec in available_plots.items()
                }
                buttons = self._create_plotter_buttons(plot_data)
                self._buttons_container.extend(buttons)
            else:
                self._buttons_container.append(
                    pn.pane.Markdown("*No plotters available for this selection*")
                )
        except Exception as e:
            self._logger.exception(
                "Error loading plotters for job %s", self._context.job
            )
            self._buttons_container.append(
                pn.pane.Markdown(f"*Error loading plotters: {e}*")
            )

    def _create_plotter_buttons(
        self, available_plots: dict[str, tuple[str, object]]
    ) -> list[pn.widgets.Button]:
        """Create buttons for each available plotter."""
        buttons = []
        for plot_name, (title, _spec) in available_plots.items():
            button = pn.widgets.Button(
                name=title,
                button_type="primary",
                sizing_mode='stretch_width',
                min_width=200,
            )
            button.on_click(lambda event, pn=plot_name: self._on_button_click(pn))
            buttons.append(button)
        return buttons

    def _on_button_click(self, plot_name: str) -> None:
        """Handle plotter button click - update context and advance."""
        self._context.plot_name = plot_name
        if self._wizard:
            self._wizard.advance()


class ConfigurationStep(WizardStep):
    """Step 3: Plot configuration."""

    def __init__(
        self,
        context: PlotterSelectionContext,
        job_service: JobService,
        plotting_controller: PlottingController,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize configuration step.

        Parameters
        ----------
        context:
            Shared wizard context
        job_service:
            Service for accessing job data
        plotting_controller:
            Controller for plot creation
        logger:
            Logger instance for error reporting
        """
        super().__init__()
        self._context = context
        self._job_service = job_service
        self._plotting_controller = plotting_controller
        self._wizard: Wizard | None = None
        self._logger = logger
        self._config_panel: ConfigurationPanel | None = None
        self._panel_container = pn.Column(sizing_mode='stretch_width')

    def set_wizard(self, wizard: Wizard) -> None:
        """Set wizard reference after construction."""
        self._wizard = wizard

    def reset(self) -> None:
        """Reset configuration panel (e.g., when going back)."""
        self._config_panel = None
        self._panel_container.clear()

    def is_valid(self) -> bool:
        """Step 3 doesn't use Next button, completion is via config panel."""
        return False

    def render(self) -> pn.Column:
        """Render configuration panel."""
        return pn.Column(
            pn.pane.HTML("<h3>Step 3: Configure Plot</h3>"),
            self._panel_container,
            sizing_mode='stretch_width',
        )

    def on_enter(self) -> None:
        """Create configuration panel when step becomes active."""
        if self._config_panel is None and self._context.job and self._context.plot_name:
            self._create_config_panel()

    def _create_config_panel(self) -> None:
        """Create the configuration panel for the selected plotter."""
        if not self._context.job or not self._context.plot_name:
            return

        job_data = self._job_service.job_data.get(self._context.job, {})
        available_sources = list(job_data.keys())

        if not available_sources:
            self._show_error('No sources available for selected job')
            return

        try:
            plot_spec = self._plotting_controller.get_spec(self._context.plot_name)
        except Exception as e:
            self._logger.exception("Error getting plot spec")
            self._show_error(f'Error getting plot spec: {e}')
            return

        config_adapter = PlotConfigurationAdapter(
            job_number=self._context.job,
            output_name=self._context.output,
            plot_spec=plot_spec,
            available_sources=available_sources,
            plotting_controller=self._plotting_controller,
            success_callback=self._on_config_complete,
        )

        self._config_panel = ConfigurationPanel(
            config=config_adapter,
            start_button_text="Create Plot",
            show_cancel_button=False,
            success_callback=self._on_panel_success,
        )

        self._panel_container.clear()
        self._panel_container.append(self._config_panel.panel)

    def _on_config_complete(self, plot, selected_sources: list[str]) -> None:
        """Handle plot creation - store in context."""
        self._context.created_plot = plot
        self._context.selected_sources = selected_sources

    def _on_panel_success(self) -> None:
        """Handle successful panel action - complete wizard."""
        if self._wizard:
            self._wizard.complete()

    def _show_error(self, message: str) -> None:
        """Display an error notification."""
        if pn.state.notifications is not None:
            pn.state.notifications.error(message, duration=3000)


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
        self._success_callback = success_callback
        self._logger = logging.getLogger(__name__)

        # Create shared context
        self._context = PlotterSelectionContext()

        # Create steps without wizard reference
        step1 = JobOutputSelectionStep(
            context=self._context,
            job_service=job_service,
        )

        step2 = PlotterSelectionStep(
            context=self._context,
            plotting_controller=plotting_controller,
            logger=self._logger,
        )

        step3 = ConfigurationStep(
            context=self._context,
            job_service=job_service,
            plotting_controller=plotting_controller,
            logger=self._logger,
        )

        # Create wizard
        self._wizard = Wizard(
            steps=[step1, step2, step3],
            context=self._context,
            on_complete=self._on_wizard_complete,
            on_cancel=self._on_wizard_cancel,
        )

        # Wire up wizard references for steps that need to call wizard methods
        step2.set_wizard(self._wizard)
        step3.set_wizard(self._wizard)

        # Store step3 reference for reset
        self._step3 = step3
        self._cancel_callback = cancel_callback

        # Create modal wrapping the wizard
        self._modal = pn.Modal(
            self._wizard.render(),
            name="Select Job and Plotter",
            margin=20,
            width=900,
            height=700,
        )

        # Watch for modal close events (X button or ESC key)
        self._modal.param.watch(self._on_modal_closed, 'open')

    def _on_wizard_complete(self, context: PlotterSelectionContext) -> None:
        """Handle wizard completion - close modal and call success callback."""
        self._modal.open = False
        if context.created_plot is not None and context.selected_sources is not None:
            self._success_callback(context.created_plot, context.selected_sources)

    def _on_wizard_cancel(self) -> None:
        """Handle wizard cancellation - close modal and call cancel callback."""
        self._modal.open = False
        self._cancel_callback()

    def _on_modal_closed(self, event) -> None:
        """Handle modal being closed via X button or ESC key."""
        if not event.new:  # Modal was closed
            # Only call cancel callback if wizard wasn't already completed/cancelled
            if self._wizard._state == WizardState.ACTIVE:
                self._cancel_callback()

    def show(self) -> None:
        """Show the modal dialog."""
        # Reset context
        self._context.job = None
        self._context.output = None
        self._context.plot_name = None
        self._context.created_plot = None
        self._context.selected_sources = None
        self._step3.reset()

        # Reset wizard and show modal
        self._wizard.reset()
        self._modal.open = True

    @property
    def modal(self) -> pn.Modal:
        """Get the modal widget."""
        return self._modal
