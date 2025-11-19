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
from ess.livedata.dashboard.plotting import PlotterSpec
from ess.livedata.dashboard.plotting_controller import PlottingController

from ..plot_configuration_adapter import PlotConfigurationAdapter
from .configuration_widget import ConfigurationPanel
from .wizard import Wizard, WizardStep


@dataclass
class JobOutputSelection:
    """Output from job/output selection step."""

    job: JobNumber
    output: str | None


@dataclass
class PlotterSelection:
    """Output from plotter selection step."""

    job: JobNumber
    output: str | None
    plot_name: str


@dataclass
class PlotResult:
    """Output from configuration step (final result)."""

    plot: Any
    selected_sources: list[str]


class JobOutputSelectionStep(WizardStep[None, JobOutputSelection]):
    """
    Step 1: Job and output selection.

    This is mostly copied from the legacy PlotCreationWidget but is considered legacy.
    The contents of this widget will be fully replaced.
    """

    def __init__(
        self,
        job_service: JobService,
    ) -> None:
        """
        Initialize job/output selection step.

        Parameters
        ----------
        job_service:
            Service for accessing job data
        """
        super().__init__()
        self._job_service = job_service
        self._table = self._create_job_output_table()
        self._selected_job: JobNumber | None = None
        self._selected_output: str | None = None

        # Set up selection watcher
        self._table.param.watch(self._on_table_selection_change, 'selection')

    @property
    def name(self) -> str:
        """Display name for this step."""
        return "Select Job and Output"

    @property
    def description(self) -> str | None:
        """Description text for this step."""
        return "Choose the job and output you want to visualize."

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
            self._selected_job = None
            self._selected_output = None
            self._notify_ready_changed(False)
            return

        # Get selected job number and output name from index
        selected_row = selection[0]
        job_number_str = self._table.value['job_number'].iloc[selected_row]
        output_name = self._table.value['output_name'].iloc[selected_row]

        self._selected_job = JobNumber(job_number_str)
        self._selected_output = output_name if output_name else None
        self._notify_ready_changed(True)

    def is_valid(self) -> bool:
        """Whether a valid job/output selection has been made."""
        return self._selected_job is not None

    def commit(self) -> JobOutputSelection | None:
        """Commit the selected job and output."""
        if self._selected_job is None:
            return None
        return JobOutputSelection(job=self._selected_job, output=self._selected_output)

    def render_content(self) -> pn.Column:
        """Render job/output selection table."""
        return pn.Column(
            self._table,
            sizing_mode='stretch_width',
        )

    def on_enter(self, input_data: None) -> None:
        """Update table data when step becomes active."""
        self._update_job_output_table()


class PlotterSelectionStep(WizardStep[JobOutputSelection, PlotterSelection]):
    """Step 2: Plotter type selection."""

    def __init__(
        self,
        plotting_controller: PlottingController,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize plotter selection step.

        Parameters
        ----------
        plotting_controller:
            Controller for determining available plotters
        logger:
            Logger instance for error reporting
        """
        super().__init__()
        self._plotting_controller = plotting_controller
        self._logger = logger
        self._radio_group: pn.widgets.RadioButtonGroup | None = None
        self._content_container = pn.Column(sizing_mode='stretch_width')
        self._job_output: JobOutputSelection | None = None
        self._selected_plot_name: str | None = None

    @property
    def name(self) -> str:
        """Display name for this step."""
        return "Select Plotter Type"

    @property
    def description(self) -> str | None:
        """Description text for this step."""
        return "Choose the type of plot you want to create."

    def is_valid(self) -> bool:
        """Step is valid when a plotter has been selected."""
        return self._selected_plot_name is not None

    def commit(self) -> PlotterSelection | None:
        """Commit the job, output, and selected plotter."""
        if self._job_output is None or self._selected_plot_name is None:
            return None
        return PlotterSelection(
            job=self._job_output.job,
            output=self._job_output.output,
            plot_name=self._selected_plot_name,
        )

    def render_content(self) -> pn.Column:
        """Render plotter selection radio buttons."""
        return self._content_container

    def on_enter(self, input_data: JobOutputSelection) -> None:
        """Update available plotters when step becomes active."""
        self._job_output = input_data
        self._update_plotter_selection()

    def _update_plotter_selection(self) -> None:
        """Update plotter selection based on job and output selection."""
        self._content_container.clear()

        if self._job_output is None:
            self._content_container.append(pn.pane.Markdown("*No job selected*"))
            self._radio_group = None
            self._notify_ready_changed(False)
            return

        available_plots = self._plotting_controller.get_available_plotters(
            self._job_output.job, self._job_output.output
        )
        if available_plots:
            self._create_radio_buttons(available_plots)
        else:
            self._content_container.append(
                pn.pane.Markdown("*No plotters available for this selection*")
            )
            self._radio_group = None
            self._notify_ready_changed(False)

    def _create_radio_buttons(self, available_plots: dict[str, PlotterSpec]) -> None:
        """Create radio button group for plotter selection."""
        # Build mapping from display title to plot name.
        # RadioButtonGroup displays keys (titles) and stores values (plot names).
        # Handle potential duplicate titles by making them unique.
        self._plot_name_map = self._make_unique_title_mapping(available_plots)
        options = self._plot_name_map

        # Select first option by default
        # When using dict options, the value must be a dict value (plot name), not a key
        initial_value = (
            next(iter(self._plot_name_map.values())) if self._plot_name_map else None
        )

        self._radio_group = pn.widgets.RadioButtonGroup(
            name="Plotter Type",
            options=options,
            value=initial_value,
            button_type="primary",
            button_style="solid",
            sizing_mode='stretch_width',
        )
        self._radio_group.param.watch(self._on_plotter_selection_change, 'value')
        self._content_container.append(self._radio_group)

        # Initialize with the selected value
        if initial_value is not None:
            # initial_value is already the plot name (dict value)
            self._selected_plot_name = initial_value
            self._notify_ready_changed(True)

    def _make_unique_title_mapping(
        self, available_plots: dict[str, PlotterSpec]
    ) -> dict[str, str]:
        """Create mapping from unique display titles to internal plot names."""
        title_counts: dict[str, int] = {}
        result: dict[str, str] = {}

        # Sort alphabetically by title for better UX
        sorted_plots = sorted(available_plots.items(), key=lambda x: x[1].title)

        for name, spec in sorted_plots:
            title = spec.title
            count = title_counts.get(title, 0)
            title_counts[title] = count + 1

            # Make title unique if we've seen it before
            unique_title = f"{title} ({count + 1})" if count > 0 else title
            result[unique_title] = name

        return result

    def _on_plotter_selection_change(self, event) -> None:
        """Handle plotter selection change."""
        if event.new is not None:
            # When using dict options, event.new is the dict value (plot name)
            self._selected_plot_name = event.new
            self._notify_ready_changed(True)
        else:
            self._selected_plot_name = None
            self._notify_ready_changed(False)


class ConfigurationStep(WizardStep[PlotterSelection, PlotResult]):
    """Step 3: Plot configuration."""

    def __init__(
        self,
        job_service: JobService,
        plotting_controller: PlottingController,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize configuration step.

        Parameters
        ----------
        job_service:
            Service for accessing job data
        plotting_controller:
            Controller for plot creation
        logger:
            Logger instance for error reporting
        """
        super().__init__()
        self._job_service = job_service
        self._plotting_controller = plotting_controller
        self._logger = logger
        self._config_panel: ConfigurationPanel | None = None
        self._panel_container = pn.Column(sizing_mode='stretch_width')
        self._plotter_selection: PlotterSelection | None = None
        # Track last configuration to detect when panel needs recreation
        self._last_job: JobNumber | None = None
        self._last_output: str | None = None
        self._last_plot_name: str | None = None
        # Store result from callback
        self._last_plot_result: PlotResult | None = None

    @property
    def name(self) -> str:
        """Display name for this step."""
        return "Configure Plot"

    def is_valid(self) -> bool:
        """Step is valid when configuration is valid."""
        if self._config_panel is None:
            return False
        is_valid, _ = self._config_panel.validate()
        return is_valid

    def commit(self) -> PlotResult | None:
        """Commit the plot configuration and create the plot."""
        if self._config_panel is None or self._plotter_selection is None:
            return None

        # Clear previous result
        self._last_plot_result = None

        # Execute action (which calls adapter, which calls our callback)
        success = self._config_panel.execute_action()
        if not success:
            return None

        # Result was captured by callback
        return self._last_plot_result

    def render_content(self) -> pn.Column:
        """Render configuration panel."""
        return self._panel_container

    def on_enter(self, input_data: PlotterSelection) -> None:
        """Create or recreate configuration panel when selection changes."""
        self._plotter_selection = input_data

        # Check if the configuration has changed
        if (
            input_data.job != self._last_job
            or input_data.output != self._last_output
            or input_data.plot_name != self._last_plot_name
        ):
            # Recreate panel with new configuration
            self._create_config_panel()
            # Track new values
            self._last_job = input_data.job
            self._last_output = input_data.output
            self._last_plot_name = input_data.plot_name

    def _create_config_panel(self) -> None:
        """Create the configuration panel for the selected plotter."""
        if self._plotter_selection is None:
            return

        job_data = self._job_service.job_data.get(self._plotter_selection.job, {})
        available_sources = list(job_data.keys())

        if not available_sources:
            self._show_error('No sources available for selected job')
            return

        try:
            plot_spec = self._plotting_controller.get_spec(
                self._plotter_selection.plot_name
            )
        except Exception as e:
            self._logger.exception("Error getting plot spec")
            self._show_error(f'Error getting plot spec: {e}')
            return

        config_state = self._plotting_controller.get_persistent_plotter_config(
            job_number=self._plotter_selection.job,
            output_name=self._plotter_selection.output,
            plot_name=plot_spec.name,
        )

        # Capture state at modal creation time to avoid reading stale instance state
        plotter_selection = self._plotter_selection

        def on_plot_created(selected_sources: list[str], params) -> None:
            """Create plot with captured state."""
            plot = self._plotting_controller.create_plot(
                job_number=plotter_selection.job,
                source_names=selected_sources,
                output_name=plotter_selection.output,
                plot_name=plotter_selection.plot_name,
                params=params,
            )
            self._last_plot_result = PlotResult(
                plot=plot, selected_sources=selected_sources
            )

        config_adapter = PlotConfigurationAdapter(
            plot_spec=plot_spec,
            source_names=available_sources,
            success_callback=on_plot_created,
            config_state=config_state,
        )

        self._config_panel = ConfigurationPanel(config=config_adapter)

        self._panel_container.clear()
        self._panel_container.append(self._config_panel.panel)

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
        self._cancel_callback = cancel_callback
        self._logger = logging.getLogger(__name__)

        # Create steps
        step1 = JobOutputSelectionStep(job_service=job_service)

        step2 = PlotterSelectionStep(
            plotting_controller=plotting_controller, logger=self._logger
        )

        step3 = ConfigurationStep(
            job_service=job_service,
            plotting_controller=plotting_controller,
            logger=self._logger,
        )

        # Create wizard
        self._wizard = Wizard(
            steps=[step1, step2, step3],
            on_complete=self._on_wizard_complete,
            on_cancel=self._on_wizard_cancel,
            action_button_label="Create Plot",
        )

        # Create modal wrapping the wizard
        self._modal = pn.Modal(
            self._wizard.render(),
            name="Select Job and Plotter",
            margin=20,
            width=900,
            height=700,
        )

        # Watch for modal close events (X button or ESC key).
        # Panel's Modal widget uses 'open' as a boolean state property:
        # when it transitions to False, the modal is closed.
        self._modal.param.watch(self._on_modal_closed, 'open')

    def _on_wizard_complete(self, result: PlotResult) -> None:
        """Handle wizard completion - close modal and call success callback."""
        self._modal.open = False
        self._success_callback(result.plot, result.selected_sources)

    def _on_wizard_cancel(self) -> None:
        """Handle wizard cancellation - close modal and call cancel callback."""
        self._modal.open = False
        self._cancel_callback()

    def _on_modal_closed(self, event) -> None:
        """Handle modal being closed via X button or ESC key."""
        if not event.new:  # Modal was closed
            # Only call cancel callback if wizard wasn't already completed/cancelled
            if not self._wizard.is_finished():
                self._cancel_callback()

    def show(self) -> None:
        """Show the modal dialog."""
        # Reset wizard and show modal
        self._wizard.reset()
        self._modal.open = True

    @property
    def modal(self) -> pn.Modal:
        """Get the modal widget."""
        return self._modal
