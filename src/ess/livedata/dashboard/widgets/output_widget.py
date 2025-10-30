# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import holoviews as hv
import panel as pn

from ess.livedata.config.workflow_spec import JobNumber, WorkflowId
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plotting import PlotterSpec
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.widgets.configuration_widget import ConfigurationModal
from ess.livedata.dashboard.widgets.plot_creation_widget import (
    PlotConfigurationAdapter,
)
from ess.livedata.dashboard.workflow_controller import WorkflowController


class OutputUIConstants:
    """Constants for Output widget UI styling and sizing."""

    # Colors
    DEFAULT_COLOR = "#C162F4"
    BUTTON_COLOR = "#007bff"

    # Sizes
    OUTPUT_INFO_WIDTH = 400
    OUTPUT_INFO_HEIGHT = 45
    BUTTON_WIDTH = 80
    BUTTON_HEIGHT = 36
    EXPAND_BUTTON_WIDTH = 25
    EXPAND_BUTTON_HEIGHT = 20
    DETAILS_WIDTH = 600

    # Margins
    STANDARD_MARGIN = (5, 5)
    INFO_MARGIN = (5, 10)
    BUTTON_MARGIN = (0, 2)
    EXPAND_MARGIN = (0, 5)
    DETAILS_MARGIN = (5, 10)


class OutputWidget:
    """Widget to display a single workflow output with plotting capabilities."""

    def __init__(
        self,
        *,
        job_number: JobNumber,
        output_name: str | None,
        output_title: str,
        output_description: str,
        source_names: list[str],
        workflow_id: WorkflowId,
        job_service: JobService,
        plotting_controller: PlottingController,
        workflow_controller: WorkflowController,
        plot_created_callback: callable | None = None,
    ) -> None:
        """
        Initialize output widget.

        Parameters
        ----------
        job_number:
            The job number this output belongs to
        output_name:
            The output name (field name), or None for single-output workflows
        output_title:
            Human-readable title for the output
        output_description:
            Description of the output (shown as tooltip)
        source_names:
            List of source names that have this output
        workflow_id:
            The workflow ID that produced this output
        job_service:
            Service for accessing job data
        plotting_controller:
            Controller for creating plots
        workflow_controller:
            Controller for accessing workflow specifications
        plot_created_callback:
            Callback to invoke when a plot is successfully created.
            Signature: callback(plot: hv.DynamicMap, selected_sources: list[str])
        """
        self._job_number = job_number
        self._output_name = output_name
        self._output_title = output_title
        self._output_description = output_description
        self._source_names = source_names
        self._workflow_id = workflow_id
        self._job_service = job_service
        self._plotting_controller = plotting_controller
        self._workflow_controller = workflow_controller
        self._plot_created_callback = plot_created_callback

        self._details_expanded = False
        self._modal_container = pn.Column()
        self._setup_widgets()
        self._create_panel()

    def _setup_widgets(self) -> None:
        """Set up the UI components."""
        # Output title with description tooltip
        title_html = f"<b>{self._output_title}</b>"
        if self._output_description:
            # Use title attribute for tooltip
            title_html = (
                f'<b title="{self._output_description}">{self._output_title}</b>'
            )

        self._output_info = pn.pane.HTML(
            title_html,
            width=OutputUIConstants.OUTPUT_INFO_WIDTH,
            height=OutputUIConstants.OUTPUT_INFO_HEIGHT,
            margin=OutputUIConstants.INFO_MARGIN,
        )

        # Plot button
        self._plot_button = pn.widgets.Button(
            name="Plot",
            button_type="primary",
            width=OutputUIConstants.BUTTON_WIDTH,
            height=OutputUIConstants.BUTTON_HEIGHT,
            margin=OutputUIConstants.BUTTON_MARGIN,
        )
        self._plot_button.on_click(self._on_plot_button_click)

        # Expand/collapse button for details
        self._expand_button = pn.widgets.Button(
            name="▼",
            button_type="light",
            width=OutputUIConstants.EXPAND_BUTTON_WIDTH,
            height=OutputUIConstants.EXPAND_BUTTON_HEIGHT,
            margin=OutputUIConstants.EXPAND_MARGIN,
        )
        self._expand_button.on_click(self._toggle_details)

        # Details panel (initially hidden)
        self._details_panel = self._create_details_panel()

    def _create_details_panel(self) -> pn.Column:
        """Create the expandable details panel."""
        workflow_text = f"{self._workflow_id.name} v{self._workflow_id.version}"
        job_number_text = str(self._job_number)[:16]  # Truncate long UUIDs
        sources_text = ", ".join(self._source_names)

        details_html = f"""
        <div style="font-size: 12px; padding: 5px;">
            <b>Workflow:</b> {workflow_text}<br>
            <b>Job Number:</b> {job_number_text}<br>
            <b>Sources:</b> {sources_text}
        </div>
        """

        return pn.Column(
            pn.pane.HTML(
                details_html,
                width=OutputUIConstants.DETAILS_WIDTH,
                margin=OutputUIConstants.DETAILS_MARGIN,
            ),
            visible=False,
        )

    def _toggle_details(self, event) -> None:
        """Toggle the visibility of the details panel."""
        self._details_expanded = not self._details_expanded
        self._details_panel.visible = self._details_expanded
        self._expand_button.name = "▲" if self._details_expanded else "▼"

    def _on_plot_button_click(self, event) -> None:
        """Handle plot button click - show plotter selection menu."""
        # Get available plotters
        try:
            available_plotters = self._plotting_controller.get_available_plotters(
                self._job_number, self._output_name
            )
        except Exception as e:
            # Show error message
            error_pane = pn.pane.Alert(
                f"Error getting available plotters: {e}",
                alert_type="danger",
                margin=OutputUIConstants.STANDARD_MARGIN,
            )
            self._modal_container.clear()
            self._modal_container.append(error_pane)
            return

        if not available_plotters:
            # Show message that no plotters are available
            info_pane = pn.pane.Alert(
                "No plotters available for this output",
                alert_type="warning",
                margin=OutputUIConstants.STANDARD_MARGIN,
            )
            self._modal_container.clear()
            self._modal_container.append(info_pane)
            return

        # Show plotter selection menu
        self._show_plotter_menu(available_plotters)

    def _show_plotter_menu(self, available_plotters: dict[str, PlotterSpec]) -> None:
        """Show a menu to select which plotter to use."""
        menu_items = []

        for plot_name, spec in available_plotters.items():

            def make_callback(name=plot_name, s=spec):
                return lambda event: self._on_plotter_selected(name, s)

            button = pn.widgets.Button(
                name=spec.title,
                button_type="light",
                sizing_mode="stretch_width",
                margin=(2, 5),
            )
            button.on_click(make_callback())

            # Add description if available
            if spec.description:
                menu_items.append(
                    pn.Column(
                        button,
                        pn.pane.HTML(
                            f'<div style="font-size: 11px; color: #666; '
                            f'margin-left: 10px;">{spec.description}</div>',
                            margin=(0, 5, 5, 5),
                        ),
                    )
                )
            else:
                menu_items.append(button)

        menu_panel = pn.Column(
            pn.pane.HTML("<h4>Select Plotter</h4>", margin=(5, 5)),
            *menu_items,
            styles={
                "border": "1px solid #dee2e6",
                "border-radius": "4px",
                "background-color": "#f8f9fa",
            },
            margin=OutputUIConstants.STANDARD_MARGIN,
            max_width=400,
        )

        self._modal_container.clear()
        self._modal_container.append(menu_panel)

    def _on_plotter_selected(self, plot_name: str, spec: PlotterSpec) -> None:
        """Handle plotter selection - show configuration modal."""
        # Clear the plotter menu
        self._modal_container.clear()

        # Create configuration adapter
        config = PlotConfigurationAdapter(
            job_number=self._job_number,
            output_name=self._output_name,
            plot_spec=spec,
            available_sources=self._source_names,
            plotting_controller=self._plotting_controller,
            success_callback=self._on_plot_created,
        )

        # Create and show configuration modal
        modal = ConfigurationModal(config=config, start_button_text="Create Plot")
        self._modal_container.append(modal.modal)
        modal.show()

    def _on_plot_created(
        self, plot: hv.DynamicMap, selected_sources: list[str]
    ) -> None:
        """Handle successful plot creation."""
        # Clear the modal
        self._modal_container.clear()

        # Invoke callback if provided
        if self._plot_created_callback:
            self._plot_created_callback(plot, selected_sources)

    def _create_panel(self) -> None:
        """Create the panel layout for this widget."""
        # Main row with output info, expand button, and plot button
        main_row = pn.Row(
            self._expand_button,
            self._output_info,
            self._plot_button,
            sizing_mode="stretch_width",
        )

        self._panel = pn.Column(
            main_row,
            self._details_panel,
            self._modal_container,
            styles={
                "border": "1px solid #dee2e6",
                "border-radius": "4px",
                "margin": "2px",
            },
            sizing_mode="stretch_width",
        )

    @property
    def job_number(self) -> JobNumber:
        """Get the job number for this widget."""
        return self._job_number

    @property
    def output_name(self) -> str | None:
        """Get the output name for this widget."""
        return self._output_name

    def panel(self) -> pn.layout.Column:
        """Get the panel layout for this widget."""
        return self._panel

    def update_source_names(self, source_names: list[str]) -> None:
        """
        Update the list of source names.

        Parameters
        ----------
        source_names:
            Updated list of source names
        """
        self._source_names = source_names
        # Recreate details panel with new source names
        old_visible = self._details_panel.visible
        self._details_panel = self._create_details_panel()
        self._details_panel.visible = old_visible
        # Update panel layout
        self._panel[1] = self._details_panel


class OutputListWidget:
    """Widget to display a list of workflow outputs with live updates."""

    def __init__(
        self,
        *,
        job_service: JobService,
        plotting_controller: PlottingController,
        workflow_controller: WorkflowController,
        plot_created_callback: callable | None = None,
    ) -> None:
        """
        Initialize output list widget.

        Parameters
        ----------
        job_service:
            Service for accessing job data
        plotting_controller:
            Controller for creating plots
        workflow_controller:
            Controller for accessing workflow specifications
        plot_created_callback:
            Callback to invoke when a plot is successfully created.
            Signature: callback(plot: hv.DynamicMap, selected_sources: list[str])
        """
        self._job_service = job_service
        self._plotting_controller = plotting_controller
        self._workflow_controller = workflow_controller
        self._plot_created_callback = plot_created_callback

        self._output_widgets: dict[tuple[JobNumber, str | None], OutputWidget] = {}
        self._widget_panels: dict[tuple[JobNumber, str | None], pn.layout.Column] = {}

        self._setup_layout()

        # Subscribe to job data updates
        self._job_service.register_job_update_subscriber(self._on_data_update)

    def _setup_layout(self) -> None:
        """Set up the main layout."""
        self._header = pn.pane.HTML("<h3>Workflow Outputs</h3>", margin=(10, 10, 5, 10))
        self._output_list = pn.Column(sizing_mode="stretch_width", margin=(0, 10))

        # Initialize with current job data
        self._refresh_widgets()

    def _get_output_metadata(
        self, job_number: JobNumber, output_name: str | None
    ) -> tuple[str, str]:
        """
        Get human-readable title and description for an output.

        Parameters
        ----------
        job_number:
            The job number to get output metadata for
        output_name:
            The raw output name (field name in the outputs model)

        Returns
        -------
        :
            Tuple of (title, description). If metadata is not available,
            returns the raw output_name as title and empty description.
        """
        workflow_id = self._job_service.job_info.get(job_number)
        if workflow_id is None:
            return output_name or "Output", ""

        workflow_spec = self._workflow_controller.get_workflow_spec(workflow_id)
        if workflow_spec is None or workflow_spec.outputs is None:
            return output_name or "Output", ""

        # Get field metadata from the outputs model
        field_info = workflow_spec.outputs.model_fields.get(output_name)
        if field_info is None:
            return output_name or "Output", ""

        # Extract title and description from field metadata
        title = field_info.title if field_info.title else (output_name or "Output")
        description = field_info.description if field_info.description else ""

        return title, description

    def _on_data_update(self) -> None:
        """Handle job data updates from the service."""
        self._refresh_widgets()

    def _refresh_widgets(self) -> None:
        """Refresh the widget list based on current job data."""
        # Build set of current (job_number, output_name) pairs
        current_outputs: set[tuple[JobNumber, str | None]] = set()

        for job_number, sources_data in self._job_service.job_data.items():
            # Get all output names from the first source (all sources have same outputs)
            if sources_data:
                first_source_data = next(iter(sources_data.values()))
                if isinstance(first_source_data, dict):
                    output_names = first_source_data.keys()
                    for output_name in output_names:
                        current_outputs.add((job_number, output_name))

        # Remove widgets that no longer exist
        widgets_to_remove = set(self._output_widgets.keys()) - current_outputs
        for widget_key in widgets_to_remove:
            self._remove_output_widget(widget_key)

        # Add or update widgets for current outputs
        for job_number, output_name in current_outputs:
            self._add_or_update_output_widget(job_number, output_name)

    def _add_or_update_output_widget(
        self, job_number: JobNumber, output_name: str | None
    ) -> None:
        """Add a new output widget or update an existing one."""
        widget_key = (job_number, output_name)

        # Get current source names for this output
        job_data = self._job_service.job_data.get(job_number, {})
        source_names = list(job_data.keys())

        if widget_key in self._output_widgets:
            # Update existing widget with new source names
            self._output_widgets[widget_key].update_source_names(source_names)
        else:
            # Create new widget
            workflow_id = self._job_service.job_info.get(job_number)
            if workflow_id is None:
                return  # Skip if workflow info not available

            title, description = self._get_output_metadata(job_number, output_name)

            widget = OutputWidget(
                job_number=job_number,
                output_name=output_name,
                output_title=title,
                output_description=description,
                source_names=source_names,
                workflow_id=workflow_id,
                job_service=self._job_service,
                plotting_controller=self._plotting_controller,
                workflow_controller=self._workflow_controller,
                plot_created_callback=self._plot_created_callback,
            )

            widget_panel = widget.panel()
            self._output_widgets[widget_key] = widget
            self._widget_panels[widget_key] = widget_panel
            self._output_list.append(widget_panel)

    def _remove_output_widget(self, widget_key: tuple[JobNumber, str | None]) -> None:
        """Remove an output widget."""
        if widget_key in self._output_widgets:
            self._output_widgets.pop(widget_key)
            if widget_key in self._widget_panels:
                widget_panel = self._widget_panels.pop(widget_key)
                try:
                    self._output_list.remove(widget_panel)
                except ValueError:
                    pass

    def panel(self) -> pn.layout.Column:
        """Get the main panel for this widget."""
        return pn.Column(self._header, self._output_list, sizing_mode="stretch_width")
