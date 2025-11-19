# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Plot configuration modal for PlotOrchestrator-based workflow.

This modal provides a wizard for configuring plots without requiring
existing data:
1. Select workflow from available workflow specs
2. Select output name from workflow outputs
3. Select plotter type based on output metadata
4. Configure plotter (source selection and parameters)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import panel as pn
import pydantic

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec
from ess.livedata.dashboard.plot_configuration_adapter import PlotConfigurationAdapter
from ess.livedata.dashboard.plot_orchestrator import PlotConfig
from ess.livedata.dashboard.plotting import PlotterSpec

from .configuration_widget import ConfigurationPanel
from .wizard import Wizard, WizardStep


@dataclass
class WorkflowSelection:
    """Output from workflow selection step."""

    workflow_id: WorkflowId


@dataclass
class OutputSelection:
    """Output from output selection step."""

    workflow_id: WorkflowId
    output_name: str


@dataclass
class PlotterSelection:
    """Output from plotter selection step."""

    workflow_id: WorkflowId
    output_name: str
    plot_name: str


class WorkflowAndOutputSelectionStep(WizardStep[None, OutputSelection]):
    """Step 1: Select workflow and output (combined for better UX)."""

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        logger: logging.Logger,
    ) -> None:
        """
        Initialize workflow and output selection step.

        Parameters
        ----------
        workflow_registry
            Registry of available workflows and their specifications.
        logger
            Logger instance for error reporting.
        """
        super().__init__()
        self._workflow_registry = dict(workflow_registry)
        self._logger = logger
        self._selected_workflow_id: WorkflowId | None = None
        self._selected_output: str | None = None

        # Initialize containers first before creating buttons (to avoid callback issues)
        self._content_container = pn.Column(sizing_mode='stretch_width')
        self._output_container = pn.Column(sizing_mode='stretch_width')
        self._output_buttons: pn.widgets.RadioButtonGroup | None = None

        # Create workflow selector radio buttons (without triggering callbacks yet)
        self._workflow_buttons = self._create_workflow_buttons()

        # Initial layout
        self._update_content()

        # Now set the initial value (after all containers are set up)
        # This will trigger _on_workflow_change callback which updates output buttons
        if self._workflow_buttons.options:
            self._workflow_buttons.value = next(
                iter(self._workflow_buttons.options.values())
            )

    @property
    def name(self) -> str:
        """Display name for this step."""
        return "Select Workflow & Output"

    @property
    def description(self) -> str | None:
        """Description text for this step."""
        return "Choose the workflow and output to visualize."

    def _create_workflow_buttons(self) -> pn.widgets.RadioButtonGroup:
        """Create workflow selection radio buttons."""
        # Sort workflows by title for better UX
        sorted_workflows = sorted(
            self._workflow_registry.items(), key=lambda item: item[1].title
        )

        # Create options mapping title -> workflow_id
        options = {spec.title: wid for wid, spec in sorted_workflows}

        buttons = pn.widgets.RadioButtonGroup(
            name='Workflow',
            options=options,
            orientation='vertical',
            button_type='primary',
            button_style='outline',
            sizing_mode='stretch_width',
        )

        # Watch for selection changes
        buttons.param.watch(self._on_workflow_change, 'value')

        return buttons

    def _on_workflow_change(self, event) -> None:
        """Handle workflow selection change."""
        if event.new is not None:
            self._selected_workflow_id = event.new
            self._selected_output = None  # Reset output selection
            self._update_output_buttons()
            self._update_content()
            self._validate()
        else:
            self._selected_workflow_id = None
            self._selected_output = None
            self._validate()

    def _update_output_buttons(self) -> None:
        """Update output radio buttons based on selected workflow."""
        self._output_container.clear()
        self._output_buttons = None

        if self._selected_workflow_id is None:
            return

        workflow_spec = self._workflow_registry.get(self._selected_workflow_id)
        if workflow_spec is None or workflow_spec.outputs is None:
            self._output_container.append(
                pn.pane.Markdown("*No outputs available for this workflow*")
            )
            return

        # Extract output names from the Pydantic model
        output_fields = workflow_spec.outputs.model_fields
        if not output_fields:
            self._output_container.append(
                pn.pane.Markdown("*No outputs defined for this workflow*")
            )
            return

        # Create options mapping from output title to output name
        options = {}
        for field_name, field_info in output_fields.items():
            title = field_info.title if field_info.title else field_name
            options[title] = field_name

        self._output_buttons = pn.widgets.RadioButtonGroup(
            name='Output',
            options=options,
            orientation='vertical',
            button_type='primary',
            button_style='outline',
            sizing_mode='stretch_width',
        )

        # Watch for selection changes
        self._output_buttons.param.watch(self._on_output_change, 'value')

        # Initialize with first selection
        if options:
            self._output_buttons.value = next(iter(options.values()))
            self._selected_output = self._output_buttons.value

        self._output_container.append(self._output_buttons)
        self._validate()

    def _on_output_change(self, event) -> None:
        """Handle output selection change."""
        if event.new is not None:
            self._selected_output = event.new
            self._validate()
        else:
            self._selected_output = None
            self._validate()

    def _update_content(self) -> None:
        """Update the content container layout with side-by-side selectors."""
        self._content_container.clear()

        # Create side-by-side layout for workflow and output
        workflow_col = pn.Column(
            pn.pane.Markdown("**Workflow**"),
            self._workflow_buttons,
            sizing_mode='stretch_both',
        )

        output_col = pn.Column(
            pn.pane.Markdown("**Output**"),
            self._output_container,
            sizing_mode='stretch_both',
        )

        side_by_side = pn.Row(workflow_col, output_col, sizing_mode='stretch_width')

        self._content_container.append(side_by_side)

    def _validate(self) -> None:
        """Update validity based on selections."""
        is_valid = (
            self._selected_workflow_id is not None and self._selected_output is not None
        )
        self._notify_ready_changed(is_valid)

    def is_valid(self) -> bool:
        """Whether both workflow and output have been selected."""
        return (
            self._selected_workflow_id is not None and self._selected_output is not None
        )

    def commit(self) -> OutputSelection | None:
        """Commit the workflow and output selection."""
        if self._selected_workflow_id is None or self._selected_output is None:
            return None
        return OutputSelection(
            workflow_id=self._selected_workflow_id,
            output_name=self._selected_output,
        )

    def render_content(self) -> pn.Column:
        """Render combined workflow and output selector."""
        return self._content_container

    def on_enter(self, input_data: None) -> None:
        """Called when step becomes active."""
        pass


class PlotterSelectionStep(WizardStep[OutputSelection, PlotterSelection]):
    """Step 2: Select plotter type based on output metadata."""

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        plotting_controller,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize plotter selection step.

        Parameters
        ----------
        workflow_registry
            Registry of available workflows and their specifications.
        plotting_controller
            Controller for determining available plotters from specs.
        logger
            Logger instance for error reporting.
        """
        super().__init__()
        self._workflow_registry = dict(workflow_registry)
        self._plotting_controller = plotting_controller
        self._logger = logger
        self._output_selection: OutputSelection | None = None
        self._selected_plot_name: str | None = None
        self._radio_group: pn.widgets.RadioButtonGroup | None = None
        self._content_container = pn.Column(sizing_mode='stretch_width')

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
        """Commit the workflow, output, and selected plotter."""
        if self._output_selection is None or self._selected_plot_name is None:
            return None
        return PlotterSelection(
            workflow_id=self._output_selection.workflow_id,
            output_name=self._output_selection.output_name,
            plot_name=self._selected_plot_name,
        )

    def render_content(self) -> pn.Column:
        """Render plotter selection radio buttons."""
        return self._content_container

    def on_enter(self, input_data: OutputSelection) -> None:
        """Update available plotters when step becomes active."""
        self._output_selection = input_data
        self._update_plotter_selection()

    def _update_plotter_selection(self) -> None:
        """Update plotter selection based on workflow and output selection."""
        self._content_container.clear()

        if self._output_selection is None:
            self._content_container.append(pn.pane.Markdown("*No output selected*"))
            self._radio_group = None
            self._notify_ready_changed(False)
            return

        workflow_spec = self._workflow_registry.get(self._output_selection.workflow_id)
        if workflow_spec is None:
            self._content_container.append(
                pn.pane.Markdown("*Workflow spec not found*")
            )
            self._radio_group = None
            self._notify_ready_changed(False)
            return

        # Get available plotters from spec (returns tuple: plotters, has_metadata)
        available_plots, has_metadata = (
            self._plotting_controller.get_available_plotters_from_spec(
                workflow_spec=workflow_spec,
                output_name=self._output_selection.output_name,
            )
        )

        if available_plots:
            # Show warning if we're in fallback mode (no metadata available)
            if not has_metadata:
                self._content_container.append(
                    pn.pane.Markdown(
                        "**⚠️ Could not determine output properties. "
                        "Some of these plotters may not work with the actual data.**",
                        styles={'color': '#ff6b35', 'margin-bottom': '10px'},
                    )
                )
            self._create_radio_buttons(available_plots)
        else:
            # This should rarely happen since fallback returns all plotters
            self._content_container.append(pn.pane.Markdown("*No plotters available.*"))
            self._radio_group = None
            self._notify_ready_changed(False)

    def _create_radio_buttons(self, available_plots: dict[str, PlotterSpec]) -> None:
        """Create radio button group for plotter selection."""
        # Build mapping from display title to plot name
        self._plot_name_map = self._make_unique_title_mapping(available_plots)
        options = self._plot_name_map

        # Select first option by default
        initial_value = (
            next(iter(self._plot_name_map.values())) if self._plot_name_map else None
        )

        self._radio_group = pn.widgets.RadioButtonGroup(
            name="Plotter Type",
            options=options,
            value=initial_value,
            orientation='vertical',
            button_type="primary",
            button_style="outline",
            sizing_mode='stretch_width',
        )
        self._radio_group.param.watch(self._on_plotter_selection_change, 'value')
        self._content_container.append(self._radio_group)

        # Initialize with the selected value
        if initial_value is not None:
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
            self._selected_plot_name = event.new
            self._notify_ready_changed(True)
        else:
            self._selected_plot_name = None
            self._notify_ready_changed(False)


class SpecBasedConfigurationStep(WizardStep[PlotterSelection, PlotConfig]):
    """Step 3: Configure plot (source selection and plotter parameters)."""

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        plotting_controller,
        logger: logging.Logger,
        initial_config: PlotConfig | None = None,
    ) -> None:
        """
        Initialize spec-based configuration step.

        Parameters
        ----------
        workflow_registry
            Registry of available workflows and their specifications.
        plotting_controller
            Controller for getting plotter specs.
        logger
            Logger instance for error reporting.
        initial_config
            Optional initial configuration for edit mode.
        """
        super().__init__()
        self._workflow_registry = dict(workflow_registry)
        self._plotting_controller = plotting_controller
        self._logger = logger
        self._initial_config = initial_config
        self._config_panel: ConfigurationPanel | None = None
        self._panel_container = pn.Column(sizing_mode='stretch_width')
        self._plotter_selection: PlotterSelection | None = None
        # Track last configuration to detect when panel needs recreation
        self._last_workflow_id: WorkflowId | None = None
        self._last_output: str | None = None
        self._last_plot_name: str | None = None
        # Store result from callback
        self._last_config_result: PlotConfig | None = None

    @property
    def name(self) -> str:
        """Display name for this step."""
        return "Configure Plot"

    @property
    def description(self) -> str | None:
        """Description text for this step."""
        return None

    def is_valid(self) -> bool:
        """Step is valid when configuration is valid."""
        if self._config_panel is None:
            return False
        is_valid, _ = self._config_panel.validate()
        return is_valid

    def commit(self) -> PlotConfig | None:
        """Commit the plot configuration."""
        if self._config_panel is None or self._plotter_selection is None:
            return None

        # Clear previous result
        self._last_config_result = None

        # Execute action (which calls adapter, which calls our callback)
        success = self._config_panel.execute_action()
        if not success:
            return None

        # Result was captured by callback
        return self._last_config_result

    def render_content(self) -> pn.Column:
        """Render configuration panel."""
        return self._panel_container

    def on_enter(self, input_data: PlotterSelection) -> None:
        """Create or recreate configuration panel when selection changes."""
        self._plotter_selection = input_data

        # Check if the configuration has changed
        if (
            input_data.workflow_id != self._last_workflow_id
            or input_data.output_name != self._last_output
            or input_data.plot_name != self._last_plot_name
        ):
            # Recreate panel with new configuration
            self._create_config_panel()
            # Track new values
            self._last_workflow_id = input_data.workflow_id
            self._last_output = input_data.output_name
            self._last_plot_name = input_data.plot_name

    def _create_config_panel(self) -> None:
        """Create the configuration panel for the selected plotter."""
        if self._plotter_selection is None:
            return

        workflow_spec = self._workflow_registry.get(self._plotter_selection.workflow_id)
        if workflow_spec is None:
            self._show_error('Workflow spec not found')
            return

        if not workflow_spec.source_names:
            self._show_error('No sources available for workflow')
            return

        try:
            plot_spec = self._plotting_controller.get_spec(
                self._plotter_selection.plot_name
            )
        except Exception as e:
            self._logger.exception("Error getting plot spec")
            self._show_error(f'Error getting plot spec: {e}')
            return

        # Create config_state from initial_config if in edit mode
        config_state = None
        if self._initial_config is not None:
            # Import ConfigurationState here to avoid circular imports
            from ess.livedata.dashboard.configuration_adapter import ConfigurationState

            config_state = ConfigurationState(
                source_names=self._initial_config.source_names,
                params=(
                    self._initial_config.params.model_dump()
                    if isinstance(self._initial_config.params, pydantic.BaseModel)
                    else self._initial_config.params
                ),
            )

        config_adapter = PlotConfigurationAdapter(
            plot_spec=plot_spec,
            source_names=workflow_spec.source_names,
            success_callback=self._on_config_collected,
            config_state=config_state,
        )

        self._config_panel = ConfigurationPanel(config=config_adapter)

        self._panel_container.clear()
        self._panel_container.append(self._config_panel.panel)

    def _on_config_collected(
        self, selected_sources: list[str], params: pydantic.BaseModel | dict[str, Any]
    ) -> None:
        """Callback from adapter - store result for commit() to return."""
        if self._plotter_selection is None:
            return
        self._last_config_result = PlotConfig(
            workflow_id=self._plotter_selection.workflow_id,
            output_name=self._plotter_selection.output_name,
            plot_name=self._plotter_selection.plot_name,
            source_names=selected_sources,
            params=params,
        )

    def _show_error(self, message: str) -> None:
        """Display an error notification."""
        if pn.state.notifications is not None:
            pn.state.notifications.error(message, duration=3000)


class PlotConfigModal:
    """
    Three-step wizard modal for configuring plots without existing data.

    This modal guides the user through:
    1. Workflow and output selection (combined for convenient UX)
    2. Plotter type selection based on output metadata
    3. Configure plotter (source selection and parameters)

    The configuration is created using workflow output metadata (dims, coords),
    enabling plotter selection before data is available. This makes it suitable
    for template-based plot grid configuration.

    When initial_config is provided (edit mode), the wizard starts at the last
    step (configuration) with pre-populated values, but users can navigate back
    to change workflow, output, or plotter type.

    Parameters
    ----------
    workflow_registry
        Registry of available workflows and their specifications.
    plotting_controller
        Controller for determining available plotters from specs.
    success_callback
        Called with PlotConfig when user completes configuration.
    cancel_callback
        Called when modal is closed or cancelled.
    initial_config
        Optional existing configuration for edit mode. When provided, the wizard
        starts at the configuration step with pre-filled values.
    """

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        plotting_controller,
        success_callback: Callable[[PlotConfig], None],
        cancel_callback: Callable[[], None],
        initial_config: PlotConfig | None = None,
    ) -> None:
        self._success_callback = success_callback
        self._cancel_callback = cancel_callback
        self._logger = logging.getLogger(__name__)
        self._initial_config = initial_config

        # Create steps
        step1 = WorkflowAndOutputSelectionStep(
            workflow_registry=workflow_registry, logger=self._logger
        )
        step2 = PlotterSelectionStep(
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
            logger=self._logger,
        )
        step3 = SpecBasedConfigurationStep(
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
            logger=self._logger,
            initial_config=initial_config,
        )

        # Create wizard
        action_label = "Update Plot" if initial_config else "Add Plot"
        self._wizard = Wizard(
            steps=[step1, step2, step3],
            on_complete=self._on_wizard_complete,
            on_cancel=self._on_wizard_cancel,
            action_button_label=action_label,
        )

        # Create modal wrapping the wizard
        modal_title = "Reconfigure Plot" if initial_config else "Configure Plot"
        self._modal = pn.Modal(
            self._wizard.render(),
            name=modal_title,
            margin=20,
            width=700,
            height=600,
        )

        # Watch for modal close events
        self._modal.param.watch(self._on_modal_closed, 'open')

    def _on_wizard_complete(self, result: PlotConfig) -> None:
        """Handle wizard completion - close modal and call success callback."""
        self._modal.open = False
        self._success_callback(result)

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
        if self._initial_config is not None:
            # Edit mode: convert config to step results and start at last step
            step_results = self._config_to_step_results(self._initial_config)
            # Start at step 2 (0-indexed), which is the last step (configuration)
            self._wizard.reset_to_step(2, step_results)
        else:
            # Add mode: reset to first step
            self._wizard.reset()
        self._modal.open = True

    def _config_to_step_results(self, config: PlotConfig) -> list[Any]:
        """
        Convert PlotConfig to step results for wizard initialization.

        Parameters
        ----------
        config
            The configuration to convert.

        Returns
        -------
        :
            List of step results [OutputSelection, PlotterSelection]
        """
        # Step 1 result: OutputSelection
        output_selection = OutputSelection(
            workflow_id=config.workflow_id,
            output_name=config.output_name,
        )

        # Step 2 result: PlotterSelection
        plotter_selection = PlotterSelection(
            workflow_id=config.workflow_id,
            output_name=config.output_name,
            plot_name=config.plot_name,
        )

        return [output_selection, plotter_selection]

    @property
    def modal(self) -> pn.Modal:
        """Get the modal widget."""
        return self._modal
