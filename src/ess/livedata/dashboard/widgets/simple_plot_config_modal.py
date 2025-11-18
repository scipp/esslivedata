# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Simplified plot configuration modal for PlotOrchestrator-based workflow.

This modal provides a 3-step wizard for configuring plots without requiring
existing data:
1. Select workflow from available workflow specs
2. Select output name from workflow outputs
3. Multi-select source names from workflow sources
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import panel as pn

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec

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
class PlotConfigResult:
    """Final result from the modal."""

    workflow_id: WorkflowId
    output_name: str
    source_names: list[str]


class WorkflowSelectionStep(WizardStep[None, WorkflowSelection]):
    """Step 1: Select workflow from available workflows."""

    def __init__(self, workflow_registry: Mapping[WorkflowId, WorkflowSpec]) -> None:
        """
        Initialize workflow selection step.

        Parameters
        ----------
        workflow_registry
            Registry of available workflows and their specifications.
        """
        super().__init__()
        self._workflow_registry = dict(workflow_registry)
        self._selected_workflow_id: WorkflowId | None = None

        # Create workflow selector
        self._workflow_selector = self._create_workflow_selector()

    @property
    def name(self) -> str:
        """Display name for this step."""
        return "Select Workflow"

    @property
    def description(self) -> str | None:
        """Description text for this step."""
        return "Choose the workflow you want to visualize."

    def _create_workflow_selector(self) -> pn.widgets.Select:
        """Create workflow selection dropdown."""
        # Sort workflows by title for better UX
        sorted_workflows = sorted(
            self._workflow_registry.items(), key=lambda item: item[1].title
        )

        # Create options mapping title -> workflow_id
        options = {spec.title: wid for wid, spec in sorted_workflows}

        selector = pn.widgets.Select(
            name='Workflow',
            options=options,
            sizing_mode='stretch_width',
        )

        # Watch for selection changes
        selector.param.watch(self._on_selection_change, 'value')

        # Initialize with first selection if available
        if options:
            selector.value = next(iter(options.values()))
            self._selected_workflow_id = selector.value
            self._notify_ready_changed(True)

        return selector

    def _on_selection_change(self, event) -> None:
        """Handle workflow selection change."""
        if event.new is not None:
            self._selected_workflow_id = event.new
            self._notify_ready_changed(True)
        else:
            self._selected_workflow_id = None
            self._notify_ready_changed(False)

    def is_valid(self) -> bool:
        """Whether a valid workflow has been selected."""
        return self._selected_workflow_id is not None

    def commit(self) -> WorkflowSelection | None:
        """Commit the selected workflow."""
        if self._selected_workflow_id is None:
            return None
        return WorkflowSelection(workflow_id=self._selected_workflow_id)

    def render_content(self) -> pn.Column:
        """Render workflow selector."""
        return pn.Column(
            self._workflow_selector,
            sizing_mode='stretch_width',
        )

    def on_enter(self, input_data: None) -> None:
        """Called when step becomes active."""
        pass


class OutputSelectionStep(WizardStep[WorkflowSelection, OutputSelection]):
    """Step 2: Select output name from workflow outputs."""

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        logger: logging.Logger,
    ) -> None:
        """
        Initialize output selection step.

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
        self._workflow_selection: WorkflowSelection | None = None
        self._selected_output: str | None = None
        self._output_selector: pn.widgets.Select | None = None
        self._content_container = pn.Column(sizing_mode='stretch_width')

    @property
    def name(self) -> str:
        """Display name for this step."""
        return "Select Output"

    @property
    def description(self) -> str | None:
        """Description text for this step."""
        return "Choose which workflow output to visualize."

    def is_valid(self) -> bool:
        """Step is valid when an output has been selected."""
        return self._selected_output is not None

    def commit(self) -> OutputSelection | None:
        """Commit the workflow and output selection."""
        if self._workflow_selection is None or self._selected_output is None:
            return None
        return OutputSelection(
            workflow_id=self._workflow_selection.workflow_id,
            output_name=self._selected_output,
        )

    def render_content(self) -> pn.Column:
        """Render output selector."""
        return self._content_container

    def on_enter(self, input_data: WorkflowSelection) -> None:
        """Update available outputs when step becomes active."""
        self._workflow_selection = input_data
        self._update_output_selection()

    def _update_output_selection(self) -> None:
        """Update output selection based on workflow selection."""
        self._content_container.clear()

        if self._workflow_selection is None:
            self._content_container.append(pn.pane.Markdown("*No workflow selected*"))
            self._output_selector = None
            self._notify_ready_changed(False)
            return

        workflow_spec = self._workflow_registry.get(
            self._workflow_selection.workflow_id
        )
        if workflow_spec is None or workflow_spec.outputs is None:
            self._content_container.append(
                pn.pane.Markdown("*No outputs available for this workflow*")
            )
            self._output_selector = None
            self._notify_ready_changed(False)
            return

        # Extract output names from the Pydantic model
        output_fields = workflow_spec.outputs.model_fields
        if not output_fields:
            self._content_container.append(
                pn.pane.Markdown("*No outputs defined for this workflow*")
            )
            self._output_selector = None
            self._notify_ready_changed(False)
            return

        # Create options mapping from output title to output name
        options = {}
        for field_name, field_info in output_fields.items():
            title = field_info.title if field_info.title else field_name
            options[title] = field_name

        self._output_selector = pn.widgets.Select(
            name='Output',
            options=options,
            sizing_mode='stretch_width',
        )

        # Watch for selection changes
        self._output_selector.param.watch(self._on_output_change, 'value')

        # Initialize with first selection
        if options:
            self._output_selector.value = next(iter(options.values()))
            self._selected_output = self._output_selector.value
            self._notify_ready_changed(True)

        self._content_container.append(self._output_selector)

    def _on_output_change(self, event) -> None:
        """Handle output selection change."""
        if event.new is not None:
            self._selected_output = event.new
            self._notify_ready_changed(True)
        else:
            self._selected_output = None
            self._notify_ready_changed(False)


class SourceSelectionStep(WizardStep[OutputSelection, PlotConfigResult]):
    """Step 3: Multi-select source names from workflow sources."""

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        logger: logging.Logger,
    ) -> None:
        """
        Initialize source selection step.

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
        self._output_selection: OutputSelection | None = None
        self._selected_sources: list[str] = []
        self._source_selector: pn.widgets.MultiSelect | None = None
        self._content_container = pn.Column(sizing_mode='stretch_width')

    @property
    def name(self) -> str:
        """Display name for this step."""
        return "Select Sources"

    @property
    def description(self) -> str | None:
        """Description text for this step."""
        return "Choose which data sources to include in the plot."

    def is_valid(self) -> bool:
        """Step is valid when at least one source has been selected."""
        return len(self._selected_sources) > 0

    def commit(self) -> PlotConfigResult | None:
        """Commit the complete configuration."""
        if self._output_selection is None or not self._selected_sources:
            return None
        return PlotConfigResult(
            workflow_id=self._output_selection.workflow_id,
            output_name=self._output_selection.output_name,
            source_names=self._selected_sources,
        )

    def render_content(self) -> pn.Column:
        """Render source selector."""
        return self._content_container

    def on_enter(self, input_data: OutputSelection) -> None:
        """Update available sources when step becomes active."""
        self._output_selection = input_data
        self._update_source_selection()

    def _update_source_selection(self) -> None:
        """Update source selection based on workflow selection."""
        self._content_container.clear()

        if self._output_selection is None:
            self._content_container.append(pn.pane.Markdown("*No output selected*"))
            self._source_selector = None
            self._notify_ready_changed(False)
            return

        workflow_spec = self._workflow_registry.get(self._output_selection.workflow_id)
        if workflow_spec is None:
            self._content_container.append(
                pn.pane.Markdown("*Workflow spec not found*")
            )
            self._source_selector = None
            self._notify_ready_changed(False)
            return

        if not workflow_spec.source_names:
            self._content_container.append(
                pn.pane.Markdown("*No sources available for this workflow*")
            )
            self._source_selector = None
            self._notify_ready_changed(False)
            return

        # Create multi-select widget with sorted source names
        self._source_selector = pn.widgets.MultiSelect(
            name='Sources',
            options=sorted(workflow_spec.source_names),
            size=min(10, len(workflow_spec.source_names)),
            sizing_mode='stretch_width',
        )

        # Watch for selection changes
        self._source_selector.param.watch(self._on_source_change, 'value')

        # Initialize with first source selected
        if workflow_spec.source_names:
            self._source_selector.value = [workflow_spec.source_names[0]]
            self._selected_sources = [workflow_spec.source_names[0]]
            self._notify_ready_changed(True)

        self._content_container.append(self._source_selector)

    def _on_source_change(self, event) -> None:
        """Handle source selection change."""
        if event.new and len(event.new) > 0:
            self._selected_sources = list(event.new)
            self._notify_ready_changed(True)
        else:
            self._selected_sources = []
            self._notify_ready_changed(False)


class SimplePlotConfigModal:
    """
    Simplified three-step wizard modal for configuring plots.

    This modal guides the user through:
    1. Workflow selection from available workflow specs
    2. Output name selection from workflow outputs
    3. Source name multi-selection from workflow sources

    The configuration is created without requiring existing data, making it
    suitable for template-based plot grid configuration.

    Parameters
    ----------
    workflow_registry
        Registry of available workflows and their specifications.
    success_callback
        Called with PlotConfigResult when user completes configuration.
    cancel_callback
        Called when modal is closed or cancelled.
    """

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        success_callback: Callable[[PlotConfigResult], None],
        cancel_callback: Callable[[], None],
    ) -> None:
        self._success_callback = success_callback
        self._cancel_callback = cancel_callback
        self._logger = logging.getLogger(__name__)

        # Create steps
        step1 = WorkflowSelectionStep(workflow_registry=workflow_registry)
        step2 = OutputSelectionStep(
            workflow_registry=workflow_registry, logger=self._logger
        )
        step3 = SourceSelectionStep(
            workflow_registry=workflow_registry, logger=self._logger
        )

        # Create wizard
        self._wizard = Wizard(
            steps=[step1, step2, step3],
            on_complete=self._on_wizard_complete,
            on_cancel=self._on_wizard_cancel,
            action_button_label="Add Plot",
        )

        # Create modal wrapping the wizard
        self._modal = pn.Modal(
            self._wizard.render(),
            name="Configure Plot",
            margin=20,
            width=700,
            height=600,
        )

        # Watch for modal close events
        self._modal.param.watch(self._on_modal_closed, 'open')

    def _on_wizard_complete(self, result: PlotConfigResult) -> None:
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
        # Reset wizard and show modal
        self._wizard.reset()
        self._modal.open = True

    @property
    def modal(self) -> pn.Modal:
        """Get the modal widget."""
        return self._modal
