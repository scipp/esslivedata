# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Widget and subwidgets for configuring, running, and stopping data reduction workflows.

Given a :py:class:`~ess.livedata.config.workflow_spec.WorkflowSpec`, this module
provides a Panel widget that allows users to configure, run, and stop data reduction
workflows. Concretely we have:

- A list-like selection widget of a workflow. Also displays a description of each
  workflow.
- A subwidget that is dynamically generated based on the selected workflow, containing
  1. A selection widget that allows for selecting multiple source names to apply the
     workflow to.
  2. A subwidget for each parameter of the workflow, allowing users to configure
     parameters of the workflow. Based on
     :py:class:`~ess.livedata.config.workflow_spec.Parameter`. If available,
     the initial value of the parameters is configured from a
     :py:class:`~ess.livedata.config.workflow_spec.WorkflowConfig`, otherwise the
     default value from the parameter is used.
- A list widget displaying running workflows, allowing users to stop them.
"""

from __future__ import annotations

import panel as pn

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec
from ess.livedata.dashboard.workflow_controller import (
    BoundWorkflowController,
    WorkflowController,
)

from .configuration_widget import ConfigurationModal
from .workflow_config_modal import WorkflowConfigurationAdapter


class WorkflowSelectorWidget:
    """Widget for selecting workflows from available specifications."""

    _no_selection = object()

    def __init__(self, controller: WorkflowController) -> None:
        """
        Initialize workflow selector.

        Parameters
        ----------
        controller
            Controller for workflow operations
        """
        self._controller = controller
        self._bound_controller: BoundWorkflowController | None = None
        self._selector = pn.widgets.Select(name="Workflow")
        self._description_pane = pn.pane.HTML(
            "Select a workflow to see its description"
        )
        self._widget = self._create_widget()
        self._setup_callbacks()
        self._on_workflows_updated(self._controller.get_workflow_specs())

    @classmethod
    def _make_workflow_options(
        cls, specs: dict[WorkflowId, WorkflowSpec] | None = None
    ) -> dict[str, WorkflowId | object]:
        """Get workflow options for selector widget."""
        specs = specs or {}
        select_text = "--- Click to select a workflow ---"
        options = {select_text: cls._no_selection}
        options.update({spec.title: workflow_id for workflow_id, spec in specs.items()})
        return options

    @classmethod
    def _is_no_selection(cls, value: WorkflowId | object) -> bool:
        """Check if the given value represents no workflow selection."""
        return value is cls._no_selection

    @classmethod
    def _get_default_workflow_selection(cls) -> object:
        """Get the default value for no workflow selection."""
        return cls._no_selection

    def _create_widget(self) -> pn.Column:
        """Create the main selector widget."""
        return pn.Column(self._selector, self._description_pane)

    def _setup_callbacks(self) -> None:
        """Setup callbacks for widget interactions."""
        self._selector.param.watch(self._on_workflow_selected, "value")

    def _on_workflow_selected(self, event) -> None:
        """Handle workflow selection change."""
        workflow_id = event.new

        # Create bound controller and UI helper for selected workflow
        if self._is_no_selection(workflow_id):
            self._bound_controller = None
            text = "Select a workflow to see its description"
        else:
            self._bound_controller = self._controller.get_bound_controller(workflow_id)
            if self._bound_controller is not None:
                description = self._bound_controller.spec.description
                text = f"<p><strong>Description:</strong> {description}</p>"
            else:
                text = "Select a workflow to see its description"

        self._description_pane.object = text

    def _on_workflows_updated(self, specs: dict[WorkflowId, WorkflowSpec]) -> None:
        """Handle workflow specs updates."""
        self._selector.options = self._make_workflow_options(specs)
        self._selector.value = self._get_default_workflow_selection()

    @property
    def widget(self) -> pn.Column:
        """Get the Panel widget."""
        return self._widget

    @property
    def selected_workflow_id(self) -> WorkflowId | None:
        """Get the currently selected workflow ID."""
        value = self._selector.value
        return None if self._is_no_selection(value) else value

    def create_modal(self) -> ConfigurationModal | None:
        if self._bound_controller is None:
            return None

        adapter = WorkflowConfigurationAdapter(self._bound_controller)
        return ConfigurationModal(config=adapter, start_button_text="Start Workflow")


class ReductionWidget:
    """Main widget for data reduction workflow configuration and control."""

    def __init__(self, controller: WorkflowController) -> None:
        """
        Initialize reduction widget.

        Parameters
        ----------
        controller
            Controller for workflow operations
        """
        self._controller = controller
        self._workflow_selector = WorkflowSelectorWidget(controller)
        self._configure_button = pn.widgets.Button(
            name="Configure & Start", button_type="primary", disabled=True
        )
        # Container for modals - they need to be part of the served structure
        self._modal_container = pn.Column()
        self._widget = self._create_widget()
        self._setup_callbacks()

    def _create_widget(self) -> pn.Column:
        """Create the main widget layout."""
        return pn.Column(
            pn.Column(
                self._workflow_selector.widget,
                self._configure_button,
                sizing_mode="stretch_width",
            ),
            self._modal_container,  # Add modal container to main structure
        )

    def _setup_callbacks(self) -> None:
        """Setup callbacks for widget interactions."""
        self._workflow_selector._selector.param.watch(
            self._on_workflow_selected, "value"
        )
        self._configure_button.on_click(self._on_configure_workflow)

    def _on_workflow_selected(self, event) -> None:
        """Handle workflow selection change."""
        workflow_id = event.new
        self._configure_button.disabled = WorkflowSelectorWidget._is_no_selection(
            workflow_id
        )

    def _on_configure_workflow(self, event) -> None:
        """Handle configure workflow button click."""
        if (modal := self._workflow_selector.create_modal()) is None:
            return

        # Clear previous modals and add the new one
        self._modal_container.clear()
        self._modal_container.append(modal.modal)
        modal.show()

    @property
    def widget(self) -> pn.Column:
        """Get the Panel widget."""
        return self._widget
