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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import panel as pn
import pydantic
import structlog

from ess.livedata.config.workflow_spec import (
    WorkflowId,
    WorkflowSpec,
    find_timeseries_outputs,
)
from ess.livedata.dashboard.correlation_plotter import CORRELATION_HISTOGRAM_PLOTTERS
from ess.livedata.dashboard.data_roles import PRIMARY, X_AXIS, Y_AXIS

if TYPE_CHECKING:
    from ess.livedata.config import Instrument
from ess.livedata.dashboard.notifications import show_error
from ess.livedata.dashboard.plot_configuration_adapter import PlotConfigurationAdapter
from ess.livedata.dashboard.plot_orchestrator import (
    DataSourceConfig,
    PlotConfig,
)
from ess.livedata.dashboard.plotter_registry import PlotterSpec

from .configuration_widget import ConfigurationPanel
from .styles import Colors, ModalSizing
from .wizard import Wizard, WizardStep

logger = structlog.get_logger(__name__)

# Synthetic workflow ID for static overlays (no actual workflow subscription)
STATIC_OVERLAY_GROUP = "static_overlay"
STATIC_OVERLAY_WORKFLOW = WorkflowId(
    instrument="static",
    name="geometric",
    version=1,
)
STATIC_OVERLAY_OUTPUT = "static"

# CSS to disable button transition animations for snappier UI response
_NO_TRANSITION_CSS = """
.bk-btn {
    transition: none !important;
}
"""


def _resolve_axis_source_titles(
    axis_sources: dict[str, DataSourceConfig],
    instrument_config: Instrument,
    workflow_registry: Mapping[WorkflowId, WorkflowSpec] | None = None,
) -> dict[str, str]:
    """Build a mapping from bins field name to display title for axis sources.

    Parameters
    ----------
    axis_sources:
        Axis role -> DataSourceConfig mapping.
    instrument_config:
        Instrument config to resolve source names to titles.
    workflow_registry:
        Optional registry used to look up output titles. When provided and the
        workflow has multiple outputs, the output title is appended to the source
        title to disambiguate (e.g. "Cave Monitor (wavelength)").

    Returns
    -------
    :
        Mapping of bins field name (e.g. 'x_axis_source') to display title.
    """
    role_to_field = {X_AXIS: 'x_axis_source', Y_AXIS: 'y_axis_source'}
    result: dict[str, str] = {}
    for role, field_name in role_to_field.items():
        if role in axis_sources and axis_sources[role].source_names:
            ds = axis_sources[role]
            source_name = ds.source_names[0]
            title = instrument_config.get_source_title(source_name)
            if workflow_registry is not None:
                spec = workflow_registry.get(ds.workflow_id)
                if (
                    spec is not None
                    and spec.outputs is not None
                    and len(spec.outputs.model_fields) > 1
                ):
                    output_title = spec.get_output_title(ds.output_name)
                    title = f"{title} ({output_title})"
            result[field_name] = title
    return result


@dataclass(frozen=True)
class OutputDisplayHints:
    """UI hints derived from inspecting a workflow output template.

    Consolidates all output-template-dependent decisions for the plot
    configuration modal so that the template is inspected in one place.
    """

    hidden_fields: frozenset[str]
    preselect_all_sources: bool


def _resolve_output_display_hints(
    *,
    is_static: bool,
    workflow_spec: WorkflowSpec | None,
    output_name: str,
) -> OutputDisplayHints:
    """Derive UI hints from the workflow output template.

    Inspects the output template once and returns all decisions that depend
    on its properties (dimensionality, coordinates, etc.).

    Parameters
    ----------
    is_static:
        Whether the plot uses a static overlay workflow.
    workflow_spec:
        Workflow specification for the selected output. May be ``None``
        for static overlays.
    output_name:
        Name of the output field.

    Returns
    -------
    :
        Display hints for the configuration UI.
    """
    if is_static:
        return OutputDisplayHints(hidden_fields=frozenset(), preselect_all_sources=True)
    from ess.livedata.dashboard.plotting_controller import output_has_time_coord

    supports_windowing = output_has_time_coord(workflow_spec, output_name)
    hidden = frozenset({'window'}) if not supports_windowing else frozenset()

    template = (
        workflow_spec.get_output_template(output_name)
        if workflow_spec is not None
        else None
    )
    preselect_all = template is None or template.ndim < 2

    return OutputDisplayHints(hidden_fields=hidden, preselect_all_sources=preselect_all)


def _inject_axis_source_titles(
    params: pydantic.BaseModel | dict,
    axis_sources: dict[str, DataSourceConfig],
    instrument_config: Instrument,
    workflow_registry: Mapping[WorkflowId, WorkflowSpec] | None = None,
) -> pydantic.BaseModel | dict:
    """Inject axis source titles into correlation histogram params.

    Updates the bins.x_axis_source and bins.y_axis_source fields with
    human-readable titles resolved from axis_sources.
    Accepts either a pydantic model or a plain dict of params.
    """
    titles = _resolve_axis_source_titles(
        axis_sources, instrument_config, workflow_registry
    )
    if not titles:
        return params

    if isinstance(params, dict):
        if 'bins' in params and isinstance(params['bins'], dict):
            params['bins'].update(titles)
        return params

    if not hasattr(params, 'bins'):
        return params
    new_bins = params.bins.model_copy(update=titles)
    return params.model_copy(update={'bins': new_bins})


def _build_timeseries_options(
    available_timeseries: list[tuple[WorkflowId, str, str]],
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
    instrument_config: Instrument | None,
) -> dict[str, tuple[WorkflowId, str, str]]:
    """Build display-name to timeseries-identity mapping for axis dropdowns.

    Parameters
    ----------
    available_timeseries:
        (workflow_id, source_name, output_name) tuples.
    workflow_registry:
        Registry to look up workflow specs.
    instrument_config:
        Instrument config for resolving source titles.

    Returns
    -------
    :
        Mapping of display name to (workflow_id, source_name, output_name).
    """
    options: dict[str, tuple[WorkflowId, str, str]] = {}
    for workflow_id, source_name, output_name in available_timeseries:
        spec = workflow_registry.get(workflow_id)
        workflow_title = spec.title if spec else workflow_id.name
        if instrument_config is not None:
            source_title = instrument_config.get_source_title(source_name)
        else:
            source_title = source_name
        display_name = f"{workflow_title}: {source_title}"
        # Append output name only when the workflow has multiple outputs,
        # to disambiguate which output is being selected.
        has_multiple_outputs = (
            spec is not None
            and spec.outputs is not None
            and len(spec.outputs.model_fields) > 1
        )
        if has_multiple_outputs:
            output_field = spec.outputs.model_fields.get(output_name) if spec else None
            output_title = (
                output_field.title
                if output_field is not None and output_field.title
                else output_name
            )
            display_name = f"{display_name} ({output_title})"
        options[display_name] = (workflow_id, source_name, output_name)
    return options


def _find_initial_axis_value(
    role: str,
    initial_axis_sources: dict[str, DataSourceConfig],
    options: dict[str, tuple[WorkflowId, str, str]],
) -> tuple[WorkflowId, str, str] | None:
    """Find the dropdown option matching the initial config for a given axis role."""
    if role not in initial_axis_sources:
        return None
    initial_ds = initial_axis_sources[role]
    for wf_id, src_name, out_name in options.values():
        if (
            wf_id == initial_ds.workflow_id
            and src_name in initial_ds.source_names
            and out_name == initial_ds.output_name
        ):
            return (wf_id, src_name, out_name)
    return None


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
    axis_sources: dict[str, DataSourceConfig] | None = None


class WorkflowOutputPrefill(Protocol):
    """Minimal prefill protocol for workflow + output selection."""

    @property
    def workflow_id(self) -> WorkflowId: ...

    @property
    def output_name(self) -> str: ...


class WorkflowAndOutputSelectionStep(WizardStep[None, OutputSelection]):
    """Step 1: Select workflow and output (combined for better UX)."""

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        initial_config: WorkflowOutputPrefill | None = None,
        *,
        include_static_overlay: bool = True,
    ) -> None:
        """
        Initialize workflow and output selection step.

        Parameters
        ----------
        workflow_registry
            Registry of available workflows and their specifications.
        initial_config
            Optional initial selection (anything exposing ``workflow_id`` and
            ``output_name``). For plot-edit mode this is a ``PlotConfig``.
        include_static_overlay
            When False, the synthetic Static Overlay namespace is hidden.
        """
        super().__init__()
        self._workflow_registry = dict(workflow_registry)
        self._initial_config = initial_config
        self._include_static_overlay = include_static_overlay
        self._selected_group: str | None = None
        self._selected_workflow_id: WorkflowId | None = None
        self._selected_output: str | None = None

        # Initialize containers
        self._content_container = pn.Column(sizing_mode='stretch_width')
        self._workflow_container = pn.Column(sizing_mode='stretch_width')
        self._output_container = pn.Column(sizing_mode='stretch_width')

        # Create all button widgets once (reuse by updating options)
        self._group_buttons = self._create_group_buttons()
        self._workflow_buttons = self._create_workflow_buttons()
        self._output_buttons = self._create_output_buttons()
        self._workflow_description = pn.pane.HTML(
            '',
            sizing_mode='stretch_width',
            styles={'font-size': '12px', 'color': Colors.TEXT_MUTED},
            visible=False,
        )
        self._output_description = pn.pane.HTML(
            '',
            sizing_mode='stretch_width',
            styles={'font-size': '12px', 'color': Colors.TEXT_MUTED},
            visible=False,
        )
        self._name_input = self._create_name_input()

        # Add buttons to containers
        self._workflow_container.append(self._workflow_buttons)
        self._workflow_container.append(self._workflow_description)
        self._output_container.append(self._output_buttons)
        self._output_container.append(self._output_description)

        # Build layout
        self._update_content()

        # Initialize selections
        self._initialize_selections(initial_config)

    @property
    def name(self) -> str:
        """Display name for this step."""
        return "Select Workflow & Output"

    @property
    def description(self) -> str | None:
        """Description text for this step."""
        return "Choose the workflow and output to visualize."

    def _group_for_workflow(self, workflow_id: WorkflowId) -> str | None:
        """Return the group identifier for a workflow id, or None if unknown."""
        if workflow_id == STATIC_OVERLAY_WORKFLOW:
            return STATIC_OVERLAY_GROUP
        spec = self._workflow_registry.get(workflow_id)
        return spec.group.name if spec is not None else None

    def _initialize_selections(
        self, initial_config: WorkflowOutputPrefill | None
    ) -> None:
        """Initialize selections and bind handlers afterward to avoid cascades."""
        # Batch all widget updates to avoid multiple browser render cycles
        with pn.io.hold():
            if initial_config is not None:
                # Edit mode: pre-select from config
                group_value = self._group_for_workflow(initial_config.workflow_id)
                self._selected_group = group_value
                self._group_buttons.value = group_value
                self._update_workflow_options()
                self._selected_workflow_id = initial_config.workflow_id
                self._workflow_buttons.value = initial_config.workflow_id
                self._update_output_options()
                self._selected_output = initial_config.output_name
                # Static overlay branch only applies when the workflow is the
                # synthetic static-overlay workflow.
                if initial_config.workflow_id == STATIC_OVERLAY_WORKFLOW:
                    self._name_input.value = initial_config.output_name
                else:
                    self._output_buttons.value = initial_config.output_name
            elif self._group_buttons.options:
                # New mode: select first available option
                group_value = next(iter(self._group_buttons.options.values()))
                self._selected_group = group_value
                self._group_buttons.value = group_value
                self._update_workflow_options()
                if self._workflow_buttons.options:
                    workflow_value = next(iter(self._workflow_buttons.options.values()))
                    self._selected_workflow_id = workflow_value
                    self._workflow_buttons.value = workflow_value
                    self._update_output_options()
                    # For static overlay, don't auto-select (require user input)
                    if self._selected_workflow_id != STATIC_OVERLAY_WORKFLOW:
                        if self._output_buttons.options:
                            output_value = next(
                                iter(self._output_buttons.options.values())
                            )
                            self._selected_output = output_value
                            self._output_buttons.value = output_value

        # Bind handlers after initial values are set
        self._group_buttons.param.watch(self._on_group_change, 'value')
        self._workflow_buttons.param.watch(self._on_workflow_change, 'value')
        self._output_buttons.param.watch(self._on_output_change, 'value')
        self._name_input.param.watch(self._on_name_input_change, 'value')
        self._update_workflow_description()
        self._update_output_description()
        self._validate()

    def _create_group_buttons(self) -> pn.widgets.RadioButtonGroup:
        """Create workflow-group selection radio buttons (handler bound later)."""
        # Deduplicate by group identifier; keep first-seen title for display.
        groups: dict[str, str] = {}
        for spec in self._workflow_registry.values():
            groups.setdefault(spec.group.name, spec.group.title)
        options = {
            title: name
            for name, title in sorted(groups.items(), key=lambda item: item[0])
        }
        # Add synthetic "Static Overlay" group for geometric overlays
        if self._include_static_overlay:
            options["Static Overlay"] = STATIC_OVERLAY_GROUP

        return pn.widgets.RadioButtonGroup(
            name='Group',
            options=options,
            orientation='horizontal',
            button_type='primary',
            button_style='outline',
            sizing_mode='stretch_width',
            stylesheets=[_NO_TRANSITION_CSS],
        )

    def _create_workflow_buttons(self) -> pn.widgets.RadioButtonGroup:
        """Create workflow selection radio buttons (handler bound later)."""
        return pn.widgets.RadioButtonGroup(
            name='Workflow',
            options={},
            orientation='vertical',
            button_type='primary',
            button_style='outline',
            sizing_mode='stretch_width',
            stylesheets=[_NO_TRANSITION_CSS],
        )

    def _create_output_buttons(self) -> pn.widgets.RadioButtonGroup:
        """Create output selection radio buttons (handler bound later)."""
        return pn.widgets.RadioButtonGroup(
            name='Output',
            options={},
            orientation='vertical',
            button_type='primary',
            button_style='outline',
            sizing_mode='stretch_width',
            stylesheets=[_NO_TRANSITION_CSS],
        )

    def _create_name_input(self) -> pn.widgets.TextInput:
        """Create text input for static overlay naming (handler bound later)."""
        return pn.widgets.TextInput(
            name='Overlay Name',
            placeholder='Enter a name for this overlay...',
            sizing_mode='stretch_width',
        )

    def _on_group_change(self, event) -> None:
        """Handle workflow-group selection change."""
        # Batch all cascading widget updates
        with pn.io.hold():
            if event.new is not None:
                self._selected_group = event.new
                self._selected_workflow_id = None
                self._selected_output = None
                self._update_workflow_options()
                # Select first workflow and trigger output update
                if self._workflow_buttons.options:
                    first_workflow = next(iter(self._workflow_buttons.options.values()))
                    self._selected_workflow_id = first_workflow
                    self._workflow_buttons.value = first_workflow
                    self._update_output_options()
                    # Skip auto-selection for static overlays (user must enter name)
                    if (
                        first_workflow != STATIC_OVERLAY_WORKFLOW
                        and self._output_buttons.options
                    ):
                        first_output = next(iter(self._output_buttons.options.values()))
                        self._selected_output = first_output
                        self._output_buttons.value = first_output
            else:
                self._selected_group = None
                self._selected_workflow_id = None
                self._selected_output = None
        self._update_workflow_description()
        self._validate()

    def _on_workflow_change(self, event) -> None:
        """Handle workflow selection change."""
        # Batch cascading widget updates
        with pn.io.hold():
            if event.new is not None:
                self._selected_workflow_id = event.new
                self._selected_output = None
                self._update_output_options()
                # Skip auto-selection for static overlays (user must enter name)
                if (
                    event.new != STATIC_OVERLAY_WORKFLOW
                    and self._output_buttons.options
                ):
                    first_output = next(iter(self._output_buttons.options.values()))
                    self._selected_output = first_output
                    self._output_buttons.value = first_output
            else:
                self._selected_workflow_id = None
                self._selected_output = None
        self._update_workflow_description()
        self._validate()

    def _on_output_change(self, event) -> None:
        """Handle output selection change."""
        if event.new is not None:
            self._selected_output = event.new
        else:
            self._selected_output = None
        self._update_output_description()
        self._validate()

    def _update_workflow_description(self) -> None:
        """Update the workflow description pane for the selected workflow."""
        desc = None
        if self._selected_workflow_id is not None:
            spec = self._workflow_registry.get(self._selected_workflow_id)
            if spec is not None:
                desc = spec.description
        if desc:
            self._workflow_description.object = desc
            self._workflow_description.visible = True
        else:
            self._workflow_description.object = ''
            self._workflow_description.visible = False

    def _update_output_description(self) -> None:
        """Update the output description pane for the selected output."""
        desc = None
        if self._selected_workflow_id is not None and self._selected_output is not None:
            spec = self._workflow_registry.get(self._selected_workflow_id)
            if spec is not None:
                desc = spec.get_output_description(self._selected_output)
        if desc:
            self._output_description.object = desc
            self._output_description.visible = True
        else:
            self._output_description.object = ''
            self._output_description.visible = False

    def _on_name_input_change(self, event) -> None:
        """Handle static overlay name input change."""
        name = event.new.strip() if event.new else ''
        # Store non-empty name, or None if empty/whitespace
        self._selected_output = name if name else None
        self._validate()

    def _update_workflow_options(self) -> None:
        """Update workflow button options based on selected group."""
        if self._selected_group is None:
            self._workflow_buttons.options = {}
            return

        # Handle synthetic static overlay group
        if self._selected_group == STATIC_OVERLAY_GROUP:
            self._workflow_buttons.options = {"Geometric": STATIC_OVERLAY_WORKFLOW}
            return

        filtered_workflows = [
            (wid, spec)
            for wid, spec in self._workflow_registry.items()
            if spec.group.name == self._selected_group
        ]

        if not filtered_workflows:
            self._workflow_buttons.options = {}
            return

        sorted_workflows = sorted(filtered_workflows, key=lambda item: item[1].title)
        options = {spec.title: wid for wid, spec in sorted_workflows}
        self._workflow_buttons.options = options

    def _update_output_options(self) -> None:
        """Update output button options based on selected workflow."""
        if self._selected_workflow_id is None:
            self._output_buttons.options = {}
            return

        # Handle synthetic static overlay workflow - show name input instead of buttons
        if self._selected_workflow_id == STATIC_OVERLAY_WORKFLOW:
            self._output_container.clear()
            self._name_input.value = ''  # Clear for new overlays
            self._output_container.append(self._name_input)
            return

        # For regular workflows, ensure output buttons are shown (in case we're
        # coming from static overlay which swapped in the name input)
        if self._name_input in self._output_container:
            self._output_container.clear()
            self._output_container.append(self._output_buttons)
            self._output_container.append(self._output_description)

        workflow_spec = self._workflow_registry.get(self._selected_workflow_id)
        if workflow_spec is None:
            self._output_buttons.options = {}
            return

        output_fields = workflow_spec.outputs.model_fields
        if not output_fields:
            self._output_buttons.options = {}
            return

        # Field definition order is intentional: spec authors put the primary output
        # first, as it will be auto-selected when the wizard opens.
        options = {workflow_spec.get_output_title(name): name for name in output_fields}
        self._output_buttons.options = options

    def _update_content(self) -> None:
        """Update content with group selector above workflow/output columns."""
        self._content_container.clear()

        # Group selector on top (horizontal)
        group_section = pn.Column(
            pn.pane.Markdown("**Group**"),
            self._group_buttons,
            sizing_mode='stretch_width',
        )

        # Workflow and output columns below
        workflow_col = pn.Column(
            pn.pane.Markdown("**Workflow**"),
            self._workflow_container,
            sizing_mode='stretch_both',
        )

        output_col = pn.Column(
            pn.pane.Markdown("**Output**"),
            self._output_container,
            sizing_mode='stretch_both',
        )

        two_columns = pn.Row(workflow_col, output_col, sizing_mode='stretch_width')

        self._content_container.append(group_section)
        self._content_container.append(two_columns)

    def _validate(self) -> None:
        """Update validity based on selections."""
        is_valid = (
            self._selected_group is not None
            and self._selected_workflow_id is not None
            and self._selected_output is not None
        )
        self._notify_ready_changed(is_valid)

    def is_valid(self) -> bool:
        """Whether group, workflow, and output have all been selected."""
        return (
            self._selected_group is not None
            and self._selected_workflow_id is not None
            and self._selected_output is not None
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


class PlotterSelectionStep(WizardStep[OutputSelection | None, PlotterSelection]):
    """Step 2: Select plotter type based on output metadata."""

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        plotting_controller,
        initial_config: PlotConfig | None = None,
        instrument_config: Instrument | None = None,
    ) -> None:
        """
        Initialize plotter selection step.

        Parameters
        ----------
        workflow_registry
            Registry of available workflows and their specifications.
        plotting_controller
            Controller for determining available plotters from specs.
        initial_config
            Optional initial configuration for edit mode.
        instrument_config
            Optional instrument configuration for source metadata lookup.
        """
        super().__init__()
        self._workflow_registry = dict(workflow_registry)
        self._plotting_controller = plotting_controller
        self._initial_config = initial_config
        self._instrument_config = instrument_config
        self._output_selection: OutputSelection | None = None
        self._selected_plot_name: str | None = None
        self._radio_group: pn.widgets.RadioButtonGroup | None = None
        self._content_container = pn.Column(sizing_mode='stretch_width')

        # Correlation axis selection state
        self._selected_axis_sources: dict[str, DataSourceConfig] = {}
        self._axis_selectors_container = pn.Column(sizing_mode='stretch_width')
        self._axis_selectors: dict[str, pn.widgets.Select] = {}
        self._available_timeseries: list[tuple[WorkflowId, str, str]] | None = None

    @property
    def name(self) -> str:
        """Display name for this step."""
        return "Select Plotter Type"

    @property
    def description(self) -> str | None:
        """Description text for this step."""
        return "Choose the type of plot you want to create."

    def is_valid(self) -> bool:
        """Step is valid when a plotter has been selected.

        For correlation histogram plotters, also requires axis selection.
        """
        if self._selected_plot_name is None:
            return False
        # For correlation histograms, require axis selection
        if self._selected_plot_name in CORRELATION_HISTOGRAM_PLOTTERS:
            required_roles = self._get_required_axis_roles()
            return all(role in self._selected_axis_sources for role in required_roles)
        return True

    def _get_required_axis_roles(self) -> list[str]:
        """Get the required axis roles for the selected plotter."""
        if self._selected_plot_name == 'correlation_histogram_1d':
            return [X_AXIS]
        elif self._selected_plot_name == 'correlation_histogram_2d':
            return [X_AXIS, Y_AXIS]
        return []

    def commit(self) -> PlotterSelection | None:
        """Commit the workflow, output, and selected plotter."""
        if self._output_selection is None or self._selected_plot_name is None:
            return None
        # Include axis sources for correlation histogram plotters
        axis_sources = (
            self._selected_axis_sources.copy()
            if self._selected_plot_name in CORRELATION_HISTOGRAM_PLOTTERS
            else None
        )
        return PlotterSelection(
            workflow_id=self._output_selection.workflow_id,
            output_name=self._output_selection.output_name,
            plot_name=self._selected_plot_name,
            axis_sources=axis_sources,
        )

    def render_content(self) -> pn.Column:
        """Render plotter selection radio buttons."""
        return self._content_container

    def on_enter(self, input_data: OutputSelection | None) -> None:
        """Update available plotters when step becomes active."""
        if input_data is not None:
            self._output_selection = input_data
        elif self._initial_config is not None:
            # Fall back to initial config when jumping to this step
            self._output_selection = OutputSelection(
                workflow_id=self._initial_config.workflow_id,
                output_name=self._initial_config.output_name,
            )
        self._update_plotter_selection()

    def _update_plotter_selection(self) -> None:
        """Update plotter selection based on workflow and output selection."""
        self._content_container.clear()

        if self._output_selection is None:
            self._content_container.append(pn.pane.Markdown("*No output selected*"))
            self._radio_group = None
            self._notify_ready_changed(False)
            return

        # Handle static overlay workflow - show static plotters
        if self._output_selection.workflow_id == STATIC_OVERLAY_WORKFLOW:
            available_plots = self._plotting_controller.get_static_plotters()
            if available_plots:
                self._create_radio_buttons(available_plots)
            else:
                self._content_container.append(
                    pn.pane.Markdown("*No static plotters available.*")
                )
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
                        "**Warning: Could not determine output properties. "
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

        # Determine initial value: use initial_config if available, else first option
        if (
            self._initial_config is not None
            and self._initial_config.plot_name in self._plot_name_map.values()
        ):
            initial_value = self._initial_config.plot_name
        else:
            initial_value = (
                next(iter(self._plot_name_map.values()))
                if self._plot_name_map
                else None
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
            # Show axis selection for correlation histogram plotters
            if initial_value in CORRELATION_HISTOGRAM_PLOTTERS:
                self._update_axis_selection()
            self._notify_ready_changed(self.is_valid())

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
            # Show axis selection for correlation histogram plotters
            if self._selected_plot_name in CORRELATION_HISTOGRAM_PLOTTERS:
                self._update_axis_selection()
            else:
                self._hide_axis_selection()
            self._notify_ready_changed(self.is_valid())
        else:
            self._selected_plot_name = None
            self._hide_axis_selection()
            self._notify_ready_changed(False)

    def _get_available_timeseries(self) -> list[tuple[WorkflowId, str, str]]:
        """Get available timeseries outputs from workflow registry (cached)."""
        if self._available_timeseries is None:
            self._available_timeseries = find_timeseries_outputs(
                self._workflow_registry
            )
        return self._available_timeseries

    def _update_axis_selection(self) -> None:
        """Create/update axis selection dropdowns for correlation histograms."""
        self._axis_selectors_container.clear()
        self._axis_selectors.clear()
        self._selected_axis_sources.clear()

        required_roles = self._get_required_axis_roles()
        if not required_roles:
            return

        available_timeseries = self._get_available_timeseries()
        if not available_timeseries:
            self._axis_selectors_container.append(
                pn.pane.Markdown("*No timeseries available for correlation.*")
            )
            return

        options = _build_timeseries_options(
            available_timeseries,
            self._workflow_registry,
            self._instrument_config,
        )

        # Map roles to display labels
        role_labels = {X_AXIS: 'X-Axis', Y_AXIS: 'Y-Axis'}
        if len(required_roles) == 1:
            role_labels[X_AXIS] = 'Correlation Axis'

        # Get initial axis sources from edit mode config (if any)
        initial_axis_sources: dict[str, DataSourceConfig] = {}
        if self._initial_config is not None:
            for role in [X_AXIS, Y_AXIS]:
                if role in self._initial_config.data_sources:
                    initial_axis_sources[role] = self._initial_config.data_sources[role]

        for role in required_roles:
            label = role_labels.get(role, role)
            initial_value = _find_initial_axis_value(
                role, initial_axis_sources, options
            )

            selector = pn.widgets.Select(
                name=f'{label} (correlate against)',
                options={'Select...': None, **options},
                value=initial_value,
                sizing_mode='stretch_width',
            )
            selector.param.watch(
                lambda event, r=role: self._on_axis_selection_change(event, r), 'value'
            )
            self._axis_selectors[role] = selector
            self._axis_selectors_container.append(selector)

            if initial_value is not None:
                workflow_id, source_name, output_name = initial_value
                self._selected_axis_sources[role] = DataSourceConfig(
                    workflow_id=workflow_id,
                    source_names=[source_name],
                    output_name=output_name,
                )

        # Add the container to the content if not already there
        if self._axis_selectors_container not in self._content_container:
            self._content_container.append(self._axis_selectors_container)

    def _hide_axis_selection(self) -> None:
        """Hide axis selection dropdowns."""
        self._axis_selectors_container.clear()
        self._axis_selectors.clear()
        self._selected_axis_sources.clear()

    def _on_axis_selection_change(self, event, role: str) -> None:
        """Handle axis selection change."""
        value = event.new
        if value is not None and isinstance(value, tuple):
            workflow_id, source_name, output_name = value
            self._selected_axis_sources[role] = DataSourceConfig(
                workflow_id=workflow_id,
                source_names=[source_name],
                output_name=output_name,
            )
        elif role in self._selected_axis_sources:
            del self._selected_axis_sources[role]
        self._notify_ready_changed(self.is_valid())


class SpecBasedConfigurationStep(WizardStep[PlotterSelection | None, PlotConfig]):
    """Step 3: Configure plot (source selection and plotter parameters)."""

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        plotting_controller,
        initial_config: PlotConfig | None = None,
        instrument_config: Instrument | None = None,
    ) -> None:
        """
        Initialize spec-based configuration step.

        Parameters
        ----------
        workflow_registry
            Registry of available workflows and their specifications.
        plotting_controller
            Controller for getting plotter specs.
        initial_config
            Optional initial configuration for edit mode.
        instrument_config
            Optional instrument configuration for source metadata lookup.
        """
        super().__init__()
        self._workflow_registry = dict(workflow_registry)
        self._plotting_controller = plotting_controller
        self._initial_config = initial_config
        self._instrument_config = instrument_config
        self._config_panel: ConfigurationPanel | None = None
        self._panel_container = pn.Column(sizing_mode='stretch_width')
        self._plotter_selection: PlotterSelection | None = None
        # Track last configuration to detect when panel needs recreation
        self._last_workflow_id: WorkflowId | None = None
        self._last_output: str | None = None
        self._last_plot_name: str | None = None
        self._last_axis_sources: dict[str, DataSourceConfig] | None = None
        # Store result from callback
        self._last_config_result: PlotConfig | None = None
        self._supports_windowing: bool = True

    @property
    def name(self) -> str:
        """Display name for this step."""
        return "Configure Plot"

    @property
    def description(self) -> str | None:
        """Description text for this step."""
        return None

    def is_valid(self) -> bool:
        """
        Step is always considered valid to keep button enabled.

        Actual validation happens in commit() to show errors when user clicks.
        """
        return True

    def commit(self) -> PlotConfig | None:
        """Commit the plot configuration."""
        if self._config_panel is None or self._plotter_selection is None:
            return None

        # Validate configuration first (shows errors if invalid)
        is_valid, _ = self._config_panel.validate()
        if not is_valid:
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

    def on_enter(self, input_data: PlotterSelection | None) -> None:
        """Create or recreate configuration panel when selection changes."""
        if input_data is not None:
            self._plotter_selection = input_data
        elif self._initial_config is not None:
            # Fall back to initial config when jumping to this step
            # Extract axis sources directly from the data_sources dict
            axis_sources = {
                role: ds
                for role, ds in self._initial_config.data_sources.items()
                if role in (X_AXIS, Y_AXIS)
            }
            self._plotter_selection = PlotterSelection(
                workflow_id=self._initial_config.workflow_id,
                output_name=self._initial_config.output_name,
                plot_name=self._initial_config.plot_name,
                axis_sources=axis_sources if axis_sources else None,
            )

        if self._plotter_selection is None:
            return

        # Check if the configuration has changed
        if (
            self._plotter_selection.workflow_id != self._last_workflow_id
            or self._plotter_selection.output_name != self._last_output
            or self._plotter_selection.plot_name != self._last_plot_name
            or self._plotter_selection.axis_sources != self._last_axis_sources
        ):
            # Recreate panel with new configuration
            self._create_config_panel()
            # Track new values
            self._last_workflow_id = self._plotter_selection.workflow_id
            self._last_output = self._plotter_selection.output_name
            self._last_plot_name = self._plotter_selection.plot_name
            self._last_axis_sources = self._plotter_selection.axis_sources

    def _create_config_panel(self) -> None:
        """Create the configuration panel for the selected plotter."""
        if self._plotter_selection is None:
            return

        # Handle static overlay workflow - no workflow_spec, empty sources
        is_static = self._plotter_selection.workflow_id == STATIC_OVERLAY_WORKFLOW
        if is_static:
            source_names: list[str] = []
        else:
            workflow_spec = self._workflow_registry.get(
                self._plotter_selection.workflow_id
            )
            if workflow_spec is None:
                show_error('Workflow spec not found')
                return
            if not workflow_spec.source_names:
                show_error('No sources available for workflow')
                return
            source_names = workflow_spec.source_names

        try:
            plot_spec = self._plotting_controller.get_spec(
                self._plotter_selection.plot_name
            )
        except Exception as e:
            logger.exception("Error getting plot spec")
            show_error(f'Error getting plot spec: {e}')
            return

        # Create config_state and initial_source_names from initial_config
        # if in edit mode
        config_state = None
        initial_source_names = None
        axis_sources = self._plotter_selection.axis_sources or {}
        if self._initial_config is not None:
            from ess.livedata.dashboard.configuration_adapter import ConfigurationState

            params_dict = (
                self._initial_config.params.model_dump(mode='json')
                if isinstance(self._initial_config.params, pydantic.BaseModel)
                else dict(self._initial_config.params)
            )
            if axis_sources:
                _inject_axis_source_titles(
                    params_dict,
                    axis_sources,
                    self._instrument_config,
                    self._workflow_registry,
                )
            config_state = ConfigurationState(params=params_dict)
            initial_source_names = self._initial_config.source_names
        elif (
            self._plotter_selection.plot_name in CORRELATION_HISTOGRAM_PLOTTERS
            and axis_sources
        ):
            # For new correlation histograms, pre-populate axis source titles
            from ess.livedata.dashboard.configuration_adapter import ConfigurationState

            titles = _resolve_axis_source_titles(
                axis_sources, self._instrument_config, self._workflow_registry
            )
            if titles:
                config_state = ConfigurationState(params={'bins': titles})

        hints = _resolve_output_display_hints(
            is_static=is_static,
            workflow_spec=workflow_spec if not is_static else None,
            output_name=self._plotter_selection.output_name,
        )
        self._supports_windowing = 'window' not in hints.hidden_fields

        # For new plots, use display hints to decide source pre-selection.
        # 2D+ outputs (images) default to empty so the user picks one source.
        if initial_source_names is None and not hints.preselect_all_sources:
            initial_source_names = []

        output_template_dims: tuple[str, ...] | None = None
        if not is_static:
            template = workflow_spec.get_output_template(
                self._plotter_selection.output_name
            )
            if template is not None:
                output_template_dims = tuple(template.dims)

        config_adapter = PlotConfigurationAdapter(
            plot_spec=plot_spec,
            source_names=source_names,
            success_callback=self._on_config_collected,
            config_state=config_state,
            initial_source_names=initial_source_names,
            instrument_config=self._instrument_config,
            hidden_fields=hints.hidden_fields,
            output_template_dims=output_template_dims,
            params_factory=self._plotting_controller.get_params_factory(
                self._plotter_selection.plot_name
            ),
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

        # Create primary data source
        primary_source = DataSourceConfig(
            workflow_id=self._plotter_selection.workflow_id,
            source_names=selected_sources,
            output_name=self._plotter_selection.output_name,
        )

        # Build data_sources dict with primary and optional axis sources
        data_sources: dict[str, DataSourceConfig] = {PRIMARY: primary_source}

        # Get axis sources for correlation histograms
        axis_sources = self._plotter_selection.axis_sources or {}

        # For correlation histograms, add axis sources and inject names into params
        if (
            self._plotter_selection.plot_name in CORRELATION_HISTOGRAM_PLOTTERS
            and axis_sources
        ):
            data_sources.update(axis_sources)
            if isinstance(params, pydantic.BaseModel):
                params = _inject_axis_source_titles(
                    params,
                    axis_sources,
                    self._instrument_config,
                    self._workflow_registry,
                )

        self._last_config_result = PlotConfig(
            data_sources=data_sources,
            plot_name=self._plotter_selection.plot_name,
            params=params,
            supports_windowing=self._supports_windowing,
        )


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
    instrument_config
        Optional instrument configuration for source metadata lookup.
    """

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        plotting_controller,
        success_callback: Callable[[PlotConfig], None],
        cancel_callback: Callable[[], None],
        initial_config: PlotConfig | None = None,
        instrument_config: Instrument | None = None,
    ) -> None:
        self._success_callback = success_callback
        self._cancel_callback = cancel_callback
        self._initial_config = initial_config

        # Create steps
        step1 = WorkflowAndOutputSelectionStep(
            workflow_registry=workflow_registry,
            initial_config=initial_config,
        )
        step2 = PlotterSelectionStep(
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
            initial_config=initial_config,
            instrument_config=instrument_config,
        )
        step3 = SpecBasedConfigurationStep(
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
            initial_config=initial_config,
            instrument_config=instrument_config,
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
            width=ModalSizing.WIDTH,
            max_height=ModalSizing.MAX_HEIGHT,
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
            # Edit mode: start at last step (steps use initial_config internally)
            self._wizard.reset_to_step(2)
        else:
            # Add mode: reset to first step
            self._wizard.reset()
        self._modal.open = True

    @property
    def modal(self) -> pn.Modal:
        """Get the modal widget."""
        return self._modal
