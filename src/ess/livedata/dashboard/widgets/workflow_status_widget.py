# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Widget for displaying and controlling workflow status.

Provides a collapsible card for each workflow showing:
- Header with title, status badge, timing info, and action buttons
- Body (when expanded):
  - Workflow description (if present)
  - Staging area with config toolbars grouped by identical configurations
  - Output chips showing available workflow outputs
  - Commit button when staged config differs from active config
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import panel as pn

from ess.livedata.config.device_contract import DeviceContract
from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec
from ess.livedata.core.job import JobState

from ..derived_devices import (
    affected_device_names,
    all_devices,
    view_device_names,
)
from ..format_utils import extract_error_summary
from ..job_orchestrator import StoppedReason
from ..notifications import show_error
from .buttons import ButtonStyles, create_tool_button
from .configuration_widget import ConfigurationModal
from .confirmation_modal import ConfirmationModal
from .derived_devices_overview import DerivedDevicesOverview
from .icons import get_icon
from .styles import Colors, ErrorBox, HoverColors, StatusColors, WarningBox

if TYPE_CHECKING:
    from ..job_orchestrator import JobConfig, JobOrchestrator
    from ..job_service import JobService
    from ..session_updater import SessionUpdater


# UI styling constants
class WorkflowWidgetStyles:
    """Styling constants for workflow status widget."""

    STATUS_COLORS: ClassVar[dict[str, str]] = {
        'active': StatusColors.SUCCESS,
        'stopped': StatusColors.MUTED,
        'error': StatusColors.ERROR,
        'warning': StatusColors.WARNING,
        'pending_context': StatusColors.PENDING,
        'pending': StatusColors.PENDING,
        'finishing': StatusColors.PENDING,
        'scheduled': StatusColors.PENDING,
    }
    MODIFIED_BORDER_COLOR = StatusColors.WARNING
    UNCONFIGURED_BG = WarningBox.BG
    UNCONFIGURED_BORDER = WarningBox.BORDER
    ERROR_BG = ErrorBox.BG
    ERROR_BORDER = ErrorBox.BORDER
    ERROR_TEXT = ErrorBox.TEXT
    WARNING_MSG_BG = WarningBox.BG
    WARNING_MSG_BORDER = WarningBox.BORDER
    WARNING_MSG_TEXT = WarningBox.TEXT

    # Output chip colors (unique to workflow outputs display)
    OUTPUT_CHIP_BG = '#e7f1ff'
    OUTPUT_CHIP_BORDER = '#b6d4fe'
    OUTPUT_CHIP_TEXT = '#0d6efd'

    # Device (NICOS derived-device) badge/marker color.
    DEVICE_COLOR = StatusColors.PRIMARY

    # Dimensions
    HEADER_HEIGHT = 40
    TOOLBAR_HEIGHT = 32


@dataclass(frozen=True)
class SourceStatus:
    """Status of a single source within a workflow."""

    source_name: str
    display_title: str
    state: JobState
    error_summary: str | None


@dataclass(frozen=True)
class MessageBox:
    """A status message to display in the workflow body, with severity."""

    text: str
    is_error: bool


# Priority of job states for worst-state aggregation; higher wins.
_JOB_STATE_PRIORITY: dict[JobState, int] = {
    JobState.error: 4,
    JobState.warning: 3,
    JobState.pending_context: 2,
    JobState.stopped: 1,
    JobState.active: 0,
}

# Display names for status states whose raw value is not user-friendly.
_JOB_STATE_DISPLAY_NAMES: dict[JobState, str] = {
    JobState.pending_context: 'WAITING FOR CONTEXT',
}


def worst_job_state(states: Iterable[JobState]) -> JobState:
    """Return the highest-priority state for aggregating a workflow's sources.

    Priority is error > warning > pending_context > stopped > active. States not
    listed in the priority table are treated as lowest priority.

    Parameters
    ----------
    states:
        Job states of the sources contributing to a single workflow.

    Returns
    -------
    :
        The aggregated worst state, defaulting to ``JobState.active`` for an
        empty input.
    """
    worst = JobState.active
    best_priority = _JOB_STATE_PRIORITY.get(worst, -1)
    for state in states:
        priority = _JOB_STATE_PRIORITY.get(state, -1)
        if priority > best_priority:
            best_priority = priority
            worst = state
    return worst


def job_state_display_name(state: JobState) -> str:
    """Return the user-facing status label for a job state."""
    return _JOB_STATE_DISPLAY_NAMES.get(state, state.value.upper())


@dataclass(frozen=True)
class ConfigGroup:
    """A group of sources sharing the same configuration."""

    source_names: tuple[str, ...]
    params: dict
    aux_source_names: dict
    is_modified: bool = False


def _make_hashable_key(config_dict: dict) -> str:
    """
    Convert a config dict to a hashable key.

    Uses JSON serialization with sorted keys to create a deterministic
    string representation that handles nested dicts, lists, etc.
    """
    return json.dumps(config_dict, sort_keys=True)


def _group_configs_by_equality(
    staged: dict[str, JobConfig],
    active: dict[str, JobConfig],
) -> list[ConfigGroup]:
    """
    Group sources by config equality for display.

    Sources with identical params and aux_source_names are grouped together.
    """
    if not staged:
        return []

    # Group by (params_key, aux_key) for hashability
    groups: dict[tuple[str, str], list[str]] = {}
    config_lookup: dict[tuple[str, str], tuple[dict, dict]] = {}

    for source_name, job_config in staged.items():
        # Create hashable key from config using JSON serialization
        params_key = _make_hashable_key(job_config.params)
        aux_key = _make_hashable_key(job_config.aux_source_names)
        key = (params_key, aux_key)

        if key not in groups:
            groups[key] = []
            config_lookup[key] = (job_config.params, job_config.aux_source_names)
        groups[key].append(source_name)

    # Convert to ConfigGroup objects
    result = []
    for key, source_names in groups.items():
        params, aux_source_names = config_lookup[key]

        # Check if this group is modified (differs from active config)
        is_modified = False
        for source_name in source_names:
            active_config = active.get(source_name)
            if active_config is None:
                # Source not in active config - consider modified
                is_modified = True
                break
            if (
                active_config.params != params
                or active_config.aux_source_names != aux_source_names
            ):
                is_modified = True
                break

        result.append(
            ConfigGroup(
                source_names=tuple(sorted(source_names)),
                params=params,
                aux_source_names=aux_source_names,
                is_modified=is_modified,
            )
        )

    # Sort by first source name for consistent ordering
    return sorted(result, key=lambda g: g.source_names[0])


def _get_unconfigured_sources(
    all_sources: list[str], staged: dict[str, JobConfig]
) -> list[str]:
    """Get sources that are not yet configured."""
    return sorted(set(all_sources) - set(staged.keys()))


class WorkflowStatusWidget:
    """
    Widget displaying status and controls for a single workflow.

    Shows a collapsible card with:
    - Header: title, status, timing, reset/stop buttons
    - Body: description (if present), config toolbars, outputs, commit button
    """

    def __init__(
        self,
        *,
        workflow_id: WorkflowId,
        workflow_spec: WorkflowSpec,
        orchestrator: JobOrchestrator,
        job_service: JobService,
        device_contract: DeviceContract | None = None,
    ) -> None:
        """
        Initialize workflow status widget.

        Parameters
        ----------
        workflow_id
            ID of the workflow this widget represents.
        workflow_spec
            Specification of the workflow.
        orchestrator
            Job orchestrator for config staging and commit.
        job_service
            Service providing job status updates.
        device_contract
            Per-instrument NICOS derived-device contract, used to derive
            device-bearing status and to gate disruptive actions. Defaults to
            an empty contract (no devices).
        """
        self._workflow_id = workflow_id
        self._workflow_spec = workflow_spec
        self._orchestrator = orchestrator
        self._job_service = job_service
        self._device_contract = device_contract or DeviceContract(())

        self._expanded = False
        self._panel: pn.Column | None = None

        # Modal container - zero height so it doesn't affect layout, but provides
        # a place in the component tree for the modal to attach to.
        self._modal_container = pn.Row(height=0, sizing_mode='stretch_width')

        # References to updatable elements (avoid full rebuild on status/expand update)
        self._status_badge: pn.pane.HTML | None = None
        self._status_dots: pn.pane.HTML | None = None
        self._timing_html: pn.pane.HTML | None = None
        self._expand_btn: pn.widgets.Button | None = None
        self._header: pn.Row | None = None
        self._body: pn.Column | None = None

        self._build_widget()
        # Initialise to the current version so the first refresh() is a no-op
        # unless state actually changed after construction.
        self._last_state_version = self._orchestrator.get_workflow_state_version(
            self._workflow_id
        )
        self._last_job_version = self._job_service.version

    @property
    def workflow_id(self) -> WorkflowId:
        """Get the workflow ID for this widget."""
        return self._workflow_id

    @property
    def _tool_css_class(self) -> str:
        """CSS class scoping this workflow's tool buttons for automation/tests."""
        return f'lt-wf-{self._workflow_id.name}'

    def _affected_device_names(self) -> list[str]:
        """Device names a disruptive action on this workflow would affect now."""
        running = self._orchestrator.get_running_workflow_sources()
        return affected_device_names(self._workflow_id, running, self._device_contract)

    def _build_widget(self) -> None:
        """Build or rebuild the entire widget.

        The body is created lazily on first expand. When collapsed (default),
        only the header is built, avoiding Bokeh model creation for hidden content.
        """
        with pn.io.hold():
            self._header = self._create_header()

            if self._expanded:
                self._body = self._create_body()
            else:
                self._body = None

            self._update_header_border()

            children = [self._header]
            if self._body is not None:
                children.append(self._body)
            children.append(self._modal_container)

            if self._panel is None:
                self._panel = pn.Column(
                    *children,
                    styles={
                        'border': f'1px solid {Colors.BORDER}',
                        'border-radius': '6px',
                        'overflow': 'hidden',
                        'background': 'white',
                    },
                    sizing_mode='stretch_width',
                    margin=(0, 0, 8, 0),
                )
            else:
                self._panel.clear()
                self._panel.extend(children)

    def _create_header(self) -> pn.Row:
        """Create the header row with expand button, title, status, and buttons."""
        # Expand/collapse button (store reference for icon updates)
        icon_name = 'chevron-down' if self._expanded else 'chevron-right'
        self._expand_btn = create_tool_button(
            icon_name=icon_name,
            button_color=StatusColors.MUTED,
            hover_color='rgba(0, 0, 0, 0.05)',
            on_click_callback=lambda: self.set_expanded(not self._expanded),
            css_classes=[self._tool_css_class, 'lt-wf-expand'],
        )

        # Workflow title
        title_html = pn.pane.HTML(
            f'<span style="font-weight: 600; font-size: 14px; '
            f'color: {Colors.TEXT_DARK};">'
            f'{self._workflow_spec.title}</span>',
            height=WorkflowWidgetStyles.HEADER_HEIGHT,
            styles={'display': 'flex', 'align-items': 'center'},
        )

        # Status badge, per-source dots, and timing info (store references for updates)
        status, status_color, timing_text, _, per_source = self._get_status_and_timing()
        self._status_badge = pn.pane.HTML(
            self._make_status_badge_html(status, status_color),
            height=WorkflowWidgetStyles.HEADER_HEIGHT,
            styles={'display': 'flex', 'align-items': 'center'},
        )

        self._status_dots = pn.pane.HTML(
            self._make_status_dots_html(per_source),
            height=WorkflowWidgetStyles.HEADER_HEIGHT,
            styles={'display': 'flex', 'align-items': 'center'},
        )

        self._timing_html = pn.pane.HTML(
            self._make_timing_html(timing_text),
            height=WorkflowWidgetStyles.HEADER_HEIGHT,
            styles={'display': 'flex', 'align-items': 'center'},
        )

        # Action buttons (only show when workflow is active)
        action_buttons = self._create_header_buttons()

        header = pn.Row(
            self._expand_btn,
            title_html,
            pn.Spacer(width=12),
            self._status_badge,
            pn.Spacer(width=8),
            self._status_dots,
            pn.Spacer(width=12),
            self._timing_html,
            pn.Spacer(sizing_mode='stretch_width'),
            action_buttons,
            height=WorkflowWidgetStyles.HEADER_HEIGHT,
            styles={
                'background': Colors.BG_LIGHT,
                'padding': '6px 12px',  # Fit 28px buttons in 40px header
            },
            sizing_mode='stretch_width',
            align='center',
        )

        return header

    def _update_header_border(self) -> None:
        """Update header border based on expanded state."""
        if self._header is not None:
            border = f'1px solid {Colors.BORDER}' if self._expanded else 'none'
            self._header.styles = {**self._header.styles, 'border-bottom': border}

    def _create_header_buttons(self) -> pn.Row:
        """Create action buttons for header (play, reset, stop)."""
        buttons = []

        active_job_number = self._orchestrator.get_active_job_number(self._workflow_id)

        # Check if there are changes to commit (same logic as _create_commit_row)
        has_changes = self._has_uncommitted_changes()

        # Show play button if there are uncommitted changes
        if has_changes:
            play_btn = create_tool_button(
                icon_name='player-play',
                button_color=WorkflowWidgetStyles.STATUS_COLORS['active'],
                hover_color=HoverColors.SUCCESS,
                on_click_callback=self._on_commit_click,
                css_classes=[self._tool_css_class],
            )
            buttons.append(play_btn)

        # Show stop and reset buttons if workflow is running
        if active_job_number is not None:
            stop_btn = create_tool_button(
                icon_name='player-stop',
                button_color=ButtonStyles.DANGER_RED,
                hover_color=ButtonStyles.DANGER_HOVER,
                on_click_callback=self._on_stop_click,
                css_classes=[self._tool_css_class],
            )
            buttons.append(stop_btn)

            reset_btn = create_tool_button(
                icon_name='backspace',
                button_color=StatusColors.MUTED,
                hover_color=HoverColors.MUTED,
                on_click_callback=self._on_reset_click,
                css_classes=[self._tool_css_class],
            )
            buttons.append(reset_btn)

        return pn.Row(*buttons, margin=0)

    def _has_uncommitted_changes(self) -> bool:
        """Check if there are staged changes that differ from active config."""
        staged = self._orchestrator.get_staged_config(self._workflow_id)
        if not staged:
            return False

        active = self._orchestrator.get_active_config(self._workflow_id)
        if not active:
            # Staged configs but no active - has changes
            return True

        # Check if any config differs from active
        config_groups = _group_configs_by_equality(staged, active)
        if any(g.is_modified for g in config_groups):
            return True

        # Check if source sets differ
        return set(staged.keys()) != set(active.keys())

    def _create_body(self) -> pn.Column:
        """Create the collapsible body content."""
        components = []

        # Error or warning message (if any)
        _, _, _, message_box, _ = self._get_status_and_timing()
        if message_box is not None:
            if message_box.is_error:
                bg = WorkflowWidgetStyles.ERROR_BG
                border = WorkflowWidgetStyles.ERROR_BORDER
                text = WorkflowWidgetStyles.ERROR_TEXT
            else:
                bg = WorkflowWidgetStyles.WARNING_MSG_BG
                border = WorkflowWidgetStyles.WARNING_MSG_BORDER
                text = WorkflowWidgetStyles.WARNING_MSG_TEXT
            components.append(
                pn.pane.HTML(
                    message_box.text,
                    sizing_mode='stretch_width',
                    margin=0,
                    styles={
                        'padding': '8px 12px',
                        'background': bg,
                        'border-bottom': f'1px solid {border}',
                        'color': text,
                        'font-size': '12px',
                    },
                )
            )

        # Workflow description (if present)
        if self._workflow_spec.description:
            components.append(
                pn.pane.HTML(
                    f'<div style="font-size: 12px; color: {Colors.TEXT_MUTED}; '
                    f'font-style: italic; line-height: 1.4;">'
                    f'{self._workflow_spec.description}</div>',
                    sizing_mode='stretch_width',
                    margin=0,
                    styles={
                        'padding': '8px 12px',
                        'background': '#ffffff',
                        'border-bottom': f'1px solid {Colors.BORDER}',
                    },
                )
            )

        # Staging area with config toolbars
        staging_area = self._create_staging_area()
        components.append(staging_area)

        # Outputs section
        outputs_section = self._create_outputs_section()
        components.append(outputs_section)

        # Commit button row (only if there are staged changes)
        commit_row = self._create_commit_row()
        if commit_row is not None:
            components.append(commit_row)

        return pn.Column(*components, sizing_mode='stretch_width', margin=0)

    def _create_staging_area(self) -> pn.Column:
        """Create the staging area with config toolbars."""
        staged = self._orchestrator.get_staged_config(self._workflow_id)
        active = self._orchestrator.get_active_config(self._workflow_id)

        config_groups = _group_configs_by_equality(staged, active)
        unconfigured = _get_unconfigured_sources(
            self._workflow_spec.source_names, staged
        )

        toolbars = []

        # Create toolbar for each config group
        for group in config_groups:
            toolbar = self._create_config_toolbar(group, is_unconfigured=False)
            toolbars.append(toolbar)

        # Create toolbar for unconfigured sources (if any)
        if unconfigured:
            unconfigured_group = ConfigGroup(
                source_names=tuple(unconfigured),
                params={},
                aux_source_names={},
                is_modified=False,
            )
            toolbar = self._create_config_toolbar(
                unconfigured_group, is_unconfigured=True
            )
            toolbars.append(toolbar)

        # Determine label based on whether there are modifications
        has_modifications = any(g.is_modified for g in config_groups) or bool(
            unconfigured
        )
        label_text = 'Configuration (staged)' if has_modifications else 'Configuration'

        label = pn.pane.HTML(
            f'<span style="font-size: 11px; '
            f'color: {Colors.TEXT_MUTED}; '
            f'text-transform: uppercase; '
            f'letter-spacing: 0.5px;">{label_text}</span>',
            margin=(0, 0, 6, 0),
        )

        return pn.Column(
            label,
            *toolbars,
            styles={
                'padding': '8px 12px',
                'border-bottom': f'1px solid {Colors.BORDER}',
            },
            sizing_mode='stretch_width',
            margin=0,
        )

    def _create_config_toolbar(
        self, group: ConfigGroup, *, is_unconfigured: bool
    ) -> pn.Row:
        """Create a toolbar row for a config group."""
        # Source tags - use display titles from orchestrator
        source_tags_html = ''.join(
            f'<span style="display: inline-block; padding: 1px 6px; '
            f'background: {WarningBox.BG if is_unconfigured else Colors.BG_MUTED}; '
            f'border-radius: 3px; margin-right: 4px; font-size: 11px;">'
            f'{self._orchestrator.get_source_title(name)}</span>'
            for name in group.source_names
        )

        if is_unconfigured:
            source_tags_html += (
                f'<span style="font-style: italic; '
                f'color: {WarningBox.TEXT};">unconfigured</span>'
            )

        source_list = pn.pane.HTML(
            f'<div style="font-size: 12px; '
            f'color: {Colors.TEXT};">'
            f'{source_tags_html}</div>',
            sizing_mode='stretch_width',
        )

        # Buttons
        gear_btn = create_tool_button(
            icon_name='settings',
            button_color=ButtonStyles.PRIMARY_BLUE,
            hover_color=ButtonStyles.PRIMARY_HOVER,
            on_click_callback=lambda: self._on_gear_click(list(group.source_names)),
            css_classes=[self._tool_css_class],
        )

        buttons = [gear_btn]

        # Add remove button (not for unconfigured sources)
        if not is_unconfigured:
            remove_btn = create_tool_button(
                icon_name='x',
                button_color=ButtonStyles.DANGER_RED,
                hover_color=ButtonStyles.DANGER_HOVER,
                on_click_callback=lambda: self._on_remove_click(
                    list(group.source_names)
                ),
                css_classes=[self._tool_css_class],
            )
            buttons.append(remove_btn)

        # Toolbar styling
        styles: dict[str, str] = {
            'padding': '6px 8px',
            'background': WorkflowWidgetStyles.UNCONFIGURED_BG
            if is_unconfigured
            else Colors.BG_LIGHT,
            'border': (
                '1px solid '
                + (
                    WorkflowWidgetStyles.UNCONFIGURED_BORDER
                    if is_unconfigured
                    else Colors.BORDER
                )
            ),
            'border-radius': '4px',
        }

        # Add modified indicator
        if group.is_modified and not is_unconfigured:
            styles['border-left'] = (
                f'3px solid {WorkflowWidgetStyles.MODIFIED_BORDER_COLOR}'
            )

        return pn.Row(
            source_list,
            pn.Row(*buttons, margin=0),
            styles=styles,
            sizing_mode='stretch_width',
            margin=(0, 0, 4, 0),
        )

    def _output_chip_tooltip(self, view) -> str:
        """Build the hover tooltip for an output chip.

        Combines the output description with the workflow parameters that
        shape it, so the user can see at a glance what each output is and
        which knobs control it.
        """
        parts: list[str] = []
        if view.description:
            parts.append(view.description)
        titles = self._workflow_spec.get_output_param_titles(view.name)
        if titles:
            parts.append(f"Controlled by: {', '.join(titles)}")
        return '\n\n'.join(parts)

    def _create_outputs_section(self) -> pn.Column:
        """Create the outputs section with chips."""
        if self._workflow_spec.outputs is None:
            return pn.Column(margin=0)

        # All declared sources are candidates for the per-output device marker;
        # the contract decides which (workflow, source, field) is a device.
        candidate_sources = set(self._workflow_spec.source_names)

        chips_html = ''
        for view in self._workflow_spec.get_output_views():
            title = view.title
            chip_bg = WorkflowWidgetStyles.OUTPUT_CHIP_BG
            chip_border = WorkflowWidgetStyles.OUTPUT_CHIP_BORDER
            chip_text = WorkflowWidgetStyles.OUTPUT_CHIP_TEXT
            device_names = view_device_names(
                self._workflow_id,
                candidate_sources,
                view,
                self._device_contract,
            )
            marker = ''
            if device_names:
                device_tooltip = 'NICOS device(s): ' + ', '.join(device_names)
                marker = (
                    f'<span title="{device_tooltip}" style="margin-left: 6px; '
                    f'padding: 0 5px; border-radius: 8px; font-size: 10px; '
                    f'font-weight: bold; color: white; '
                    f'background: {WorkflowWidgetStyles.DEVICE_COLOR};">NICOS</span>'
                )
            tooltip = self._output_chip_tooltip(view)
            title_attr = f' title="{html.escape(tooltip)}"' if tooltip else ''
            chips_html += (
                f'<span{title_attr} style="display: inline-flex; align-items: center; '
                f'padding: 4px 10px; background: {chip_bg}; '
                f'border: 1px solid {chip_border}; '
                f'border-radius: 16px; font-size: 12px; '
                f'color: {chip_text}; '
                f'margin-right: 6px; margin-bottom: 6px;">{title}{marker}</span>'
            )

        label = pn.pane.HTML(
            f'<span style="font-size: 11px; '
            f'color: {Colors.TEXT_MUTED}; '
            f'text-transform: uppercase; '
            f'letter-spacing: 0.5px;">Outputs</span>',
            margin=(0, 0, 6, 0),
        )

        chips = pn.pane.HTML(
            f'<div style="display: flex; flex-wrap: wrap;">{chips_html}</div>',
            sizing_mode='stretch_width',
        )

        return pn.Column(
            label,
            chips,
            styles={'padding': '8px 12px'},
            sizing_mode='stretch_width',
            margin=0,
        )

    def _create_commit_row(self) -> pn.Row | None:
        """Create commit button row if there are uncommitted changes."""
        staged = self._orchestrator.get_staged_config(self._workflow_id)
        active = self._orchestrator.get_active_config(self._workflow_id)

        # Check if there are changes to commit
        has_changes = False

        # Check for unconfigured sources that are now configured
        if staged and not active:
            has_changes = True
        elif staged:
            # Check if any config differs from active
            config_groups = _group_configs_by_equality(staged, active)
            has_changes = any(g.is_modified for g in config_groups)

            # Also check if source sets differ
            if set(staged.keys()) != set(active.keys()):
                has_changes = True

        if not has_changes or not staged:
            return None

        # Determine button text
        button_text = 'Commit & Restart' if active else 'Start'

        commit_btn = pn.widgets.Button(
            label=button_text,
            color='primary',
            margin=0,
        )
        commit_btn.on_click(lambda e: self._on_commit_click())

        return pn.Row(
            pn.Spacer(sizing_mode='stretch_width'),
            commit_btn,
            styles={
                'padding': '8px 12px',
                'background': Colors.BG_LIGHT,
                'border-top': f'1px solid {Colors.BORDER}',
            },
            sizing_mode='stretch_width',
            margin=0,
        )

    def _make_status_badge_html(self, status: str, status_color: str) -> str:
        """Generate HTML for the status badge."""
        return (
            f'<span style="padding: 2px 8px; border-radius: 3px; font-size: 11px; '
            f'font-weight: bold; color: white; background: {status_color};">'
            f'{status}</span>'
        )

    @staticmethod
    def _make_timing_html(timing_text: str) -> str:
        """Generate HTML for the timing display."""
        return (
            f'<span style="font-size: 12px; '
            f'color: {Colors.TEXT_MUTED};">'
            f'{timing_text}</span>'
        )

    @staticmethod
    def _make_status_dots_html(sources: list[SourceStatus]) -> str:
        """Generate HTML for per-source status indicator dots.

        Each dot represents one source, colored by its job state.
        """
        if not sources:
            return ''
        dots = []
        for s in sources:
            color = WorkflowWidgetStyles.STATUS_COLORS.get(
                s.state.value, WorkflowWidgetStyles.STATUS_COLORS['stopped']
            )
            tooltip = f'{s.display_title}: {s.state.value}'
            if s.error_summary:
                tooltip += f' \u2014 {s.error_summary}'
            dots.append(
                f'<span title="{tooltip}" style="'
                f'display: inline-block; width: 8px; height: 8px; '
                f'border-radius: 50%; background: {color}; '
                f'cursor: default;'
                f'"></span>'
            )
        return (
            '<span style="display: inline-flex; align-items: center; gap: 4px;">'
            + ''.join(dots)
            + '</span>'
        )

    def _get_status_and_timing(
        self,
    ) -> tuple[str, str, str, MessageBox | None, list[SourceStatus]]:
        """Get workflow status, color, timing, message, and per-source statuses.

        Iterates job statuses once to extract aggregated and per-source info.

        Returns
        -------
        :
            Tuple of (status_text, status_color, timing_text, message_box,
            per_source_statuses). ``message_box`` carries the first backend error
            or, absent any error, the first backend warning message.
        """
        # Deferred import: datetime is only needed for active workflows with
        # a start_time, and importing at module level would add unnecessary
        # startup cost for a rarely-used module.
        from datetime import UTC, datetime

        active_job_number = self._orchestrator.get_active_job_number(self._workflow_id)

        if active_job_number is None:
            reason = self._orchestrator.get_stopped_reason(self._workflow_id)
            timing = (
                'Backend shut down' if reason == StoppedReason.backend_shutdown else ''
            )
            return (
                'STOPPED',
                WorkflowWidgetStyles.STATUS_COLORS['stopped'],
                timing,
                None,
                [],
            )

        # Single pass over job statuses to collect status, timing, and messages
        has_fresh_backend_status = False
        states: list[JobState] = []
        earliest_start = None
        first_error = None
        first_warning = None
        per_source: dict[str, SourceStatus] = {}

        for job_status in self._job_service.job_statuses.values():
            if job_status.workflow_id != self._workflow_id:
                continue
            if job_status.job_id.job_number != active_job_number:
                continue
            if self._job_service.is_status_stale(job_status.job_id):
                continue

            has_fresh_backend_status = True

            # Collect per-source status
            error_summary = (
                extract_error_summary(job_status.error_message)
                if job_status.error_message
                else None
            )
            source_name = job_status.job_id.source_name
            per_source[source_name] = SourceStatus(
                source_name=source_name,
                display_title=self._orchestrator.get_source_title(source_name),
                state=job_status.state,
                error_summary=error_summary,
            )

            states.append(job_status.state)

            # Timing: track earliest start
            start = job_status.start_time
            if start is not None:
                if earliest_start is None or start < earliest_start:
                    earliest_start = start

            # Capture first error and first warning message
            if first_error is None and error_summary:
                first_error = error_summary
            if first_warning is None and job_status.warning_message:
                first_warning = job_status.warning_message

        worst_state = worst_job_state(states)
        if first_error is not None:
            message_box = MessageBox(text=f'Error: {first_error}', is_error=True)
        elif first_warning is not None:
            message_box = MessageBox(text=f'Warning: {first_warning}', is_error=False)
        else:
            message_box = None

        # Order per-source statuses by workflow spec order
        spec_order = {
            name: i for i, name in enumerate(self._workflow_spec.source_names)
        }
        per_source_list = sorted(
            per_source.values(),
            key=lambda s: spec_order.get(s.source_name, len(spec_order)),
        )

        if not has_fresh_backend_status:
            # Show expected sources as pending dots
            pending_sources = [
                SourceStatus(
                    source_name=name,
                    display_title=self._orchestrator.get_source_title(name),
                    state=JobState.scheduled,
                    error_summary=None,
                )
                for name in self._workflow_spec.source_names
                if name in self._orchestrator.get_active_config(self._workflow_id)
            ]
            return (
                'PENDING',
                WorkflowWidgetStyles.STATUS_COLORS['pending'],
                'Waiting for backend...',
                None,
                pending_sources,
            )

        status_text = job_state_display_name(worst_state)
        status_color = WorkflowWidgetStyles.STATUS_COLORS.get(
            worst_state.value, WorkflowWidgetStyles.STATUS_COLORS['stopped']
        )

        # Build timing text
        if earliest_start is None:
            timing_text = 'Starting...'
        else:
            start_dt = earliest_start.to_datetime(tz=UTC)
            now = datetime.now(tz=UTC)
            duration_secs = int((now - start_dt).total_seconds())

            if duration_secs < 60:
                duration_str = f'{duration_secs}s'
            elif duration_secs < 3600:
                mins = duration_secs // 60
                secs = duration_secs % 60
                duration_str = f'{mins}m {secs}s'
            else:
                hours = duration_secs // 3600
                mins = (duration_secs % 3600) // 60
                duration_str = f'{hours}h {mins}m'

            timing_text = f'{start_dt.strftime("%H:%M:%S")} ({duration_str})'

        return status_text, status_color, timing_text, message_box, per_source_list

    def _on_header_click(self, event) -> None:
        """Handle header click to toggle expand/collapse."""
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        """Set expand/collapse state programmatically."""
        if self._expanded != expanded:
            self._expanded = expanded
            self._update_expand_state()

    def _update_expand_state(self) -> None:
        """Update UI elements for expand/collapse without full rebuild.

        On first expand, creates the body lazily and inserts it into the panel.
        On collapse, removes the body from the panel to free Bokeh models.
        """
        with pn.io.hold():
            # Update expand button indicator
            if self._expand_btn is not None:
                icon_name = 'chevron-down' if self._expanded else 'chevron-right'
                self._expand_btn.icon = get_icon(icon_name)

            if self._expanded:
                # Create body lazily on first expand (or recreate after collapse)
                self._body = self._create_body()
                if self._panel is not None:
                    # Insert body between header and modal_container
                    self._panel.insert(1, self._body)
            else:
                # Remove body from panel to free Bokeh models
                if self._body is not None and self._panel is not None:
                    self._panel.remove(self._body)
                self._body = None

            # Update header border
            self._update_header_border()

    def _on_gear_click(self, source_names: list[str]) -> None:
        """Handle gear button click - show configuration modal."""
        try:
            adapter = self._orchestrator.create_workflow_adapter(
                self._workflow_id, selected_sources=source_names, commit=False
            )

            modal = ConfigurationModal(
                config=adapter,
                start_button_text="Apply",
                success_callback=self._cleanup_modal,
            )

            # Add modal to container so it renders, then show
            self._modal_container.clear()
            self._modal_container.append(modal.modal)
            modal.show()
        except Exception:
            import structlog

            structlog.get_logger().exception(
                "Failed to create workflow configuration modal"
            )
            show_error("Failed to open configuration")

    def _cleanup_modal(self) -> None:
        """Clean up modal after completion."""
        self._modal_container.clear()

    def _on_remove_click(self, source_names: list[str]) -> None:
        """Handle remove button click - remove sources from staged config."""
        staged = self._orchestrator.get_staged_config(self._workflow_id)

        # Remove the specified sources and re-stage the rest in a transaction.
        # Transaction ensures only a single version increment.
        with self._orchestrator.staging_transaction(self._workflow_id):
            self._orchestrator.clear_staged_configs(self._workflow_id)
            for source_name, config in staged.items():
                if source_name not in source_names:
                    self._orchestrator.stage_config(
                        self._workflow_id,
                        source_name=source_name,
                        params=config.params,
                        aux_source_names=config.aux_source_names,
                    )

    def _gate_disruptive_action(
        self, action_label: str, action: Callable[[], None]
    ) -> None:
        """Run ``action``, gating it behind a confirmation when device-bearing.

        Re-reads live device-bearing state at click time (not stale widget
        state). When the workflow currently exposes a NICOS device and the
        global gate is enabled, opens a confirmation modal naming the affected
        devices; the action runs only on confirm. Otherwise the action runs
        immediately.

        Parameters
        ----------
        action_label
            Lower-case verb phrase for the dialog (e.g. ``"stop"``).
        action
            The orchestrator call to perform on confirm / immediately.
        """
        device_names = self._affected_device_names()
        if not device_names or not self._orchestrator.gate_enabled:
            action()
            return

        names = ''.join(f'<li>{name}</li>' for name in device_names)
        message = (
            f'This workflow currently exposes the following device(s) to NICOS:'
            f'<ul style="margin: 4px 0 8px 0; padding-left: 20px; '
            f'font-weight: 600;">{names}</ul>'
            f'Choosing to {action_label} is disruptive <i>if</i> NICOS is using '
            f'the device — we cannot tell whether a scan is active. Proceed?'
        )
        modal = ConfirmationModal(
            title=f'Confirm {action_label}',
            message=message,
            on_confirm=action,
            confirm_label=action_label.capitalize(),
            confirm_css_classes=[self._tool_css_class, 'lt-confirm-proceed'],
            on_close=self._cleanup_modal,
        )
        self._modal_container.clear()
        self._modal_container.append(modal.modal)
        modal.show()

    def _on_reset_click(self) -> None:
        """Handle reset button click - resets accumulated data for the workflow.

        The reset command clears accumulated data in the backend but keeps
        the workflow running. No widget rebuild is needed - the data streams
        will reflect the reset automatically. Gated when device-bearing.
        """
        self._gate_disruptive_action(
            'reset', lambda: self._orchestrator.reset_workflow(self._workflow_id)
        )

    def _on_stop_click(self) -> None:
        """Handle stop button click - stops the workflow. Gated when device-bearing."""
        self._gate_disruptive_action(
            'stop', lambda: self._orchestrator.stop_workflow(self._workflow_id)
        )

    def _on_commit_click(self) -> None:
        """Handle commit button click. Gated when device-bearing."""
        self._gate_disruptive_action(
            'restart with modified config',
            lambda: self._orchestrator.commit_workflow(self._workflow_id),
        )

    def is_stale(self) -> bool:
        """True when a refresh would show something the browser has not seen.

        Two independent sources: the workflow's state version (structural
        changes -> full rebuild) and the job-status version (badge, per-source
        dots and timing text). The latter matters on its own because a
        committed job reaching RUNNING bumps no workflow state version, so
        without it the row would sit on its post-commit rendering until the
        session's next unconditional full pass.
        """
        return (
            self._orchestrator.get_workflow_state_version(self._workflow_id)
            != self._last_state_version
            or self._job_service.version != self._last_job_version
        )

    def refresh(self) -> None:
        """Refresh widget from current state.

        Checks the orchestrator's workflow state version to detect structural
        changes (staging, commit, stop). On version change, does a full rebuild.
        Otherwise, updates only the status badge and timing text, skipping
        assignments when the values haven't changed.
        """
        self._last_job_version = self._job_service.version
        current_version = self._orchestrator.get_workflow_state_version(
            self._workflow_id
        )
        if current_version != self._last_state_version:
            self._last_state_version = current_version
            self._build_widget()
            return

        status, status_color, timing_text, _, per_source = self._get_status_and_timing()

        if self._status_badge is not None:
            new_badge = self._make_status_badge_html(status, status_color)
            if self._status_badge.object != new_badge:
                self._status_badge.object = new_badge

        if self._status_dots is not None:
            new_dots = self._make_status_dots_html(per_source)
            if self._status_dots.object != new_dots:
                self._status_dots.object = new_dots

        if self._timing_html is not None:
            new_timing = self._make_timing_html(timing_text)
            if self._timing_html.object != new_timing:
                self._timing_html.object = new_timing

    def panel(self) -> pn.Column:
        """Get the panel layout for this widget."""
        if self._panel is None:
            self._build_widget()
        return self._panel


class WorkflowStatusListWidget:
    """
    Widget displaying a list of workflow status widgets.

    Creates one WorkflowStatusWidget per workflow managed by the orchestrator.
    """

    def __init__(
        self,
        *,
        orchestrator: JobOrchestrator,
        job_service: JobService,
        device_contract: DeviceContract | None = None,
    ) -> None:
        """
        Initialize workflow status list widget.

        Parameters
        ----------
        orchestrator
            Job orchestrator for config staging and commit.
            Provides the workflow registry.
        job_service
            Service providing job status updates.
        device_contract
            Per-instrument NICOS derived-device contract, threaded to each
            per-workflow widget for device-bearing derivation and gating.
            Defaults to an empty contract (no devices).
        """
        self._orchestrator = orchestrator
        self._job_service = job_service
        self._device_contract = device_contract or DeviceContract(())
        self._is_visible: Callable[[], bool] | None = None
        self._populated: bool = False
        self._gate_checkbox: pn.widgets.Checkbox | None = None
        self._gate_banner: pn.pane.HTML | None = None
        self._last_gate_version: int = orchestrator.get_gate_version()

        self._derived_devices = DerivedDevicesOverview(
            orchestrator=orchestrator,
            device_contract=self._device_contract,
        )

        workflow_registry = orchestrator.get_workflow_registry()
        self._widgets: dict[WorkflowId, WorkflowStatusWidget] = {
            workflow_id: WorkflowStatusWidget(
                workflow_id=workflow_id,
                workflow_spec=spec,
                orchestrator=orchestrator,
                job_service=job_service,
                device_contract=device_contract,
            )
            for workflow_id, spec in sorted(
                workflow_registry.items(), key=lambda x: x[1].title
            )
        }
        # Zero-height container the derived-devices modal is mounted into on
        # first open. Mounting a pn.Modal at render time inside the dynamic
        # Workflows tab re-registers its callbacks on tab activation; lazy
        # mounting on user action (as the per-workflow modals do) avoids that.
        self._modal_container = pn.Row(height=0, sizing_mode='stretch_width')
        header_row = self._create_header_row()
        gate_banner = [self._gate_banner] if self._gate_banner is not None else []
        self._panel = pn.Column(
            header_row,
            *gate_banner,
            self._modal_container,
            sizing_mode='stretch_width',
            margin=(10, 10),
        )

    def _create_header_row(self) -> pn.Row:
        """Create the header row with expand/collapse all buttons."""
        compact_btn_css = [
            f"""
            button {{
                font-size: 12px !important;
                padding: 4px 8px !important;
                border: 1px solid {Colors.BORDER} !important;
                border-radius: 4px !important;
            }}
            button:hover {{
                background-color: {Colors.BG_MUTED} !important;
            }}
            """
        ]

        expand_all_btn = pn.widgets.Button(
            label='Expand all',
            color='light',
            width=90,
            height=28,
            margin=(0, 4, 0, 0),
            stylesheets=compact_btn_css,
        )
        expand_all_btn.on_click(lambda e: self._expand_all())

        collapse_all_btn = pn.widgets.Button(
            label='Collapse all',
            color='light',
            width=90,
            height=28,
            margin=0,
            stylesheets=compact_btn_css,
        )
        collapse_all_btn.on_click(lambda e: self._collapse_all())

        left: list[pn.viewable.Viewable] = []
        if len(self._device_contract) > 0:
            devices_btn = pn.widgets.Button(
                label='NICOS devices',
                color='light',
                width=120,
                height=28,
                margin=(0, 4, 0, 0),
                stylesheets=compact_btn_css,
                css_classes=['lt-derived-devices-open'],
            )
            devices_btn.on_click(lambda e: self._show_derived_devices())

            self._gate_checkbox = pn.widgets.Checkbox(
                label='Confirm before disrupting a device NICOS may be using',
                value=self._orchestrator.gate_enabled,
                margin=(6, 0, 0, 8),
                css_classes=['lt-gate-toggle'],
            )
            self._gate_checkbox.param.watch(self._on_gate_toggle, 'value')
            gate_help = pn.widgets.TooltipIcon(
                value=(
                    'Applies to all sessions. Uncheck to skip confirmations '
                    'while no NICOS scan is running.'
                ),
                margin=(4, 0, 0, 0),
            )
            # Full-width banner above the list, shown only while the gate is off:
            # the warning is about the device-bearing workflows below and their
            # stop/reset/gear actions, not about the header itself.
            self._gate_banner = pn.pane.HTML(
                self._gate_banner_html(),
                sizing_mode='stretch_width',
                margin=(0, 0, 8, 0),
            )
            self._apply_gate_style()
            left.extend([devices_btn, self._gate_checkbox, gate_help])

        return pn.Row(
            *left,
            pn.Spacer(sizing_mode='stretch_width'),
            expand_all_btn,
            collapse_all_btn,
            sizing_mode='stretch_width',
            margin=(0, 0, 8, 0),
        )

    def _on_gate_toggle(self, event) -> None:
        """Apply a checkbox toggle to the shared, cross-session gate flag."""
        self._orchestrator.set_gate_enabled(event.new)
        self._apply_gate_style()

    def _sync_gate_checkbox(self) -> None:
        """Reflect the shared gate flag, picking up toggles from other sessions.

        Assigning an unchanged value re-enters :meth:`_on_gate_toggle`, but
        :meth:`JobOrchestrator.set_gate_enabled` is a no-op for equal values, so
        there is no feedback loop.
        """
        self._last_gate_version = self._orchestrator.get_gate_version()
        if self._gate_checkbox is None:
            return
        enabled = self._orchestrator.gate_enabled
        if self._gate_checkbox.value != enabled:
            self._gate_checkbox.value = enabled
        self._apply_gate_style()

    def _apply_gate_style(self) -> None:
        """Show the danger banner above the list while the gate is off."""
        if self._gate_banner is None or self._gate_checkbox is None:
            return
        self._gate_banner.visible = not self._gate_checkbox.value

    def _gate_banner_html(self) -> str:
        """Warning banner naming the device-bearing workflows the gate guards.

        The named set is static (the contract's workflows), so the banner needs
        no run-state refresh; whether each is currently online is visible in the
        list below.
        """
        registry = self._orchestrator.get_workflow_registry()
        titles: list[str] = []
        for device in all_devices(self._device_contract):
            spec = registry.get(device.workflow_id)
            title = spec.title if spec is not None else str(device.workflow_id)
            if title not in titles:
                titles.append(title)
        named = ', '.join(f'<strong>{title}</strong>' for title in titles)
        device_clause = (
            'takes its NICOS device offline'
            if len(titles) == 1
            else 'takes their NICOS devices offline'
        )
        return (
            f'<div style="background: {WarningBox.BG}; '
            f'border-left: 4px solid {WarningBox.BORDER}; '
            f'color: {WarningBox.TEXT}; padding: 8px 12px; '
            'border-radius: 4px; font-size: 13px; line-height: 1.45;">'
            'Confirmations are <strong>off</strong>: stopping, resetting, or '
            f'reconfiguring {named} {device_clause} immediately, '
            'with no prompt.</div>'
        )

    def _show_derived_devices(self) -> None:
        """Mount (once) and open the derived-devices overview modal."""
        if self._derived_devices.modal not in self._modal_container:
            self._modal_container.append(self._derived_devices.modal)
        self._derived_devices.open()

    def _expand_all(self) -> None:
        """Expand all workflow widgets."""
        with pn.io.hold():
            for widget in self._widgets.values():
                widget.set_expanded(True)

    def _collapse_all(self) -> None:
        """Collapse all workflow widgets."""
        with pn.io.hold():
            for widget in self._widgets.values():
                widget.set_expanded(False)

    def rebuild_widget(self, workflow_id: WorkflowId) -> None:
        """Rebuild a specific workflow widget after config changes."""
        widget = self._widgets.get(workflow_id)
        if widget is not None:
            widget._build_widget()

    def register_periodic_refresh(
        self,
        session_updater: SessionUpdater,
        *,
        is_visible: Callable[[], bool] | None = None,
    ) -> None:
        """Register for periodic refresh of workflow status displays.

        Parameters
        ----------
        session_updater:
            The session updater to register the refresh handler with.
        is_visible:
            Optional predicate returning whether this tab is currently visible.
            When provided, refreshes are skipped while the tab is hidden.
        """
        self._is_visible = is_visible
        session_updater.register_custom_handler(
            self._refresh_all, has_work=self._has_pending_work
        )
        self._derived_devices.register_periodic_refresh(session_updater)
        # Populate as soon as the session is connected. onload callbacks run
        # with pn.state.curdoc set (unlike session-factory code), so
        # pn.io.hold() is effective and all workflow panels are inserted
        # atomically rather than one at a time.
        pn.state.onload(self._populate_pending)

    def _populate_pending(self) -> None:
        """Insert all workflow panels into the column in one batch.

        Called from pn.state.onload() where pn.io.hold() is effective.
        Also called as a fallback from _refresh_all() in non-server contexts
        (e.g., tests) where onload may not fire.
        """
        if not self._populated:
            with pn.io.hold():
                self._panel.extend([w.panel() for w in self._widgets.values()])
            self._populated = True

    def _has_pending_work(self) -> bool:
        """Wake-tick gate: True when a refresh would change something.

        Covers structural changes (per-workflow state versions, gate toggle)
        and observed job status (badges, dots, timing). Purely wall-clock
        staleness -- text that ages with no message behind it -- has no
        counter and rides the unconditional full pass instead.

        Returns False while the tab is hidden, matching :meth:`_refresh_all`.
        Becoming visible again is not a change any predicate can observe, so
        the tab switch requests a *full* tick (see ``PlotGridTabs``).
        """
        if self._is_visible is not None and not self._is_visible():
            return False
        if not self._populated:
            return True
        if self._orchestrator.get_gate_version() != self._last_gate_version:
            return True
        return any(widget.is_stale() for widget in self._widgets.values())

    def _refresh_all(self) -> None:
        """Refresh all workflow status widgets.

        Skips refresh when the tab is not visible, since updates would touch
        Bokeh models that the user cannot see.
        """
        if self._is_visible is not None and not self._is_visible():
            return
        if not self._populated:
            # Fallback: onload didn't fire (non-server context).
            self._populate_pending()
        self._sync_gate_checkbox()
        for widget in self._widgets.values():
            widget.refresh()

    def panel(self) -> pn.Column:
        """Get the main panel for this widget."""
        return self._panel
