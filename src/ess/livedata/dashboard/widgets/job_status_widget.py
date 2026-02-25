# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

import panel as pn

from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.core.job_manager import JobAction
from ess.livedata.dashboard.format_utils import extract_error_summary
from ess.livedata.dashboard.job_controller import JobController
from ess.livedata.dashboard.job_service import JobService

from .icons import get_icon

if TYPE_CHECKING:
    from ess.livedata.dashboard.session_updater import SessionUpdater


# UI Constants
class UIConstants:
    """Constants for UI styling and sizing."""

    # Colors
    COLORS: ClassVar[dict[JobState, str]] = {
        JobState.scheduled: "#6c757d",  # Gray
        JobState.active: "#28a745",  # Green
        JobState.paused: "#ffc107",  # Yellow
        JobState.finishing: "#17a2b8",  # Blue
        JobState.stopped: "#343a40",  # Dark gray
        JobState.error: "#dc3545",  # Red
        JobState.warning: "#fd7e14",  # Orange
    }
    DEFAULT_COLOR = "#C162F4"
    ERROR_COLOR = "#dc3545"
    WARNING_COLOR = "#fd7e14"

    # Sizes
    JOB_INFO_WIDTH = 300
    JOB_INFO_HEIGHT = 45
    STATUS_INDICATOR_WIDTH = 90
    STATUS_INDICATOR_HEIGHT = 25
    TIMING_INFO_WIDTH = 150
    TIMING_INFO_HEIGHT = 25
    BUTTON_WIDTH = 46
    BUTTON_HEIGHT = 36
    EXPAND_BUTTON_WIDTH = 25
    EXPAND_BUTTON_HEIGHT = 20
    ERROR_BRIEF_WIDTH = 400
    ERROR_BRIEF_HEIGHT = 30
    ERROR_DETAILS_WIDTH = 600

    # Margins
    STANDARD_MARGIN = (5, 5)
    INFO_MARGIN = (5, 10)
    ERROR_MARGIN = (0, 10)
    BUTTON_MARGIN = (0, 2)
    EXPAND_MARGIN = (0, 5)
    DETAILS_MARGIN = (5, 10)

    # Text
    JOB_NUMBER_MAX_LENGTH = 8

    # Button icons (Tabler icon names)
    RESET_ICON = 'backspace'
    PAUSE_ICON = 'player-pause'
    PLAY_ICON = 'player-play'
    STOP_ICON = 'player-stop'
    REMOVE_ICON = 'trash'


class JobStatusWidget:
    """Widget to display the status of a job."""

    def __init__(self, job_status: JobStatus, job_controller: JobController) -> None:
        self._job_status = job_status
        self._job_controller = job_controller
        self._panel = None  # Store panel reference for dynamic updates
        self._checkbox = None  # Checkbox for multi-selection
        self._setup_widgets()
        self._create_panel()

    def _setup_widgets(self) -> None:
        """Set up the UI components."""
        # Checkbox for multi-selection
        self._checkbox = pn.widgets.Checkbox(
            value=False,
            width=25,
            height=25,
            margin=UIConstants.STANDARD_MARGIN,
        )

        # Job info row
        source_name = self._job_status.job_id.source_name
        job_number = str(self._job_status.job_id.job_number)
        job_number_short = job_number[: UIConstants.JOB_NUMBER_MAX_LENGTH]
        job_id_text = f"{source_name} ({job_number_short})"

        workflow_name = self._job_status.workflow_id.name
        workflow_version = self._job_status.workflow_id.version
        workflow_text = f"{workflow_name} v{workflow_version}"

        self._job_info = pn.pane.HTML(
            f"<b>{job_id_text}</b><br>{workflow_text}",
            width=UIConstants.JOB_INFO_WIDTH,
            height=UIConstants.JOB_INFO_HEIGHT,
            margin=UIConstants.INFO_MARGIN,
        )

        # Status indicator with color
        self._status_indicator = self._create_status_indicator()

        # Timing info
        self._timing_info = self._create_timing_info()
        # Error/warning message
        self._error_display = self._create_error_display()

        # Action buttons
        self._action_buttons = self._create_action_buttons()

    def _create_status_indicator(self) -> pn.pane.HTML:
        """Create the status indicator widget."""
        status_color = self._get_status_color(self._job_status.state)
        status_text = self._job_status.state.value.upper()
        status_style = self._get_status_style(status_color)
        return pn.pane.HTML(
            f'<div style="{status_style}">{status_text}</div>',
            width=UIConstants.STATUS_INDICATOR_WIDTH,
            height=UIConstants.STATUS_INDICATOR_HEIGHT,
            margin=UIConstants.STANDARD_MARGIN,
        )

    def _create_timing_info(self) -> pn.pane.HTML:
        """Create the timing info widget."""
        timing_text = self._format_timing()
        return pn.pane.HTML(
            timing_text,
            width=UIConstants.TIMING_INFO_WIDTH,
            height=UIConstants.TIMING_INFO_HEIGHT,
            margin=UIConstants.STANDARD_MARGIN,
        )

    def _create_error_display(self) -> pn.pane.HTML:
        """Set up error/warning message display."""
        error_text = self._format_error_message()
        return pn.pane.HTML(
            error_text,
            height=UIConstants.ERROR_BRIEF_HEIGHT,
            width=UIConstants.ERROR_BRIEF_WIDTH,
            margin=UIConstants.ERROR_MARGIN,
        )

    def _create_button(self, icon_name: str, callback) -> pn.widgets.Button:
        """Create a button with consistent styling."""
        button = pn.widgets.Button(
            name='',
            icon=get_icon(icon_name),
            button_type="light",
            width=UIConstants.BUTTON_WIDTH,
            height=UIConstants.BUTTON_HEIGHT,
            margin=UIConstants.BUTTON_MARGIN,
        )
        button.on_click(callback)
        return button

    def _get_button_widgets(self) -> list[pn.widgets.Button]:
        """Get button widgets based on current job state."""
        buttons = []

        # Stop/Remove button - dual purpose
        if self._job_status.state == JobState.stopped:
            remove_btn = self._create_button(
                UIConstants.REMOVE_ICON,
                lambda event: self._send_action(JobAction.remove),
            )
            buttons.append(remove_btn)
        elif self._job_status.state not in [JobState.stopped]:
            stop_btn = self._create_button(
                UIConstants.STOP_ICON, lambda event: self._send_action(JobAction.stop)
            )
            buttons.append(stop_btn)

        # Reset button - always available for non-removed jobs
        reset_btn = self._create_button(
            UIConstants.RESET_ICON, lambda event: self._send_action(JobAction.reset)
        )
        buttons.append(reset_btn)

        # TODO Do we need to be able to pause jobs that have errors/warnings?
        # Currently JobState is insufficient to determine if a job is running or
        # paused while in error/warning state.

        # Pause/Resume button - only for active/paused jobs
        if self._job_status.state in [JobState.active, JobState.paused]:
            if self._job_status.state == JobState.active:
                pause_btn = self._create_button(
                    UIConstants.PAUSE_ICON,
                    lambda event: self._send_action(JobAction.pause),
                )
                buttons.append(pause_btn)
            else:  # paused
                resume_btn = self._create_button(
                    UIConstants.PLAY_ICON,
                    lambda event: self._send_action(JobAction.resume),
                )
                buttons.append(resume_btn)

        return buttons

    def _create_action_buttons(self) -> pn.layout.Row:
        """Create action buttons based on job state."""
        buttons = self._get_button_widgets()
        return pn.Row(*buttons, margin=UIConstants.STANDARD_MARGIN)

    def _send_action(self, action: JobAction) -> None:
        """Send job action via the controller."""
        self._job_controller.send_job_action(self._job_status.job_id, action)

    def _get_status_color(self, state: JobState) -> str:
        """Get color for job state."""
        return UIConstants.COLORS.get(state, UIConstants.DEFAULT_COLOR)

    def _get_status_style(self, color: str) -> str:
        """Get CSS style for status indicator."""
        return (
            f"background-color: {color}; color: white; "
            f"padding: 2px 8px; border-radius: 3px; text-align: center; "
            f"font-weight: bold;"
        )

    def _get_error_style(self, color: str) -> str:
        """Get CSS style for error details."""
        return f"color: {color}; font-size: 12px; margin: 5px 0; white-space: pre-wrap;"

    def _format_timing(self) -> str:
        """Format timing information."""
        if self._job_status.start_time is None:
            return "Not started"

        start_dt = datetime.fromtimestamp(self._job_status.start_time / 1e9, tz=UTC)
        start_str = start_dt.strftime("%H:%M:%S")

        if self._job_status.end_time is not None:
            duration = (self._job_status.end_time - self._job_status.start_time) / 1e9
            return f"{start_str} ({duration:.1f}s)"
        else:
            return f"Started: {start_str}"

    def _format_error_message(self) -> str:
        """Format error/warning message as brief summary."""
        if self._job_status.error_message:
            msg = self._job_status.error_message
            color = UIConstants.ERROR_COLOR
        elif self._job_status.warning_message:
            msg = self._job_status.warning_message
            color = UIConstants.WARNING_COLOR
        else:
            return ""  # No error or warning message

        summary = extract_error_summary(msg)
        return f'<div style="{self._get_error_style(color)}">Error: {summary}</div>'

    def _get_error_color(self) -> str:
        """Get error color based on error type."""
        return (
            UIConstants.ERROR_COLOR
            if self._job_status.has_error
            else UIConstants.WARNING_COLOR
        )

    def _has_error_or_warning(self) -> bool:
        """Check if there's an error or warning message to display."""
        return bool(self._job_status.error_message or self._job_status.warning_message)

    def _create_panel(self) -> None:
        """Create the panel layout for this widget."""
        # Main row with checkbox, job info, status, timing, and actions
        main_row = pn.Row(
            self._checkbox,
            self._job_info,
            self._status_indicator,
            self._timing_info,
            self._action_buttons,
            sizing_mode="stretch_both",
        )

        # Only include error display if there's an error or warning message
        components = [main_row]
        if self._has_error_or_warning():
            components.append(self._error_display)

        self._panel = pn.Column(
            *components,
            styles={
                "border": "1px solid #dee2e6",
                "border-radius": "4px",
                "margin": "2px",
            },
            sizing_mode="stretch_both",
        )

    def update_status(self, job_status: JobStatus) -> None:
        """Update the widget with new job status, only changing what's necessary."""
        old_status = self._job_status
        self._job_status = job_status

        # Check if error/warning state changed (affects panel layout)
        error_state_changed = bool(
            old_status.error_message or old_status.warning_message
        ) != bool(job_status.error_message or job_status.warning_message)

        # Update status indicator if state changed
        if old_status.state != job_status.state:
            self._update_status_indicator()
            # Update action buttons when state changes
            self._update_action_buttons()

        # Update timing info if times changed
        if (
            old_status.start_time != job_status.start_time
            or old_status.end_time != job_status.end_time
        ):
            self._update_timing_info()

        # Update error/warning display if messages changed
        if (
            old_status.error_message != job_status.error_message
            or old_status.warning_message != job_status.warning_message
        ):
            self._update_error_display()
            # If error state changed, recreate the panel to add/remove error widget
            if error_state_changed:
                self._recreate_panel()

    def _update_status_indicator(self) -> None:
        """Update just the status indicator."""
        status_color = self._get_status_color(self._job_status.state)
        status_text = self._job_status.state.value.upper()
        status_style = self._get_status_style(status_color)
        self._status_indicator.object = (
            f'<div style="{status_style}">{status_text}</div>'
        )

    def _update_timing_info(self) -> None:
        """Update just the timing information."""
        timing_text = self._format_timing()
        self._timing_info.object = timing_text

    def _update_error_display(self) -> None:
        """Update just the error display."""
        error_text = self._format_error_message()
        self._error_display.object = error_text

    def _update_action_buttons(self) -> None:
        """Update action buttons in place when job state changes."""
        # Clear existing buttons
        self._action_buttons.clear()

        # Add new buttons based on current state
        new_buttons = self._get_button_widgets()
        self._action_buttons.extend(new_buttons)

    def _recreate_panel(self) -> None:
        """Recreate the panel layout when error/warning state changes."""
        # Clear the existing panel
        self._panel.clear()

        # Create new components
        main_row = pn.Row(
            self._checkbox,
            self._job_info,
            self._status_indicator,
            self._timing_info,
            self._action_buttons,
            sizing_mode="stretch_both",
        )

        # Add components to the panel
        self._panel.append(main_row)
        if self._has_error_or_warning():
            self._panel.append(self._error_display)

    @property
    def job_id(self):
        """Get the job ID for this widget."""
        return self._job_status.job_id

    @property
    def is_selected(self) -> bool:
        """Check if this job is selected via checkbox."""
        return self._checkbox.value

    def set_selected(self, selected: bool) -> None:
        """Set the selection state of this job."""
        self._checkbox.value = selected

    def panel(self) -> pn.layout.Column:
        """Get the panel layout for this widget."""
        return self._panel


class JobStatusListWidget:
    """Widget to display a list of job statuses with live updates."""

    def __init__(self, job_service: JobService, job_controller: JobController) -> None:
        self._job_service = job_service
        self._job_controller = job_controller
        self._status_widgets: dict[str, JobStatusWidget] = {}
        self._widget_panels: dict[str, pn.layout.Column] = {}
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Set up the main layout."""
        self._header = pn.pane.HTML("<h3>Job Status</h3>", margin=(10, 10, 5, 10))

        # Selection controls
        bnt_margin = (0, 2)
        self._select_all_btn = pn.widgets.Button(
            name="Select All", button_type="primary", margin=bnt_margin
        )
        self._select_all_btn.on_click(self._select_all)

        self._select_none_btn = pn.widgets.Button(
            name="Select None", button_type="primary", margin=bnt_margin
        )
        self._select_none_btn.on_click(self._select_none)

        self._invert_selection_btn = pn.widgets.Button(
            name="Invert", button_type="primary", margin=bnt_margin
        )
        self._invert_selection_btn.on_click(self._invert_selection)

        # Bulk action buttons
        self._bulk_reset_btn = pn.widgets.Button(
            name="Reset Selected",
            icon=get_icon(UIConstants.RESET_ICON),
            button_type="light",
            margin=bnt_margin,
        )
        self._bulk_reset_btn.on_click(self._bulk_reset)

        self._bulk_stop_btn = pn.widgets.Button(
            name="Stop Selected",
            icon=get_icon(UIConstants.STOP_ICON),
            button_type="light",
            margin=bnt_margin,
        )
        self._bulk_stop_btn.on_click(self._bulk_stop)

        # Controls row
        self._controls = pn.Row(
            self._select_all_btn,
            self._select_none_btn,
            self._invert_selection_btn,
            pn.Spacer(width=20),
            self._bulk_stop_btn,
            self._bulk_reset_btn,
            margin=(5, 10),
        )

        self._job_list = pn.Column(sizing_mode="stretch_width", margin=(0, 10))

        # Initialize with current job statuses (snapshot to avoid race conditions)
        for job_status in list(self._job_service.job_statuses.values()):
            self._add_or_update_job_widget(job_status)

    def refresh(self) -> None:
        """Refresh job status display from current job statuses."""
        # Snapshot to avoid race conditions with background threads
        current_statuses = list(self._job_service.job_statuses.values())
        current_job_keys = {
            f"{status.job_id.source_name}:{status.job_id.job_number}"
            for status in current_statuses
        }

        # Remove widgets for jobs that no longer exist
        widgets_to_remove = set(self._status_widgets.keys()) - current_job_keys
        for job_key in widgets_to_remove:
            self._remove_job_widget(job_key)

        # Add or update widgets for current jobs
        for job_status in current_statuses:
            self._add_or_update_job_widget(job_status)

    def _add_or_update_job_widget(self, job_status: JobStatus) -> None:
        """Add a new job widget or update an existing one."""
        job_key = f"{job_status.job_id.source_name}:{job_status.job_id.job_number}"

        if job_key in self._status_widgets:
            # Update existing widget
            self._status_widgets[job_key].update_status(job_status)
        else:
            # Create new widget
            widget = JobStatusWidget(job_status, self._job_controller)
            widget_panel = widget.panel()
            self._status_widgets[job_key] = widget
            self._widget_panels[job_key] = widget_panel
            self._job_list.append(widget_panel)

    def _remove_job_widget(self, job_key: str) -> None:
        """Remove a job widget."""
        if job_key in self._status_widgets:
            self._status_widgets.pop(job_key)
            if job_key in self._widget_panels:
                widget_panel = self._widget_panels.pop(job_key)
                # Remove from the job list
                try:
                    self._job_list.remove(widget_panel)
                except ValueError:
                    # Panel might not be in the list, e.g., if there is a mismatch
                    # between what job statuses we get from Kafka and which have been
                    # added here. This might be avoided if we control topic offsets
                    # better.
                    pass

    def _select_all(self, event) -> None:
        """Select all job widgets."""
        for widget in self._status_widgets.values():
            widget.set_selected(True)

    def _select_none(self, event) -> None:
        """Deselect all job widgets."""
        for widget in self._status_widgets.values():
            widget.set_selected(False)

    def _invert_selection(self, event) -> None:
        """Invert selection of all job widgets."""
        for widget in self._status_widgets.values():
            widget.set_selected(not widget.is_selected)

    def _get_selected_job_ids(self) -> list:
        """Get job IDs for all selected widgets."""
        return [
            widget.job_id
            for widget in self._status_widgets.values()
            if widget.is_selected
        ]

    def _bulk_reset(self, event) -> None:
        """Reset all selected jobs."""
        selected_job_ids = self._get_selected_job_ids()
        if selected_job_ids:
            self._job_controller.send_job_actions_batch(
                selected_job_ids, JobAction.reset
            )

    def _bulk_stop(self, event) -> None:
        """Stop all selected jobs."""
        selected_job_ids = self._get_selected_job_ids()
        if selected_job_ids:
            self._job_controller.send_job_actions_batch(
                selected_job_ids, JobAction.stop
            )

    def register_periodic_refresh(self, session_updater: SessionUpdater) -> None:
        """Register for periodic refresh of job status display.

        Parameters
        ----------
        session_updater:
            The session updater to register the refresh handler with.
        """
        session_updater.register_custom_handler(self.refresh)

    def panel(self) -> pn.layout.Column:
        """Get the main panel for this widget."""
        return pn.Column(
            self._header, self._controls, self._job_list, sizing_mode="stretch_width"
        )
