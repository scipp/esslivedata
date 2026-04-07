# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Widget to display backend service worker status."""

from __future__ import annotations

from typing import ClassVar

import panel as pn

from ess.livedata.core.job import ServiceState, ServiceStatus, StreamStats
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.dashboard.service_registry import ServiceRegistry

from .buttons import ButtonStyles, create_tool_button
from .styles import Colors, StatusColors


class WorkerUIConstants:
    """Constants for worker status UI styling and sizing."""

    # Colors for service states
    COLORS: ClassVar[dict[ServiceState, str]] = {
        ServiceState.starting: StatusColors.MUTED,
        ServiceState.running: StatusColors.SUCCESS,
        ServiceState.stopping: StatusColors.WARNING,
        ServiceState.stopped: StatusColors.MUTED,
        ServiceState.error: StatusColors.ERROR,
    }
    DEFAULT_COLOR = StatusColors.MUTED
    STALE_COLOR = StatusColors.ERROR

    # Sizes
    NAMESPACE_WIDTH = 200
    WORKER_ID_WIDTH = 150
    STATUS_WIDTH = 90
    VERSION_WIDTH = 200
    UPTIME_WIDTH = 120
    STATS_WIDTH = 240

    # Margins
    STANDARD_MARGIN = (2, 5)
    HEADER_MARGIN = (10, 10, 5, 10)


def _format_duration(seconds: float | None) -> str:
    """Format a duration in human-readable form."""
    if seconds is None:
        return "Unknown"

    if seconds < 60:
        value = int(seconds)
        unit = "second" if value == 1 else "seconds"
        return f"{value} {unit}"
    elif seconds < 3600:
        value = seconds / 60
        unit = "minute" if value == 1 else "minutes"
        return f"{value:.1f} {unit}"
    elif seconds < 86400:
        value = seconds / 3600
        unit = "hour" if value == 1 else "hours"
        return f"{value:.1f} {unit}"
    else:
        value = seconds / 86400
        unit = "day" if value == 1 else "days"
        return f"{value:.1f} {unit}"


def _format_messages(count: int) -> str:
    """Format message count in human-readable form."""
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}k"
    else:
        return f"{count / 1_000_000:.1f}M"


def _format_stream_stats_summary(stats: StreamStats | None) -> str:
    """Format compact stream stats summary for the worker status row."""
    if stats is None:
        return "Msgs: -"
    total = sum(s.count for s in stats.streams)
    window = f"{stats.window_seconds:.0f}s"
    return f"Msgs: {_format_messages(total)}/{window}"


def _format_stream_stats_details(stats: StreamStats | None) -> str:
    """Format expandable per-stream details table.

    Always returns visible content so the row height stays consistent
    regardless of whether stream data has arrived yet.
    """
    style = "margin: 0; font-size: 11px;"
    if stats is None:
        return (
            f'<span style="{style}; color: {Colors.TEXT_MUTED}">No stream data</span>'
        )
    if not stats.streams:
        return (
            f'<details style="{style}">'
            f"<summary>Streams ({stats.window_seconds:.0f}s window)</summary>"
            "<i>No streams received</i>"
            "</details>"
        )
    unmapped_count = sum(1 for s in stats.streams if s.stream is None)
    summary_extra = ""
    if unmapped_count:
        summary_extra = (
            f' — <span style="color: {StatusColors.ERROR}">'
            f"{unmapped_count} unmapped</span>"
        )
    rows = []
    for s in stats.streams:
        stream_cell = s.stream or (
            f'<span style="color: {StatusColors.ERROR}">unmapped</span>'
        )
        rows.append(
            f"<tr><td>{s.topic}</td><td>{s.source_name}</td>"
            f"<td>{stream_cell}</td><td style='text-align:right'>"
            f"{_format_messages(s.count)}</td></tr>"
        )
    header_style = f"text-align:left; border-bottom:1px solid {Colors.BORDER}"
    count_style = f"text-align:right; border-bottom:1px solid {Colors.BORDER}"
    table = (
        '<table style="width:100%; border-collapse:collapse; font-size:11px;">'
        "<tr>"
        f'<th style="{header_style}">Topic</th>'
        f'<th style="{header_style}">Source</th>'
        f'<th style="{header_style}">Stream</th>'
        f'<th style="{count_style}">Count</th>'
        "</tr>" + "".join(rows) + "</table>"
    )
    return (
        f'<details style="{style}">'
        f"<summary>{len(stats.streams)} streams / "
        f"{stats.window_seconds:.0f}s window{summary_extra}</summary>"
        f"{table}"
        "</details>"
    )


class WorkerStatusRow:
    """Widget to display the status of a single backend worker.

    Uses stable pane references that are updated in-place to avoid flicker.
    """

    def __init__(
        self,
        status: ServiceStatus,
        is_stale: bool,
        last_seen_seconds_ago: float | None = None,
    ) -> None:
        # Create stable pane references
        self._namespace_pane = pn.pane.HTML(
            width=WorkerUIConstants.NAMESPACE_WIDTH,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )
        self._worker_id_pane = pn.pane.HTML(
            width=WorkerUIConstants.WORKER_ID_WIDTH,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )
        self._status_pane = pn.pane.HTML(
            width=WorkerUIConstants.STATUS_WIDTH,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )
        self._version_pane = pn.pane.HTML(
            width=WorkerUIConstants.VERSION_WIDTH,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )
        self._uptime_pane = pn.pane.HTML(
            width=WorkerUIConstants.UPTIME_WIDTH,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )
        self._stats_pane = pn.pane.HTML(
            width=WorkerUIConstants.STATS_WIDTH,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )

        row = pn.Row(
            self._namespace_pane,
            self._worker_id_pane,
            self._status_pane,
            self._version_pane,
            self._uptime_pane,
            self._stats_pane,
            sizing_mode="stretch_width",
            margin=0,
        )
        self._details_pane = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(0, 10, 2, 10),
        )
        self._panel = pn.Column(
            row,
            self._details_pane,
            sizing_mode="stretch_width",
            styles={"border-bottom": f"1px solid {Colors.BORDER}"},
            margin=0,
        )

        self._last_stream_stats: StreamStats | None = None

        # Set initial content
        self.update(status, is_stale, last_seen_seconds_ago)

    def _get_status_color(self, status: ServiceStatus, is_stale: bool) -> str:
        """Get color for worker state, considering staleness."""
        if is_stale:
            # Graceful shutdown (inferred from timed-out stopping): show gray
            if status.state == ServiceState.stopping:
                return WorkerUIConstants.COLORS[ServiceState.stopped]
            return WorkerUIConstants.STALE_COLOR
        return WorkerUIConstants.COLORS.get(
            status.state, WorkerUIConstants.DEFAULT_COLOR
        )

    def _create_status_style(self, color: str) -> str:
        """Create CSS style for status indicator."""
        return (
            f"background-color: {color}; color: white; "
            f"padding: 2px 8px; border-radius: 3px; text-align: center; "
            f"font-weight: bold; font-size: 11px;"
        )

    def update(
        self,
        status: ServiceStatus,
        is_stale: bool,
        last_seen_seconds_ago: float | None = None,
    ) -> None:
        """Update the row content in-place."""
        # Namespace
        namespace_text = f"{status.instrument}:{status.namespace}"
        self._namespace_pane.object = f"<b>{namespace_text}</b>"

        # Worker ID (truncated)
        worker_id_short = status.worker_id[:8]
        self._worker_id_pane.object = f"<code>{worker_id_short}</code>"

        # Status indicator
        status_color = self._get_status_color(status, is_stale)
        if is_stale:
            # Distinguish graceful shutdown from unexpected disappearance
            is_graceful = status.state == ServiceState.stopping
            status_text = "STOPPED" if is_graceful else "STALE"
        else:
            status_text = status.state.value.upper()
        status_style = self._create_status_style(status_color)
        self._status_pane.object = f'<div style="{status_style}">{status_text}</div>'

        # Version
        self._version_pane.object = f"<code>{status.version}</code>"

        # Time info: show "Last seen X ago" for non-running workers, uptime otherwise
        show_last_seen = is_stale or status.state in (
            ServiceState.stopping,
            ServiceState.stopped,
            ServiceState.error,
        )
        if show_last_seen:
            time_text = f"Seen: {_format_duration(last_seen_seconds_ago)} ago"
        else:
            started_at = status.started_at
            uptime = self._calculate_uptime(started_at) if started_at else None
            time_text = f"Up: {_format_duration(uptime)}"
        self._uptime_pane.object = f"<span>{time_text}</span>"

        # Cache stream stats when a new snapshot arrives
        if status.stream_stats is not None:
            self._last_stream_stats = status.stream_stats

        # Stats
        jobs_text = f"Jobs: {status.active_job_count}"
        batch_text = f"Batch: {status.batch_interval_s:.0f}s"
        msgs_text = _format_stream_stats_summary(self._last_stream_stats)
        self._stats_pane.object = (
            f"<span>{jobs_text} | {msgs_text} | {batch_text}</span>"
        )

        # Full-width expandable stream details
        self._details_pane.object = _format_stream_stats_details(
            self._last_stream_stats
        )

    def _calculate_uptime(self, started_at: Timestamp) -> float:
        """Calculate uptime in seconds from started_at timestamp."""
        return (Timestamp.now() - started_at).to_seconds()

    @property
    def panel(self) -> pn.Column:
        """Get the panel for this widget."""
        return self._panel


class BackendStatusWidget:
    """Widget to display a list of backend service worker statuses."""

    def __init__(self, service_registry: ServiceRegistry) -> None:
        self._service_registry = service_registry
        self._worker_rows: dict[str, WorkerStatusRow] = {}
        self._empty_placeholder: pn.pane.HTML | None = None
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Set up the main layout."""
        self._clear_button = create_tool_button(
            icon_name='trash',
            button_color=ButtonStyles.DANGER_RED,
            hover_color=ButtonStyles.DANGER_HOVER,
            on_click_callback=self._on_clear_stopped,
        )
        self._clear_button.disabled = True
        self._clear_button.description = "Clear stopped workers"

        self._header = pn.pane.HTML(
            "<h3>Backend Workers</h3>",
            margin=WorkerUIConstants.HEADER_MARGIN,
        )

        # Summary row with clear button
        self._summary = pn.pane.HTML(
            self._format_summary(),
            margin=(5, 10),
        )
        self._summary_row = pn.Row(
            self._summary,
            pn.Spacer(),
            self._clear_button,
            sizing_mode="stretch_width",
            align="center",
            margin=0,
        )

        # Table header
        header_style = (
            "font-weight: bold; font-size: 12px; "
            f"background-color: {Colors.BG_LIGHT}; padding: 5px 0;"
        )
        self._table_header = pn.Row(
            pn.pane.HTML(
                f'<span style="{header_style}">Namespace</span>',
                width=WorkerUIConstants.NAMESPACE_WIDTH,
                margin=WorkerUIConstants.STANDARD_MARGIN,
            ),
            pn.pane.HTML(
                f'<span style="{header_style}">Worker ID</span>',
                width=WorkerUIConstants.WORKER_ID_WIDTH,
                margin=WorkerUIConstants.STANDARD_MARGIN,
            ),
            pn.pane.HTML(
                f'<span style="{header_style}">Status</span>',
                width=WorkerUIConstants.STATUS_WIDTH,
                margin=WorkerUIConstants.STANDARD_MARGIN,
            ),
            pn.pane.HTML(
                f'<span style="{header_style}">Version</span>',
                width=WorkerUIConstants.VERSION_WIDTH,
                margin=WorkerUIConstants.STANDARD_MARGIN,
            ),
            pn.pane.HTML(
                f'<span style="{header_style}">Uptime</span>',
                width=WorkerUIConstants.UPTIME_WIDTH,
                margin=WorkerUIConstants.STANDARD_MARGIN,
            ),
            pn.pane.HTML(
                f'<span style="{header_style}">Stats</span>',
                width=WorkerUIConstants.STATS_WIDTH,
                margin=WorkerUIConstants.STANDARD_MARGIN,
            ),
            styles={"border-bottom": f"2px solid {Colors.BORDER}"},
            sizing_mode="stretch_width",
            margin=(0, 10),
        )

        self._worker_list = pn.Column(sizing_mode="stretch_width", margin=(0, 10))

        # Initialize with current worker statuses
        self._update_worker_list()

    def _format_summary(self) -> str:
        """Format the summary text."""
        total = len(self._service_registry.worker_statuses)

        if total == 0:
            return "<i>No backend workers connected</i>"

        # Count workers by display state (considering staleness)
        starting_count = 0
        running_count = 0
        stopping_count = 0
        stopped_count = 0
        stale_count = 0
        error_count = 0

        for worker_key, status in self._service_registry.worker_statuses.items():
            is_stale = self._service_registry.is_status_stale(worker_key)

            if is_stale:
                # Distinguish graceful shutdown from unexpected disappearance
                if status.state in (ServiceState.stopping, ServiceState.stopped):
                    stopped_count += 1
                else:
                    stale_count += 1
            elif status.state == ServiceState.starting:
                starting_count += 1
            elif status.state == ServiceState.running:
                running_count += 1
            elif status.state == ServiceState.stopping:
                stopping_count += 1
            elif status.state == ServiceState.stopped:
                stopped_count += 1
            elif status.state == ServiceState.error:
                error_count += 1

        def _span(color: str, count: int, label: str) -> str:
            return f'<span style="color: {color}">{count} {label}</span>'

        colors = WorkerUIConstants.COLORS
        parts = [f"<b>{total}</b> workers"]
        if starting_count:
            parts.append(
                _span(colors[ServiceState.starting], starting_count, "starting")
            )
        if running_count:
            parts.append(_span(colors[ServiceState.running], running_count, "running"))
        if stopping_count:
            parts.append(
                _span(colors[ServiceState.stopping], stopping_count, "stopping")
            )
        if stopped_count:
            parts.append(_span(colors[ServiceState.stopped], stopped_count, "stopped"))
        if stale_count:
            parts.append(_span(WorkerUIConstants.STALE_COLOR, stale_count, "stale"))
        if error_count:
            parts.append(_span(colors[ServiceState.error], error_count, "error"))

        return " | ".join(parts)

    def _on_clear_stopped(self) -> None:
        """Handle clear stopped workers button click."""
        self._service_registry.remove_inactive_workers()

    def refresh(self) -> None:
        """Refresh the display with current worker states."""
        with pn.io.hold():
            self._summary.object = self._format_summary()
            self._update_worker_list()
            has_inactive = bool(self._service_registry.get_inactive_worker_keys())
            self._clear_button.disabled = not has_inactive

    def _update_worker_list(self) -> None:
        """Update the worker list, reusing existing rows where possible."""
        current_keys = set(self._service_registry.worker_statuses.keys())
        existing_keys = set(self._worker_rows.keys())

        # Remove rows for workers that no longer exist
        removed_keys = existing_keys - current_keys
        for worker_key in removed_keys:
            row = self._worker_rows.pop(worker_key)
            self._worker_list.remove(row.panel)

        # Remove empty placeholder if we have workers
        if current_keys and self._empty_placeholder is not None:
            self._worker_list.remove(self._empty_placeholder)
            self._empty_placeholder = None

        # Update existing rows and add new ones
        for worker_key, status in self._service_registry.worker_statuses.items():
            is_stale = self._service_registry.is_status_stale(worker_key)
            last_seen = self._service_registry.get_last_seen_seconds_ago(worker_key)

            if worker_key in self._worker_rows:
                # Update existing row in-place
                self._worker_rows[worker_key].update(status, is_stale, last_seen)
            else:
                # Create new row
                row = WorkerStatusRow(status, is_stale, last_seen)
                self._worker_rows[worker_key] = row
                self._worker_list.append(row.panel)

        # Show empty placeholder if no workers
        if not current_keys and self._empty_placeholder is None:
            color = WorkerUIConstants.DEFAULT_COLOR
            self._empty_placeholder = pn.pane.HTML(
                f'<div style="text-align: center; padding: 20px; color: {color};">'
                "Waiting for backend workers to connect..."
                "</div>",
                sizing_mode="stretch_width",
            )
            self._worker_list.append(self._empty_placeholder)

    def panel(self) -> pn.Column:
        """Get the main panel for this widget."""
        return pn.Column(
            self._header,
            self._summary_row,
            self._table_header,
            self._worker_list,
            sizing_mode="stretch_width",
        )
