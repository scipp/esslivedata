# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Widget to display backend service worker status."""

from __future__ import annotations

from typing import ClassVar

import panel as pn

from ess.livedata.core.job import ServiceState, ServiceStatus
from ess.livedata.dashboard.service_registry import ServiceRegistry


class WorkerUIConstants:
    """Constants for worker status UI styling and sizing."""

    # Colors for service states
    COLORS: ClassVar[dict[ServiceState, str]] = {
        ServiceState.starting: "#6c757d",  # Gray
        ServiceState.running: "#28a745",  # Green
        ServiceState.stopping: "#ffc107",  # Yellow
        ServiceState.error: "#dc3545",  # Red
    }
    DEFAULT_COLOR = "#6c757d"
    STALE_COLOR = "#dc3545"  # Red for stale workers

    # Sizes
    NAMESPACE_WIDTH = 200
    WORKER_ID_WIDTH = 150
    STATUS_WIDTH = 90
    UPTIME_WIDTH = 100
    STATS_WIDTH = 180
    ROW_HEIGHT = 35

    # Margins
    STANDARD_MARGIN = (5, 5)
    HEADER_MARGIN = (10, 10, 5, 10)


def _format_uptime(uptime_seconds: float | None) -> str:
    """Format uptime in human-readable form."""
    if uptime_seconds is None:
        return "Unknown"

    if uptime_seconds < 60:
        value = int(uptime_seconds)
        unit = "second" if value == 1 else "seconds"
        return f"{value} {unit}"
    elif uptime_seconds < 3600:
        value = uptime_seconds / 60
        unit = "minute" if value == 1 else "minutes"
        return f"{value:.1f} {unit}"
    elif uptime_seconds < 86400:
        value = uptime_seconds / 3600
        unit = "hour" if value == 1 else "hours"
        return f"{value:.1f} {unit}"
    else:
        value = uptime_seconds / 86400
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


class WorkerStatusRow:
    """Widget to display the status of a single backend worker.

    Uses stable pane references that are updated in-place to avoid flicker.
    """

    def __init__(self, status: ServiceStatus, is_stale: bool) -> None:
        # Create stable pane references
        self._namespace_pane = pn.pane.HTML(
            width=WorkerUIConstants.NAMESPACE_WIDTH,
            height=WorkerUIConstants.ROW_HEIGHT,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )
        self._worker_id_pane = pn.pane.HTML(
            width=WorkerUIConstants.WORKER_ID_WIDTH,
            height=WorkerUIConstants.ROW_HEIGHT,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )
        self._status_pane = pn.pane.HTML(
            width=WorkerUIConstants.STATUS_WIDTH,
            height=WorkerUIConstants.ROW_HEIGHT,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )
        self._uptime_pane = pn.pane.HTML(
            width=WorkerUIConstants.UPTIME_WIDTH,
            height=WorkerUIConstants.ROW_HEIGHT,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )
        self._stats_pane = pn.pane.HTML(
            width=WorkerUIConstants.STATS_WIDTH,
            height=WorkerUIConstants.ROW_HEIGHT,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )

        self._panel = pn.Row(
            self._namespace_pane,
            self._worker_id_pane,
            self._status_pane,
            self._uptime_pane,
            self._stats_pane,
            styles={"border-bottom": "1px solid #dee2e6"},
            sizing_mode="stretch_width",
        )

        # Set initial content
        self.update(status, is_stale)

    def _get_status_color(self, status: ServiceStatus, is_stale: bool) -> str:
        """Get color for worker state, considering staleness."""
        if is_stale:
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

    def update(self, status: ServiceStatus, is_stale: bool) -> None:
        """Update the row content in-place."""
        # Namespace
        namespace_text = f"{status.instrument}:{status.namespace}"
        self._namespace_pane.object = f"<b>{namespace_text}</b>"

        # Worker ID (truncated)
        worker_id_short = status.worker_id[:8]
        self._worker_id_pane.object = f"<code>{worker_id_short}</code>"

        # Status indicator
        status_color = self._get_status_color(status, is_stale)
        status_text = "STALE" if is_stale else status.state.value.upper()
        status_style = self._create_status_style(status_color)
        self._status_pane.object = f'<div style="{status_style}">{status_text}</div>'

        # Uptime - get from registry at update time
        uptime_text = _format_uptime(
            self._calculate_uptime(status.started_at) if status.started_at else None
        )
        self._uptime_pane.object = f"<span>Up: {uptime_text}</span>"

        # Stats
        jobs_text = f"Jobs: {status.active_job_count}"
        msgs_text = f"Msgs: {_format_messages(status.messages_processed)}"
        self._stats_pane.object = f"<span>{jobs_text} | {msgs_text}</span>"

    def _calculate_uptime(self, started_at_ns: int) -> float:
        """Calculate uptime in seconds from started_at timestamp."""
        import time

        current_time_ns = time.time_ns()
        uptime_ns = current_time_ns - started_at_ns
        return uptime_ns / 1_000_000_000

    @property
    def panel(self) -> pn.Row:
        """Get the panel for this widget."""
        return self._panel


class BackendStatusWidget:
    """Widget to display a list of backend service worker statuses."""

    def __init__(self, service_registry: ServiceRegistry) -> None:
        self._service_registry = service_registry
        self._worker_rows: dict[str, WorkerStatusRow] = {}
        self._empty_placeholder: pn.pane.HTML | None = None
        self._setup_layout()

        # Subscribe to service registry updates
        self._service_registry.register_update_subscriber(self._on_status_update)

    def _setup_layout(self) -> None:
        """Set up the main layout."""
        self._header = pn.pane.HTML(
            "<h3>Backend Workers</h3>",
            margin=WorkerUIConstants.HEADER_MARGIN,
        )

        # Summary row
        self._summary = pn.pane.HTML(
            self._format_summary(),
            margin=(5, 10),
        )

        # Table header
        header_style = (
            "font-weight: bold; font-size: 12px; "
            "background-color: #f8f9fa; padding: 5px 0;"
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
                f'<span style="{header_style}">Uptime</span>',
                width=WorkerUIConstants.UPTIME_WIDTH,
                margin=WorkerUIConstants.STANDARD_MARGIN,
            ),
            pn.pane.HTML(
                f'<span style="{header_style}">Stats</span>',
                width=WorkerUIConstants.STATS_WIDTH,
                margin=WorkerUIConstants.STANDARD_MARGIN,
            ),
            styles={"border-bottom": "2px solid #dee2e6"},
            sizing_mode="stretch_width",
        )

        self._worker_list = pn.Column(sizing_mode="stretch_width", margin=(0, 10))

        # Initialize with current worker statuses
        self._update_worker_list()

    def _format_summary(self) -> str:
        """Format the summary text."""
        total = len(self._service_registry.worker_statuses)
        stale = len(self._service_registry.get_stale_workers())

        if total == 0:
            return "<i>No backend workers connected</i>"

        status_counts: dict[ServiceState, int] = {}
        for status in self._service_registry.worker_statuses.values():
            status_counts[status.state] = status_counts.get(status.state, 0) + 1

        parts = [f"<b>{total}</b> workers"]
        if running := status_counts.get(ServiceState.running, 0):
            parts.append(f'<span style="color: #28a745">{running} running</span>')
        if stale > 0:
            parts.append(f'<span style="color: #dc3545">{stale} stale</span>')
        if error := status_counts.get(ServiceState.error, 0):
            parts.append(f'<span style="color: #dc3545">{error} error</span>')

        return " | ".join(parts)

    def _on_status_update(self) -> None:
        """Handle worker status updates from the registry."""
        with pn.io.hold():
            self._summary.object = self._format_summary()
            self._update_worker_list()

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

            if worker_key in self._worker_rows:
                # Update existing row in-place
                self._worker_rows[worker_key].update(status, is_stale)
            else:
                # Create new row
                row = WorkerStatusRow(status, is_stale)
                self._worker_rows[worker_key] = row
                self._worker_list.append(row.panel)

        # Show empty placeholder if no workers
        if not current_keys and self._empty_placeholder is None:
            self._empty_placeholder = pn.pane.HTML(
                '<div style="text-align: center; padding: 20px; color: #6c757d;">'
                "Waiting for backend workers to connect..."
                "</div>",
                sizing_mode="stretch_width",
            )
            self._worker_list.append(self._empty_placeholder)

    def panel(self) -> pn.Column:
        """Get the main panel for this widget."""
        return pn.Column(
            self._header,
            self._summary,
            self._table_header,
            self._worker_list,
            sizing_mode="stretch_width",
        )
