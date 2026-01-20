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
        return f"{uptime_seconds:.0f}s"
    elif uptime_seconds < 3600:
        minutes = uptime_seconds / 60
        return f"{minutes:.1f}m"
    elif uptime_seconds < 86400:
        hours = uptime_seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = uptime_seconds / 86400
        return f"{days:.1f}d"


def _format_messages(count: int) -> str:
    """Format message count in human-readable form."""
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1_000_000:.1f}M"


class WorkerStatusWidget:
    """Widget to display the status of a single backend worker."""

    def __init__(
        self, status: ServiceStatus, is_stale: bool, uptime_seconds: float | None
    ) -> None:
        self._status = status
        self._is_stale = is_stale
        self._uptime_seconds = uptime_seconds
        self._panel = self._create_panel()

    def _get_status_color(self) -> str:
        """Get color for worker state, considering staleness."""
        if self._is_stale:
            return WorkerUIConstants.STALE_COLOR
        return WorkerUIConstants.COLORS.get(
            self._status.state, WorkerUIConstants.DEFAULT_COLOR
        )

    def _create_status_style(self, color: str) -> str:
        """Create CSS style for status indicator."""
        return (
            f"background-color: {color}; color: white; "
            f"padding: 2px 8px; border-radius: 3px; text-align: center; "
            f"font-weight: bold; font-size: 11px;"
        )

    def _create_panel(self) -> pn.Row:
        """Create the panel layout for this worker."""
        # Namespace display
        namespace_text = f"{self._status.instrument}:{self._status.namespace}"
        namespace_pane = pn.pane.HTML(
            f"<b>{namespace_text}</b>",
            width=WorkerUIConstants.NAMESPACE_WIDTH,
            height=WorkerUIConstants.ROW_HEIGHT,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )

        # Worker ID (truncated)
        worker_id_short = self._status.worker_id[:8]
        worker_id_pane = pn.pane.HTML(
            f"<code>{worker_id_short}</code>",
            width=WorkerUIConstants.WORKER_ID_WIDTH,
            height=WorkerUIConstants.ROW_HEIGHT,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )

        # Status indicator
        status_color = self._get_status_color()
        status_text = "STALE" if self._is_stale else self._status.state.value.upper()
        status_style = self._create_status_style(status_color)
        status_pane = pn.pane.HTML(
            f'<div style="{status_style}">{status_text}</div>',
            width=WorkerUIConstants.STATUS_WIDTH,
            height=WorkerUIConstants.ROW_HEIGHT,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )

        # Uptime
        uptime_text = _format_uptime(self._uptime_seconds)
        uptime_pane = pn.pane.HTML(
            f"<span>Up: {uptime_text}</span>",
            width=WorkerUIConstants.UPTIME_WIDTH,
            height=WorkerUIConstants.ROW_HEIGHT,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )

        # Stats
        jobs_text = f"Jobs: {self._status.active_job_count}"
        msgs_text = f"Msgs: {_format_messages(self._status.messages_processed)}"
        stats_pane = pn.pane.HTML(
            f"<span>{jobs_text} | {msgs_text}</span>",
            width=WorkerUIConstants.STATS_WIDTH,
            height=WorkerUIConstants.ROW_HEIGHT,
            margin=WorkerUIConstants.STANDARD_MARGIN,
        )

        # Error indicator if present
        error_pane = None
        if self._status.error:
            error_style = (
                "color: #dc3545; font-size: 11px; "
                "white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"
            )
            error_text = self._status.error.split('\n')[0][:50]
            error_pane = pn.pane.HTML(
                f'<span style="{error_style}">{error_text}</span>',
                margin=WorkerUIConstants.STANDARD_MARGIN,
            )

        row_components = [
            namespace_pane,
            worker_id_pane,
            status_pane,
            uptime_pane,
            stats_pane,
        ]
        if error_pane:
            row_components.append(error_pane)

        return pn.Row(
            *row_components,
            styles={
                "border-bottom": "1px solid #dee2e6",
            },
            sizing_mode="stretch_width",
        )

    def panel(self) -> pn.Row:
        """Get the panel for this widget."""
        return self._panel


class BackendStatusWidget:
    """Widget to display a list of backend service worker statuses."""

    def __init__(self, service_registry: ServiceRegistry) -> None:
        self._service_registry = service_registry
        self._worker_widgets: dict[str, WorkerStatusWidget] = {}
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
        self._rebuild_worker_list()

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
            self._rebuild_worker_list()

    def _rebuild_worker_list(self) -> None:
        """Rebuild the worker list from current registry state."""
        self._worker_list.clear()
        self._worker_widgets.clear()

        # Sort workers by namespace then worker_id for consistent display
        workers = sorted(
            self._service_registry.worker_statuses.items(),
            key=lambda x: (x[1].instrument, x[1].namespace, x[1].worker_id),
        )

        for worker_key, status in workers:
            is_stale = self._service_registry.is_status_stale(worker_key)
            uptime = self._service_registry.get_worker_uptime_seconds(worker_key)
            widget = WorkerStatusWidget(status, is_stale, uptime)
            self._worker_widgets[worker_key] = widget
            self._worker_list.append(widget.panel())

        if not workers:
            self._worker_list.append(
                pn.pane.HTML(
                    '<div style="text-align: center; padding: 20px; color: #6c757d;">'
                    "Waiting for backend workers to connect..."
                    "</div>",
                    sizing_mode="stretch_width",
                )
            )

    def panel(self) -> pn.Column:
        """Get the main panel for this widget."""
        return pn.Column(
            self._header,
            self._summary,
            self._table_header,
            self._worker_list,
            sizing_mode="stretch_width",
        )
