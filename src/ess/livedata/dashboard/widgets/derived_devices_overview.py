# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Read-only "Derived devices" overview, opened from the sidebar.

Lists the NICOS derived devices currently exposed (contract ∩ running jobs):
device name, source, owning workflow, and run state. There is no per-card
deep-link mechanism in the dashboard today, so no link to the owning card is
shown.

The overview holds no persisted state. Its contents are derived from the
orchestrator's live running-sources snapshot intersected with the device
contract, and rebuilt on the workflow state-version counter so all sessions
stay in sync (see ``.claude/rules/dashboard-widgets.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import panel as pn

from ess.livedata.config.device_contract import DeviceContract

from ..derived_devices import exposed_devices
from .buttons import create_tool_button
from .styles import Colors, StatusColors

if TYPE_CHECKING:
    from ..job_orchestrator import JobOrchestrator
    from ..session_updater import SessionUpdater


class DerivedDevicesOverview:
    """Sidebar button + modal listing currently-exposed NICOS devices.

    Parameters
    ----------
    orchestrator:
        Job orchestrator providing the live running-sources snapshot, the
        workflow registry (for display titles), and state versions.
    device_contract:
        Per-instrument device contract intersected with running jobs.
    """

    def __init__(
        self,
        *,
        orchestrator: JobOrchestrator,
        device_contract: DeviceContract,
    ) -> None:
        self._orchestrator = orchestrator
        self._device_contract = device_contract
        self._last_signature: tuple[int, ...] | None = None

        self._body = pn.Column(sizing_mode='stretch_width')
        self._modal = pn.Modal(
            pn.Column(
                pn.pane.Markdown('### Derived devices', margin=(0, 0, 5, 0)),
                self._body,
                sizing_mode='stretch_width',
            ),
            name='Derived devices',
            margin=20,
            width=520,
        )

        open_btn = create_tool_button(
            icon_name='stack',
            button_color=StatusColors.PRIMARY,
            hover_color='rgba(0, 123, 255, 0.1)',
            on_click_callback=self._open,
            css_classes=['lt-derived-devices-open'],
        )
        label = pn.pane.HTML(
            f'<span style="font-size: 13px; color: {Colors.TEXT};">'
            'Derived devices</span>',
            styles={'display': 'flex', 'align-items': 'center'},
        )
        self.panel = pn.Column(
            pn.Row(open_btn, label, align='center'),
            self._modal,
            sizing_mode='stretch_width',
        )
        self._rebuild_body()

    def register_periodic_refresh(self, session_updater: SessionUpdater) -> None:
        """Rebuild the device list when the exposed-device set may have changed."""
        session_updater.register_custom_handler(self._refresh)

    def _open(self) -> None:
        self._rebuild_body()
        self._modal.open = True

    def _signature(self) -> tuple[int, ...]:
        """Cheap change signature: per-workflow state versions.

        Captures every staging/commit/stop, which is the only way the exposed
        device set can change, so rebuilding only on signature change avoids
        per-tick churn.
        """
        return tuple(
            self._orchestrator.get_workflow_state_version(workflow_id)
            for workflow_id in self._orchestrator.get_workflow_registry()
        )

    def _refresh(self) -> None:
        signature = self._signature()
        if signature != self._last_signature:
            self._rebuild_body()

    def _rebuild_body(self) -> None:
        self._last_signature = self._signature()
        running = self._orchestrator.get_running_workflow_sources()
        devices = exposed_devices(running, self._device_contract)
        registry = self._orchestrator.get_workflow_registry()

        if not devices:
            rows: list[pn.viewable.Viewable] = [
                pn.pane.HTML(
                    f'<span style="font-size: 13px; color: {Colors.TEXT_MUTED};">'
                    'No devices are currently exposed. A device is exposed only '
                    'while its owning workflow job is running.</span>',
                    sizing_mode='stretch_width',
                )
            ]
        else:
            rows = [self._header_row()]
            for device in devices:
                spec = registry.get(device.workflow_id)
                workflow_title = (
                    spec.title if spec is not None else str(device.workflow_id)
                )
                source_title = self._orchestrator.get_source_title(device.source_name)
                rows.append(
                    self._device_row(device.device_name, source_title, workflow_title)
                )

        with pn.io.hold():
            self._body[:] = rows

    @staticmethod
    def _header_row() -> pn.pane.HTML:
        return pn.pane.HTML(
            '<div style="display: grid; '
            'grid-template-columns: 1.2fr 1fr 1.4fr 0.8fr; gap: 8px; '
            f'font-size: 11px; color: {Colors.TEXT_MUTED}; '
            'text-transform: uppercase; letter-spacing: 0.5px; '
            f'padding: 4px 0; border-bottom: 1px solid {Colors.BORDER};">'
            '<span>Device</span><span>Source</span>'
            '<span>Workflow</span><span>State</span></div>',
            sizing_mode='stretch_width',
        )

    @staticmethod
    def _device_row(device: str, source: str, workflow: str) -> pn.pane.HTML:
        return pn.pane.HTML(
            '<div style="display: grid; '
            'grid-template-columns: 1.2fr 1fr 1.4fr 0.8fr; gap: 8px; '
            f'font-size: 12px; color: {Colors.TEXT}; '
            f'padding: 4px 0; border-bottom: 1px solid {Colors.BORDER};">'
            f'<span style="font-weight: 600;">{device}</span>'
            f'<span>{source}</span><span>{workflow}</span>'
            f'<span style="color: {StatusColors.SUCCESS}; font-weight: 600;">'
            'running</span></div>',
            sizing_mode='stretch_width',
        )
