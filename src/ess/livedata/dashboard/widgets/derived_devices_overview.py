# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Read-only "Derived devices" overview, opened from the workflow-list toolbar.

Lists every NICOS derived device the contract declares: device name, source,
owning workflow, and live state (running iff its owning job is running). There
is no per-card deep-link mechanism in the dashboard today, so no link to the
owning card is shown.

The overview holds no persisted state. Its contents are the static device
contract annotated with the orchestrator's live running-sources snapshot, and
rebuilt on the workflow state-version counter so all sessions stay in sync (see
``.claude/rules/dashboard-widgets.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import panel as pn

from ess.livedata.config.device_contract import DeviceContract

from ..derived_devices import all_devices
from .styles import Colors, StatusColors

if TYPE_CHECKING:
    from ..job_orchestrator import JobOrchestrator
    from ..session_updater import SessionUpdater

_DESCRIPTION = (
    'Selected workflow outputs are published as devices to NICOS. A device is '
    'readable by NICOS only while its owning workflow job is running. Stopping, '
    'resetting, or reconfiguring that workflow takes the device offline and will '
    'disrupt a count command or scan command in NICOS that relies on a given device.'
)


class DerivedDevicesOverview:
    """Modal listing every NICOS device the contract declares, with live state.

    Owns only the modal and its content; the trigger button lives in the
    workflow-list toolbar and calls :meth:`open`.

    Parameters
    ----------
    orchestrator:
        Job orchestrator providing the live running-sources snapshot, the
        workflow registry (for display titles), and state versions.
    device_contract:
        Per-instrument device contract.
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

        self._gate_checkbox = pn.widgets.Checkbox(
            label='Confirm before disrupting a device NICOS may be using',
            value=orchestrator.gate_enabled,
            css_classes=['lt-gate-toggle'],
        )
        self._gate_checkbox.param.watch(self._on_gate_toggle, 'value')

        self._body = pn.Column(sizing_mode='stretch_width')
        self.modal = pn.Modal(
            pn.Column(
                pn.pane.Markdown('### NICOS derived devices', margin=(0, 0, 5, 0)),
                pn.pane.HTML(
                    f'<div style="font-size: 12px; color: {Colors.TEXT_MUTED}; '
                    f'line-height: 1.5; margin-bottom: 10px;">{_DESCRIPTION}</div>',
                    sizing_mode='stretch_width',
                ),
                self._gate_checkbox,
                pn.pane.HTML(
                    f'<div style="font-size: 11px; color: {Colors.TEXT_MUTED}; '
                    f'line-height: 1.4; margin: 2px 0 12px 0;">Applies to all '
                    'sessions. Uncheck to skip confirmations while no NICOS scan '
                    'is running.</div>',
                    sizing_mode='stretch_width',
                ),
                self._body,
                sizing_mode='stretch_width',
            ),
            name='NICOS derived devices',
            margin=20,
            width=520,
        )
        self._rebuild_body()

    def register_periodic_refresh(self, session_updater: SessionUpdater) -> None:
        """Rebuild the device list when the live-device set may have changed."""
        session_updater.register_custom_handler(self._refresh)

    def open(self) -> None:
        """Rebuild the list and open the modal."""
        self._rebuild_body()
        self._sync_gate_checkbox()
        self.modal.open = True

    def _on_gate_toggle(self, event) -> None:
        """Apply a checkbox toggle to the shared, cross-session gate flag."""
        self._orchestrator.set_gate_enabled(event.new)

    def _sync_gate_checkbox(self) -> None:
        """Reflect the shared gate flag, picking up toggles from other sessions.

        Assigning an unchanged value re-enters :meth:`_on_gate_toggle`, but
        :meth:`JobOrchestrator.set_gate_enabled` is a no-op for equal values, so
        there is no feedback loop.
        """
        enabled = self._orchestrator.gate_enabled
        if self._gate_checkbox.value != enabled:
            self._gate_checkbox.value = enabled

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
        self._sync_gate_checkbox()

    def _rebuild_body(self) -> None:
        self._last_signature = self._signature()
        running = self._orchestrator.get_running_workflow_sources()
        devices = all_devices(self._device_contract)
        registry = self._orchestrator.get_workflow_registry()

        if not devices:
            content = pn.pane.HTML(
                f'<span style="font-size: 13px; color: {Colors.TEXT_MUTED};">'
                'No derived devices are declared for this instrument.</span>',
                sizing_mode='stretch_width',
            )
        else:
            items = []
            for device in devices:
                spec = registry.get(device.workflow_id)
                workflow_title = (
                    spec.title if spec is not None else str(device.workflow_id)
                )
                items.append(
                    self._device_item_html(
                        device.device_name,
                        self._orchestrator.get_source_title(device.source_name),
                        workflow_title,
                        is_running=device.is_running_in(running),
                    )
                )
            content = pn.pane.HTML(''.join(items), sizing_mode='stretch_width')

        with pn.io.hold():
            self._body[:] = [content]

    @staticmethod
    def _device_item_html(
        device: str, source: str, workflow: str, *, is_running: bool
    ) -> str:
        """One device as a two-line list item: name, then ``source · workflow``.

        State is right-aligned so the badges line up regardless of name length;
        the device name gets the full row width and rarely needs to wrap.
        """
        if is_running:
            state = (
                f'<span style="color: {StatusColors.SUCCESS}; font-weight: 600;">'
                'running</span>'
            )
        else:
            state = f'<span style="color: {Colors.TEXT_MUTED};">stopped</span>'
        return (
            '<div style="display: flex; justify-content: space-between; '
            'align-items: baseline; gap: 12px; padding: 8px 0; '
            f'border-bottom: 1px solid {Colors.BORDER};">'
            '<div style="min-width: 0;">'
            f'<div style="font-weight: 600; color: {Colors.TEXT}; '
            f'font-size: 13px; overflow-wrap: anywhere;">{device}</div>'
            f'<div style="font-size: 11px; color: {Colors.TEXT_MUTED}; '
            f'margin-top: 2px;">{source} · {workflow}</div>'
            '</div>'
            f'<span style="flex: none; font-size: 12px;">{state}</span>'
            '</div>'
        )
