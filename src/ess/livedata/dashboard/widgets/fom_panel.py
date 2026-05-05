# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Figure-of-merit panel: per-slot rows, slot configuration wizard, and
modal-based UX for the dedicated FOM tab.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import panel as pn
import pydantic
import scipp as sc
import structlog

from ess.livedata.config.workflow_spec import (
    JobId,
    ResultKey,
    WorkflowId,
    WorkflowSpec,
)

from ..configuration_adapter import ConfigurationState
from ..data_service import DataService
from ..fom_orchestrator import FOMOrchestrator, FOMSlot, FOMSlotState
from ..job_service import JobService
from ..notifications import show_error
from ..workflow_configuration_adapter import WorkflowConfigurationAdapter
from .buttons import ButtonStyles, create_tool_button
from .configuration_widget import ConfigurationPanel
from .plot_config_modal import (
    OutputSelection,
    WorkflowAndOutputSelectionStep,
)
from .styles import Colors, HoverColors, ModalSizing, StatusColors, WarningBox
from .wizard import Wizard, WizardStep
from .workflow_status_widget import (
    WorkflowWidgetStyles,
    derive_aggregate_status,
    make_status_badge_html,
    make_status_dots_html,
    make_timing_html,
)

logger = structlog.get_logger(__name__)


_NICOS_SCAN_WARNING = (
    "If NICOS is currently using this figure-of-merit to control a scan, "
    "this action will likely break or interfere with the scan. "
    "Double-check that no scan is in progress before continuing."
)


def _show_dangerous_confirm(
    *,
    modal_container: pn.layout.ListLike,
    title: str,
    warning: str,
    notes: str | None = None,
    action_label: str,
    on_confirm: Callable[[], object],
) -> None:
    """Open a styled confirm modal for an action that risks wasting beam time."""
    confirm_btn = pn.widgets.Button(name=action_label, button_type="danger")
    cancel_btn = pn.widgets.Button(name="Cancel", button_type="light")

    title_html = pn.pane.HTML(
        f'<div style="font-size: 16px; font-weight: 700; '
        f'color: {Colors.TEXT_DARK}; margin-bottom: 4px;">{title}</div>'
    )
    warning_html = pn.pane.HTML(
        f'<div style="background: {WarningBox.BG}; '
        f'border-left: 4px solid {WarningBox.BORDER}; '
        f'color: {WarningBox.TEXT}; padding: 10px 12px; border-radius: 4px; '
        f'font-size: 13px; line-height: 1.45;">'
        f'<strong>Caution &mdash; potential beam-time impact.</strong><br>'
        f'{warning}</div>',
        sizing_mode='stretch_width',
    )

    items: list = [title_html, warning_html]
    if notes:
        items.append(
            pn.pane.HTML(
                f'<div style="font-size: 12px; color: {Colors.TEXT_MUTED};">'
                f'{notes}</div>'
            )
        )
    items.append(
        pn.Row(pn.Spacer(), cancel_btn, confirm_btn, sizing_mode='stretch_width')
    )

    modal = pn.Modal(
        pn.Column(*items, sizing_mode='stretch_width'),
        name=action_label,
        margin=20,
        width=460,
    )

    def _dismiss() -> None:
        modal.open = False
        if modal in modal_container:
            modal_container.remove(modal)

    def _on_confirm_click(_event) -> None:
        _dismiss()
        on_confirm()

    def _on_cancel_click(_event) -> None:
        _dismiss()

    confirm_btn.on_click(_on_confirm_click)
    cancel_btn.on_click(_on_cancel_click)

    modal_container.append(modal)
    modal.open = True


@dataclass(frozen=True)
class _FOMPrefill:
    """Adapter exposing the WorkflowOutputPrefill protocol for an FOM slot."""

    workflow_id: WorkflowId
    output_name: str


@dataclass(frozen=True)
class _ParamCommit:
    """Output of the parameter step."""

    source_names: list[str]
    parameter_values: pydantic.BaseModel | None
    aux_source_names: dict[str, str]


class _ParameterStep(WizardStep[OutputSelection, _ParamCommit]):
    """Step 2: workflow parameter configuration for a chosen workflow+output."""

    def __init__(
        self,
        *,
        workflow_registry: dict[WorkflowId, WorkflowSpec],
        prefill_state: ConfigurationState | None = None,
        prefill_sources: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._workflow_registry = workflow_registry
        self._prefill_state = prefill_state
        self._prefill_sources = prefill_sources
        self._panel_container = pn.Column(sizing_mode='stretch_width')
        self._config_panel: ConfigurationPanel | None = None
        self._adapter: WorkflowConfigurationAdapter | None = None
        self._captured: _ParamCommit | None = None
        self._workflow_id: WorkflowId | None = None
        self._output_name: str | None = None

    @property
    def name(self) -> str:
        return "Configure Parameters"

    @property
    def description(self) -> str | None:
        return (
            "Pick one or more sources and configure workflow parameters. "
            "Selecting multiple sources binds them all to this slot's alias; "
            "the consumer (e.g. NICOS) aggregates the substreams."
        )

    def is_valid(self) -> bool:
        return self._config_panel is not None

    def render_content(self) -> pn.Column:
        return self._panel_container

    def on_enter(self, input_data: OutputSelection | None) -> None:
        if input_data is None:
            return
        if (
            input_data.workflow_id == self._workflow_id
            and input_data.output_name == self._output_name
            and self._config_panel is not None
        ):
            return
        self._workflow_id = input_data.workflow_id
        self._output_name = input_data.output_name
        self._build_panel()

    def commit(self) -> _ParamCommit | None:
        if self._config_panel is None:
            return None
        self._captured = None
        if not self._config_panel.validate_and_execute():
            return None
        return self._captured

    def _build_panel(self) -> None:
        if self._workflow_id is None:
            return
        spec = self._workflow_registry.get(self._workflow_id)
        if spec is None:
            self._panel_container.clear()
            self._panel_container.append(pn.pane.Markdown("*Workflow not found.*"))
            self._adapter = None
            self._config_panel = None
            self._notify_ready_changed(False)
            return

        if self._prefill_sources:
            initial_sources = [
                s for s in self._prefill_sources if s in spec.source_names
            ] or None
        else:
            initial_sources = None

        def _on_start(
            selected_sources: list[str],
            parameter_values: pydantic.BaseModel,
            aux_source_names: dict[str, str] | None,
        ) -> None:
            if not selected_sources:
                return
            self._captured = _ParamCommit(
                source_names=list(selected_sources),
                parameter_values=parameter_values,
                aux_source_names=dict(aux_source_names or {}),
            )

        self._adapter = WorkflowConfigurationAdapter(
            spec=spec,
            config_state=self._prefill_state,
            start_callback=_on_start,
            initial_source_names=initial_sources,
        )
        self._config_panel = ConfigurationPanel(config=self._adapter)
        self._panel_container.clear()
        self._panel_container.append(self._config_panel.panel)
        self._notify_ready_changed(True)


@dataclass(frozen=True)
class _CommitArgs:
    workflow_id: WorkflowId
    source_names: list[str]
    output_name: str
    params: pydantic.BaseModel | None
    aux_source_names: dict[str, str]


class FOMConfigWizard:
    """Modal hosting the two-step FOM wizard for a single slot."""

    def __init__(
        self,
        *,
        slot: FOMSlot,
        orchestrator: FOMOrchestrator,
        modal_container: pn.layout.Column | pn.layout.Row,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._slot = slot
        self._orchestrator = orchestrator
        self._modal_container = modal_container
        self._on_close = on_close
        self._modal: pn.Modal | None = None
        self._pending_commit: _CommitArgs | None = None
        self._build()

    @property
    def modal(self) -> pn.Modal:
        if self._modal is None:
            raise RuntimeError("FOMConfigWizard has not been built")
        return self._modal

    def show(self) -> None:
        if self._modal is not None:
            self._modal.open = True

    def _build(self) -> None:
        registry = dict(self._orchestrator.get_workflow_registry())
        state = self._orchestrator.get_slot_state(self._slot)

        prefill: _FOMPrefill | None = None
        prefill_state: ConfigurationState | None = None
        prefill_sources: list[str] | None = None
        if state is not None:
            prefill = _FOMPrefill(
                workflow_id=state.workflow_id,
                output_name=state.output_name,
            )
            prefill_state = ConfigurationState(
                params=dict(state.params),
                aux_source_names=dict(state.aux_source_names),
            )
            prefill_sources = list(state.source_names)

        step1 = WorkflowAndOutputSelectionStep(
            registry, initial_config=prefill, include_static_overlay=False
        )
        step2 = _ParameterStep(
            workflow_registry=registry,
            prefill_state=prefill_state,
            prefill_sources=prefill_sources,
        )

        self._wizard = Wizard(
            steps=[step1, step2],
            on_complete=self._on_complete,
            on_cancel=self._on_cancel,
            action_button_label="Apply",
        )
        self._wizard.reset()

        self._modal = pn.Modal(
            self._wizard.render(),
            name=f"Configure {self._slot}",
            margin=20,
            width=ModalSizing.WIDTH,
        )

    def _on_complete(self, result: _ParamCommit) -> None:
        # The wizard's last step is the parameter step; result == _ParamCommit.
        # Step 1's selection is captured in step 2 via on_enter.
        # We pull workflow_id/output_name from step 1's last commit.
        step1 = self._wizard._steps[0]  # type: ignore[attr-defined]
        out: OutputSelection | None = step1.commit()
        if out is None or result is None:
            return

        commit_args = _CommitArgs(
            workflow_id=out.workflow_id,
            source_names=list(result.source_names),
            output_name=out.output_name,
            params=result.parameter_values,
            aux_source_names=result.aux_source_names,
        )

        existing = self._orchestrator.get_slot_state(self._slot)
        if existing is not None and existing.is_running:
            self._pending_commit = commit_args
            self._show_replace_confirm()
        else:
            self._do_commit(commit_args)

    def _on_cancel(self) -> None:
        self._close_modal()

    def _close_modal(self) -> None:
        if self._modal is not None:
            self._modal.open = False
        if self._on_close is not None:
            self._on_close()

    def _show_replace_confirm(self) -> None:
        existing = self._orchestrator.get_slot_state(self._slot)
        existing_label = (
            existing.workflow_id.name if existing is not None else self._slot
        )
        existing_sources = (
            ', '.join(existing.source_names)
            if existing is not None and existing.source_names
            else '—'
        )

        def _do_replace() -> None:
            args = self._pending_commit
            self._pending_commit = None
            if args is not None:
                self._do_commit(args)

        _show_dangerous_confirm(
            modal_container=self._modal_container,
            title=(
                f"Replace slot `{self._slot}` (currently bound to "
                f"`{existing_label}` on {existing_sources})?"
            ),
            warning=_NICOS_SCAN_WARNING,
            notes="Replacing will stop the running jobs and start new ones.",
            action_label="Replace",
            on_confirm=_do_replace,
        )

    def _do_commit(self, args: _CommitArgs) -> None:
        params_dict = (
            args.params.model_dump(mode='json')
            if isinstance(args.params, pydantic.BaseModel)
            else (args.params or {})
        )
        try:
            self._orchestrator.commit_slot(
                self._slot,
                workflow_id=args.workflow_id,
                source_names=args.source_names,
                output_name=args.output_name,
                params=params_dict,
                aux_source_names=args.aux_source_names,
            )
        except Exception:
            logger.exception("fom_commit_slot_failed", slot=self._slot)
            show_error(f"Failed to commit FOM slot {self._slot}")
            return
        self._close_modal()


def _format_value_text(value) -> str:
    """Render a scipp DataArray scalar (or other) as a single-line readout."""
    try:
        if isinstance(value, sc.DataArray):
            data = value.data
            if data.ndim == 0:
                unit = '' if data.unit is None else f' {data.unit}'
                return f'{float(data.value):.4g}{unit}'
            return f'{data.ndim}D ({list(data.dims)}, shape={list(data.shape)})'
    except Exception:
        logger.exception("fom_value_render_failed")
    return str(value)


class FOMSlotWidget:
    """Per-slot row: status, timing, live readout, and action buttons."""

    def __init__(
        self,
        *,
        slot: FOMSlot,
        orchestrator: FOMOrchestrator,
        job_service: JobService,
        data_service: DataService,
        modal_container: pn.layout.Column | pn.layout.Row,
    ) -> None:
        self._slot = slot
        self._orchestrator = orchestrator
        self._job_service = job_service
        self._data_service = data_service
        self._modal_container = modal_container

        self._status_badge: pn.pane.HTML | None = None
        self._dots: pn.pane.HTML | None = None
        self._timing: pn.pane.HTML | None = None
        self._readout: pn.pane.HTML | None = None
        self._buttons: pn.Row | None = None

        self._panel: pn.Column = pn.Column(sizing_mode='stretch_width')
        self._wizard: FOMConfigWizard | None = None

        self._last_state_version = -1
        self._build()

    def panel(self) -> pn.Column:
        return self._panel

    def refresh(self) -> None:
        version = self._orchestrator.get_slot_state_version(self._slot)
        if version != self._last_state_version:
            self._last_state_version = version
            self._build()
            return
        # Status & readout updates only.
        self._update_dynamic()

    def _build(self) -> None:
        with pn.io.hold():
            state = self._orchestrator.get_slot_state(self._slot)
            title_html = pn.pane.HTML(
                f'<span style="font-weight: 600; font-size: 14px; '
                f'color: {Colors.TEXT_DARK};">{self._slot}</span>',
                styles={'display': 'flex', 'align-items': 'center'},
            )

            status, color, timing_text, dots_html = self._derive_status(state)
            self._status_badge = pn.pane.HTML(make_status_badge_html(status, color))
            self._dots = pn.pane.HTML(dots_html)
            self._timing = pn.pane.HTML(make_timing_html(timing_text))
            self._readout = pn.pane.HTML(self._render_readout(state))
            self._buttons = self._make_buttons(state)

            description = pn.pane.HTML(
                self._render_binding_summary(state),
                sizing_mode='stretch_width',
                styles={'font-size': '12px', 'color': Colors.TEXT_MUTED},
            )

            header = pn.Row(
                title_html,
                pn.Spacer(width=12),
                self._status_badge,
                pn.Spacer(width=8),
                self._dots,
                pn.Spacer(width=12),
                self._timing,
                pn.Spacer(sizing_mode='stretch_width'),
                self._buttons,
                styles={
                    'background': Colors.BG_LIGHT,
                    'padding': '6px 12px',
                    'border-bottom': f'1px solid {Colors.BORDER}',
                },
                sizing_mode='stretch_width',
                align='center',
                height=WorkflowWidgetStyles.HEADER_HEIGHT,
            )

            body = pn.Column(
                description,
                self._readout,
                styles={'padding': '12px', 'background': 'white'},
                sizing_mode='stretch_width',
            )

            self._panel.clear()
            self._panel.extend(
                [
                    pn.Column(
                        header,
                        body,
                        styles={
                            'border': f'1px solid {Colors.BORDER}',
                            'border-radius': '6px',
                            'overflow': 'hidden',
                            'background': 'white',
                        },
                        sizing_mode='stretch_width',
                        margin=(0, 0, 8, 0),
                    )
                ]
            )

    def _update_dynamic(self) -> None:
        state = self._orchestrator.get_slot_state(self._slot)
        status, color, timing_text, dots_html = self._derive_status(state)
        with pn.io.hold():
            if self._status_badge is not None:
                new = make_status_badge_html(status, color)
                if self._status_badge.object != new:
                    self._status_badge.object = new
            if self._dots is not None and self._dots.object != dots_html:
                self._dots.object = dots_html
            if self._timing is not None:
                new = make_timing_html(timing_text)
                if self._timing.object != new:
                    self._timing.object = new
            if self._readout is not None:
                new = self._render_readout(state)
                if self._readout.object != new:
                    self._readout.object = new

    def _derive_status(self, state: FOMSlotState | None) -> tuple[str, str, str, str]:
        if state is None:
            return (
                'EMPTY',
                WorkflowWidgetStyles.STATUS_COLORS['stopped'],
                '',
                '',
            )
        if not state.is_running:
            return (
                'STOPPED',
                WorkflowWidgetStyles.STATUS_COLORS['stopped'],
                '',
                '',
            )
        status, color, timing_text, _, per_source = derive_aggregate_status(
            job_service=self._job_service,
            job_ids=state.job_ids,
        )
        return status, color, timing_text, make_status_dots_html(per_source)

    def _render_readout(self, state: FOMSlotState | None) -> str:
        if state is None:
            return (
                f'<div style="font-size: 13px; color: {Colors.TEXT_MUTED};">'
                f'Slot is empty. Click <em>Configure</em> to bind a workflow output.'
                f'</div>'
            )
        if not state.is_running:
            return (
                f'<div style="font-family: monospace; font-size: 18px; '
                f'color: {Colors.TEXT_DARK};">— (stopped)</div>'
            )
        rendered = [
            f'<tr><td style="padding: 1px 8px 1px 0; '
            f'color: {Colors.TEXT_MUTED}; white-space: nowrap;">'
            f'{job_id.source_name}</td>'
            f'<td style="padding: 1px 0; font-family: monospace;">'
            f'{self._render_value(job_id, state)}</td></tr>'
            for job_id in state.job_ids
        ]
        return (
            f'<table style="border-collapse: collapse; font-size: 14px; '
            f'color: {Colors.TEXT_DARK};">' + ''.join(rendered) + '</table>'
        )

    def _render_value(self, job_id: JobId, state: FOMSlotState) -> str:
        result_key = ResultKey(
            workflow_id=state.workflow_id,
            job_id=job_id,
            output_name=state.output_name,
        )
        try:
            value = self._data_service[result_key]
        except KeyError:
            return '— (no value yet)'
        return _format_value_text(value)

    def _render_binding_summary(self, state: FOMSlotState | None) -> str:
        if state is None:
            return ''
        sources_text = ', '.join(state.source_names) or '—'
        rows = [
            ('WORKFLOW', state.workflow_id.name),
            ('SOURCES', sources_text),
            ('OUTPUT', state.output_name),
        ]
        row_html = ''.join(
            f'<tr>'
            f'<td style="padding: 1px 8px 1px 0; '
            f'color: {Colors.TEXT_MUTED}; white-space: nowrap;">{label}</td>'
            f'<td style="padding: 1px 0;"><code>{value}</code></td>'
            f'</tr>'
            for label, value in rows
        )
        return (
            f'<table style="border-collapse: collapse; font-size: 12px;">'
            f'{row_html}'
            f'</table>'
        )

    def _make_buttons(self, state: FOMSlotState | None) -> pn.Row:
        configure_btn = create_tool_button(
            icon_name='settings',
            button_color=ButtonStyles.PRIMARY_BLUE,
            hover_color=ButtonStyles.PRIMARY_HOVER,
            on_click_callback=self._on_configure_click,
        )
        buttons: list = [configure_btn]
        if state is not None and state.is_running:
            reset_btn = create_tool_button(
                icon_name='backspace',
                button_color=StatusColors.MUTED,
                hover_color=HoverColors.MUTED,
                on_click_callback=self._on_reset_click,
            )
            stop_btn = create_tool_button(
                icon_name='player-stop',
                button_color=ButtonStyles.DANGER_RED,
                hover_color=ButtonStyles.DANGER_HOVER,
                on_click_callback=self._on_stop_click,
            )
            buttons.extend([reset_btn, stop_btn])
        elif state is not None:
            start_btn = create_tool_button(
                icon_name='player-play',
                button_color=WorkflowWidgetStyles.STATUS_COLORS['active'],
                hover_color=HoverColors.SUCCESS,
                on_click_callback=self._on_start_click,
            )
            clear_btn = create_tool_button(
                icon_name='x',
                button_color=ButtonStyles.DANGER_RED,
                hover_color=ButtonStyles.DANGER_HOVER,
                on_click_callback=self._on_clear_click,
            )
            buttons.extend([start_btn, clear_btn])
        return pn.Row(*buttons, margin=0)

    def _on_configure_click(self) -> None:
        self._wizard = FOMConfigWizard(
            slot=self._slot,
            orchestrator=self._orchestrator,
            modal_container=self._modal_container,
            on_close=self._cleanup_modal,
        )
        self._modal_container.append(self._wizard.modal)
        self._wizard.show()

    def _cleanup_modal(self) -> None:
        if self._wizard is not None and self._wizard.modal in self._modal_container:
            self._modal_container.remove(self._wizard.modal)
        self._wizard = None

    def _on_reset_click(self) -> None:
        _show_dangerous_confirm(
            modal_container=self._modal_container,
            title=f"Reset accumulated data for slot `{self._slot}`?",
            warning=_NICOS_SCAN_WARNING,
            action_label="Reset",
            on_confirm=lambda: self._orchestrator.reset_slot(self._slot),
        )

    def _on_stop_click(self) -> None:
        _show_dangerous_confirm(
            modal_container=self._modal_container,
            title=f"Stop the running jobs in slot `{self._slot}`?",
            warning=_NICOS_SCAN_WARNING,
            notes="Slot configuration is retained and can be restarted.",
            action_label="Stop",
            on_confirm=lambda: self._orchestrator.release_slot(self._slot),
        )

    def _on_start_click(self) -> None:
        self._orchestrator.start_slot(self._slot)

    def _on_clear_click(self) -> None:
        self._orchestrator.clear_slot(self._slot)


class FOMPanel:
    """Top-level FOM panel hosting one row per slot."""

    def __init__(
        self,
        *,
        orchestrator: FOMOrchestrator,
        job_service: JobService,
        data_service: DataService,
    ) -> None:
        self._orchestrator = orchestrator
        self._modal_container = pn.Column(height=0, sizing_mode='stretch_width')

        self._slot_widgets: dict[FOMSlot, FOMSlotWidget] = {
            slot: FOMSlotWidget(
                slot=slot,
                orchestrator=orchestrator,
                job_service=job_service,
                data_service=data_service,
                modal_container=self._modal_container,
            )
            for slot in orchestrator.slot_names
        }

        slot_list = pn.Column(
            *(w.panel() for w in self._slot_widgets.values()),
            sizing_mode='stretch_width',
        )

        header = pn.pane.HTML(
            '<h3 style="margin: 0 0 12px 0;">Figures of Merit</h3>'
            '<p style="margin: 0 0 12px 0; color: #666; font-size: 12px;">'
            'Each slot binds a stable alias (consumed by NICOS) to one '
            'workflow output. Selecting multiple sources binds them all '
            'under the same alias; the consumer aggregates the substreams.'
            '</p>',
            sizing_mode='stretch_width',
        )

        self._panel = pn.Column(
            header,
            slot_list,
            self._modal_container,
            sizing_mode='stretch_width',
            margin=(10, 10),
        )

    @property
    def panel(self) -> pn.Column:
        return self._panel

    def register_periodic_refresh(
        self, session_updater, is_visible: Callable[[], bool] | None = None
    ) -> None:
        """Register periodic refresh on the session updater."""

        def _refresh() -> None:
            if is_visible is not None and not is_visible():
                return
            for w in self._slot_widgets.values():
                w.refresh()

        session_updater.register_custom_handler(_refresh)
