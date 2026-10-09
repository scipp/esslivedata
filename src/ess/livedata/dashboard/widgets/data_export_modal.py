# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Wizard for exporting buffered data to a file.

Modeled on the plot configuration wizard:

1. Select workflow and output (the plot wizard's first step).
2. Select what to export: the current value, or for a time-series output its
   buffered history, optionally correlated with up to two other time series.
3. Select sources and export params.
4. Preview what is buffered, and download the file.

What can be exported is described in :mod:`ess.livedata.dashboard.data_export`.
"""

from __future__ import annotations

import html
import io
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import panel as pn
import scipp as sc
import structlog

from ess.livedata.config.workflow_spec import (
    DataKey,
    WorkflowId,
    WorkflowSpec,
    find_timeseries_outputs,
)

from ..configuration_adapter import ConfigurationAdapter
from ..data_export import (
    MAX_EXPORT_BYTES,
    CurrentValueExportParams,
    ExportError,
    ExportItem,
    ExportParams,
    ExportRequest,
    HistoryExportParams,
    collect_export,
    export_filename,
    write_export,
)
from ..data_roles import PRIMARY, X_AXIS, Y_AXIS
from ..data_service import DataService
from ..plot_orchestrator import DataSourceConfig
from ..plotting_controller import hidden_window_fields, since_start_available
from .configuration_widget import ConfigurationPanel
from .plot_config_modal import (
    OutputSelection,
    WorkflowAndOutputSelectionStep,
    build_timeseries_options,
)
from .styles import Colors, HoverColors, ModalSizing, StatusColors
from .wizard import Wizard, WizardStep

if TYPE_CHECKING:
    from ess.livedata.config import Instrument

logger = structlog.get_logger(__name__)

# Panel 1.9 renders FileDownload with the default button type whatever its
# ``button_type``, so the primary look is applied here.
_DOWNLOAD_BUTTON_CSS = f"""
:host(.solid) button.bk-btn.bk-btn-default {{
    color: {StatusColors.PRIMARY};
    border: 1px solid {StatusColors.PRIMARY};
}}
:host(.solid) button.bk-btn.bk-btn-default:hover {{
    background-color: {HoverColors.PRIMARY};
}}
"""

_HISTORY = 'Time series'
_CURRENT = 'Current value'
_KINDS: dict[str, type[ExportParams]] = {
    _HISTORY: HistoryExportParams,
    _CURRENT: CurrentValueExportParams,
}
_KIND_DESCRIPTIONS = {
    _HISTORY: (
        'Every update of the output still buffered by the dashboard, optionally '
        'with the values of other time series at the time of each update.'
    ),
    _CURRENT: (
        'The value of the output as a plot would show it: since the run started, '
        'the latest update, or updates over a time window aggregated. A window '
        'must not reach back further than the dashboard buffers.'
    ),
}


@dataclass(frozen=True)
class ExportTypeSelection:
    """Output of the export type step."""

    workflow_id: WorkflowId
    view_name: str
    params_class: type[ExportParams]
    axis_sources: dict[str, DataSourceConfig] = field(default_factory=dict)


class _OutputSelectionStep(WorkflowAndOutputSelectionStep):
    def __init__(self, workflow_registry: Mapping[WorkflowId, WorkflowSpec]) -> None:
        super().__init__(workflow_registry, include_static_overlay=False)

    @property
    def description(self) -> str | None:
        return "Choose the workflow and output to export."


class ExportTypeStep(WizardStep[OutputSelection | None, ExportTypeSelection]):
    """Step 2: choose between current value and buffered history."""

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        instrument_config: Instrument | None,
    ) -> None:
        super().__init__()
        self._workflow_registry = workflow_registry
        self._timeseries = find_timeseries_outputs(workflow_registry)
        self._axis_options = build_timeseries_options(
            self._timeseries, workflow_registry, instrument_config
        )
        self._output: OutputSelection | None = None
        self._kind = pn.widgets.RadioButtonGroup(
            options=list(_KINDS),
            orientation='vertical',
            color='primary',
            variant='outline',
            sizing_mode='stretch_width',
        )
        self._kind_description = pn.pane.HTML(
            '', styles={'font-size': '12px', 'color': Colors.TEXT_MUTED}
        )
        self._axes = {
            role: pn.widgets.Select(
                label=label,
                options={'None': None, **self._axis_options},
                value=None,
                sizing_mode='stretch_width',
            )
            for role, label in (
                (X_AXIS, 'Correlate with (optional)'),
                (Y_AXIS, 'Second correlation axis (optional)'),
            )
        }
        self._axes_section = pn.Column(
            *self._axes.values(), sizing_mode='stretch_width'
        )
        self._kind.param.watch(self._on_change, 'value')
        self._axes[X_AXIS].param.watch(self._on_change, 'value')
        self._content = pn.Column(
            self._kind,
            self._kind_description,
            self._axes_section,
            sizing_mode='stretch_width',
        )

    @property
    def name(self) -> str:
        return "Select Export Type"

    @property
    def description(self) -> str | None:
        return "Only data the dashboard still buffers can be exported."

    def render_content(self) -> pn.Column:
        return self._content

    def is_valid(self) -> bool:
        return self._output is not None and self._kind.value in self._kind.options

    def on_enter(self, input_data: OutputSelection | None) -> None:
        if input_data is None or input_data == self._output:
            return
        self._output = input_data
        is_timeseries = any(
            workflow_id == input_data.workflow_id and view == input_data.view_name
            for workflow_id, _, view in self._timeseries
        )
        with pn.io.hold():
            self._kind.options = [_HISTORY, _CURRENT] if is_timeseries else [_CURRENT]
            self._kind.value = self._kind.options[0]
            for selector in self._axes.values():
                selector.value = None
            self._update()

    def commit(self) -> ExportTypeSelection | None:
        if self._output is None:
            return None
        params_class = _KINDS[self._kind.value]
        axis_sources = {}
        if params_class is HistoryExportParams:
            for role, selector in self._axes.items():
                if selector.value is not None:
                    workflow_id, source_name, view_name = selector.value
                    axis_sources[role] = DataSourceConfig(
                        workflow_id=workflow_id,
                        source_names=[source_name],
                        view_name=view_name,
                    )
        return ExportTypeSelection(
            workflow_id=self._output.workflow_id,
            view_name=self._output.view_name,
            params_class=params_class,
            axis_sources=axis_sources,
        )

    def _on_change(self, event: Any) -> None:
        self._update()

    def _update(self) -> None:
        self._kind_description.object = _KIND_DESCRIPTIONS.get(self._kind.value, '')
        self._axes_section.visible = self._kind.value == _HISTORY
        # A second axis without a first would leave the X axis role unfilled.
        has_x_axis = self._axes[X_AXIS].value is not None
        self._axes[Y_AXIS].disabled = not has_x_axis
        if not has_x_axis:
            self._axes[Y_AXIS].value = None
        self._notify_ready_changed(self.is_valid())


class _ExportConfigurationAdapter(ConfigurationAdapter):
    """Feeds source selection and export params to the generic config form."""

    def __init__(
        self,
        *,
        selection: ExportTypeSelection,
        workflow_spec: WorkflowSpec,
        instrument_config: Instrument | None,
        on_collected: Callable[[list[str], ExportParams], None],
    ) -> None:
        super().__init__()
        self._selection = selection
        self._workflow_spec = workflow_spec
        self._instrument_config = instrument_config
        self._on_collected = on_collected

    @property
    def title(self) -> str:
        view = self._workflow_spec.get_output_view(self._selection.view_name)
        title = view.title if view is not None else self._selection.view_name
        return f"Export {self._workflow_spec.title}: {title}"

    @property
    def description(self) -> str:
        return "Select the sources to export."

    @property
    def hidden_fields(self) -> frozenset[str]:
        return hidden_window_fields(
            self._selection.params_class,
            self._workflow_spec,
            self._selection.view_name,
        )

    def model_class(self) -> type[ExportParams]:
        return self._selection.params_class

    @property
    def source_names(self) -> list[str]:
        return self._workflow_spec.source_names

    def get_source_title(self, source_name: str) -> str:
        if self._instrument_config is None:
            return source_name
        return self._instrument_config.get_source_title(source_name)

    def start_action(
        self, selected_sources: list[str], parameter_values: ExportParams
    ) -> None:
        self._on_collected(selected_sources, parameter_values)


class ExportConfigurationStep(WizardStep[ExportTypeSelection | None, ExportRequest]):
    """Step 3: choose sources and export params."""

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        instrument_config: Instrument | None,
    ) -> None:
        super().__init__()
        self._workflow_registry = workflow_registry
        self._instrument_config = instrument_config
        self._selection: ExportTypeSelection | None = None
        self._panel: ConfigurationPanel | None = None
        self._content = pn.Column(sizing_mode='stretch_width')
        self._request: ExportRequest | None = None

    @property
    def name(self) -> str:
        return "Configure Export"

    def render_content(self) -> pn.Column:
        return self._content

    def is_valid(self) -> bool:
        # Validation runs in commit() so that errors are shown on click.
        return True

    def on_enter(self, input_data: ExportTypeSelection | None) -> None:
        if input_data is None or input_data == self._selection:
            return
        self._selection = input_data
        self._panel = ConfigurationPanel(
            config=_ExportConfigurationAdapter(
                selection=input_data,
                workflow_spec=self._workflow_registry[input_data.workflow_id],
                instrument_config=self._instrument_config,
                on_collected=self._on_collected,
            )
        )
        self._content.objects = [self._panel.panel]

    def commit(self) -> ExportRequest | None:
        if self._panel is None:
            return None
        is_valid, _ = self._panel.validate()
        if not is_valid:
            return None
        self._request = None
        if not self._panel.execute_action():
            return None
        return self._request

    def _on_collected(self, sources: list[str], params: ExportParams) -> None:
        selection = self._selection
        if (
            isinstance(params, CurrentValueExportParams)
            and params.windowing() == 'since_start'
            and not since_start_available(
                self._workflow_registry[selection.workflow_id], selection.view_name
            )
        ):
            # Raised into ConfigurationPanel, which shows it in the form.
            raise ValueError(
                "'Since run start' is not available for this output: it has no "
                "cumulative stream. Choose window mode."
            )
        self._request = ExportRequest(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=selection.workflow_id,
                    source_names=sources,
                    view_name=selection.view_name,
                ),
                **selection.axis_sources,
            },
            params=params,
        )


class ExportPreviewStep(WizardStep[ExportRequest | None, None]):
    """Step 4: read the data, summarize it, and offer the file for download.

    The data is read when the step is entered; the download holds exactly what
    the summary describes. Going back and forth reads it anew.
    """

    def __init__(
        self,
        *,
        data_service: DataService[DataKey, sc.DataArray],
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        instrument: str,
        instrument_config: Instrument | None,
    ) -> None:
        super().__init__()
        self._data_service = data_service
        self._workflow_registry = workflow_registry
        self._instrument = instrument
        self._instrument_config = instrument_config
        self._content = pn.Column(sizing_mode='stretch_width')

    @property
    def name(self) -> str:
        return "Download"

    def render_content(self) -> pn.Column:
        return self._content

    def is_valid(self) -> bool:
        return True

    def commit(self) -> None:
        return None

    def on_enter(self, input_data: ExportRequest | None) -> None:
        if input_data is None:
            return
        try:
            items = collect_export(
                input_data, self._data_service, self._workflow_registry
            )
        except ExportError as error:
            self._show_message(str(error))
            return
        summary = pn.pane.HTML(_summary_html(items), sizing_mode='stretch_width')
        # Checked before writing too, so an oversized export costs no write.
        size = sum(sys.getsizeof(item.data) for item in items.values())
        if size > MAX_EXPORT_BYTES:
            self._content.objects = [summary, self._too_large(size)]
            return
        created = datetime.now(tz=UTC)
        filename = self._filename(input_data, created)
        try:
            content = write_export(
                items,
                file_format=input_data.params.file.format,
                title=filename.rsplit('.', 1)[0],
            )
        except ValueError as error:
            # Data the format cannot hold, e.g. binned data in NeXus.
            self._content.objects = [summary, self._message(str(error))]
            return
        if len(content) > MAX_EXPORT_BYTES:
            self._content.objects = [summary, self._too_large(len(content))]
            return
        size = f'{len(content) / 1e6:.2f} MB'
        download = pn.widgets.FileDownload(
            file=io.BytesIO(content),
            filename=filename,
            label=f'Download {filename} ({size})',
            embed=False,
            sizing_mode='stretch_width',
            css_classes=['lt-export-download'],
            stylesheets=[_DOWNLOAD_BUTTON_CSS],
        )
        self._content.objects = [summary, download]

    def _filename(self, request: ExportRequest, created: datetime) -> str:
        primary = request.data_sources[PRIMARY]
        spec = self._workflow_registry[primary.workflow_id]
        view = spec.get_output_view(primary.view_name)
        get_title = (
            self._instrument_config.get_source_title
            if self._instrument_config is not None
            else str
        )
        return export_filename(
            instrument=self._instrument,
            source_titles=[get_title(name) for name in primary.source_names],
            output_title=view.title if view is not None else primary.view_name,
            file_format=request.params.file.format,
            created=created,
        )

    def _too_large(self, size: int) -> pn.pane.Alert:
        return self._message(
            f'The file would be {size / 1e6:.0f} MB, more than the '
            f'{MAX_EXPORT_BYTES / 1e6:.0f} MB the dashboard can send. '
            'Select fewer sources.'
        )

    def _show_message(self, text: str) -> None:
        self._content.objects = [self._message(text)]

    @staticmethod
    def _message(text: str) -> pn.pane.Alert:
        return pn.pane.Alert(text, alert_type='warning', sizing_mode='stretch_width')


def _summary_html(items: Mapping[str, ExportItem]) -> str:
    rows = ''.join(
        f'<tr><td>{html.escape(name)}</td><td>{_shape(item.data)}</td>'
        f'<td>{_time_range(item.data)}</td></tr>'
        for name, item in items.items()
    )
    return (
        '<table style="width: 100%; font-size: 13px; border-collapse: collapse">'
        '<tr style="text-align: left"><th>Group</th><th>Shape</th>'
        f'<th>Time range (local)</th></tr>{rows}</table>'
    )


def _shape(da: sc.DataArray) -> str:
    return ', '.join(f'{dim}: {size}' for dim, size in da.sizes.items()) or 'scalar'


def _time_range(da: sc.DataArray) -> str:
    if 'time' in da.dims:
        times = da.coords['time']
        start, end = times[0], times[-1]
    elif 'start_time' in da.coords and 'end_time' in da.coords:
        start, end = da.coords['start_time'], da.coords['end_time']
    else:
        return ''
    return f'{_local(start)} to {_local(end)}'


def _local(time: sc.Variable) -> str:
    ns = time.to(unit='ns').value.astype('int64')
    utc = datetime.fromtimestamp(int(ns) / 1e9, tz=UTC)
    return utc.astimezone().strftime('%Y-%m-%d %H:%M:%S')


class DataExportModal:
    """Modal holding the export wizard."""

    def __init__(
        self,
        *,
        data_service: DataService[DataKey, sc.DataArray],
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        instrument: str,
        instrument_config: Instrument | None,
        on_close: Callable[[], None],
    ) -> None:
        self._on_close = on_close
        steps = [
            _OutputSelectionStep(workflow_registry),
            ExportTypeStep(workflow_registry, instrument_config),
            ExportConfigurationStep(workflow_registry, instrument_config),
            ExportPreviewStep(
                data_service=data_service,
                workflow_registry=workflow_registry,
                instrument=instrument,
                instrument_config=instrument_config,
            ),
        ]
        # The last step offers its own download button, so the wizard has no
        # action button; Cancel closes the modal.
        self._wizard = Wizard(
            steps=steps, on_complete=lambda _: self._close(), on_cancel=self._close
        )
        self.modal = pn.Modal(
            self._wizard.render(),
            name='Export Data',
            margin=20,
            width=ModalSizing.WIDTH,
        )
        self.modal.param.watch(self._on_modal_open_changed, 'open')

    def show(self) -> None:
        self._wizard.reset()
        self.modal.open = True

    def _close(self) -> None:
        self.modal.open = False

    def _on_modal_open_changed(self, event: Any) -> None:
        if not event.new:
            self._on_close()


class DataExportLauncher:
    """Opens the export wizard in a modal it holds.

    ``panel`` holds the modal and takes no space, so the launcher can sit
    anywhere in the page, e.g. next to the button that calls :meth:`open`.
    """

    def __init__(
        self,
        *,
        data_service: DataService[DataKey, sc.DataArray],
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        instrument: str,
        instrument_config: Instrument | None,
    ) -> None:
        self._data_service = data_service
        self._workflow_registry = workflow_registry
        self._instrument = instrument
        self._instrument_config = instrument_config
        # Zero height: the modal renders as an overlay, but must be in the
        # document to render at all. Stretching rather than fixed width, since a
        # fixed-size child makes Panel infer a fixed size for its parent layout.
        self.panel = pn.Row(height=0, sizing_mode='stretch_width')

    def open(self) -> None:
        """Show a fresh export wizard."""
        modal = DataExportModal(
            data_service=self._data_service,
            workflow_registry=self._workflow_registry,
            instrument=self._instrument,
            instrument_config=self._instrument_config,
            on_close=self.panel.clear,
        )
        self.panel.objects = [modal.modal]
        modal.show()
