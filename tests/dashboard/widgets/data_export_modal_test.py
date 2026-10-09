# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import io
import warnings

import panel as pn
import pydantic
import pytest
import scipp as sc
import scippnexus as snx
from panel.util.warnings import PanelUserWarning

from ess.livedata.config.workflow_spec import (
    REDUCTION,
    CumulativeOutput,
    DataKey,
    OutputView,
    SeriesOutput,
    WindowOutput,
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.dashboard.data_export import (
    CurrentValueExportParams,
    ExportRequest,
    HistoryExportParams,
    TimeseriesRetention,
    timeseries_keys,
)
from ess.livedata.dashboard.data_roles import PRIMARY, X_AXIS
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.plot_orchestrator import DataSourceConfig
from ess.livedata.dashboard.plot_params import TimeWindowMode
from ess.livedata.dashboard.widgets.data_export_modal import (
    DataExportLauncher,
    ExportConfigurationStep,
    ExportPreviewStep,
    ExportTypeSelection,
    ExportTypeStep,
)
from ess.livedata.dashboard.widgets.plot_config_modal import (
    STATIC_OVERLAY_GROUP,
    OutputSelection,
    WorkflowAndOutputSelectionStep,
)

MONITOR = WorkflowId(instrument='test', name='monitor', version=1)
MOTION = WorkflowId(instrument='test', name='motion', version=1)


class _Params(pydantic.BaseModel):
    pass


class MonitorOutputs(WorkflowOutputsBase):
    output_views = (
        OutputView(name='counts', title='Counts', fields=('counts_total', 'counts')),
        OutputView(
            name='spectrum', title='Spectrum', fields=('spectrum_total', 'spectrum')
        ),
        # No cumulative field: 'since_start' cannot be served.
        OutputView(name='profile', title='Profile', fields=('profile',)),
    )

    counts_total: CumulativeOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0.0, unit='counts'))
    )
    counts: WindowOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0.0, unit='counts'))
    )
    spectrum_total: CumulativeOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.zeros(dims=['tof'], shape=[0]))
    )
    spectrum: WindowOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.zeros(dims=['tof'], shape=[0]))
    )
    profile: WindowOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.zeros(dims=['x'], shape=[0]))
    )


class MotionOutputs(WorkflowOutputsBase):
    position: SeriesOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0.0, unit='mm'))
    )


def _spec(workflow_id: WorkflowId, outputs, source_names: list[str]) -> WorkflowSpec:
    return WorkflowSpec(
        instrument=workflow_id.instrument,
        name=workflow_id.name,
        version=workflow_id.version,
        title=workflow_id.name.title(),
        description='',
        source_names=source_names,
        params=_Params,
        aux_sources=None,
        outputs=outputs,
        group=REDUCTION,
    )


@pytest.fixture
def registry() -> dict[WorkflowId, WorkflowSpec]:
    return {
        MONITOR: _spec(MONITOR, MonitorOutputs, ['monitor_1', 'monitor_2']),
        MOTION: _spec(MOTION, MotionOutputs, ['motor_x', 'motor_y']),
    }


@pytest.fixture
def data_service(registry) -> DataService[DataKey, sc.DataArray]:
    service = DataService[DataKey, sc.DataArray]()
    service.register_subscriber(TimeseriesRetention(timeseries_keys(registry)))
    return service


def _counts(value: float, *, start: float, end: float) -> sc.DataArray:
    return sc.DataArray(
        sc.scalar(value, unit='counts'),
        coords={
            'time': sc.scalar(int(end * 1e9), unit='ns'),
            'start_time': sc.scalar(int(start * 1e9), unit='ns'),
        },
    )


@pytest.fixture
def counts_buffered(data_service) -> None:
    key = DataKey(workflow_id=MONITOR, source_name='monitor_1', output_name='counts')
    for i in range(3):
        data_service[key] = _counts(float(i), start=i, end=i + 1)


def _selection(
    params_class=HistoryExportParams, view_name: str = 'counts'
) -> ExportTypeSelection:
    return ExportTypeSelection(
        workflow_id=MONITOR, view_name=view_name, params_class=params_class
    )


def _axis_label(step: ExportTypeStep) -> str:
    """Label of the motor_x option in the first correlation selector."""
    [x_axis, _] = _axis_selectors(step)
    return next(label for label in x_axis.options if 'motor_x' in label)


def _axis_selectors(step: ExportTypeStep) -> list[pn.widgets.Select]:
    return list(step.render_content().select(pn.widgets.Select))


def _kind(step: ExportTypeStep) -> pn.widgets.RadioButtonGroup:
    [kind] = step.render_content().select(pn.widgets.RadioButtonGroup)
    return kind


class TestExportTypeStep:
    @pytest.fixture
    def step(self, registry) -> ExportTypeStep:
        return ExportTypeStep(registry, instrument_config=None)

    def test_not_valid_before_an_output_is_chosen(self, step) -> None:
        assert not step.is_valid()
        assert step.commit() is None

    def test_time_series_output_offers_history_first(self, step) -> None:
        step.on_enter(OutputSelection(MONITOR, 'counts'))

        kind = _kind(step)
        assert kind.options == ['Time series', 'Current value']
        assert kind.value == 'Time series'
        assert step.is_valid()

    def test_other_output_offers_only_the_current_value(self, step) -> None:
        step.on_enter(OutputSelection(MONITOR, 'spectrum'))

        kind = _kind(step)
        assert kind.options == ['Current value']
        assert kind.value == 'Current value'
        assert step.is_valid()

    def test_history_commits_with_the_chosen_axis(self, step) -> None:
        step.on_enter(OutputSelection(MONITOR, 'counts'))
        [x_axis, _] = _axis_selectors(step)
        x_axis.value = x_axis.options[_axis_label(step)]

        selection = step.commit()

        assert selection == ExportTypeSelection(
            workflow_id=MONITOR,
            view_name='counts',
            params_class=HistoryExportParams,
            axis_sources={
                X_AXIS: DataSourceConfig(
                    workflow_id=MOTION,
                    source_names=['motor_x'],
                    view_name='position',
                )
            },
        )

    def test_second_axis_is_disabled_until_the_first_is_chosen(self, step) -> None:
        step.on_enter(OutputSelection(MONITOR, 'counts'))
        [x_axis, y_axis] = _axis_selectors(step)
        assert y_axis.disabled

        x_axis.value = x_axis.options[_axis_label(step)]
        assert not y_axis.disabled

        x_axis.value = None
        assert y_axis.disabled

    def test_clearing_the_first_axis_clears_the_second(self, step) -> None:
        step.on_enter(OutputSelection(MONITOR, 'counts'))
        [x_axis, y_axis] = _axis_selectors(step)
        x_axis.value = x_axis.options[_axis_label(step)]
        y_axis.value = y_axis.options[_axis_label(step)]

        x_axis.value = None

        assert y_axis.value is None
        assert step.commit().axis_sources == {}

    def test_current_value_ignores_a_chosen_axis(self, step) -> None:
        step.on_enter(OutputSelection(MONITOR, 'counts'))
        [x_axis, _] = _axis_selectors(step)
        x_axis.value = x_axis.options[_axis_label(step)]
        _kind(step).value = 'Current value'

        selection = step.commit()

        assert selection.params_class is CurrentValueExportParams
        assert selection.axis_sources == {}

    def test_ready_state_is_reported_to_the_wizard(self, step) -> None:
        ready: list[bool] = []
        step.on_ready_changed(ready.append)

        step.on_enter(OutputSelection(MONITOR, 'counts'))

        assert ready[-1] is True


class TestExportConfigurationStep:
    @pytest.fixture
    def step(self, registry) -> ExportConfigurationStep:
        return ExportConfigurationStep(registry, instrument_config=None)

    def test_not_committable_before_a_selection_arrives(self, step) -> None:
        assert step.commit() is None

    def test_commit_requests_the_preselected_sources(self, step) -> None:
        step.on_enter(_selection())

        request = step.commit()

        assert isinstance(request.params, HistoryExportParams)
        assert request.data_sources == {
            PRIMARY: DataSourceConfig(
                workflow_id=MONITOR,
                source_names=['monitor_1', 'monitor_2'],
                view_name='counts',
            )
        }

    def test_commit_carries_the_axis_sources_of_the_selection(self, step) -> None:
        axis = DataSourceConfig(
            workflow_id=MOTION, source_names=['motor_x'], view_name='position'
        )
        step.on_enter(
            ExportTypeSelection(
                workflow_id=MONITOR,
                view_name='counts',
                params_class=HistoryExportParams,
                axis_sources={X_AXIS: axis},
            )
        )

        request = step.commit()

        assert request.data_sources[X_AXIS] == axis
        assert request.data_sources[PRIMARY].view_name == 'counts'

    def test_current_value_defaults_to_the_latest_update(self, step) -> None:
        step.on_enter(_selection(CurrentValueExportParams, 'spectrum'))

        request = step.commit()

        assert request.params == CurrentValueExportParams()

    def test_since_start_without_a_cumulative_field_is_refused(self, step) -> None:
        step.on_enter(_selection(CurrentValueExportParams, 'profile'))
        _mode_selector(step).value = TimeWindowMode.since_start

        assert step.commit() is None

    def test_since_start_is_accepted_with_a_cumulative_field(self, step) -> None:
        step.on_enter(_selection(CurrentValueExportParams, 'spectrum'))
        _mode_selector(step).value = TimeWindowMode.since_start

        request = step.commit()

        assert request.params.windowing() == 'since_start'


def _mode_selector(step: ExportConfigurationStep) -> pn.widgets.Select:
    [mode] = [
        w for w in step.render_content().select(pn.widgets.Select) if w.name == 'Mode'
    ]
    return mode


class TestExportPreviewStep:
    @pytest.fixture
    def step(self, data_service, registry) -> ExportPreviewStep:
        return ExportPreviewStep(
            data_service=data_service,
            workflow_registry=registry,
            instrument='test',
            instrument_config=None,
        )

    @pytest.fixture
    def request_(self) -> ExportRequest:
        return ExportRequest(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=MONITOR,
                    source_names=['monitor_1'],
                    view_name='counts',
                )
            },
            params=HistoryExportParams(),
        )

    def test_offers_a_nexus_file_holding_the_buffered_data(
        self, step, request_, counts_buffered
    ) -> None:
        step.on_enter(request_)

        [download] = step.render_content().select(pn.widgets.FileDownload)
        assert download.filename.endswith('.nxs')
        entry = snx.load(io.BytesIO(download.file.getvalue()))['entry']
        assert entry['monitor_1'].sizes == {'time': 3}

    def test_summarizes_the_exported_groups(
        self, step, request_, counts_buffered
    ) -> None:
        step.on_enter(request_)

        [summary] = step.render_content().select(pn.pane.HTML)
        assert 'monitor_1' in summary.object
        assert 'time: 3' in summary.object

    def test_warns_if_nothing_is_buffered(self, step, request_) -> None:
        step.on_enter(request_)

        [alert] = step.render_content().select(pn.pane.Alert)
        assert 'No data' in alert.object
        assert not step.render_content().select(pn.widgets.FileDownload)

    def test_entering_again_reads_the_data_anew(
        self, step, request_, data_service
    ) -> None:
        key = DataKey(
            workflow_id=MONITOR, source_name='monitor_1', output_name='counts'
        )
        data_service[key] = _counts(1.0, start=0, end=1)
        step.on_enter(request_)
        data_service[key] = _counts(2.0, start=1, end=2)

        step.on_enter(request_)

        [download] = step.render_content().select(pn.widgets.FileDownload)
        entry = snx.load(io.BytesIO(download.file.getvalue()))['entry']
        assert entry['monitor_1'].sizes == {'time': 2}


class TestDataExportLauncher:
    @pytest.fixture
    def launcher(self, data_service, registry) -> DataExportLauncher:
        return DataExportLauncher(
            data_service=data_service,
            workflow_registry=registry,
            instrument='test',
            instrument_config=None,
        )

    @pytest.fixture(autouse=True)
    def headless(self):
        """Opening or closing a ``pn.Modal`` outside a server warns."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PanelUserWarning)
            yield

    def test_panel_is_empty_until_opened(self, launcher) -> None:
        assert len(launcher.panel) == 0

    def test_open_shows_an_open_modal(self, launcher) -> None:
        launcher.open()

        [modal] = launcher.panel.objects
        assert isinstance(modal, pn.Modal)
        assert modal.open

    def test_closing_the_modal_empties_the_panel(self, launcher) -> None:
        launcher.open()
        [modal] = launcher.panel.objects

        modal.open = False

        assert len(launcher.panel) == 0

    def test_can_be_reopened_after_closing(self, launcher) -> None:
        launcher.open()
        launcher.panel.objects[0].open = False

        launcher.open()

        [modal] = launcher.panel.objects
        assert modal.open


class TestOutputSelectionGroups:
    @staticmethod
    def _group_options(step: WorkflowAndOutputSelectionStep) -> dict[str, str]:
        [group] = [
            w
            for w in step.render_content().select(pn.widgets.RadioButtonGroup)
            if w.name == 'Group'
        ]
        return group.options

    def test_static_overlay_is_offered_by_default(self, registry) -> None:
        step = WorkflowAndOutputSelectionStep(registry)

        assert STATIC_OVERLAY_GROUP in self._group_options(step).values()

    def test_static_overlay_can_be_left_out(self, registry) -> None:
        step = WorkflowAndOutputSelectionStep(registry, include_static_overlay=False)

        assert STATIC_OVERLAY_GROUP not in self._group_options(step).values()
