# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import io
from datetime import UTC, datetime

import pydantic
import pytest
import scipp as sc
import scipp.testing
import scippnexus as snx

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
    ExportError,
    ExportFormat,
    ExportRequest,
    HistoryExportParams,
    TimeseriesRetention,
    collect_export,
    export_filename,
    timeseries_keys,
    write_export,
)
from ess.livedata.dashboard.data_roles import PRIMARY, X_AXIS, Y_AXIS
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.plot_orchestrator import DataSourceConfig
from ess.livedata.dashboard.plot_params import TimeWindowMode, TimeWindowParams

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


class MotionOutputs(WorkflowOutputsBase):
    position: SeriesOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0.0, unit='mm'))
    )
    speed: SeriesOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0.0, unit='mm/s'))
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


def _key(workflow_id: WorkflowId, source_name: str, output_name: str) -> DataKey:
    return DataKey(
        workflow_id=workflow_id, source_name=source_name, output_name=output_name
    )


def _ns(seconds: float) -> sc.Variable:
    return sc.scalar(int(seconds * 1e9), unit='ns')


def _window(value: float, *, start: float, end: float) -> sc.DataArray:
    """A per-update message covering ``[start, end]`` seconds since the epoch."""
    return sc.DataArray(
        sc.scalar(value, unit='counts'),
        coords={'time': _ns(end), 'start_time': _ns(start)},
    )


def _sample(value: float, *, at: float) -> sc.DataArray:
    return sc.DataArray(sc.scalar(value, unit='mm'), coords={'time': _ns(at)})


def _datetimes(*seconds: float) -> sc.Variable:
    return sc.epoch(unit='ns') + sc.array(
        dims=['time'], values=[int(s * 1e9) for s in seconds], unit='ns'
    )


def _history_request(
    *sources: str, x_axis: str | None = None, y_axis: str | None = None
) -> ExportRequest:
    data_sources = {
        PRIMARY: DataSourceConfig(
            workflow_id=MONITOR, source_names=list(sources), view_name='counts'
        )
    }
    for role, axis in ((X_AXIS, x_axis), (Y_AXIS, y_axis)):
        if axis is not None:
            data_sources[role] = DataSourceConfig(
                workflow_id=MOTION, source_names=[axis], view_name='position'
            )
    return ExportRequest(data_sources=data_sources, params=HistoryExportParams())


def _current_request(time_window: TimeWindowParams) -> ExportRequest:
    return ExportRequest(
        data_sources={
            PRIMARY: DataSourceConfig(
                workflow_id=MONITOR, source_names=['monitor_1'], view_name='spectrum'
            )
        },
        params=CurrentValueExportParams(time_window=time_window),
    )


def test_timeseries_keys_are_the_per_update_fields_of_0d_series(registry) -> None:
    assert set(timeseries_keys(registry)) == {
        _key(MONITOR, 'monitor_1', 'counts'),
        _key(MONITOR, 'monitor_2', 'counts'),
        _key(MOTION, 'motor_x', 'position'),
        _key(MOTION, 'motor_y', 'position'),
        _key(MOTION, 'motor_x', 'speed'),
        _key(MOTION, 'motor_y', 'speed'),
    }


class TestHistoryExport:
    def test_exports_every_buffered_update_of_an_unplotted_series(
        self, data_service, registry
    ) -> None:
        key = _key(MONITOR, 'monitor_1', 'counts')
        for i in range(3):
            data_service[key] = _window(float(i), start=i, end=i + 1)

        items = collect_export(_history_request('monitor_1'), data_service, registry)

        assert list(items) == ['monitor_1']
        assert items['monitor_1'].key == key
        exported = items['monitor_1'].data
        sc.testing.assert_identical(
            exported.data,
            sc.array(dims=['time'], values=[0.0, 1.0, 2.0], unit='counts'),
        )

    def test_time_coords_are_utc_datetimes(self, data_service, registry) -> None:
        key = _key(MONITOR, 'monitor_1', 'counts')
        data_service[key] = _window(1.0, start=10, end=11)
        data_service[key] = _window(2.0, start=11, end=12)

        exported = collect_export(
            _history_request('monitor_1'), data_service, registry
        )['monitor_1'].data

        sc.testing.assert_identical(exported.coords['time'], _datetimes(11, 12))
        sc.testing.assert_identical(exported.coords['start_time'], _datetimes(10)[0])
        sc.testing.assert_identical(exported.coords['end_time'], _datetimes(12)[0])

    def test_sources_without_data_are_left_out(self, data_service, registry) -> None:
        data_service[_key(MONITOR, 'monitor_1', 'counts')] = _window(
            1.0, start=0, end=1
        )

        items = collect_export(
            _history_request('monitor_1', 'monitor_2'), data_service, registry
        )

        assert list(items) == ['monitor_1']

    def test_raises_if_no_source_has_data(self, data_service, registry) -> None:
        with pytest.raises(ExportError, match='No data'):
            collect_export(_history_request('monitor_1'), data_service, registry)


class TestCorrelatedHistoryExport:
    @pytest.fixture
    def counts(self, data_service) -> None:
        key = _key(MONITOR, 'monitor_1', 'counts')
        for i in range(4):
            data_service[key] = _window(float(i), start=i, end=i + 1)

    def test_axis_value_in_effect_at_each_point_becomes_a_coord(
        self, data_service, registry, counts
    ) -> None:
        motor = _key(MOTION, 'motor_x', 'position')
        data_service[motor] = _sample(10.0, at=1.5)
        data_service[motor] = _sample(20.0, at=3.5)

        exported = collect_export(
            _history_request('monitor_1', x_axis='motor_x'), data_service, registry
        )['monitor_1'].data

        # The update at t=1 predates the first motor reading and is dropped.
        sc.testing.assert_identical(exported.coords['time'], _datetimes(2, 3, 4))
        sc.testing.assert_identical(
            exported.coords['motor_x'],
            sc.array(dims=['time'], values=[10.0, 10.0, 20.0], unit='mm'),
        )

    def test_axes_are_exported_as_their_own_series(
        self, data_service, registry, counts
    ) -> None:
        for axis in ('motor_x', 'motor_y'):
            data_service[_key(MOTION, axis, 'position')] = _sample(1.0, at=0.5)

        items = collect_export(
            _history_request('monitor_1', x_axis='motor_x', y_axis='motor_y'),
            data_service,
            registry,
        )

        assert set(items) == {'monitor_1', 'motor_x', 'motor_y'}
        assert {'motor_x', 'motor_y'} <= set(items['monitor_1'].data.coords)
        sc.testing.assert_identical(
            items['motor_x'].data.coords['time'], _datetimes(0.5)
        )

    def test_raises_if_axis_has_no_data(self, data_service, registry, counts) -> None:
        with pytest.raises(ExportError, match='correlation axis'):
            collect_export(
                _history_request('monitor_1', x_axis='motor_x'), data_service, registry
            )

    def test_raises_if_no_point_follows_the_first_axis_reading(
        self, data_service, registry, counts
    ) -> None:
        data_service[_key(MOTION, 'motor_x', 'position')] = _sample(1.0, at=100)

        with pytest.raises(ExportError, match='first reading'):
            collect_export(
                _history_request('monitor_1', x_axis='motor_x'), data_service, registry
            )


class TestCurrentValueExport:
    @pytest.fixture
    def spectra(self, data_service) -> None:
        def spectrum(value: float, *, end: float) -> sc.DataArray:
            return sc.DataArray(
                sc.full(dims=['tof'], shape=[2], value=value, unit='counts'),
                coords={'time': _ns(end), 'start_time': _ns(0)},
            )

        data_service[_key(MONITOR, 'monitor_1', 'spectrum_total')] = spectrum(
            10.0, end=2
        )
        data_service[_key(MONITOR, 'monitor_1', 'spectrum')] = spectrum(3.0, end=2)

    def test_since_start_exports_the_cumulative_field(
        self, data_service, registry, spectra
    ) -> None:
        items = collect_export(
            _current_request(TimeWindowParams(mode=TimeWindowMode.since_start)),
            data_service,
            registry,
        )

        assert items['monitor_1'].key.output_name == 'spectrum_total'
        assert sc.identical(
            items['monitor_1'].data.data,
            sc.full(dims=['tof'], shape=[2], value=10.0, unit='counts'),
        )

    def test_window_exports_the_latest_update(
        self, data_service, registry, spectra
    ) -> None:
        item = collect_export(
            _current_request(TimeWindowParams(mode=TimeWindowMode.window)),
            data_service,
            registry,
        )['monitor_1']

        assert item.key.output_name == 'spectrum'
        assert item.data.dims == ('tof',)
        sc.testing.assert_identical(item.data.coords['end_time'], _datetimes(2)[0])


def test_names_carry_the_output_when_a_source_appears_twice(
    data_service, registry
) -> None:
    position = _key(MOTION, 'motor_x', 'position')
    speed = _key(MOTION, 'motor_x', 'speed')
    data_service[speed] = _sample(0.0, at=0)
    data_service[position] = _sample(1.0, at=1)
    request = ExportRequest(
        data_sources={
            PRIMARY: DataSourceConfig(
                workflow_id=MOTION, source_names=['motor_x'], view_name='position'
            ),
            X_AXIS: DataSourceConfig(
                workflow_id=MOTION, source_names=['motor_x'], view_name='speed'
            ),
        },
        params=HistoryExportParams(),
    )

    items = collect_export(request, data_service, registry)

    assert {name: item.key for name, item in items.items()} == {
        'motor_x_position': position,
        'motor_x_speed': speed,
    }
    assert 'motor_x_speed' in items['motor_x_position'].data.coords


def test_written_nexus_file_holds_the_exported_data(data_service, registry) -> None:
    key = _key(MONITOR, 'monitor_1', 'counts')
    data_service[key] = _window(1.0, start=0, end=1)
    data_service[key] = _window(2.0, start=1, end=2)
    items = collect_export(_history_request('monitor_1'), data_service, registry)

    content = write_export(items, file_format=ExportFormat.nexus, title='scan 7')

    entry = snx.load(io.BytesIO(content))['entry']
    sc.testing.assert_identical(entry['monitor_1'], items['monitor_1'].data)
    assert entry['title'] == 'scan 7'


def test_export_filename_is_descriptive_and_timestamped() -> None:
    name = export_filename(
        instrument='dream',
        source_titles=['Monitor 1'],
        output_title='Counts',
        file_format=ExportFormat.nexus,
        created=datetime(2026, 10, 7, 14, 30, 5, tzinfo=UTC),
    )

    assert name == 'DREAM_Counts_Monitor-1_20261007T143005.nxs'
