# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid
from collections.abc import Iterator

import pytest
import scipp as sc
from pydantic import Field

from ess.livedata.config import instrument_registry
from ess.livedata.config.acknowledgement import (
    AcknowledgementResponse,
    CommandAcknowledgement,
)
from ess.livedata.config.models import Interval, PolygonROI, RectangleROI
from ess.livedata.config.roi_names import ROIGeometryType, get_roi_mapper
from ess.livedata.config.workflow_spec import (
    REDUCTION,
    CumulativeOutput,
    JobId,
    ResultKey,
    SeriesOutput,
    WindowOutput,
    WorkflowConfig,
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.core.message import (
    RESPONSES_STREAM_ID,
    STATUS_STREAM_ID,
    StreamKind,
)
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.fake_backend import (
    FakeBackend,
    FakeBackendTransport,
    expand_template,
)
from ess.livedata.dashboard.roi_publisher import ROIPublisher
from ess.livedata.dashboard.transport import DashboardResources


class Outputs1D(WorkflowOutputsBase):
    histogram: sc.DataArray = Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['x'], shape=[0], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[], unit='m')},
        ),
        title='Histogram',
    )


class Outputs2D(WorkflowOutputsBase):
    image: sc.DataArray = Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['y', 'x'], shape=[0, 0], unit='counts')
        ),
        title='Image',
    )


class Outputs3D(WorkflowOutputsBase):
    volume: sc.DataArray = Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['z', 'y', 'x'], shape=[0, 0, 0], unit='counts')
        ),
        title='Volume',
    )


class OutputsTimeseries(WorkflowOutputsBase):
    reading: sc.DataArray = Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=[], shape=[], unit='K'),
            coords={'time': sc.scalar(0, unit='ns', dtype='int64')},
        ),
        title='Reading',
    )


class OutputsNoTemplate(WorkflowOutputsBase):
    result: sc.DataArray = Field(title='Result')


def _scalar_template() -> sc.DataArray:
    return sc.DataArray(sc.zeros(dims=[], shape=[], unit='counts'))


class OutputsTemporality(WorkflowOutputsBase):
    per_update: WindowOutput = Field(default_factory=_scalar_template, title='Window')
    total: CumulativeOutput = Field(default_factory=_scalar_template, title='Total')
    reading: SeriesOutput = Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=[], shape=[], unit='K'),
            coords={'time': sc.scalar(0, unit='ns', dtype='int64')},
        ),
        title='Reading',
    )


def _outputs(messages) -> dict[str, sc.DataArray]:
    """Map output name to emitted data for the data messages in ``messages``."""
    return {
        ResultKey.model_validate_json(m.stream.name).output_name: m.value
        for m in messages
        if m.stream.kind is StreamKind.LIVEDATA_DATA
    }


def _spec(outputs: type[WorkflowOutputsBase], name: str) -> WorkflowSpec:
    return WorkflowSpec(
        instrument='test',
        name=name,
        version=1,
        title=name,
        description='',
        outputs=outputs,
        group=REDUCTION,
        source_names=['source1'],
    )


def _registry(*specs: WorkflowSpec) -> dict[WorkflowId, WorkflowSpec]:
    return {spec.get_id(): spec for spec in specs}


def _config(spec: WorkflowSpec, message_id: str = 'm1') -> WorkflowConfig:
    return WorkflowConfig.from_params(
        workflow_id=spec.get_id(),
        job_id=JobId(source_name='source1', job_number=uuid.uuid4()),
        message_id=message_id,
    )


class TestExpandTemplate:
    def test_expands_empty_dim_and_preserves_unit(self) -> None:
        template = Outputs1D().histogram
        out = expand_template(template, update=0, timestamp_ns=0)
        assert out.sizes == {'x': 64}
        assert out.unit == sc.Unit('counts')
        assert out.coords['x'].sizes == {'x': 64}
        assert out.coords['x'].unit == sc.Unit('m')

    def test_two_dimensional(self) -> None:
        out = expand_template(Outputs2D().image, update=0, timestamp_ns=0)
        assert out.sizes == {'y': 64, 'x': 64}

    def test_three_dimensional_varies_along_every_axis(self) -> None:
        # The slicer steps through one dim at a time, so a volume that is flat
        # along any axis would render as an unchanging image.
        out = expand_template(Outputs3D().volume, update=0, timestamp_ns=0)
        assert out.sizes == {'z': 64, 'y': 64, 'x': 64}
        for dim in out.dims:
            profile = out.data
            for other in set(out.dims) - {dim}:
                profile = profile.sum(other)
            assert profile.max().value > 2.0 * profile.min().value

    def test_values_are_finite_and_nonnegative(self) -> None:
        out = expand_template(Outputs1D().histogram, update=2, timestamp_ns=0)
        assert sc.all(sc.isfinite(out.data)).value
        assert (out.data.values >= 0).all()

    def test_update_counter_changes_values(self) -> None:
        a = expand_template(Outputs1D().histogram, update=0, timestamp_ns=0)
        b = expand_template(Outputs1D().histogram, update=5, timestamp_ns=0)
        assert not sc.allclose(a.data, b.data)

    def test_scalar_timeseries_output_stamps_time(self) -> None:
        out = expand_template(
            OutputsTimeseries().reading, update=3, timestamp_ns=1_700_000_000
        )
        assert out.ndim == 0
        assert out.unit == sc.Unit('K')
        time = out.coords['time']
        assert time.ndim == 0
        assert time.value == 1_700_000_000
        assert time.unit == sc.Unit('ns')


class TestFakeBackend:
    def test_workflow_config_yields_ack_and_active_status(self) -> None:
        spec = _spec(Outputs1D, 'wf1d')
        backend = FakeBackend(_registry(spec))
        config = _config(spec)

        backend.submit(config)
        messages = backend.poll()

        acks = [m.value for m in messages if m.stream == RESPONSES_STREAM_ID]
        assert len(acks) == 1
        ack = acks[0]
        assert isinstance(ack, CommandAcknowledgement)
        assert ack.message_id == config.message_id
        assert ack.response is AcknowledgementResponse.ACK

        statuses = [m.value for m in messages if m.stream == STATUS_STREAM_ID]
        assert len(statuses) == 1
        status = statuses[0]
        assert isinstance(status, JobStatus)
        assert status.job_id == config.job_id
        assert status.state is JobState.active
        # start_time drives the dashboard's runtime clock; without it the
        # workflow stays stuck at "Starting...".
        assert status.start_time is not None

    def test_status_reemitted_as_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Zero update period makes every poll due, so each poll re-emits status.
        monkeypatch.setattr(
            'ess.livedata.dashboard.fake_backend._UPDATE_PERIOD_SECONDS', 0.0
        )
        spec = _spec(Outputs1D, 'wf1d')
        backend = FakeBackend(_registry(spec))
        config = _config(spec)
        backend.submit(config)

        for _ in range(3):
            statuses = [m.value for m in backend.poll() if m.stream == STATUS_STREAM_ID]
            assert len(statuses) == 1
            assert statuses[0].state is JobState.active
            assert statuses[0].start_time is not None

    def test_emits_data_with_matching_result_key(self) -> None:
        spec = _spec(Outputs1D, 'wf1d')
        backend = FakeBackend(_registry(spec))
        config = _config(spec)

        backend.submit(config)
        data = [m for m in backend.poll() if m.stream.kind is StreamKind.LIVEDATA_DATA]

        assert len(data) == 1
        key = ResultKey.model_validate_json(data[0].stream.name)
        assert key.workflow_id == config.identifier
        assert key.job_id == config.job_id
        assert key.output_name == 'histogram'
        assert isinstance(data[0].value, sc.DataArray)
        assert data[0].value.sizes == {'x': 64}

    def test_distinct_sources_yield_distinct_data(self) -> None:
        spec = _spec(Outputs1D, 'wf1d')
        backend = FakeBackend(_registry(spec))
        job_number = uuid.uuid4()  # same job, two sources -> overlaid lines
        for source in ('monitor1', 'monitor2'):
            backend.submit(
                WorkflowConfig.from_params(
                    workflow_id=spec.get_id(),
                    job_id=JobId(source_name=source, job_number=job_number),
                    message_id=source,
                )
            )
        data = {
            ResultKey.model_validate_json(m.stream.name).job_id.source_name: m.value
            for m in backend.poll()
            if m.stream.kind is StreamKind.LIVEDATA_DATA
        }
        assert set(data) == {'monitor1', 'monitor2'}
        assert not sc.allclose(data['monitor1'].data, data['monitor2'].data)

    def test_emits_one_data_message_per_output_field(self) -> None:
        class MultiOutputs(WorkflowOutputsBase):
            a: sc.DataArray = Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x'], shape=[0], unit='counts')
                ),
                title='A',
            )
            b: sc.DataArray = Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x'], shape=[0], unit='counts')
                ),
                title='B',
            )

        spec = _spec(MultiOutputs, 'multi')
        backend = FakeBackend(_registry(spec))
        backend.submit(_config(spec))

        data = [m for m in backend.poll() if m.stream.kind is StreamKind.LIVEDATA_DATA]
        output_names = {
            ResultKey.model_validate_json(m.stream.name).output_name for m in data
        }
        assert output_names == {'a', 'b'}

    def test_output_without_template_is_skipped(self) -> None:
        spec = _spec(OutputsNoTemplate, 'no_template')
        backend = FakeBackend(_registry(spec))
        backend.submit(_config(spec))

        data = [m for m in backend.poll() if m.stream.kind is StreamKind.LIVEDATA_DATA]
        assert data == []

    def test_unknown_workflow_yields_error_ack(self) -> None:
        backend = FakeBackend({})
        unknown = WorkflowId(instrument='test', name='ghost', version=1)
        config = WorkflowConfig.from_params(
            workflow_id=unknown,
            job_id=JobId(source_name='source1', job_number=uuid.uuid4()),
            message_id='m1',
        )

        backend.submit(config)
        messages = backend.poll()

        acks = [m.value for m in messages if m.stream == RESPONSES_STREAM_ID]
        assert len(acks) == 1
        assert acks[0].response is AcknowledgementResponse.ERR
        assert not [m for m in messages if m.stream.kind is StreamKind.LIVEDATA_DATA]

    def test_stop_command_halts_data_emission(self) -> None:
        spec = _spec(Outputs1D, 'wf1d')
        backend = FakeBackend(_registry(spec))
        config = _config(spec)
        backend.submit(config)
        backend.poll()  # drain initial ack/status/data

        backend.submit(
            JobCommand(job_id=config.job_id, action=JobAction.stop, message_id='m2')
        )
        messages = backend.poll()

        assert not [m for m in messages if m.stream.kind is StreamKind.LIVEDATA_DATA]

    def test_stop_command_yields_terminal_stopped_status(self) -> None:
        spec = _spec(Outputs1D, 'wf1d')
        backend = FakeBackend(_registry(spec))
        config = _config(spec)
        backend.submit(config)
        backend.poll()  # drain initial ack/status/data

        backend.submit(
            JobCommand(job_id=config.job_id, action=JobAction.stop, message_id='m2')
        )
        statuses = [m.value for m in backend.poll() if m.stream == STATUS_STREAM_ID]

        assert len(statuses) == 1
        assert statuses[0].job_id == config.job_id
        assert statuses[0].state is JobState.stopped
        # The status is terminal: the job is gone and never heartbeats again.
        assert not backend.poll()

    def test_poll_is_empty_without_active_jobs(self) -> None:
        backend = FakeBackend({})
        assert backend.poll() == []


class TestFailedJob:
    """A faulted job mirrors a workflow raising in the backend."""

    @pytest.fixture
    def running(self, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeBackend, JobId]:
        """A running job, its ack/status/data already drained."""
        # Zero update period makes every poll due, so each poll heartbeats.
        monkeypatch.setattr(
            'ess.livedata.dashboard.fake_backend._UPDATE_PERIOD_SECONDS', 0.0
        )
        spec = _spec(Outputs1D, 'wf1d')
        backend = FakeBackend(_registry(spec))
        config = _config(spec)
        backend.submit(config)
        backend.poll()
        return backend, config.job_id

    def test_reports_error_state_with_message(self, running) -> None:
        backend, job_id = running
        backend.fail_job(job_id, 'workflow blew up')

        statuses = [m.value for m in backend.poll() if m.stream == STATUS_STREAM_ID]

        assert len(statuses) == 1
        assert statuses[0].state is JobState.error
        assert statuses[0].error_message == 'workflow blew up'

    def test_keeps_heartbeating_the_error(self, running) -> None:
        backend, job_id = running
        backend.fail_job(job_id, 'workflow blew up')

        for _ in range(3):
            statuses = [m.value for m in backend.poll() if m.stream == STATUS_STREAM_ID]
            assert [s.state for s in statuses] == [JobState.error]

    def test_stops_emitting_data(self, running) -> None:
        backend, job_id = running
        backend.fail_job(job_id, 'workflow blew up')

        messages = backend.poll()

        assert not [m for m in messages if m.stream.kind is StreamKind.LIVEDATA_DATA]

    def test_unknown_job_is_rejected(self, running) -> None:
        backend, _ = running
        unknown = JobId(source_name='source1', job_number=uuid.uuid4())
        with pytest.raises(KeyError, match='No running job'):
            backend.fail_job(unknown, 'workflow blew up')


_DETECTOR_SOURCE = 'panel_0'


def _rectangle(x: float, y: float) -> RectangleROI:
    return RectangleROI(
        x=Interval(min=x, max=x + 10.0), y=Interval(min=y, max=y + 10.0)
    )


def _polygon(x: float) -> PolygonROI:
    return PolygonROI(x=[x, x + 5.0, x], y=[0.0, 5.0, 10.0], x_unit=None, y_unit=None)


class _DetectorSession:
    """A started detector-view job, driven through the fake transport.

    Uses the real :class:`CommandService` and :class:`ROIPublisher` so the ROIs
    take the same route as those drawn in the UI.
    """

    def __init__(self, resources: DashboardResources) -> None:
        self._source = resources.message_source
        self.workflow_id = next(
            workflow_id
            for workflow_id in instrument_registry['dummy'].workflow_factory
            if workflow_id.name == 'panel_0_xy'
        )
        job_id = JobId(source_name=_DETECTOR_SOURCE, job_number=uuid.uuid4())
        self.publisher = ROIPublisher(sink=resources.roi_sink)
        self.publisher.set_job_number_resolver(lambda _: job_id.job_number)
        CommandService(sink=resources.command_sink).send(
            WorkflowConfig.from_params(workflow_id=self.workflow_id, job_id=job_id)
        )

    def publish(self, rois: dict, geometry_type: ROIGeometryType = 'rectangle') -> None:
        """Publish the full ROI set for one geometry, as the UI does."""
        self.publisher.publish(
            self.workflow_id,
            _DETECTOR_SOURCE,
            rois,
            get_roi_mapper().geometry_for_type(geometry_type),
        )

    def results(self) -> dict[str, sc.DataArray]:
        """Poll the backend, returning the freshest result per output name."""
        return {
            ResultKey.model_validate_json(m.stream.name).output_name: m.value
            for m in self._source.get_messages()
            if m.stream.kind is StreamKind.LIVEDATA_DATA
        }


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> Iterator[_DetectorSession]:
    # Zero update period makes every poll due, so each poll yields fresh results.
    monkeypatch.setattr(
        'ess.livedata.dashboard.fake_backend._UPDATE_PERIOD_SECONDS', 0.0
    )
    with FakeBackendTransport(instrument='dummy') as resources:
        yield _DetectorSession(resources)


class TestROILoopback:
    def test_spectra_are_empty_until_an_roi_is_drawn(
        self, session: _DetectorSession
    ) -> None:
        results = session.results()
        spectra = results['roi_spectra_cumulative']
        assert spectra.sizes['roi'] == 0
        assert spectra.sizes['time_of_arrival'] == 64
        assert len(results['roi_rectangle']) == 0

    def test_roi_dim_follows_published_rois(self, session: _DetectorSession) -> None:
        session.publish({0: _rectangle(0.0, 0.0), 1: _rectangle(20.0, 20.0)})
        assert session.results()['roi_spectra_cumulative'].sizes['roi'] == 2

        session.publish({0: _rectangle(0.0, 0.0)})
        spectra = session.results()['roi_spectra_cumulative']
        assert spectra.sizes['roi'] == 1
        assert spectra.coords['roi'].values.tolist() == [0]

        session.publish({})
        assert session.results()['roi_spectra_cumulative'].sizes['roi'] == 0

    def test_roi_coord_carries_published_indices(
        self, session: _DetectorSession
    ) -> None:
        session.publish({1: _rectangle(0.0, 0.0), 3: _rectangle(20.0, 20.0)})
        spectra = session.results()['roi_spectra_cumulative']
        assert spectra.coords['roi'].values.tolist() == [1, 3]

    def test_geometries_share_the_roi_dim(self, session: _DetectorSession) -> None:
        session.publish({0: _rectangle(0.0, 0.0)})
        session.publish({4: _polygon(0.0)}, geometry_type='polygon')
        spectra = session.results()['roi_spectra_cumulative']
        assert spectra.coords['roi'].values.tolist() == [0, 4]

    def test_spectra_follow_roi_geometry(self, session: _DetectorSession) -> None:
        # Comparing rows within one update isolates the ROI dependence from the
        # update counter, which also varies the data.
        session.publish({0: _rectangle(0.0, 0.0), 1: _rectangle(0.0, 0.0)})
        spectra = session.results()['roi_spectra_cumulative']
        assert sc.identical(spectra['roi', 0].data, spectra['roi', 1].data)

        session.publish({0: _rectangle(0.0, 0.0), 1: _rectangle(20.0, 20.0)})
        spectra = session.results()['roi_spectra_cumulative']
        assert not sc.allclose(spectra['roi', 0].data, spectra['roi', 1].data)

    def test_current_spectra_are_stamped_with_time(
        self, session: _DetectorSession
    ) -> None:
        session.publish({0: _rectangle(0.0, 0.0)})
        spectra = session.results()['roi_spectra_current']
        assert spectra.sizes['roi'] == 1
        assert spectra.coords['time'].ndim == 0

    def test_readback_echoes_published_rois(self, session: _DetectorSession) -> None:
        rois = {0: _rectangle(0.0, 0.0), 2: _rectangle(20.0, 20.0)}
        session.publish(rois)
        readback = session.results()['roi_rectangle']
        assert RectangleROI.from_concatenated_data_array(readback) == rois

    def test_rois_of_other_jobs_are_ignored(self, session: _DetectorSession) -> None:
        session.publisher.set_job_number_resolver(lambda _: uuid.uuid4())
        session.publish({0: _rectangle(0.0, 0.0)})
        assert session.results()['roi_spectra_cumulative'].sizes['roi'] == 0


class TestEmittedTimeCoords:
    """The dashboard keys and ages its buffers by these coords."""

    @pytest.fixture
    def polls(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, sc.DataArray]]:
        # Zero update period makes every poll due, so each poll emits fresh data.
        monkeypatch.setattr(
            'ess.livedata.dashboard.fake_backend._UPDATE_PERIOD_SECONDS', 0.0
        )
        spec = _spec(OutputsTemporality, 'temporal')
        backend = FakeBackend(_registry(spec))
        backend.submit(_config(spec))
        return [_outputs(backend.poll()) for _ in range(2)]

    @pytest.mark.parametrize('output_name', ['per_update', 'total'])
    def test_bounded_outputs_carry_start_time_and_time(self, polls, output_name: str):
        coords = polls[0][output_name].coords
        assert coords['start_time'].unit == sc.Unit('ns')
        assert coords['time'].value >= coords['start_time'].value

    def test_series_output_carries_time_but_no_start_time(self, polls) -> None:
        coords = polls[0]['reading'].coords
        assert 'time' in coords
        assert 'start_time' not in coords

    def test_cumulative_start_time_stays_pinned(self, polls) -> None:
        first, second = (poll['total'].coords for poll in polls)
        assert sc.identical(first['start_time'], second['start_time'])
        assert second['time'].value >= first['time'].value

    def test_window_covers_interval_since_previous_update(self, polls) -> None:
        first, second = (poll['per_update'].coords for poll in polls)
        assert sc.identical(second['start_time'], first['time'])


@pytest.mark.parametrize('update', [0, 1, 7])
def test_data_emitted_each_poll_when_due(update: int) -> None:
    # next_emit starts at 0, so each poll after the period emits fresh data.
    spec = _spec(Outputs1D, 'wf1d')
    backend = FakeBackend(_registry(spec))
    backend.submit(_config(spec))
    data = [m for m in backend.poll() if m.stream.kind is StreamKind.LIVEDATA_DATA]
    assert len(data) == 1
