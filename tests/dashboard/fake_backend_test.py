# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import pytest
import scipp as sc
from pydantic import Field

from ess.livedata.config.acknowledgement import (
    AcknowledgementResponse,
    CommandAcknowledgement,
)
from ess.livedata.config.workflow_spec import (
    REDUCTION,
    JobId,
    ResultKey,
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
from ess.livedata.dashboard.fake_backend import FakeBackend, expand_template


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


def _spec(outputs: type[WorkflowOutputsBase], name: str) -> WorkflowSpec:
    return WorkflowSpec(
        instrument='test',
        name=name,
        version=1,
        title=name,
        description='',
        params=None,
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

    def test_poll_is_empty_without_active_jobs(self) -> None:
        backend = FakeBackend({})
        assert backend.poll() == []


@pytest.mark.parametrize('update', [0, 1, 7])
def test_data_emitted_each_poll_when_due(update: int) -> None:
    # next_emit starts at 0, so each poll after the period emits fresh data.
    spec = _spec(Outputs1D, 'wf1d')
    backend = FakeBackend(_registry(spec))
    backend.submit(_config(spec))
    data = [m for m in backend.poll() if m.stream.kind is StreamKind.LIVEDATA_DATA]
    assert len(data) == 1
