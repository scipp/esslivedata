# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""End-to-end test of the streamed wavelength lookup table on DREAM.

Runs the real chain with real specs: the lookup-table job computes per-component
tables, they are extracted as context messages, serialized to da00, ingested
back through the Kafka route, and delivered to a wavelength-mode monitor job as
context. Nothing here is stubbed except the transport itself, which is exercised
by round-tripping the actual serializer and adapter.

This is the test that fails if any single link in ADR 0010's feedback edge is
mis-wired -- the rendered stream name, the da00 source name, the ingest route,
the reassembly provider, or the gate.
"""

from __future__ import annotations

import uuid

import pytest
import scipp as sc

from ess.livedata.config.chopper import delay_setpoint_stream, speed_setpoint_stream
from ess.livedata.config.instrument import Instrument, instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.config.workflow_spec import JobId, WorkflowConfig
from ess.livedata.core.context_outputs import ContextOutputExtractor
from ess.livedata.core.job import JobData
from ess.livedata.core.job_manager import JobFactory
from ess.livedata.core.message import StreamKind
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.kafka.message_adapter import FakeKafkaMessage
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.kafka.sink_serializers import make_default_sink_serializer
from ess.livedata.workflows.detector_view_specs import CoordinateModeSettings
from ess.livedata.workflows.wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    lut_stream_name,
)

pytestmark = pytest.mark.slow

MONITOR = 'monitor_bunker'


@pytest.fixture(scope='module')
def dream() -> Instrument:
    get_config('dream')
    instrument = instrument_registry['dream']
    instrument.load_factories()
    return instrument


def _nxlog(value: float, unit) -> sc.DataArray:
    time = sc.epoch(unit='ns') + sc.arange('time', 3, unit='ns')
    return sc.DataArray(
        sc.full(value=value, sizes={'time': 3}, unit=unit), coords={'time': time}
    )


def _spec_id(instrument: Instrument, name: str):
    return next(w for w in instrument.workflow_factory if w.name == name)


def _run_lut_job(instrument: Instrument):
    """Run the lookup-table job once and return its result."""
    workflow_id = _spec_id(instrument, 'wavelength_lut')
    job_id = JobId(source_name=CHOPPER_CASCADE_SOURCE, job_number=uuid.uuid4())
    config = WorkflowConfig.from_params(
        workflow_id=workflow_id, job_id=job_id, params=None, aux_source_names=None
    )
    service = instrument.workflow_factory.get_service(workflow_id)
    job = JobFactory(instrument, service_name=service).create(
        job_id=job_id, config=config
    )
    aux = {}
    for chopper in instrument.choppers:
        aux[speed_setpoint_stream(chopper)] = _nxlog(
            14.0, instrument.streams[speed_setpoint_stream(chopper)].units
        )
        aux[delay_setpoint_stream(chopper)] = _nxlog(
            0.0, instrument.streams[delay_setpoint_stream(chopper)].units
        )
    data = JobData(
        start_time=Timestamp.from_ns(0),
        end_time=Timestamp.from_ns(1),
        primary_data={CHOPPER_CASCADE_SOURCE: _nxlog(1.0, None)},
        aux_data=aux,
    )
    reply, result = job.process(data, finalize=True)
    assert not reply.has_error, reply.error_message
    assert result.error_message is None, result.error_message
    return result


@pytest.fixture(scope='module')
def ingested(dream: Instrument) -> dict[str, sc.DataArray]:
    """Per-component tables as they arrive at a consuming service."""
    result = _run_lut_job(dream)
    messages = ContextOutputExtractor(registry=dream.workflow_factory).extract([result])
    serializer = make_default_sink_serializer(instrument='dream')
    adapter = (
        RoutingAdapterBuilder(
            stream_mapping=get_stream_mapping(instrument='dream', dev=True)
        )
        .with_livedata_context_route()
        .build()
    )
    out = {}
    for message in messages:
        serialized = serializer.serialize(message)
        received = adapter.adapt(
            FakeKafkaMessage(value=serialized.value, topic=serialized.topic)
        )
        assert received.stream.kind == StreamKind.LIVEDATA_CONTEXT
        out[received.stream.name] = received.value
    return out


def test_every_placeable_component_publishes_a_table(
    dream: Instrument, ingested: dict[str, sc.DataArray]
) -> None:
    # DREAM has no moving components, so every detector and monitor has a
    # derivable flight-path range and therefore a table.
    expected = {
        lut_stream_name(name) for name in (*dream.detector_names, *dream.monitors)
    }

    assert set(ingested) == expected


def test_tables_survive_the_wire_with_their_provenance(
    ingested: dict[str, sc.DataArray],
) -> None:
    table = ingested[lut_stream_name(MONITOR)]

    assert table.dims == ('distance', 'event_time_offset')
    assert table.unit == 'angstrom'
    for coord in (
        'pulse_period',
        'pulse_stride',
        'distance_resolution',
        'time_resolution',
    ):
        assert coord in table.coords, coord


def test_each_table_covers_only_its_own_component(
    ingested: dict[str, sc.DataArray],
) -> None:
    """The point of one table per component: the bunker monitor sits ~6.6 m from
    the source while the detectors are ~78 m away, and one table spanning both
    would be almost entirely empty rows."""
    monitor = ingested[lut_stream_name(MONITOR)].coords['distance']
    detector = ingested[lut_stream_name('mantle_detector')].coords['distance']

    assert monitor.max() < sc.scalar(10.0, unit='m')
    assert detector.min() > sc.scalar(70.0, unit='m')


def test_wavelength_job_gates_on_its_table_and_toa_job_does_not(
    dream: Instrument,
) -> None:
    workflow_id = _spec_id(dream, 'monitor_histogram')
    params_model = dream.workflow_factory.registration(workflow_id).spec.params

    toa = dream.resolve_context_keys(workflow_id, MONITOR, params_model())
    wavelength = dream.resolve_context_keys(
        workflow_id,
        MONITOR,
        params_model(coordinate_mode=CoordinateModeSettings(mode='wavelength')),
    )

    assert toa == {}
    assert set(wavelength) == {lut_stream_name(MONITOR)}


def test_wavelength_monitor_job_consumes_the_streamed_table(
    dream: Instrument, ingested: dict[str, sc.DataArray]
) -> None:
    """The whole chain: a job created in wavelength mode takes the table that
    came off the wire as context and reduces with it, with no file anywhere."""
    workflow_id = _spec_id(dream, 'monitor_histogram')
    params_model = dream.workflow_factory.registration(workflow_id).spec.params
    job_id = JobId(source_name=MONITOR, job_number=uuid.uuid4())
    config = WorkflowConfig.from_params(
        workflow_id=workflow_id,
        job_id=job_id,
        params=params_model(
            coordinate_mode=CoordinateModeSettings(mode='wavelength')
        ).model_dump(),
        aux_source_names=None,
    )
    service = dream.workflow_factory.get_service(workflow_id)
    job = JobFactory(dream, service_name=service).create(job_id=job_id, config=config)

    assert job.gating_streams == {lut_stream_name(MONITOR)}

    # Binned events in the shape ToNXevent_data hands to the workflow.
    toa = sc.array(dims=['event'], values=[1.0, 2.0, 3.0, 4.0], unit='ns')
    weights = sc.ones(sizes={'event': 4}, dtype='float64', unit='counts')
    events = sc.DataArray(data=weights, coords={'event_time_offset': toa})
    sizes = sc.array(dims=['event_time_zero'], values=[4], unit=None, dtype='int64')
    binned = sc.DataArray(
        sc.bins(begin=sc.cumsum(sizes, mode='exclusive'), dim='event', data=events)
    )
    data = JobData(
        start_time=Timestamp.from_ns(0),
        end_time=Timestamp.from_ns(1),
        primary_data={MONITOR: binned},
        aux_data={lut_stream_name(MONITOR): ingested[lut_stream_name(MONITOR)]},
    )

    reply, result = job.process(data, finalize=True)

    assert not reply.has_error, reply.error_message
    assert result.error_message is None, result.error_message
    assert result.data['cumulative'].unit == 'counts'


@pytest.fixture(scope='module')
def loki() -> Instrument:
    get_config('loki')
    instrument = instrument_registry['loki']
    instrument.load_factories()
    return instrument


class TestLokiMotionAndRoles:
    """LOKI exercises the two cases DREAM cannot.

    Its rear bank rides a declared axis, and ``beam_monitor_m4`` rides an
    undeclared one; and its I(Q) reduction needs a table per sciline
    ``Component`` rather than one per source.
    """

    def test_declared_axis_makes_the_rear_bank_placeable(
        self, loki: Instrument
    ) -> None:
        assert 'loki_detector_0' in loki.lut_components

    def test_component_on_an_undeclared_axis_has_no_table(
        self, loki: Instrument
    ) -> None:
        # No nominal position, so no range, so no table -- and therefore no
        # binding anywhere, rather than a table placed at a guessed distance.
        assert 'beam_monitor_m4' not in loki.lut_components

    def test_nothing_gates_on_a_table_that_is_never_published(
        self, loki: Instrument
    ) -> None:
        """The reason the previous test matters: binding an unpublishable
        stream would wedge every job that could have selected that monitor."""
        workflow_id = _spec_id(loki, 'i_of_q')

        declared = loki.declared_context_keys(workflow_id, 'loki_detector_0')

        assert lut_stream_name('beam_monitor_m4') not in declared

    def test_reduction_gates_on_every_candidate_monitor(self, loki: Instrument) -> None:
        """Which monitor fills the incident or transmission role is a per-job
        aux selection, so all candidates are bound. They come from the same LUT
        job and arrive together, so gating on all of them opens the gate at the
        same instant as gating on one."""
        workflow_id = _spec_id(loki, 'i_of_q')

        declared = loki.declared_context_keys(workflow_id, 'loki_detector_0')

        assert {name for name in declared if name.startswith('wavelength_lut/')} == {
            lut_stream_name(name)
            for name in (
                'loki_detector_0',
                'beam_monitor_m0',
                'beam_monitor_m1',
                'beam_monitor_m2',
                'beam_monitor_m3',
            )
        }

    def test_reduction_binds_only_its_own_detector(self, loki: Instrument) -> None:
        # The detector is fixed per job; gating on all nine banks' tables would
        # be over-gating with no upside.
        declared = loki.declared_context_keys(
            _spec_id(loki, 'i_of_q'), 'loki_detector_5'
        )

        assert lut_stream_name('loki_detector_5') in declared
        assert lut_stream_name('loki_detector_0') not in declared

    def test_reduction_has_no_coordinate_mode_so_always_gates(
        self, loki: Instrument
    ) -> None:
        workflow_id = _spec_id(loki, 'i_of_q')
        params_model = loki.workflow_factory.registration(workflow_id).spec.params

        resolved = loki.resolve_context_keys(
            workflow_id, 'loki_detector_0', params_model()
        )

        assert lut_stream_name('loki_detector_0') in resolved
