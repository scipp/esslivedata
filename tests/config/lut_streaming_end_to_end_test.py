# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""End-to-end test of the streamed wavelength lookup table on DREAM.

Runs the real chain with real specs: the lookup-table job computes the detector
and monitor tables, they are extracted as context messages, serialized to da00,
ingested back through the Kafka route, and delivered to a wavelength-mode
monitor job as context. Nothing here is stubbed except the transport itself,
which is exercised by round-tripping the actual serializer and adapter.

This is the test that fails if any single link in ADR 0010's feedback edge is
mis-wired -- the stream name, the da00 source name, the ingest route, the block
selection, the reassembly provider, or the gate.
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
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename
from ess.livedata.workflows.detector_view_specs import CoordinateModeSettings
from ess.livedata.workflows.lut_blocks import block_ranges, select_block
from ess.livedata.workflows.lut_ranges import component_ltotal_range
from ess.livedata.workflows.wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    DETECTOR_LUT_OUTPUT,
    LUT_STREAM_NAMES,
    MONITOR_LUT_OUTPUT,
)

pytestmark = pytest.mark.slow

MONITOR = 'monitor_bunker'
DETECTOR_STREAM = LUT_STREAM_NAMES[DETECTOR_LUT_OUTPUT]
MONITOR_STREAM = LUT_STREAM_NAMES[MONITOR_LUT_OUTPUT]


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
    """The group tables as they arrive at a consuming service."""
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


def test_publishes_one_table_per_group(ingested: dict[str, sc.DataArray]) -> None:
    # Two messages for seven components: what the components share is a
    # beamline, and a table is a function of position on it.
    assert set(ingested) == {DETECTOR_STREAM, MONITOR_STREAM}


def test_tables_survive_the_wire_with_their_scalar_fields(
    ingested: dict[str, sc.DataArray],
) -> None:
    table = ingested[MONITOR_STREAM]

    assert table.dims == ('distance', 'event_time_offset')
    assert table.unit == 'angstrom'
    for coord in (
        'pulse_period',
        'pulse_stride',
        'distance_resolution',
        'time_resolution',
    ):
        assert coord in table.coords, coord


def test_detectors_share_one_dense_block(
    dream: Instrument, ingested: dict[str, sc.DataArray]
) -> None:
    """DREAM's five banks sit within two metres of each other, so one block
    covers them all -- against four metres of range if each got its own."""
    (block,) = block_ranges(ingested[DETECTOR_STREAM])

    assert (block[1] - block[0]) < sc.scalar(2.5, unit='m')


def test_monitors_get_a_block_each_with_nothing_in_between(
    dream: Instrument, ingested: dict[str, sc.DataArray]
) -> None:
    """The bunker monitor sits ~6.6 m from the source and the cave monitor
    ~72 m; a table spanning both would be 65 m of empty rows."""
    blocks = block_ranges(ingested[MONITOR_STREAM])

    assert len(blocks) == len(set(dream.monitors) & dream.lut_components)
    assert all((upper - lower) < sc.scalar(1.0, unit='m') for lower, upper in blocks)


def test_each_monitor_selects_its_own_block_off_the_wire(
    ingested: dict[str, sc.DataArray],
) -> None:
    """What replaces a stream per monitor: the roles differ by flight path, not
    by the stream they bind."""
    table = ingested[MONITOR_STREAM]
    bunker, cave = block_ranges(table)

    near = select_block(table, 0.5 * (bunker[0] + bunker[1]))
    far = select_block(table, 0.5 * (cave[0] + cave[1]))

    assert near.coords['distance'].max() < far.coords['distance'].min()


def test_every_placeable_component_has_a_block_covering_it(
    dream: Instrument, ingested: dict[str, sc.DataArray]
) -> None:
    """The invariant the two sides never exchange: producer and consumer derive
    the same flight paths from the same artifact, in different services. A
    component whose block went missing would gate fine and then fail at every
    recompute."""
    filename = str(get_nexus_geometry_filename('dream'))
    for names, stream, is_monitor in (
        (dream.detector_names, DETECTOR_STREAM, False),
        (dream.monitors, MONITOR_STREAM, True),
    ):
        for name in set(names) & dream.lut_components:
            lower, upper = component_ltotal_range(
                filename,
                name,
                is_monitor=is_monitor,
                axis_ranges=dream.axis_ranges,
            )
            block = select_block(ingested[stream], 0.5 * (lower + upper))
            assert block.sizes['distance'] > 1, name


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
    assert set(wavelength) == {MONITOR_STREAM}


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

    assert job.gating_streams == {MONITOR_STREAM}

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
        aux_data={MONITOR_STREAM: ingested[MONITOR_STREAM]},
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

    def test_reduction_binds_one_stream_per_group(self, loki: Instrument) -> None:
        """Both monitor roles read the shared monitor table, so the reduction
        declares two streams whatever it selects -- and an unpublishable table
        (``beam_monitor_m4``) cannot be gated on by declaration at all."""
        declared = loki.declared_context_keys(
            _spec_id(loki, 'i_of_q'), 'loki_detector_0'
        )

        assert {name for name in declared if name.startswith('wavelength_lut/')} == {
            DETECTOR_STREAM,
            MONITOR_STREAM,
        }

    def test_reduction_gates_on_both_tables_whatever_it_selects(
        self, loki: Instrument
    ) -> None:
        workflow_id = _spec_id(loki, 'i_of_q')
        params_model = loki.workflow_factory.registration(workflow_id).spec.params

        resolved = loki.resolve_context_keys(
            workflow_id,
            'loki_detector_0',
            params_model(),
            aux_source_names={
                'incident_monitor': 'beam_monitor_m1',
                'transmission_monitor': 'beam_monitor_m3',
            },
        )

        assert {name for name in resolved if name.startswith('wavelength_lut/')} == {
            DETECTOR_STREAM,
            MONITOR_STREAM,
        }

    def test_selecting_a_monitor_without_a_table_fails_at_job_creation(
        self, loki: Instrument
    ) -> None:
        """``beam_monitor_m4`` has no block in the monitor table. The gate would
        open on the table's arrival and the job would then fail at every
        recompute, so the factory rejects the selection before the job exists."""
        registration = loki.workflow_factory.registration(_spec_id(loki, 'i_of_q'))

        with pytest.raises(ValueError, match='beam_monitor_m4'):
            registration.factory(
                'loki_detector_0',
                registration.spec.params(),
                {
                    'incident_monitor': 'beam_monitor_m4',
                    'transmission_monitor': 'beam_monitor_m3',
                },
            )

    def test_default_monitor_selection_is_placeable(self, loki: Instrument) -> None:
        """A default naming an unplaceable monitor would break I(Q) for anyone
        who never touches the monitor selectors, so the defaults must stay on
        the placeable side of the guard above."""
        aux_sources = loki.workflow_factory.registration(
            _spec_id(loki, 'i_of_q')
        ).spec.aux_sources

        defaults = {inp.default for inp in aux_sources.inputs.values()}

        assert defaults <= loki.lut_components

    def test_reduction_has_no_coordinate_mode_so_always_gates(
        self, loki: Instrument
    ) -> None:
        workflow_id = _spec_id(loki, 'i_of_q')
        params_model = loki.workflow_factory.registration(workflow_id).spec.params

        resolved = loki.resolve_context_keys(
            workflow_id,
            'loki_detector_0',
            params_model(),
            aux_source_names={
                'incident_monitor': 'beam_monitor_m1',
                'transmission_monitor': 'beam_monitor_m3',
            },
        )

        assert DETECTOR_STREAM in resolved
