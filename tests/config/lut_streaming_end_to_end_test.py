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


def _params_model(instrument: Instrument, name: str):
    registration = instrument.workflow_factory.registration(_spec_id(instrument, name))
    return registration.spec.params


def _create_job(
    instrument: Instrument,
    name: str,
    source_name: str,
    params=None,
    aux_source_names: dict[str, str] | None = None,
):
    """Create a job the way the backend does, params and all.

    Job creation is where a gate is decided, so a test that wants to know what
    a job waits for has to build one: which context streams a job requests is a
    property of the graph its params produce, not of its spec (ADR 0010).
    """
    workflow_id = _spec_id(instrument, name)
    job_id = JobId(source_name=source_name, job_number=uuid.uuid4())
    config = WorkflowConfig.from_params(
        workflow_id=workflow_id,
        job_id=job_id,
        params=None if params is None else params.model_dump(),
        aux_source_names=aux_source_names,
    )
    service = instrument.workflow_factory.get_service(workflow_id)
    return JobFactory(instrument, service_name=service).create(
        job_id=job_id, config=config
    )


#: A DREAM phasing whose cascade transmits, in nanoseconds of chopper delay at
#: 14 Hz. Zero delay throughout closes the beam a few millimetres past the
#: pulse-shaping pair, giving an all-NaN table that a consumer now refuses, so
#: the consumer half of this chain needs a cascade that actually lets neutrons
#: through. Found by scanning each chopper's delay over one rotation in beam
#: order and keeping what maximised transmission downstream; re-run that scan
#: if regenerating the geometry artifact moves a chopper and these stop working.
_TRANSMITTING_DELAYS_NS = {
    'pulse_shaping_chopper2': 45_600_000.0,
    'overlap_chopper': 17_200_000.0,
}


def _events_across_one_frame(count: int = 200) -> sc.DataArray:
    """Binned events spanning a pulse, in the shape ``ToNXevent_data`` hands over.

    Spread across the whole frame so some fall in whatever window the cascade
    leaves open; events bunched at the start of the frame convert to NaN under
    any realistic phasing and would say nothing about the table.
    """
    toa = sc.linspace('event', 0.0, 1e9 / 14.0, count, unit='ns')
    events = sc.DataArray(
        data=sc.ones(sizes={'event': count}, dtype='float64', unit='counts'),
        coords={'event_time_offset': toa},
    )
    sizes = sc.array(dims=['event_time_zero'], values=[count], unit=None, dtype='int64')
    return sc.DataArray(
        sc.bins(begin=sc.cumsum(sizes, mode='exclusive'), dim='event', data=events)
    )


def _run_lut_job(instrument: Instrument, speeds: dict[str, float] | None = None):
    """Run the lookup-table job once and return its result.

    ``speeds`` overrides individual chopper rotation-speed setpoints; the rest
    run at the source frequency.
    """
    speeds = speeds or {}
    job = _create_job(instrument, 'wavelength_lut', CHOPPER_CASCADE_SOURCE)
    aux = {}
    for chopper in instrument.choppers:
        aux[speed_setpoint_stream(chopper)] = _nxlog(
            speeds.get(chopper, 14.0),
            instrument.streams[speed_setpoint_stream(chopper)].units,
        )
        aux[delay_setpoint_stream(chopper)] = _nxlog(
            _TRANSMITTING_DELAYS_NS.get(chopper, 0.0),
            instrument.streams[delay_setpoint_stream(chopper)].units,
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


def _ingest(instrument: Instrument, result) -> dict[str, sc.DataArray]:
    """Put a lookup-table result on the wire and take it off again."""
    messages = ContextOutputExtractor(registry=instrument.workflow_factory).extract(
        [result]
    )
    serializer = make_default_sink_serializer(instrument=instrument.name)
    adapter = (
        RoutingAdapterBuilder(
            stream_mapping=get_stream_mapping(instrument=instrument.name, dev=True)
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


@pytest.fixture(scope='module')
def ingested(dream: Instrument) -> dict[str, sc.DataArray]:
    """The group tables as they arrive at a consuming service."""
    return _ingest(dream, _run_lut_job(dream))


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
    """Nothing declares this: one params model reaches the table provider and
    the other reduces straight from time of arrival, and the gate follows."""
    params_model = _params_model(dream, 'monitor_histogram')

    toa = _create_job(dream, 'monitor_histogram', MONITOR, params_model())
    wavelength = _create_job(
        dream,
        'monitor_histogram',
        MONITOR,
        params_model(coordinate_mode=CoordinateModeSettings(mode='wavelength')),
    )

    assert toa.gating_streams == set()
    assert wavelength.gating_streams == {MONITOR_STREAM}


def test_wavelength_monitor_job_consumes_the_streamed_table(
    dream: Instrument, ingested: dict[str, sc.DataArray]
) -> None:
    """The whole chain: a job created in wavelength mode takes the table that
    came off the wire as context and reduces with it, with no file anywhere."""
    params_model = _params_model(dream, 'monitor_histogram')
    job = _create_job(
        dream,
        'monitor_histogram',
        MONITOR,
        params_model(coordinate_mode=CoordinateModeSettings(mode='wavelength')),
    )

    assert job.gating_streams == {MONITOR_STREAM}

    data = JobData(
        start_time=Timestamp.from_ns(0),
        end_time=Timestamp.from_ns(1),
        primary_data={MONITOR: _events_across_one_frame()},
        aux_data={MONITOR_STREAM: ingested[MONITOR_STREAM]},
    )

    reply, result = job.process(data, finalize=True)

    assert not reply.has_error, reply.error_message
    assert result.error_message is None, result.error_message
    assert result.data['cumulative'].unit == 'counts'
    # Events were actually converted, not dropped as NaN. Without this the test
    # passes on a table that assigns no wavelength at all, which is what it did
    # while the cascade was closed.
    assert result.data['cumulative'].sum().value > 0


def test_chopper_out_of_phase_stops_the_consumer(dream: Instrument) -> None:
    """DREAM's PROD failure, end to end (#1309).

    An overlap chopper at 5 Hz cannot be phase-locked to a 14 Hz source. The
    lookup-table job publishes a table that lets nothing through rather than
    raising, and the consumer refuses that table rather than reducing with it.
    Both halves matter: raising would publish nothing and leave the consumer on
    the table it was last given, and reducing would leave it republishing its
    unchanged accumulator with a fresh timestamp, so a plot frozen since the
    choppers moved would still read as current. Publishing nothing is what lets
    the freshness indicator age.
    """
    result = _run_lut_job(dream, speeds={'overlap_chopper': 5.0})
    ingested = _ingest(dream, result)

    params_model = _params_model(dream, 'monitor_histogram')
    job = _create_job(
        dream,
        'monitor_histogram',
        MONITOR,
        params_model(coordinate_mode=CoordinateModeSettings(mode='wavelength')),
    )

    data = JobData(
        start_time=Timestamp.from_ns(0),
        end_time=Timestamp.from_ns(1),
        primary_data={MONITOR: _events_across_one_frame()},
        aux_data={MONITOR_STREAM: ingested[MONITOR_STREAM]},
    )

    reply, result = job.process(data, finalize=True)

    assert reply.has_error
    assert 'no wavelength' in reply.error_message
    # OrchestratingProcessor drops results carrying an error, so nothing of this
    # job reaches the sink and no plot is refreshed.
    assert result.error_message is not None


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

    @pytest.mark.parametrize(
        'monitors',
        [
            {
                'incident_monitor': 'beam_monitor_m2',
                'transmission_monitor': 'beam_monitor_m3',
            },
            {
                'incident_monitor': 'beam_monitor_m1',
                'transmission_monitor': 'beam_monitor_m3',
            },
        ],
        ids=['default_monitors', 'other_incident_monitor'],
    )
    def test_reduction_gates_on_one_stream_per_group(
        self, loki: Instrument, monitors: dict[str, str]
    ) -> None:
        """Two streams for three components, whichever monitors are selected.

        Both monitor roles read blocks of the shared monitor table, so the gate
        is a property of the groups the graph reaches, not of the aux selection.
        A reduction has no coordinate mode to opt out with either: it always
        reduces to wavelength, so it always waits for both tables.
        """
        job = _create_job(
            loki,
            'i_of_q',
            'loki_detector_0',
            _params_model(loki, 'i_of_q')(),
            aux_source_names=monitors,
        )

        assert {
            name for name in job.gating_streams if name.startswith('wavelength_lut/')
        } == {DETECTOR_STREAM, MONITOR_STREAM}

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
