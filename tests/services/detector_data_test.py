# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import logging
import uuid

import pytest
import scipp as sc
from structlog.testing import capture_logs

from ess.livedata.config import instrument_registry, workflow_spec
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.core.message import StreamKind
from ess.livedata.services.detector_data import make_detector_service_builder
from tests.helpers.livedata_app import LivedataApp


def _job_id(source: str) -> JobId:
    return JobId(source_name=source, job_number=uuid.uuid4())


def _data_messages(sink) -> list:
    """Workflow result messages, excluding the NICOS device stream."""
    return [m for m in sink.messages if m.stream.kind == StreamKind.LIVEDATA_DATA]


def _get_workflow_from_registry(
    instrument: str, name: str | None = None
) -> tuple[workflow_spec.WorkflowId, workflow_spec.WorkflowSpec]:
    # Assume we can just use the first registered workflow.
    namespace = 'detector_data'
    instrument_config = instrument_registry[instrument]
    workflow_registry = instrument_config.workflow_factory
    for wid, spec in workflow_registry.items():
        if spec.group.name == namespace:
            if name is None or name == spec.name:
                return wid, spec
    raise ValueError(f"Namespace {namespace} not found in specs")


def make_detector_app(instrument: str) -> LivedataApp:
    builder = make_detector_service_builder(instrument=instrument)
    return LivedataApp.from_service_builder(builder)


detector_source_name = {
    'dummy': 'panel_0',
    'dream': 'mantle_detector',
    'bifrost': 'unified_detector',
    'loki': 'loki_detector_0',
    'nmx': 'detector_panel_0',
    'beer': 'beer_detector_s2',
}


@pytest.mark.parametrize(
    "instrument", ['beer', 'bifrost', 'dummy', 'dream', 'loki', 'nmx']
)
def test_can_configure_and_stop_detector_workflow(
    instrument: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    app = make_detector_app(instrument)
    sink = app.sink
    service = app.service
    name = 'detector_projection' if instrument == 'dream' else None
    workflow_id, _ = _get_workflow_from_registry(instrument, name=name)

    source_name = detector_source_name[instrument]
    workflow_config = workflow_spec.WorkflowConfig(
        identifier=workflow_id, job_id=_job_id(source_name)
    )
    # Trigger workflow start
    app.publish_config_message(workflow_config)
    service.step()
    # Config ack lands on response_messages, not the data sink
    assert len(_data_messages(sink)) == 0

    if instrument == 'loki':
        # LOKI rear bank consumes the merged detector_carriage device stream;
        # the synthesizer emits only after all three substreams (RBV, VAL,
        # DMOV) have been observed at least once.
        app.publish_log_message(
            source_name='detector_carriage/target_value', time=1, value=5000.0
        )
        app.publish_log_message(
            source_name='detector_carriage/idle_flag', time=1, value=1
        )
        app.publish_log_message(
            source_name='detector_carriage/value', time=1, value=5000.0
        )
    # Each workflow call returns 10 results by default: cumulative, current,
    # counts_total, counts_in_toa, counts_total_cumulative,
    # counts_in_toa_range_cumulative, roi_spectra_cumulative,
    # roi_spectra_current, roi_rectangle, roi_polygon. Instruments that enable
    # a unified spectrum output add one additional spectrum_view message.
    n_out = 11 if instrument == 'bifrost' else 10
    app.publish_events(size=2000, time=2)
    service.step()
    assert len(_data_messages(sink)) == n_out
    assert _data_messages(sink)[0].value.nansum().value == 2000  # cumulative
    assert _data_messages(sink)[1].value.nansum().value == 2000  # current
    # No data -> no data published
    service.step()
    assert len(_data_messages(sink)) == n_out

    app.publish_events(size=3000, time=4)
    service.step()
    assert len(_data_messages(sink)) == 2 * n_out
    assert _data_messages(sink)[n_out].value.nansum().value == 5000  # cumulative
    assert _data_messages(sink)[n_out + 1].value.nansum().value == 3000  # current

    # More events but the same time
    app.publish_events(size=1000, time=4)
    # Later time
    app.publish_events(size=1000, time=5)
    service.step()
    assert len(_data_messages(sink)) == 3 * n_out
    assert _data_messages(sink)[2 * n_out].value.nansum().value == 7000  # cumulative
    assert _data_messages(sink)[2 * n_out + 1].value.nansum().value == 2000  # current

    # Stop workflow
    command = JobCommand(action=JobAction.stop)
    app.publish_config_message(command)
    app.publish_events(size=1000, time=10)
    service.step()
    app.publish_events(size=1000, time=20)
    service.step()
    assert len(_data_messages(sink)) == 3 * n_out


def test_loki_cumulative_resets_when_detector_carriage_moves() -> None:
    """A detector move resets the cumulative histogram in TOA mode.

    LOKI's rear bank (``loki_detector_0``) rides the ``detector_carriage``, whose
    f144 position patches the ``depends_on`` transform live. The geometric xy
    projection stamps that transform as the ``DETECTOR_TRANSFORM`` coord, so a
    move resets the cumulative accumulator instead of summing across the shifted
    screen edges. The view is in its default TOA mode -- position matters here
    even without wavelength, because the move shifts the projection's screen bins.
    """
    app = make_detector_app('loki')
    sink = app.sink
    service = app.service
    # The movable xy projection, not tube_view (which ignores carriage position).
    workflow_id, _ = _get_workflow_from_registry('loki', name='detector_xy_projection')

    source_name = 'loki_detector_0'
    workflow_config = workflow_spec.WorkflowConfig(
        identifier=workflow_id, job_id=_job_id(source_name)
    )
    app.publish_config_message(workflow_config)
    service.step()

    def prime_carriage(*, position: float, time: int) -> None:
        # The rear bank consumes the merged detector_carriage device stream; the
        # synthesizer emits only once all three substreams (RBV, VAL, DMOV) have
        # been observed at least once.
        app.publish_log_message(
            source_name='detector_carriage/target_value', time=time, value=position
        )
        app.publish_log_message(
            source_name='detector_carriage/idle_flag', time=time, value=1
        )
        app.publish_log_message(
            source_name='detector_carriage/value', time=time, value=position
        )

    def move_carriage(*, position: float, time: int) -> None:
        # Once primed, the readback (VAL) alone drives the position. Each substream
        # update emits a merged sample, so the move must use a fresh timestamp: a
        # sample whose time equals the last one is dropped (see ToNXlog.add).
        app.publish_log_message(
            source_name='detector_carriage/value', time=time, value=position
        )

    n_out = 10

    # Cycle 1: park, accumulate a first batch.
    prime_carriage(position=5000.0, time=1)
    app.publish_events(size=2000, time=2)
    service.step()
    assert len(_data_messages(sink)) == n_out
    assert _data_messages(sink)[0].value.nansum().value == 2000  # cumulative
    assert _data_messages(sink)[1].value.nansum().value == 2000  # current

    # Cycle 2: no move -> cumulative keeps accumulating (no reset).
    app.publish_events(size=3000, time=4)
    service.step()
    assert len(_data_messages(sink)) == 2 * n_out
    assert _data_messages(sink)[n_out].value.nansum().value == 5000  # cumulative
    assert _data_messages(sink)[n_out + 1].value.nansum().value == 3000  # current

    # Cycle 3: move the carriage, then accumulate -> cumulative resets.
    move_carriage(position=6000.0, time=10)
    app.publish_events(size=1000, time=11)
    service.step()
    assert len(_data_messages(sink)) == 3 * n_out
    cumulative = _data_messages(sink)[2 * n_out].value
    current = _data_messages(sink)[2 * n_out + 1].value
    # The pre-move 5000 counts are discarded: cumulative restarts from the move.
    assert cumulative.nansum().value == 1000
    assert sc.allclose(cumulative.data, current.data)


def test_odin_cumulative_resets_when_the_readout_resolution_changes() -> None:
    """A readout reconfiguration resets the cumulative image.

    ODIN's Timepix3 is ingested at reduced resolution: event ids are remapped
    onto a 512x512 grid in the preprocessor, using a stride derived from the
    resolution the detector is currently streaming. That resolution is
    reconfigurable and is not announced on any stream, so it is inferred from
    the ids -- which means it can be revised while a job runs. Counts remapped
    with the old stride land in different pixels than counts remapped with the
    new one, so summing across the revision would produce a blended image.
    ``DownsamplePixelIds`` stamps the resolution as a coord and the cumulative
    accumulator resets on it, exactly as it does on a detector move.
    """
    app = make_detector_app('odin')
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry('odin')

    source_name = 'timepix3'
    app.publish_config_message(
        workflow_spec.WorkflowConfig(
            identifier=workflow_id, job_id=_job_id(source_name)
        )
    )
    service.step()

    n_out = 10
    # Only the first rows of the 4096x4096 panel light up, so the ids seen so
    # far are consistent with a much smaller readout.
    app.publish_events(size=2000, time=2, id_range=(0, 100 * 4096))
    service.step()
    assert _data_messages(sink)[0].value.nansum().value == 2000  # cumulative

    # Still nothing above the inferred resolution -> plain accumulation.
    app.publish_events(size=3000, time=4, id_range=(0, 100 * 4096))
    service.step()
    assert _data_messages(sink)[n_out].value.nansum().value == 5000  # cumulative

    # An id from the far corner proves the panel is the full 4096, so the
    # stride changes and the 5000 counts mapped with the old one are discarded.
    app.publish_events(size=1000, time=6, id_range=(4000 * 4096, 4096 * 4096 - 1))
    service.step()
    cumulative = _data_messages(sink)[2 * n_out].value
    current = _data_messages(sink)[2 * n_out + 1].value
    assert cumulative.nansum().value == 1000
    assert sc.allclose(cumulative.data, current.data)


def test_magic_projection_stays_gated_until_rotation_readback_arrives() -> None:
    """MAGIC's banks ride rotation stages, so the projection waits for the readback.

    The binding names the merged ``detector_a_rotation`` device stream, which the
    synthesizer emits only once all three substreams have been seen -- exactly what
    the dev fake log producer publishes.
    """
    app = make_detector_app('magic')
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry('magic', name='detector_projection')

    source_name = 'magic_detector_a'
    app.publish_config_message(
        workflow_spec.WorkflowConfig(
            identifier=workflow_id, job_id=_job_id(source_name)
        )
    )
    service.step()

    # Without the rotation context the job is gated: events accumulate, nothing is
    # published.
    app.publish_events(size=1000, time=1)
    service.step()
    assert len(_data_messages(sink)) == 0

    for substream, value in (
        ('detector_a_rotation/target_value', 0.0),
        ('detector_a_rotation/idle_flag', 1.0),
        ('detector_a_rotation/value', 0.0),
    ):
        app.publish_log_message(source_name=substream, time=2, value=value)
    app.publish_events(size=1000, time=3)
    service.step()
    assert len(_data_messages(sink)) > 0


def test_service_can_recover_after_bad_workflow_id_was_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    app = make_detector_app(instrument='dummy')
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry('dummy')

    identifier = workflow_spec.WorkflowId(
        instrument='dummy', name='abcde12345', version=1
    )
    bad_workflow_id = workflow_spec.WorkflowConfig(
        identifier=identifier,  # Invalid workflow ID
        job_id=_job_id('panel_0'),
    )
    # Trigger workflow start
    app.publish_config_message(bad_workflow_id)

    app.publish_events(size=2000, time=2)
    service.step()
    service.step()
    app.publish_events(size=3000, time=4)
    service.step()

    # No error ack sent when message_id not set (error is logged server-side)
    assert len(_data_messages(sink)) == 0

    good_workflow_config = workflow_spec.WorkflowConfig(
        identifier=workflow_id, job_id=_job_id('panel_0')
    )
    # Trigger workflow start
    app.publish_config_message(good_workflow_config)
    app.publish_events(size=1000, time=5)
    service.step()
    # Service recovered; data only -- the ack is on response_messages
    # First finalize sends 10 data messages (8 + 2 initial ROI readbacks)
    assert len(_data_messages(sink)) == 10


def test_active_workflow_keeps_running_when_bad_workflow_id_was_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    app = make_detector_app(instrument='dummy')
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry('dummy')

    # Start a valid workflow first
    workflow_config = workflow_spec.WorkflowConfig(
        identifier=workflow_id,
        job_id=_job_id(detector_source_name['dummy']),
    )
    app.publish_config_message(workflow_config)
    service.step()
    # Config ack lands on response_messages, not the data sink
    assert len(_data_messages(sink)) == 0

    # Add events and verify workflow is running
    app.publish_events(size=2000, time=2)
    service.step()
    # cumulative, current, roi_spectra_current, roi_spectra_cumulative,
    # counts_total, counts_in_toa, counts_total_cumulative,
    # counts_in_toa_range_cumulative, roi_rectangle, roi_polygon
    assert len(_data_messages(sink)) == 10
    assert _data_messages(sink)[0].value.values.sum() == 2000

    # Try to set an invalid workflow ID
    bad_workflow_id = workflow_spec.WorkflowConfig(
        identifier=workflow_spec.WorkflowId(
            instrument='dummy', name='abcde12345', version=1
        ),  # Invalid workflow ID
        job_id=_job_id(detector_source_name['dummy']),
    )
    app.publish_config_message(bad_workflow_id)

    # Add more events and verify the original workflow is still running
    app.publish_events(size=3000, time=4)
    service.step()
    # No error ack without message_id, just data messages (10 + 10)
    assert len(_data_messages(sink)) == 20
    assert _data_messages(sink)[10].value.values.sum() == 5000  # cumulative


@pytest.fixture
def configured_dummy_detector() -> LivedataApp:
    app = make_detector_app(instrument='dummy')
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry('dummy')

    workflow_config = workflow_spec.WorkflowConfig(
        identifier=workflow_id, job_id=_job_id('panel_0')
    )
    # Trigger workflow start
    app.publish_config_message(workflow_config)
    # Process config message before data arrives. Without calling step() the order of
    # processing of config vs data messages is not guaranteed.
    service.step()
    sink.messages.clear()  # Clear workflow start message
    return app


def test_detector_counts_are_published_as_a_nicos_device(
    configured_dummy_detector: LivedataApp,
) -> None:
    """A running detector view exposes its bank total on the NICOS device topic.

    The device is keyed by the contracted device name rather than by the
    ``ResultKey``, so the job's random ``job_number`` does not reach NICOS, and it
    carries ``start_time`` as the generation marker (see ADR 0006).
    """
    app = configured_dummy_detector
    app.publish_events(size=2000, time=2)
    app.step()

    devices = [
        m for m in app.sink.messages if m.stream.kind == StreamKind.LIVEDATA_NICOS_DATA
    ]
    assert [m.stream.name for m in devices] == ['panel_0_counts_total']
    value = devices[0].value
    assert value.value == 2000
    assert 'start_time' in value.coords
    assert 'end_time' in value.coords


def test_message_with_unknown_schema_is_ignored(
    configured_dummy_detector: LivedataApp,
) -> None:
    app = configured_dummy_detector
    sink = app.sink

    app.publish_events(size=1000, time=0, reuse_events=True)
    # Unknown schema, should be skipped
    app.publish_data(topic=app.detector_topic, time=1, data=b'corrupt data')
    app.publish_events(size=1000, time=1, reuse_events=True)

    with capture_logs() as captured:
        app.step()

    # cumulative, current, roi_spectra_current, roi_spectra_cumulative,
    # counts_total, counts_in_toa, counts_total_cumulative,
    # counts_in_toa_range_cumulative + 2 initial ROI readbacks
    assert len(_data_messages(sink)) == 10
    assert _data_messages(sink)[0].value.values.sum() == 2000

    # Check log messages for warnings
    warning_logs = [log for log in captured if log['log_level'] == 'warning']
    assert any("unknown schema" in log['event'] for log in warning_logs)


def test_message_that_cannot_be_decoded_is_ignored(
    configured_dummy_detector: LivedataApp,
) -> None:
    app = configured_dummy_detector
    sink = app.sink

    app.publish_events(size=1000, time=0, reuse_events=True)
    # Correct schema but invalid data, should be skipped
    app.publish_data(topic=app.detector_topic, time=1, data=b'1234ev44data')
    app.publish_events(size=1000, time=1, reuse_events=True)

    with capture_logs() as captured:
        app.step()

    # cumulative, current, roi_spectra_current, roi_spectra_cumulative,
    # counts_total, counts_in_toa, counts_total_cumulative,
    # counts_in_toa_range_cumulative + 2 initial ROI readbacks
    assert len(_data_messages(sink)) == 10
    assert _data_messages(sink)[0].value.values.sum() == 2000

    # Check log messages for exceptions
    error_logs = [log for log in captured if log['log_level'] == 'error']
    assert any("Error adapting message" in log['event'] for log in error_logs)
    assert any("unpack_from requires a buffer" in str(log) for log in error_logs)
