# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import logging

import pytest
from structlog.testing import capture_logs

from ess.livedata.config import instrument_registry, workflow_spec
from ess.livedata.config.models import ConfigKey
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.services.detector_data import make_detector_service_builder
from tests.helpers.livedata_app import LivedataApp


def _get_workflow_from_registry(
    instrument: str, name: str | None = None
) -> tuple[workflow_spec.WorkflowId, workflow_spec.WorkflowSpec]:
    # Assume we can just use the first registered workflow.
    namespace = 'detector_data'
    instrument_config = instrument_registry[instrument]
    workflow_registry = instrument_config.workflow_factory
    for wid, spec in workflow_registry.items():
        if spec.namespace == namespace:
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
}


@pytest.mark.parametrize("instrument", ['bifrost', 'dummy', 'dream', 'loki', 'nmx'])
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
    config_key = ConfigKey(
        source_name=source_name, service_name="detector_data", key="workflow_config"
    )
    workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    service.step()
    # No ack when message_id not set
    assert len(sink.messages) == 0

    app.publish_events(size=2000, time=2)
    service.step()
    # Each workflow call returns 10 results: cumulative, current,
    # roi_spectra_current, roi_spectra_cumulative, counts_total, counts_in_toa,
    # counts_total_cumulative, counts_in_toa_range_cumulative,
    # roi_rectangle, roi_polygon
    assert len(sink.messages) == 10
    assert sink.messages[0].value.nansum().value == 2000  # cumulative
    assert sink.messages[1].value.nansum().value == 2000  # current
    # No data -> no data published
    service.step()
    assert len(sink.messages) == 10

    app.publish_events(size=3000, time=4)
    service.step()
    assert len(sink.messages) == 20  # 10 + 10
    assert sink.messages[10].value.nansum().value == 5000  # cumulative
    assert sink.messages[11].value.nansum().value == 3000  # current

    # More events but the same time
    app.publish_events(size=1000, time=4)
    # Later time
    app.publish_events(size=1000, time=5)
    service.step()
    assert len(sink.messages) == 30  # 20 + 10
    assert sink.messages[20].value.nansum().value == 7000  # cumulative
    assert sink.messages[21].value.nansum().value == 2000  # current

    # Stop workflow
    command = JobCommand(action=JobAction.stop)
    config_key = ConfigKey(key=command.key)
    stop = command.model_dump()
    app.publish_config_message(key=config_key, value=stop)
    app.publish_events(size=1000, time=10)
    service.step()
    app.publish_events(size=1000, time=20)
    service.step()
    assert len(sink.messages) == 30


def test_service_can_recover_after_bad_workflow_id_was_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    app = make_detector_app(instrument='dummy')
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry('dummy')

    config_key = ConfigKey(
        source_name='panel_0', service_name="detector_data", key="workflow_config"
    )
    identifier = workflow_spec.WorkflowId(
        instrument='dummy', namespace='detector_data', name='abcde12345', version=1
    )
    bad_workflow_id = workflow_spec.WorkflowConfig(
        identifier=identifier,  # Invalid workflow ID
    )
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=bad_workflow_id.model_dump())

    app.publish_events(size=2000, time=2)
    service.step()
    service.step()
    app.publish_events(size=3000, time=4)
    service.step()

    # No error ack sent when message_id not set (error is logged server-side)
    assert len(sink.messages) == 0

    good_workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=good_workflow_config.model_dump())
    app.publish_events(size=1000, time=5)
    service.step()
    # Service recovered; get data only (no ack without message_id)
    # First finalize sends 10 data messages (8 + 2 initial ROI readbacks)
    assert len(sink.messages) == 10


def test_active_workflow_keeps_running_when_bad_workflow_id_was_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    app = make_detector_app(instrument='dummy')
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry('dummy')

    # Start a valid workflow first
    config_key = ConfigKey(
        source_name=detector_source_name['dummy'],
        service_name="detector_data",
        key="workflow_config",
    )
    workflow_config = workflow_spec.WorkflowConfig(
        identifier=workflow_id,
    )
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    service.step()
    # No ack without message_id
    assert len(sink.messages) == 0

    # Add events and verify workflow is running
    app.publish_events(size=2000, time=2)
    service.step()
    # cumulative, current, roi_spectra_current, roi_spectra_cumulative,
    # counts_total, counts_in_toa, counts_total_cumulative,
    # counts_in_toa_range_cumulative, roi_rectangle, roi_polygon
    assert len(sink.messages) == 10
    assert sink.messages[0].value.values.sum() == 2000

    # Try to set an invalid workflow ID
    bad_workflow_id = workflow_spec.WorkflowConfig(
        identifier=workflow_spec.WorkflowId(
            instrument='dummy', namespace='detector_data', name='abcde12345', version=1
        )  # Invalid workflow ID
    )
    app.publish_config_message(key=config_key, value=bad_workflow_id.model_dump())

    # Add more events and verify the original workflow is still running
    app.publish_events(size=3000, time=4)
    service.step()
    # No error ack without message_id, just data messages (10 + 10)
    assert len(sink.messages) == 20
    assert sink.messages[10].value.values.sum() == 5000  # cumulative


@pytest.fixture
def configured_dummy_detector() -> LivedataApp:
    app = make_detector_app(instrument='dummy')
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry('dummy')

    config_key = ConfigKey(
        source_name='panel_0', service_name="detector_data", key="workflow_config"
    )
    workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    # Process config message before data arrives. Without calling step() the order of
    # processing of config vs data messages is not guaranteed.
    service.step()
    sink.messages.clear()  # Clear workflow start message
    return app


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
    assert len(sink.messages) == 10
    assert sink.messages[0].value.values.sum() == 2000

    # Check log messages for warnings
    warning_logs = [log for log in captured if log['log_level'] == 'warning']
    assert any(
        "has an unknown schema. Skipping." in log['event'] for log in warning_logs
    )


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
    assert len(sink.messages) == 10
    assert sink.messages[0].value.values.sum() == 2000

    # Check log messages for exceptions
    error_logs = [log for log in captured if log['log_level'] == 'error']
    assert any("Error adapting message" in log['event'] for log in error_logs)
    assert any("unpack_from requires a buffer" in str(log) for log in error_logs)
