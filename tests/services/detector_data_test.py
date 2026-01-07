# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import logging

import pytest

from ess.livedata.config import instrument_registry, workflow_spec
from ess.livedata.config.models import ConfigKey
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.services.detector_data import make_detector_service_builder
from tests.helpers.livedata_app import LivedataApp


def _get_workflow_from_registry(
    instrument: str,
) -> tuple[workflow_spec.WorkflowId, workflow_spec.WorkflowSpec]:
    # Assume we can just use the first registered workflow.
    namespace = 'detector_data'
    instrument_config = instrument_registry[instrument]
    workflow_registry = instrument_config.workflow_factory
    for wid, spec in workflow_registry.items():
        if spec.namespace == namespace:
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
    workflow_id, _ = _get_workflow_from_registry(instrument)

    source_name = detector_source_name[instrument]
    config_key = ConfigKey(
        source_name=source_name, service_name="detector_data", key="workflow_config"
    )
    workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    service.step()
    assert len(sink.messages) == 1  # Workflow status message
    sink.messages.clear()

    app.publish_events(size=2000, time=2)
    service.step()
    # Each workflow call returns 6 results: cumulative, current,
    # roi_spectra_current, roi_spectra_cumulative, counts_total, counts_in_toa
    # First finalize also sends 2 initial ROI readbacks (rectangles, polygons)
    assert len(sink.messages) == 8
    assert sink.messages[0].value.nansum().value == 2000  # cumulative
    assert sink.messages[1].value.nansum().value == 2000  # current
    # No data -> no data published
    service.step()
    assert len(sink.messages) == 8

    app.publish_events(size=3000, time=4)
    service.step()
    assert len(sink.messages) == 14  # 8 + 6 (no initial readbacks on subsequent calls)
    assert sink.messages[8].value.nansum().value == 5000  # cumulative
    assert sink.messages[9].value.nansum().value == 3000  # current

    # More events but the same time
    app.publish_events(size=1000, time=4)
    # Later time
    app.publish_events(size=1000, time=5)
    service.step()
    assert len(sink.messages) == 20
    assert sink.messages[14].value.nansum().value == 7000  # cumulative
    assert sink.messages[15].value.nansum().value == 2000  # current

    # Stop workflow
    command = JobCommand(action=JobAction.stop)
    config_key = ConfigKey(key=command.key)
    stop = command.model_dump()
    app.publish_config_message(key=config_key, value=stop)
    app.publish_events(size=1000, time=10)
    service.step()
    app.publish_events(size=1000, time=20)
    service.step()
    assert len(sink.messages) == 20


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

    assert len(sink.messages) == 1  # Workflow not started, just an error message
    status = sink.messages[0].value.value
    assert status.status == workflow_spec.WorkflowStatusType.STARTUP_ERROR
    sink.messages.clear()  # Clear the error message

    good_workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=good_workflow_config.model_dump())
    app.publish_events(size=1000, time=5)
    service.step()
    # Service recovered and started the workflow, get status and data
    # First finalize sends 8 data messages (6 + 2 initial ROI readbacks)
    assert len(sink.messages) == 9  # status + 8 data messages


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
    sink.messages.clear()  # Clear the workflow status message

    # Add events and verify workflow is running
    app.publish_events(size=2000, time=2)
    service.step()
    # cumulative, current, roi_spectra_current, roi_spectra_cumulative,
    # counts_total, counts_in_toa + 2 initial ROI readbacks
    assert len(sink.messages) == 8
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
    assert len(sink.messages) == 14 + 1  # 8 + 6 + 1 for the workflow status message
    assert sink.messages[9].value.values.sum() == 5000  # cumulative (after status msg)


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
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    app = configured_dummy_detector
    sink = app.sink

    app.publish_events(size=1000, time=0, reuse_events=True)
    # Unknown schema, should be skipped
    app.publish_data(topic=app.detector_topic, time=1, data=b'corrupt data')
    app.publish_events(size=1000, time=1, reuse_events=True)

    app.step()
    # cumulative, current, roi_spectra_current, roi_spectra_cumulative,
    # counts_total, counts_in_toa + 2 initial ROI readbacks
    assert len(sink.messages) == 8
    assert sink.messages[0].value.values.sum() == 2000

    # Check log messages for exceptions
    assert "has an unknown schema. Skipping." in caplog.text
    warning_records = [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and "ess.livedata.kafka.message_adapter" in r.name
    ]
    assert any("has an unknown schema. Skipping." in r.message for r in warning_records)


def test_message_that_cannot_be_decoded_is_ignored(
    configured_dummy_detector: LivedataApp,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    app = configured_dummy_detector
    sink = app.sink

    app.publish_events(size=1000, time=0, reuse_events=True)
    # Correct schema but invalid data, should be skipped
    app.publish_data(topic=app.detector_topic, time=1, data=b'1234ev44data')
    app.publish_events(size=1000, time=1, reuse_events=True)

    app.step()
    # cumulative, current, roi_spectra_current, roi_spectra_cumulative,
    # counts_total, counts_in_toa + 2 initial ROI readbacks
    assert len(sink.messages) == 8
    assert sink.messages[0].value.values.sum() == 2000

    # Check log messages for exceptions
    assert "Error adapting message" in caplog.text
    assert "unpack_from requires a buffer" in caplog.text
    error_records = [
        r
        for r in caplog.records
        if r.levelname == "ERROR" and "ess.livedata.kafka.message_adapter" in r.name
    ]
    assert any("unpack_from requires a buffer" in r.message for r in error_records)
