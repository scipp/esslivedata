# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import json
import logging

import numpy as np
import pytest
import scipp as sc
from streaming_data_types import eventdata_ev44
from structlog.testing import capture_logs

from ess.livedata.config import instrument_registry, workflow_spec
from ess.livedata.config.models import ConfigKey
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.services.data_reduction import make_reduction_service_builder
from tests.helpers.livedata_app import LivedataApp


def _get_workflow_from_registry(
    instrument: str, name: str
) -> tuple[workflow_spec.WorkflowId, workflow_spec.WorkflowSpec]:
    instrument_config = instrument_registry[instrument]
    workflow_registry = instrument_config.workflow_factory
    for wid, spec in workflow_registry.items():
        if spec.name == name:
            return wid, spec
    raise ValueError(f"Workflow {name} not found in specs")


def make_reduction_app(instrument: str) -> LivedataApp:
    builder = make_reduction_service_builder(instrument=instrument)
    return LivedataApp.from_service_builder(builder)


first_source_name = {
    'dummy': 'panel_0',
    'dream': 'mantle_detector',
    'bifrost': 'unified_detector',
    'loki': 'loki_detector_0',
}


@pytest.mark.parametrize(
    "instrument",
    [
        pytest.param('bifrost', marks=pytest.mark.slow),
        'dummy',
        'dream',
    ],
)
def test_can_configure_and_stop_workflow_with_detector(
    instrument: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    app = make_reduction_app(instrument=instrument)
    sink = app.sink
    service = app.service
    workflow_name = {
        'bifrost': 'spectrum_view',
        'dummy': 'total_counts',
        'dream': 'powder_reduction',
    }[instrument]
    n_target = {'bifrost': 1, 'dummy': 1, 'dream': 3}[instrument]
    check_counts = instrument != 'dream'
    # WorkflowSpec (second arg) unused here since the workflow does not take params.
    workflow_id, _ = _get_workflow_from_registry(instrument, workflow_name)

    source_name = first_source_name[instrument]
    config_key = ConfigKey(
        source_name=source_name, service_name="data_reduction", key="workflow_config"
    )
    workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    service.step()
    # No acknowledgement sent when message_id is not set (backward compatibility)
    assert len(sink.messages) == 0

    app.publish_events(size=2000, time=2)
    service.step()
    assert len(sink.messages) == 1 * n_target
    if check_counts:
        # Events before workflow config was published should not be included
        assert sink.messages[0 * n_target].value.values.sum() == 2000
    service.step()
    assert len(sink.messages) == 1 * n_target
    app.publish_events(size=3000, time=4)
    service.step()
    assert len(sink.messages) == 2 * n_target
    if check_counts:
        assert sink.messages[1 * n_target].value.values.sum() == 5000

    # More events but the same time
    app.publish_events(size=1000, time=4)
    # Later time
    app.publish_events(size=1000, time=5)
    service.step()
    assert len(sink.messages) == 3 * n_target
    if check_counts:
        # Result should include previous events with duplicate time
        assert sink.messages[2 * n_target].value.values.sum() == 7000

    # Stop workflow
    command = JobCommand(action=JobAction.stop)
    config_key = ConfigKey(key=command.key)
    stop = command.model_dump()
    app.publish_config_message(key=config_key, value=stop)
    app.publish_events(size=1000, time=10)
    service.step()
    app.publish_events(size=1000, time=20)
    service.step()
    assert len(sink.messages) == 3 * n_target


def test_dream_powder_reduction_produces_tof_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """DREAM powder reduction outputs I(TOF) alongside I(d) and I(d, 2theta)."""
    caplog.set_level(logging.INFO)
    app = make_reduction_app(instrument='dream')
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry('dream', 'powder_reduction')

    source_name = first_source_name['dream']
    config_key = ConfigKey(
        source_name=source_name, service_name="data_reduction", key="workflow_config"
    )
    workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    service.step()

    app.publish_events(size=2000, time=2)
    service.step()

    outputs: dict[str, sc.DataArray] = {}
    for msg in sink.messages:
        result_key = json.loads(msg.stream.name)
        outputs[result_key['output_name']] = msg.value

    assert 'focussed_data_tof' in outputs
    tof_output = outputs['focussed_data_tof']
    assert isinstance(tof_output, sc.DataArray)
    assert tof_output.dims == ('tof',)
    assert 'tof' in tof_output.coords
    assert tof_output.coords['tof'].unit == 'us'
    assert len(tof_output.coords['tof']) > 0


@pytest.mark.parametrize(
    "instrument",
    [
        'dream',
        pytest.param('loki', marks=pytest.mark.slow),
    ],
)
def test_can_configure_and_stop_workflow_with_detector_and_monitors(
    instrument: str, caplog: pytest.LogCaptureFixture
) -> None:
    if instrument == 'dream':
        # Note we are instead currently testing a workflow without monitors instead.
        # See above.
        pytest.skip("Dream requires upstream fixes for event-mode monitors.")
    caplog.set_level(logging.INFO)
    app = make_reduction_app(instrument=instrument)
    sink = app.sink
    service = app.service
    workflow_name = {
        'dream': 'powder_reduction',
        'loki': 'i_of_q',
    }[instrument]
    n_target = {'dream': 2, 'loki': 2}[instrument]
    workflow_id, spec = _get_workflow_from_registry(instrument, workflow_name)

    source_name = first_source_name[instrument]
    config_key = ConfigKey(
        source_name=source_name, service_name="data_reduction", key="workflow_config"
    )
    # Get default aux_source_names from the workflow spec
    aux_source_names = {}
    if spec.aux_sources is not None:
        aux_source_names = spec.aux_sources().model_dump()
    workflow_config = workflow_spec.WorkflowConfig(
        identifier=workflow_id, aux_source_names=aux_source_names
    )
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    service.step()
    sink.messages.clear()  # Clear the workflow status message(s), one per source name.

    app.publish_events(size=2000, time=2)
    service.step()
    # No monitor data yet, so the workflow was not able to produce a result yet
    assert len(sink.messages) == 0

    # Monitor messages triggers and passes the workflow
    app.publish_monitor_events(size=2000, time=3)
    service.step()
    assert len(sink.messages) == 1 * n_target

    # Once we have monitor data the worklow works even if only detector data comes in.
    # There is currently no "smart" mechanism to check if we have monitor and detector
    # data for the same time (intervals).
    app.publish_events(size=3000, time=4)
    service.step()
    assert len(sink.messages) == 2 * n_target

    # More events but the same time
    app.publish_events(size=1000, time=4)
    # Later time
    app.publish_events(size=1000, time=5)
    service.step()
    assert len(sink.messages) == 3 * n_target

    # Stop workflow
    command = JobCommand(action=JobAction.stop)
    config_key = ConfigKey(key=command.key)
    stop = command.model_dump()
    app.publish_config_message(key=config_key, value=stop)
    app.publish_events(size=1000, time=10)
    service.step()
    app.publish_events(size=1000, time=20)
    service.step()
    assert len(sink.messages) == 3 * n_target


def test_can_clear_workflow_via_config(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    app = make_reduction_app(instrument='dummy')
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry('dummy', 'total_counts')

    app.publish_events(size=1000, time=0)
    service.step()

    config_key = ConfigKey(
        source_name='panel_0', service_name="data_reduction", key="workflow_config"
    )
    workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    app.publish_events(size=2000, time=1)
    app.publish_events(size=3000, time=2)
    service.step()
    # Only data reduction result (no ack when message_id not set)
    assert len(sink.messages) == 1
    assert sink.messages[-1].value.values.sum() == 5000

    model = JobCommand(action=JobAction.reset)
    config_key = ConfigKey(key=model.key)
    app.publish_config_message(key=config_key, value=model.model_dump())

    app.publish_events(size=1000, time=4)
    service.step()
    assert sink.messages[-1].value.values.sum() == 1000
    app.publish_events(size=1000, time=6)
    service.step()
    assert sink.messages[-1].value.values.sum() == 2000

    app.publish_config_message(key=config_key, value=model.model_dump())

    app.publish_events(size=100, time=9)
    service.step()
    assert sink.messages[-1].value.values.sum() == 100
    app.publish_events(size=100, time=10)
    service.step()
    assert sink.messages[-1].value.values.sum() == 200


def test_service_can_recover_after_bad_workflow_id_was_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    app = make_reduction_app(instrument='dummy')
    sink = app.sink
    service = app.service
    workflow_id, _spec = _get_workflow_from_registry('dummy', 'total_counts')

    config_key = ConfigKey(
        source_name='panel_0', service_name="data_reduction", key="workflow_config"
    )
    identifier = workflow_spec.WorkflowId(
        instrument='dummy', namespace='data_reduction', name='abcde12345', version=1
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

    valid_workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=valid_workflow_config.model_dump())
    app.publish_events(size=1000, time=5)
    service.step()
    # Service recovered; get data only (no ack without message_id)
    assert len(sink.messages) == 1


def test_service_can_recover_after_bad_workflow_param_was_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    app = make_reduction_app(instrument='dummy')
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry('dummy', 'total_counts')

    config_key = ConfigKey(
        source_name='panel_0', service_name="data_reduction", key="workflow_config"
    )
    bad_param_value = workflow_spec.WorkflowConfig(
        identifier=workflow_id, params={'does_not_exist': 1}
    )
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=bad_param_value.model_dump())

    app.publish_events(size=2000, time=2)
    service.step()
    service.step()
    app.publish_events(size=3000, time=4)
    service.step()
    # No error ack sent when message_id not set (error is logged server-side)
    assert len(sink.messages) == 0

    valid_config = workflow_spec.WorkflowConfig(identifier=workflow_id, params={})
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=valid_config.model_dump())
    app.publish_events(size=1000, time=5)
    service.step()
    # Service recovered; get data only (no ack without message_id)
    assert len(sink.messages) == 1


def test_active_workflow_keeps_running_when_bad_workflow_id_or_params_were_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    app = make_reduction_app(instrument='dummy')
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry('dummy', 'total_counts')

    # Start a valid workflow first
    config_key = ConfigKey(
        source_name=first_source_name['dummy'],
        service_name="data_reduction",
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
    assert len(sink.messages) == 1
    assert sink.messages[0].value.values.sum() == 2000

    # Try to set an invalid workflow ID
    identifier = workflow_spec.WorkflowId(
        instrument='dummy', namespace='data_reduction', name='abcde12345', version=1
    )
    bad_workflow_id = workflow_spec.WorkflowConfig(
        identifier=identifier,  # Invalid workflow ID
        params={},
    )
    app.publish_config_message(key=config_key, value=bad_workflow_id.model_dump())

    # Add more events and verify the original workflow is still running
    app.publish_events(size=3000, time=4)
    service.step()
    # No error ack without message_id, just data messages
    assert len(sink.messages) == 2
    assert sink.messages[1].value.values.sum() == 5000

    # Try to set a workflow with invalid parameters
    bad_param_value = workflow_spec.WorkflowConfig(
        identifier=workflow_id,
        params={'does_not_exist': 1},  # Invalid parameter
    )
    app.publish_config_message(key=config_key, value=bad_param_value.model_dump())

    # Add more events and verify the original workflow is still running
    app.publish_events(size=1000, time=6)
    service.step()
    # No error ack without message_id, just data messages
    assert len(sink.messages) == 3
    assert sink.messages[2].value.values.sum() == 6000


@pytest.mark.parametrize(
    "data_before_config",
    [False, True],
    ids=["config_before_data", "data_before_config"],
)
def test_workflow_starts_with_specific_source_name(
    data_before_config: bool, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    app = make_reduction_app(instrument='dummy')
    sink = app.sink
    service = app.service
    workflow_id, spec = _get_workflow_from_registry('dummy', 'total_counts')
    source_name = spec.source_names[0]

    # This branch ensures that the service configures the workflow even for source names
    # it has not "seen" yet.
    if data_before_config:
        app.publish_events(size=1000, time=0)
        service.step()
        assert len(sink.messages) == 0

    config_key = ConfigKey(
        source_name=source_name, service_name="data_reduction", key="workflow_config"
    )
    workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    # Process config message before data arrives. Without calling step() the order of
    # processing of config vs data messages is not guaranteed.
    service.step()
    sink.messages.clear()  # Clear workflow status message

    app.publish_events(size=2000, time=2)
    service.step()
    assert len(sink.messages) == 1
    # Events before workflow config was published should not be included
    assert sink.messages[0].value.values.sum() == 2000


@pytest.fixture
def configured_dummy_reduction() -> LivedataApp:
    app = make_reduction_app(instrument='dummy')
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry('dummy', 'total_counts')

    config_key = ConfigKey(
        source_name='panel_0', service_name="data_reduction", key="workflow_config"
    )
    workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    # Process config message before data arrives. Without calling step() the order of
    # processing of config vs data messages is not guaranteed.
    service.step()
    sink.messages.clear()  # Clear workflow start message
    return app


@pytest.mark.parametrize('n_msg', [0, 1, 10, 100, 1_234])
@pytest.mark.parametrize('n_event', [0, 1, 10, 100, 1_000, 10_000, 100_000])
def test_fully_consumes_long_chain_of_event_messages(
    n_msg: int,
    n_event: int,
    configured_dummy_reduction: LivedataApp,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    app = configured_dummy_reduction
    sink = app.sink

    for i in range(n_msg):
        app.publish_events(size=n_event, time=i, reuse_events=True)
    n_step = 0
    while n_step < n_msg:
        app.step()
        n_step += 1
        accumulated_counts = sink.messages[-1].value.values.sum()
        if accumulated_counts == n_msg * n_event:
            break
    # Fuzzy limit, depends on how many messages the service can consume in one step.
    # Currently it is configured to consume up to 100.
    assert n_step <= max(1, n_msg // 20)


def test_message_with_unknown_schema_is_ignored(
    configured_dummy_reduction: LivedataApp,
) -> None:
    app = configured_dummy_reduction
    sink = app.sink

    app.publish_events(size=1000, time=0, reuse_events=True)
    # Unknown schema, should be skipped
    app.publish_data(topic=app.detector_topic, time=1, data=b'corrupt data')
    app.publish_events(size=1000, time=1, reuse_events=True)

    with capture_logs() as captured:
        app.step()

    assert len(sink.messages) == 1
    assert sink.messages[0].value.values.sum() == 2000

    # Check log messages for warnings
    warning_logs = [log for log in captured if log['log_level'] == 'warning']
    assert any(
        "has an unknown schema. Skipping." in log['event'] for log in warning_logs
    )


def test_message_that_cannot_be_decoded_is_ignored(
    configured_dummy_reduction: LivedataApp,
) -> None:
    app = configured_dummy_reduction
    sink = app.sink

    app.publish_events(size=1000, time=0, reuse_events=True)
    # Correct schema but invalid data, should be skipped
    app.publish_data(topic=app.detector_topic, time=1, data=b'1234ev44data')
    app.publish_events(size=1000, time=1, reuse_events=True)

    with capture_logs() as captured:
        app.step()

    assert len(sink.messages) == 1
    assert sink.messages[0].value.values.sum() == 2000

    # Check log messages for exceptions
    error_logs = [log for log in captured if log['log_level'] == 'error']
    assert any("Error adapting message" in log['event'] for log in error_logs)
    assert any("unpack_from requires a buffer" in str(log) for log in error_logs)


def test_message_with_bad_ev44_is_ignored(
    configured_dummy_reduction: LivedataApp,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    app = configured_dummy_reduction
    sink = app.sink

    app.publish_events(size=1000, time=0, reuse_events=True)
    bad_events = eventdata_ev44.serialise_ev44(
        source_name='panel_0',
        message_id=0,
        reference_time=[],
        reference_time_index=0,
        time_of_flight=[1, 2],
        pixel_id=[1],  # Invalid, should be the same length as time_of_flight
    )
    app.publish_data(topic=app.detector_topic, time=1, data=bad_events)
    app.publish_events(size=1000, time=1, reuse_events=True)

    # We should have valid data in the *same* batch of messages, but only the bad
    # message should be skipped.
    app.step()
    assert len(sink.messages) == 1
    assert sink.messages[0].value.values.sum() == 2000


def test_message_with_bad_timestamp_is_ignored(
    configured_dummy_reduction: LivedataApp,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    app = configured_dummy_reduction
    sink = app.sink

    app.publish_events(size=1000, time=0, reuse_events=True)
    bad_events = eventdata_ev44.serialise_ev44(
        source_name='panel_0',
        message_id=0,
        reference_time=[],
        reference_time_index=0,
        time_of_flight=[1, 2],
        pixel_id=[1, 2],
    )

    app.publish_data(topic=app.detector_topic, time=np.array([1, 2]), data=bad_events)
    app.publish_events(size=1000, time=1, reuse_events=True)

    # We should have valid data in the *same* batch of messages, but only the bad
    # message should be skipped.
    app.step()
    assert len(sink.messages) == 1
    assert sink.messages[0].value.values.sum() == 2000
