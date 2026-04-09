# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the backend shutdown lifecycle.

Verifies that the shutdown sequence (shutdown → thread join → report_stopped)
sends correct job and service status heartbeats at each stage.
"""

from ess.livedata.config import instrument_registry, workflow_spec
from ess.livedata.config.models import ConfigKey
from ess.livedata.core.job import JobState, JobStatus, ServiceState, ServiceStatus
from ess.livedata.services.detector_data import make_detector_service_builder
from tests.helpers.livedata_app import LivedataApp


def _make_detector_app() -> LivedataApp:
    builder = make_detector_service_builder(instrument='dummy')
    return LivedataApp.from_service_builder(builder)


def _get_workflow_id() -> workflow_spec.WorkflowId:
    instrument_config = instrument_registry['dummy']
    for wid, spec in instrument_config.workflow_factory.items():
        if spec.namespace == 'detector_data':
            return wid
    raise ValueError("detector_data workflow not found")


def _start_job(app: LivedataApp) -> None:
    """Configure a workflow job and process some data so the job becomes active."""
    workflow_id = _get_workflow_id()
    config_key = ConfigKey(
        source_name='panel_0', service_name='detector_data', key='workflow_config'
    )
    workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    app.step()
    app.publish_events(size=1000, time=2)
    app.step()


def _extract_statuses(
    app: LivedataApp,
) -> tuple[list[JobStatus], list[ServiceStatus]]:
    job_statuses = [
        msg.value
        for msg in app.sink.status_messages
        if isinstance(msg.value, JobStatus)
    ]
    service_statuses = [
        msg.value
        for msg in app.sink.status_messages
        if isinstance(msg.value, ServiceStatus)
    ]
    return job_statuses, service_statuses


def test_shutdown_does_not_mark_jobs_stopped():
    """shutdown() must not mark jobs as stopped (worker loop still running)."""
    app = _make_detector_app()
    _start_job(app)

    processor = app.service._processor
    app.sink.status_messages.clear()

    processor.shutdown()

    job_statuses, service_statuses = _extract_statuses(app)
    # shutdown() sends only a service heartbeat, no job statuses
    assert len(job_statuses) == 0
    assert len(service_statuses) == 1
    assert service_statuses[0].state == ServiceState.stopping


def test_report_stopped_marks_jobs_stopped():
    """report_stopped() marks all jobs as stopped and sends the final heartbeat."""
    app = _make_detector_app()
    _start_job(app)

    processor = app.service._processor
    processor.shutdown()
    app.sink.status_messages.clear()

    processor.report_stopped()

    job_statuses, service_statuses = _extract_statuses(app)
    assert len(job_statuses) >= 1
    assert all(s.state == JobState.stopped for s in job_statuses)
    assert len(service_statuses) == 1
    assert service_statuses[0].state == ServiceState.stopped


def test_report_error_marks_jobs_stopped():
    """report_error() marks all jobs as stopped and sends the final heartbeat."""
    app = _make_detector_app()
    _start_job(app)

    processor = app.service._processor
    app.sink.status_messages.clear()

    processor.report_error("fatal crash")

    job_statuses, service_statuses = _extract_statuses(app)
    assert len(job_statuses) >= 1
    assert all(s.state == JobState.stopped for s in job_statuses)
    assert len(service_statuses) == 1
    assert service_statuses[0].state == ServiceState.error
