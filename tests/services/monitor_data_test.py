# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Test for the monitor data service.

Note that this uses mostly the same logic as the data reduction service, so the tests
are similar to those in `tests/services/data_reduction_test.py`. Many tests are not
duplicated here.
"""

import json
import logging
import uuid

import pytest

from ess.livedata.config import instrument_registry, workflow_spec
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.services.monitor_data import make_monitor_service_builder
from tests.helpers.livedata_app import LivedataApp


def _job_id(source: str) -> JobId:
    return JobId(source_name=source, job_number=uuid.uuid4())


def _get_workflow_from_registry(
    instrument: str,
) -> tuple[str, workflow_spec.WorkflowSpec]:
    # Currently only one workflow for monitor data, so we can hardcode the name.
    namespace = 'monitor_data'
    instrument_config = instrument_registry[instrument]
    workflow_registry = instrument_config.workflow_factory
    for wid, spec in workflow_registry.items():
        if spec.group.name == namespace:
            return wid, spec
    raise ValueError(f"Namespace {namespace} not found in specs")


def make_monitor_app(instrument: str) -> LivedataApp:
    builder = make_monitor_service_builder(instrument=instrument)
    return LivedataApp.from_service_builder(builder)


first_monitor_source_name = {
    'dummy': 'monitor1',
    'dream': 'monitor_bunker',
    'bifrost': 'psc_monitor',
    'loki': 'beam_monitor_m1',
}


@pytest.mark.parametrize("instrument", ['bifrost', 'dummy', 'dream', 'loki'])
def test_can_configure_and_stop_monitor_workflow(
    instrument: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    app = make_monitor_app(instrument)
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry(instrument)

    source_name = first_monitor_source_name[instrument]
    workflow_config = workflow_spec.WorkflowConfig(
        identifier=workflow_id, job_id=_job_id(source_name)
    )
    # Trigger workflow start
    app.publish_config_message(workflow_config)
    service.step()
    # Config ack lands on response_messages, not the data sink
    assert len(sink.messages) == 0

    # Each workflow call returns six results: the cumulative/current histograms
    # plus per-update and cumulative scalar totals (counts_total[_cumulative],
    # counts_in_toa_range[_cumulative]).
    n_target = 6

    def latest_outputs() -> dict[str, object]:
        outputs: dict[str, object] = {}
        for msg in sink.messages[-n_target:]:
            result_key = json.loads(msg.stream.name)
            outputs[result_key['output_name']] = msg.value
        return outputs

    app.publish_monitor_events(size=2000, time=2)
    service.step()
    assert len(sink.messages) == n_target
    outputs = latest_outputs()
    assert outputs['cumulative'].values.sum() == 2000
    assert outputs['current'].values.sum() == 2000
    assert outputs['counts_total'].value == 2000
    assert outputs['counts_total_cumulative'].value == 2000
    # No data -> no data published
    service.step()
    assert len(sink.messages) == n_target

    app.publish_monitor_events(size=3000, time=4)
    service.step()
    assert len(sink.messages) == 2 * n_target
    outputs = latest_outputs()
    assert outputs['cumulative'].values.sum() == 5000
    assert outputs['current'].values.sum() == 3000
    assert outputs['counts_total'].value == 3000  # per-update resets
    assert outputs['counts_total_cumulative'].value == 5000  # keeps accumulating

    # More events but the same time
    app.publish_monitor_events(size=1000, time=4)
    # Later time
    app.publish_monitor_events(size=1000, time=5)
    service.step()
    assert len(sink.messages) == 3 * n_target
    assert latest_outputs()['cumulative'].values.sum() == 7000

    # Stop workflow
    command = JobCommand(action=JobAction.stop)
    app.publish_config_message(command)
    app.publish_monitor_events(size=1000, time=10)
    service.step()
    app.publish_monitor_events(size=1000, time=20)
    service.step()
    assert len(sink.messages) == 3 * n_target
