# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Test for the timeseries data service.

Note that this uses mostly the same logic as the data reduction service, so the tests
are similar to those in `tests/services/data_reduction_test.py`. Many tests are not
duplicated here.
"""

import logging

import pytest

from ess.livedata.config import instrument_registry, workflow_spec
from ess.livedata.config.models import ConfigKey
from ess.livedata.services.timeseries import make_timeseries_service_builder
from tests.helpers.livedata_app import LivedataApp


def _get_workflow_from_registry(
    instrument: str,
) -> tuple[workflow_spec.WorkflowId, workflow_spec.WorkflowSpec]:
    # Assume we can just use the first registered workflow.
    namespace = 'timeseries'
    instrument_config = instrument_registry[instrument]
    workflow_registry = instrument_config.workflow_factory
    for wid, spec in workflow_registry.items():
        if spec.namespace == namespace:
            return wid, spec
    raise ValueError(f"Namespace {namespace} not found in specs")


def make_timeseries_app(instrument: str) -> LivedataApp:
    builder = make_timeseries_service_builder(instrument=instrument)
    return LivedataApp.from_service_builder(builder, use_naive_message_batcher=False)


first_motion_source_name = {'dummy': 'motion1'}


@pytest.mark.parametrize("instrument", ['dummy'])
def test_updates_are_published_immediately(
    instrument: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    builder = make_timeseries_service_builder(instrument=instrument)
    # We set use_naive_message_batcher to False here (which would otherwise be used by
    # the test helper LivedataApp), as we want to test that the actual message batcher
    # behaves as expected.
    app = LivedataApp.from_service_builder(builder, use_naive_message_batcher=False)
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry(instrument)

    source_name = first_motion_source_name[instrument]
    config_key = ConfigKey(
        source_name=source_name, service_name="timeseries", key="workflow_config"
    )
    workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    # Trigger workflow start
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    service.step()
    assert len(sink.messages) == 1  # Workflow status message
    sink.messages.clear()

    app.publish_log_message(source_name=source_name, time=1, value=1.5)
    service.step()
    # Each workflow call returns only new data since last finalize (delta)
    assert len(sink.messages) == 1
    assert sink.messages[-1].value.values.sum() == 1.5
    # No data -> no data published
    service.step()
    assert len(sink.messages) == 1

    # Just a tiny bit later, but batcher does not delay processing this.
    app.publish_log_message(source_name=source_name, time=1.0001, value=0.5)
    service.step()
    assert len(sink.messages) == 2
    # Expect only the new data point (delta), not cumulative
    assert sink.messages[-1].value.values.sum() == 0.5
