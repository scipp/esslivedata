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
second_motion_source_name = {'dummy': 'motion2'}


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
    # No ack when message_id not set
    assert len(sink.messages) == 0

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


def test_duplicate_f144_messages_do_not_trigger_workflow() -> None:
    """Re-sent f144 values with identical timestamps do not trigger workflow execution.

    Upstream systems may re-send f144 values periodically (e.g., every 10s) even when
    the value has not changed. These duplicates should be dropped at the accumulator
    level and must not propagate through the pipeline as unnecessary workflow
    executions.
    """
    instrument = 'dummy'
    app = make_timeseries_app(instrument)
    sink = app.sink
    service = app.service
    workflow_id, _ = _get_workflow_from_registry(instrument)

    source_name = first_motion_source_name[instrument]
    config_key = ConfigKey(
        source_name=source_name, service_name="timeseries", key="workflow_config"
    )
    workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    service.step()

    # First message — accepted, workflow produces a result
    app.publish_log_message(source_name=source_name, time=1, value=1.5)
    service.step()
    assert len(sink.messages) == 1

    # Re-send same (time, value) — should be silently dropped, no new result
    app.publish_log_message(source_name=source_name, time=1, value=1.5)
    service.step()
    assert len(sink.messages) == 1  # unchanged

    # Multiple re-sends in one batch — still no new result
    app.publish_log_message(source_name=source_name, time=1, value=1.5)
    app.publish_log_message(source_name=source_name, time=1, value=1.5)
    app.publish_log_message(source_name=source_name, time=1, value=1.5)
    service.step()
    assert len(sink.messages) == 1  # unchanged

    # New timestamp — accepted, workflow produces a result
    app.publish_log_message(source_name=source_name, time=2, value=3.0)
    service.step()
    assert len(sink.messages) == 2
    assert sink.messages[-1].value.values.sum() == 3.0


class TestDataBeforeConfig:
    """Tests for the race condition where f144 data arrives before WorkflowConfig.

    When the timeseries service starts, Kafka may deliver f144 data before the
    dashboard's WorkflowConfig arrives. The ToNXlog accumulator consumes this data
    but no jobs exist yet. These tests verify that jobs still receive the data
    via context seeding once they are created.

    The bug only manifests with multiple streams: if stream A has data before
    config but no new data after, and stream B has new data after config, only
    B's job receives data. A's job "hangs" because:
    - The preprocessor only outputs data for streams with messages in the current
      batch (A has none)
    - The duplicate guard in ToNXlog rejects resends (same timestamp)
    - Context seeding was limited to auxiliary streams, not primary streams
    """

    @staticmethod
    def _make_config_key(
        source_name: str,
    ) -> ConfigKey:
        return ConfigKey(
            source_name=source_name,
            service_name="timeseries",
            key="workflow_config",
        )

    @staticmethod
    def _setup() -> tuple:
        instrument = 'dummy'
        app = make_timeseries_app(instrument)
        workflow_id, _ = _get_workflow_from_registry(instrument)
        source_a = first_motion_source_name[instrument]
        source_b = second_motion_source_name[instrument]
        workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
        return app, source_a, source_b, workflow_config

    def test_data_in_accumulator_without_resend(self) -> None:
        """Data for stream A arrives before config. After config, only stream B
        gets new data. Stream A's job should receive historical data via context
        seeding when stream B's data triggers job activation."""
        app, source_a, source_b, workflow_config = self._setup()

        # Data arrives first for both streams — accumulators store it, no jobs
        app.publish_log_message(source_name=source_a, time=1, value=42.0)
        app.publish_log_message(source_name=source_b, time=1, value=10.0)
        app.service.step()
        assert len(app.sink.messages) == 0

        # Config arrives for both sources — jobs scheduled
        app.publish_config_message(
            key=self._make_config_key(source_a),
            value=workflow_config.model_dump(),
        )
        app.publish_config_message(
            key=self._make_config_key(source_b),
            value=workflow_config.model_dump(),
        )
        app.service.step()

        # Only stream B gets new data — this triggers activation of ALL
        # scheduled jobs. Context seeding should provide stream A's historical
        # data to its job.
        app.publish_log_message(source_name=source_b, time=2, value=11.0)
        app.service.step()

        # Both jobs should have produced results
        result_streams = {msg.stream.name for msg in app.sink.messages}
        assert source_a in str(result_streams), (
            f"Job for {source_a} should have received data via context seeding"
        )

    def test_data_in_accumulator_with_resend(self) -> None:
        """Data for stream A arrives before config. After config, only a resend
        (same timestamp) arrives for A plus new data for B. The resend is
        rejected by ToNXlog's duplicate guard, but A's job should still
        receive historical data via context seeding."""
        app, source_a, source_b, workflow_config = self._setup()

        # Data for stream A arrives first
        app.publish_log_message(source_name=source_a, time=1, value=42.0)
        app.service.step()
        assert len(app.sink.messages) == 0

        # Config for both sources
        app.publish_config_message(
            key=self._make_config_key(source_a),
            value=workflow_config.model_dump(),
        )
        app.publish_config_message(
            key=self._make_config_key(source_b),
            value=workflow_config.model_dump(),
        )
        app.service.step()

        # Stream A: resend (same timestamp, rejected by accumulator)
        # Stream B: new data (triggers job activation)
        app.publish_log_message(source_name=source_a, time=1, value=42.0)
        app.publish_log_message(source_name=source_b, time=2, value=11.0)
        app.service.step()

        # Both jobs should have produced results
        result_streams = {msg.stream.name for msg in app.sink.messages}
        assert source_a in str(result_streams), (
            f"Job for {source_a} should have received data via context seeding "
            "even though resend was rejected"
        )

    def test_data_not_in_accumulator_with_resend(self) -> None:
        """Data was never consumed by accumulator. Resend with old timestamp
        arrives after job is created — accumulator accepts it normally."""
        app, source_a, _, workflow_config = self._setup()

        # Config arrives first — job is created, no data yet
        app.publish_config_message(
            key=self._make_config_key(source_a),
            value=workflow_config.model_dump(),
        )
        app.service.step()
        assert len(app.sink.messages) == 0

        # "Resend" arrives — accumulator has never seen it, accepted normally
        app.publish_log_message(source_name=source_a, time=1, value=42.0)
        app.service.step()
        assert len(app.sink.messages) == 1
        result = app.sink.messages[-1].value
        assert result.sizes == {'time': 1}
        assert result.values[0] == 42.0
