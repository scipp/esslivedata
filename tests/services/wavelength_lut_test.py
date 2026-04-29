# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for the chopperless wavelength lookup-table service."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import scipp as sc

from ess.livedata.config import instrument_registry, workflow_spec
from ess.livedata.config.models import ConfigKey
from ess.livedata.handlers.wavelength_lut_workflow_specs import CHOPPER_CASCADE_SOURCE
from ess.livedata.services.wavelength_lut import make_wavelength_lut_service_builder
from tests.helpers.livedata_app import LivedataApp


def _get_workflow_id(instrument: str, name: str) -> workflow_spec.WorkflowId:
    cfg = instrument_registry[instrument]
    for wid, spec in cfg.workflow_factory.items():
        if spec.namespace == 'wavelength_lut' and spec.name == name:
            return wid
    raise AssertionError(f"workflow {name!r} not registered for {instrument!r}")


@pytest.fixture
def app(caplog: pytest.LogCaptureFixture) -> LivedataApp:
    caplog.set_level(logging.INFO)
    builder = make_wavelength_lut_service_builder(instrument='dummy')
    return LivedataApp.from_service_builder(builder)


def _config_message() -> tuple[ConfigKey, dict]:
    workflow_id = _get_workflow_id('dummy', 'wavelength_lut')
    config_key = ConfigKey(
        source_name=CHOPPER_CASCADE_SOURCE,
        service_name='wavelength_lut',
        key='workflow_config',
    )
    params = {'simulation': {'num_simulated_neutrons': 50_000}}
    config = workflow_spec.WorkflowConfig(identifier=workflow_id, params=params)
    return config_key, config.model_dump()


def test_chopperless_workflow_publishes_wavelength_lut_after_job_start(
    app: LivedataApp,
) -> None:
    """End-to-end: scheduling a job triggers exactly one wavelength-lut publish.

    Realistic ordering: the synthesizer emits its 'chopper_cascade' tick on
    the *first* poll, when no job is yet scheduled. The tick is cached via
    ToNXlog (context accumulator). The operator publishes a workflow config
    *later*, in a poll with no further data. The orchestrator's empty-batch
    activation path then replays the cached tick to activate the job and
    fire the workflow once.
    """
    # First step: synthesizer's first get_messages emits the tick.
    # No job is scheduled yet, so nothing should be published.
    app.service.step()
    assert all(not isinstance(m.value, sc.DataArray) for m in app.sink.messages)

    # Second step: operator schedules the job. data_messages is empty, so
    # the activation must come from the empty-batch context-replay path.
    key, value = _config_message()
    app.publish_config_message(key=key, value=value)
    app.service.step()

    # Wavelength-lut message should now be on the sink.
    data_messages = [m for m in app.sink.messages if isinstance(m.value, sc.DataArray)]
    assert len(data_messages) == 1
    table = data_messages[0].value
    assert table.dims == ('distance', 'event_time_offset')
    assert table.unit == sc.units.angstrom
    assert np.isfinite(table.values).any()
    for name in (
        'pulse_period',
        'pulse_stride',
        'distance_resolution',
        'time_resolution',
    ):
        assert name in table.coords


def test_subsequent_steps_do_not_republish(app: LivedataApp) -> None:
    """After the workflow has fired once, additional steps with no new
    primary input must not trigger a re-publish.
    """
    app.service.step()  # Synthesizer's tick gets cached.
    key, value = _config_message()
    app.publish_config_message(key=key, value=value)
    app.service.step()  # Job activates and fires.
    initial_count = sum(
        1 for m in app.sink.messages if isinstance(m.value, sc.DataArray)
    )
    assert initial_count == 1

    for _ in range(3):
        app.service.step()

    final_count = sum(1 for m in app.sink.messages if isinstance(m.value, sc.DataArray))
    assert final_count == 1
