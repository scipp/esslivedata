# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Instrument-level integration test for LOKI's chopper wavelength-LUT workflow.

This is the first production user of spec-scope ``SpecHandle.add_context_binding``
(ADR 0003). It exercises the full path the unit tests cannot: the spec-scope
bindings declared in ``loki/factories.py`` flowing through ``JobFactory.create``
into the job's gating set and the workflow's ``set_context`` keys, against the
real LOKI geometry artifact.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
import scipp as sc

from ess.livedata.config.chopper import delay_setpoint_stream, speed_setpoint_stream
from ess.livedata.config.instrument import instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.workflow_spec import JobId, WorkflowConfig
from ess.livedata.core.job import JobData
from ess.livedata.core.job_manager import JobFactory
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.handlers.wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    WAVELENGTH_LUT_OUTPUT,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def loki():
    get_config('loki')  # register instrument
    instrument = instrument_registry['loki']
    instrument.load_factories()
    return instrument


def _create_lut_job(loki) -> tuple:
    workflow_id = next(
        w for w in loki.workflow_factory if w.name == WAVELENGTH_LUT_OUTPUT
    )
    job_id = JobId(source_name=CHOPPER_CASCADE_SOURCE, job_number=uuid.uuid4())
    config = WorkflowConfig.from_params(
        workflow_id=workflow_id, job_id=job_id, params=None, aux_source_names=None
    )
    service = loki.workflow_factory.get_service(workflow_id)
    return JobFactory(loki, service_name=service).create(job_id=job_id, config=config)


def _nxlog(value: float, unit: str | None) -> sc.DataArray:
    t = sc.epoch(unit='ns') + sc.arange('time', 3, unit='ns')
    return sc.DataArray(
        sc.full(value=value, sizes={'time': 3}, unit=unit), coords={'time': t}
    )


def test_spec_scope_bindings_define_gating_set(loki) -> None:
    job = _create_lut_job(loki)
    expected = {
        stream(chopper)
        for chopper in loki.choppers
        for stream in (speed_setpoint_stream, delay_setpoint_stream)
    }
    # Nothing seen yet → every per-chopper setpoint stream gates the job.
    assert job.missing_context(set()) == expected


def test_chopper_lut_computes_from_context_and_trigger(loki) -> None:
    job = _create_lut_job(loki)
    aux = {}
    for chopper in loki.choppers:
        aux[speed_setpoint_stream(chopper)] = _nxlog(14.0, 'Hz')
        aux[delay_setpoint_stream(chopper)] = _nxlog(0.0, 'ns')
    data = JobData(
        start_time=Timestamp.from_ns(0),
        end_time=Timestamp.from_ns(1),
        primary_data={CHOPPER_CASCADE_SOURCE: _nxlog(1.0, None)},
        aux_data=aux,
    )

    reply, result = job.process(data, finalize=True)

    assert not reply.has_error, reply.error_message
    assert result is not None
    assert result.error_message is None, result.error_message
    lut = result.data[WAVELENGTH_LUT_OUTPUT]
    assert lut.dims == ('distance', 'event_time_offset')
    assert lut.unit == sc.units.angstrom
    assert np.isfinite(lut.values).any()
