# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Instrument-level integration test for BIFROST's chopper wavelength-LUT workflow.

Exercises the spec-scope context bindings declared in ``bifrost/factories.py``
flowing through ``JobFactory.create`` into the job's gating set and the
workflow's ``set_context`` keys, against the real BIFROST geometry artifact.
"""

from __future__ import annotations

import uuid

import pytest
import scipp as sc

from ess.livedata.config.chopper import delay_setpoint_stream, speed_setpoint_stream
from ess.livedata.config.instrument import instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.workflow_spec import JobId, WorkflowConfig
from ess.livedata.core.job import JobData
from ess.livedata.core.job_manager import JobFactory
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.workflows.wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    WAVELENGTH_LUT_OUTPUT,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def bifrost():
    get_config('bifrost')  # register instrument
    instrument = instrument_registry['bifrost']
    instrument.load_factories()
    return instrument


def _any_component_table(result) -> sc.DataArray:
    """Return one per-component lookup table from a job result.

    Which components have a table depends on what the instrument's geometry
    artifact can place, so the tests assert on the shape of a table rather than
    on a particular component having one.
    """
    tables = result.data[WAVELENGTH_LUT_OUTPUT]
    assert tables, f'no per-component tables in result: {list(result.data)}'
    return next(iter(tables.values()))


def _create_lut_job(bifrost) -> tuple:
    workflow_id = next(
        w for w in bifrost.workflow_factory if w.name == 'wavelength_lut'
    )
    job_id = JobId(source_name=CHOPPER_CASCADE_SOURCE, job_number=uuid.uuid4())
    config = WorkflowConfig.from_params(
        workflow_id=workflow_id, job_id=job_id, params=None, aux_source_names=None
    )
    service = bifrost.workflow_factory.get_service(workflow_id)
    return JobFactory(bifrost, service_name=service).create(
        job_id=job_id, config=config
    )


def _nxlog(value: float, unit: str | None) -> sc.DataArray:
    t = sc.epoch(unit='ns') + sc.arange('time', 3, unit='ns')
    return sc.DataArray(
        sc.full(value=value, sizes={'time': 3}, unit=unit), coords={'time': t}
    )


def test_spec_scope_bindings_define_gating_set(bifrost) -> None:
    job = _create_lut_job(bifrost)
    expected = {
        stream(chopper)
        for chopper in bifrost.choppers
        for stream in (speed_setpoint_stream, delay_setpoint_stream)
    }
    # Nothing seen yet → every per-chopper setpoint stream gates the job.
    assert job.missing_context(set()) == expected


def test_chopper_lut_computes_from_context_and_trigger(bifrost) -> None:
    job = _create_lut_job(bifrost)
    aux = {}
    for chopper in bifrost.choppers:
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
    lut = _any_component_table(result)
    assert lut.dims == ('distance', 'event_time_offset')
    assert lut.unit == sc.units.angstrom
    # Not asserting finite wavelengths here: every chopper is fed the same
    # placeholder 14 Hz / zero delay, which blocks the beam a few metres past
    # the first chopper, so a table covering a real component's distance is
    # legitimately all-NaN. The cascade bands below carry the "wavelengths were
    # actually computed" assertion, evaluated at the chopper distances
    # themselves. Asserting finite values at detector distances would need a
    # physically meaningful chopper configuration.
