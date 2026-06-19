# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Instrument-level integration test for chopper wavelength-LUT workflows.

Exercises the full path the unit tests cannot, for every instrument that
declares ``choppers``: the spec-scope ``SpecHandle.add_context_binding``\\ s
(ADR 0003) auto-wired in ``Instrument.load_factories`` flow through
``JobFactory.create`` into the job's gating set and the workflow's
``set_context`` keys, against the real geometry artifact (which must carry the
``NXdisk_chopper`` groups with resolvable positions).
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
import scipp as sc

from ess.livedata.config.chopper import (
    delay_setpoint_stream,
    speed_setpoint_stream,
)
from ess.livedata.config.instrument import instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.workflow_spec import JobId, WorkflowConfig
from ess.livedata.core.job import JobData
from ess.livedata.core.job_manager import JobFactory
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.handlers.wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    WAVELENGTH_BANDS_OUTPUT,
    WAVELENGTH_LUT_OUTPUT,
    CascadeBands,
    WavelengthLutParams,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module', params=['loki', 'dream'])
def instrument(request):
    get_config(request.param)  # register instrument
    inst = instrument_registry[request.param]
    inst.load_factories()
    return inst


def _create_lut_job(instrument, params=None) -> tuple:
    workflow_id = next(
        w for w in instrument.workflow_factory if w.name == WAVELENGTH_LUT_OUTPUT
    )
    job_id = JobId(source_name=CHOPPER_CASCADE_SOURCE, job_number=uuid.uuid4())
    config = WorkflowConfig.from_params(
        workflow_id=workflow_id, job_id=job_id, params=params, aux_source_names=None
    )
    service = instrument.workflow_factory.get_service(workflow_id)
    return JobFactory(instrument, service_name=service).create(
        job_id=job_id, config=config
    )


def _nxlog(value: float, unit: str | None) -> sc.DataArray:
    t = sc.epoch(unit='ns') + sc.arange('time', 3, unit='ns')
    return sc.DataArray(
        sc.full(value=value, sizes={'time': 3}, unit=unit), coords={'time': t}
    )


def test_spec_scope_bindings_define_gating_set(instrument) -> None:
    job = _create_lut_job(instrument)
    expected = {
        stream(chopper)
        for chopper in instrument.choppers
        for stream in (speed_setpoint_stream, delay_setpoint_stream)
    }
    # Nothing seen yet → every per-chopper setpoint stream gates the job.
    assert job.missing_context(set()) == expected


def test_chopper_lut_computes_from_context_and_trigger(instrument) -> None:
    params = WavelengthLutParams(
        cascade_bands=CascadeBands(distances='7.0, 12.0')
    ).model_dump()
    job = _create_lut_job(instrument, params=params)
    aux = {}
    for chopper in instrument.choppers:
        speed_unit = instrument.streams[speed_setpoint_stream(chopper)].units
        delay_unit = instrument.streams[delay_setpoint_stream(chopper)].units
        aux[speed_setpoint_stream(chopper)] = _nxlog(14.0, speed_unit)
        aux[delay_setpoint_stream(chopper)] = _nxlog(0.0, delay_unit)
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

    bands = result.data[WAVELENGTH_BANDS_OUTPUT]
    assert bands.dims == ('distance', 'event_time_offset')
    assert bands.unit == sc.units.angstrom
    assert np.isfinite(bands.values).any()
    distances = bands.coords['distance']
    # Source + one row per chopper + one row per configured cut distance.
    assert distances.sizes['distance'] == len(instrument.choppers) + 1 + 2
    # Rows are ordered by ascending distance.
    assert sc.allsorted(distances, 'distance')
