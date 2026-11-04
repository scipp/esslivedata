# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for LOKI ContextBinding declarations."""

import uuid

import pytest

from ess.livedata.config.instruments.loki import factories, specs
from ess.livedata.config.workflow_spec import JobId, WorkflowConfig
from ess.livedata.core.job import Job
from ess.livedata.core.job_manager import JobFactory


@pytest.fixture(scope="module", autouse=True)
def _loki_factories_loaded() -> None:
    """Load LOKI factories before the tests run.

    Without this the tests depend on a sibling module having loaded the
    factories first (flaky under ``-n auto``). ``load_factories`` is
    idempotent.
    """
    specs.instrument.load_factories()


def test_setup_factories_declares_detector_carriage_context_binding() -> None:
    """The rear bank's f144 carriage stream is declared at instrument scope.

    Declared as a chain-patch :class:`ContextBinding` on ``loki_detector_0`` only
    (``workflow_key`` is :class:`factories.DetectorCarriageLog`, a :class:`ValueLog`
    subclass).
    """
    instrument = specs.instrument

    matching = [
        ci
        for ci in instrument.context_bindings
        if ci.stream_name == 'detector_carriage'
        and ci.workflow_key is factories.DetectorCarriageLog
        and ci.dependent_sources == frozenset({'loki_detector_0', 'beam_monitor_m4'})
    ]
    assert matching, instrument.context_bindings
    assert (
        instrument.chain_patch_path(matching[0])
        == '/entry/instrument/detector_carriage/value'
    )


def _create_job(workflow_name: str, params: dict | None = None) -> Job:
    instrument = specs.instrument
    factory = instrument.workflow_factory
    workflow_id = next(wid for wid in factory if wid.name == workflow_name)
    job_id = JobId(source_name='loki_detector_0', job_number=uuid.uuid4())
    config = WorkflowConfig(identifier=workflow_id, job_id=job_id, params=params or {})
    job_factory = JobFactory(instrument, service_name=factory.get_service(workflow_id))
    return job_factory.create(job_id=job_id, config=config)


_NO_MONITOR_TABLE = pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="#1441: the default I(Q) monitors have no wavelength-table block",
)


@pytest.mark.parametrize(
    ('workflow_name', 'gated'),
    [
        ('tube_view', False),
        ('detector_xy_projection', True),
        pytest.param('i_of_q', True, marks=_NO_MONITOR_TABLE),
    ],
)
def test_job_gates_on_carriage_only_if_it_computes_bank_position(
    workflow_name: str, gated: bool
) -> None:
    """tube_view sums over straw/pixel and never reads the bank's position."""
    job = _create_job(workflow_name)

    assert ('detector_carriage' in job.gating_streams) is gated


@_NO_MONITOR_TABLE
@pytest.mark.parametrize('mode', ['constant', 'current_run'])
def test_i_of_q_reads_transmission_monitor_position_only_in_current_run_mode(
    mode: str,
) -> None:
    """Whether a job reads an aux monitor's chain depends on its params."""
    job = _create_job('i_of_q', {'transmission': {'mode': mode}})

    assert {'beam_monitor_m2_positioner', 'sample_stack_z'} <= job.gating_streams
    assert ('beam_monitor_m3_positioner' in job.gating_streams) is (
        mode == 'current_run'
    )
