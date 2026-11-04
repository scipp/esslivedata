# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for LOKI ContextBinding declarations."""

import uuid

import pytest

from ess.livedata.config.instruments.loki import factories, specs
from ess.livedata.config.workflow_spec import JobId, WorkflowConfig
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
        and ci.dependent_sources == frozenset({'loki_detector_0'})
    ]
    assert matching, instrument.context_bindings
    assert (
        instrument.chain_patch_path(matching[0])
        == '/entry/instrument/detector_carriage/value'
    )


@pytest.mark.parametrize(
    ('workflow_name', 'gated'),
    [('tube_view', False), ('detector_xy_projection', True), ('i_of_q', True)],
)
def test_job_gates_on_carriage_only_if_it_computes_bank_position(
    workflow_name: str, gated: bool
) -> None:
    """tube_view sums over straw/pixel and never reads the bank's position."""
    instrument = specs.instrument
    factory = instrument.workflow_factory
    workflow_id = next(wid for wid in factory if wid.name == workflow_name)
    job_id = JobId(source_name='loki_detector_0', job_number=uuid.uuid4())
    config = WorkflowConfig(identifier=workflow_id, job_id=job_id)
    job_factory = JobFactory(instrument, service_name=factory.get_service(workflow_id))

    job = job_factory.create(job_id=job_id, config=config)

    assert ('detector_carriage' in job.gating_streams) is gated
