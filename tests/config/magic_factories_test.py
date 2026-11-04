# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Tests for MAGIC chain-patch gating."""

import uuid

import pytest

from ess.livedata.config.instruments.magic import specs
from ess.livedata.config.workflow_spec import JobId, WorkflowConfig
from ess.livedata.core.job_manager import JobFactory


@pytest.fixture(scope="module", autouse=True)
def _magic_factories_loaded() -> None:
    specs.instrument.load_factories()


@pytest.mark.parametrize(
    ('workflow_name', 'gated'),
    [('wire_view', False), ('strip_view', False), ('detector_projection', True)],
)
def test_job_gates_on_rotation_only_if_it_computes_bank_position(
    workflow_name: str, gated: bool
) -> None:
    """The wire and strip views fold ``detector_number`` and read no position."""
    instrument = specs.instrument
    factory = instrument.workflow_factory
    workflow_id = next(wid for wid in factory if wid.name == workflow_name)
    job_id = JobId(source_name='magic_detector_a', job_number=uuid.uuid4())
    config = WorkflowConfig(identifier=workflow_id, job_id=job_id)
    job_factory = JobFactory(instrument, service_name=factory.get_service(workflow_id))

    job = job_factory.create(job_id=job_id, config=config)

    assert ('detector_a_rotation' in job.gating_streams) is gated
