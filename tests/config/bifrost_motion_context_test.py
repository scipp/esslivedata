# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""BIFROST's rotation streams gate exactly the specs whose graphs use them.

The tank and sample rotations are offered instrument-wide
(``Instrument.offer_context_stream``), so which jobs wait on them is derived
from the graph each job builds rather than declared per spec. These tests pin
that derivation against the real BIFROST workflows: the Q-maps read the angles,
the detector view (which sums over banks) and the ratemeter (counts only) do
not.
"""

from __future__ import annotations

import uuid

import pytest

from ess.livedata.config.instrument import instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.workflow_spec import JobId, WorkflowConfig
from ess.livedata.core.job_manager import JobFactory

pytestmark = pytest.mark.slow

ROTATION_STREAMS = {'detector_tank_angle_r0', 'rotation_stage'}


@pytest.fixture(scope='module')
def bifrost():
    get_config('bifrost')  # register instrument
    instrument = instrument_registry['bifrost']
    instrument.load_factories()
    return instrument


def _gating_streams(bifrost, spec_name: str) -> set[str]:
    workflow_id = next(w for w in bifrost.workflow_factory if w.name == spec_name)
    spec = bifrost.workflow_factory[workflow_id]
    (source_name,) = spec.source_names
    job_id = JobId(source_name=source_name, job_number=uuid.uuid4())
    config = WorkflowConfig.from_params(
        workflow_id=workflow_id,
        job_id=job_id,
        params=None,
        aux_source_names=(
            spec.aux_sources.get_defaults() if spec.aux_sources is not None else None
        ),
    )
    service = bifrost.workflow_factory.get_service(workflow_id)
    job = JobFactory(bifrost, service_name=service).create(job_id=job_id, config=config)
    return job.missing_context(set())


@pytest.mark.parametrize('spec_name', ['qmap', 'elastic_qmap', 'elastic_qmap_custom'])
def test_qmap_specs_gate_on_both_rotations(bifrost, spec_name: str) -> None:
    assert ROTATION_STREAMS <= _gating_streams(bifrost, spec_name)


@pytest.mark.parametrize('spec_name', ['unified_detector_view', 'detector_ratemeter'])
def test_rotation_independent_specs_do_not_gate(bifrost, spec_name: str) -> None:
    """Without this, a fallback view would wait forever on a motionless axis."""
    assert not ROTATION_STREAMS & _gating_streams(bifrost, spec_name)
