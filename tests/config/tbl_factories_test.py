# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for TBL workflow factories."""

import pytest
import scipp as sc

from ess.livedata.config.instruments.tbl import factories, specs
from ess.livedata.core.timestamp import Timestamp


@pytest.fixture(scope="module", autouse=True)
def _tbl_factories_loaded() -> None:
    factories.setup_factories(specs.instrument)


def test_orca_view_processes_a_frame() -> None:
    """The ORCA view must survive a frame, not just construction.

    ``fold_image`` is shared with the ``add_logical_view`` registration path, so
    the two paths must agree on the transform's call convention. When they did
    not, every batch raised ``TypeError`` inside ``Job.add``, which turns the
    traceback into a job error message rather than a crash -- so the view was
    silently dead while still paying the full ingestion cost.
    """
    registration = specs.instrument.workflow_factory.registration(
        specs.orca_view_handle.workflow_id
    )
    workflow = registration.factory(source_name='orca_detector')
    frame = sc.DataArray(sc.ones(dims=['dim_0', 'dim_1'], shape=[2048, 2048]))

    workflow.accumulate(
        {'orca_detector': frame},
        start_time=Timestamp.from_ns(1000),
        end_time=Timestamp.from_ns(2000),
    )
    result = workflow.finalize()

    assert result['cumulative'].shape == (512, 512)
    assert result['current'].shape == (512, 512)
