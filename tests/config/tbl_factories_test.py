# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for TBL workflow factories."""

import pytest
import scipp as sc

from ess.livedata.config.instruments.tbl import factories, specs
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.preprocessors.frame_batch import FrameBatchAccumulator


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


def test_orca_transmission_is_gated_on_the_image_key_stream() -> None:
    workflow_id = specs.orca_transmission_handle.workflow_id

    context_keys = specs.instrument.bound_context_keys(workflow_id, 'orca_detector')

    assert list(context_keys) == ['orca_detector/image_key']
    assert 'orca_detector/image_key' in specs.instrument.streams


def test_orca_transmission_processes_frames() -> None:
    workflow_id = specs.orca_transmission_handle.workflow_id
    workflow = specs.instrument.workflow_factory.registration(workflow_id).factory(
        source_name='orca_detector'
    )
    workflow.build(
        context_keys=specs.instrument.bound_context_keys(workflow_id, 'orca_detector')
    )
    key_log = sc.DataArray(
        sc.array(dims=['time'], values=[2, 1, 0], unit='dimensionless'),
        coords={
            'time': sc.epoch(unit='ns')
            + sc.array(dims=['time'], values=[0, 100, 200], unit='ns', dtype='int64')
        },
    )
    frames = FrameBatchAccumulator()
    for time_ns, value in [(10, 1.0), (110, 5.0), (210, 3.0)]:
        frames.add(
            Timestamp.from_ns(time_ns),
            sc.DataArray(
                sc.full(dims=['dim_0', 'dim_1'], shape=[512, 512], value=value)
            ),
        )

    workflow.accumulate(
        {'orca_detector': frames.get(), 'orca_detector/image_key': key_log},
        start_time=Timestamp.from_ns(0),
        end_time=Timestamp.from_ns(1),
    )
    result = workflow.finalize()

    assert result['transmission'].shape == (512, 512)
    assert result['transmission'].values[0, 0] == pytest.approx(0.5)
