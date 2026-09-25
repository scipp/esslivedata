# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for DREAM workflow factories."""

import uuid

import numpy as np
import pytest
import scipp as sc

from ess.livedata.config.instruments.dream import specs
from ess.livedata.config.workflow_spec import JobId, WorkflowConfig
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.workflows.workflow_factory import SpecHandle


@pytest.fixture(scope="module", autouse=True)
def _dream_factories_loaded() -> None:
    specs.instrument.load_factories()


def _events(source_name: str, size: int) -> sc.DataArray:
    """Events on random pixels of a bank, grouped by pixel."""
    rng = np.random.default_rng(seed=1)
    detector_number = specs.instrument.get_detector_number(source_name)
    ids = rng.choice(detector_number.values.ravel(), size=size)
    events = sc.DataArray(
        sc.ones(dims=['event'], shape=[size], unit='counts'),
        coords={
            'detector_number': sc.array(
                dims=['event'], values=ids, unit=detector_number.unit
            ),
            'event_time_offset': sc.array(
                dims=['event'], values=rng.uniform(0, 7e7, size), unit='ns'
            ),
        },
    )
    return events.group(detector_number.flatten(to='detector_number'))


@pytest.mark.parametrize(
    ('handle', 'source_name', 'sizes'),
    [
        (specs.wire_view_handle, 'mantle_detector', (60, 32)),
        (specs.wire_view_handle, 'endcap_backward_detector', (16, 616)),
        (specs.wire_view_handle, 'endcap_forward_detector', (16, 280)),
        (specs.wire_view_handle, 'high_resolution_detector', (16, 528)),
        (specs.wire_view_handle, 'sans_detector', (16, 576)),
        (specs.strip_view_handle, 'mantle_detector', (30, 256)),
        (specs.strip_view_handle, 'endcap_backward_detector', (16, 308)),
        (specs.strip_view_handle, 'endcap_forward_detector', (16, 140)),
        (specs.strip_view_handle, 'high_resolution_detector', (32, 264)),
        (specs.strip_view_handle, 'sans_detector', (32, 288)),
    ],
)
def test_logical_view_counts_every_event(
    handle: SpecHandle, source_name: str, sizes: tuple[int, int]
) -> None:
    factory = specs.instrument.workflow_factory
    workflow = factory.create(
        source_name=source_name,
        config=WorkflowConfig(
            identifier=handle.workflow_id,
            job_id=JobId(source_name=source_name, job_number=uuid.uuid4()),
        ),
        params=factory[handle.workflow_id].params(),
    )
    workflow.accumulate(
        {source_name: _events(source_name, size=1000)},
        start_time=Timestamp.from_ns(1000),
        end_time=Timestamp.from_ns(2000),
    )
    result = workflow.finalize()

    assert result['cumulative'].shape == sizes
    assert result['cumulative'].sum().value == 1000
