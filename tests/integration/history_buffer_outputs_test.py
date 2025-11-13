# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Integration tests for history buffer functionality with backend outputs.

These tests verify that certain backend outputs have a 'time' coordinate
and work correctly with DataServiceSubscriber using FullHistoryExtractor
or WindowAggregatingExtractor.

The tests also verify that:
- Multiple data points arrive over time
- Time values are monotonically increasing
- History accumulates properly with FullHistoryExtractor
"""

import time

import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import ResultKey, WorkflowId
from ess.livedata.dashboard.data_subscriber import (
    DataSubscriber,
    MergingStreamAssembler,
)
from ess.livedata.dashboard.extractors import FullHistoryExtractor
from ess.livedata.handlers.detector_view_specs import DetectorViewParams
from ess.livedata.handlers.monitor_workflow_specs import MonitorDataParams
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import wait_for_job_data


class FakePipe:
    """Fake pipe for testing data extraction."""

    def __init__(self, data=None):
        self.init_data = data
        self.sent_data = []

    def send(self, data):
        self.sent_data.append(data)


def create_subscriber_with_full_history_extractor(
    keys: set[ResultKey],
) -> tuple[DataSubscriber, FakePipe]:
    """
    Create a test subscriber with FullHistoryExtractor.

    Parameters
    ----------
    keys:
        Set of result keys to subscribe to.

    Returns
    -------
    :
        Tuple of (subscriber, pipe) for accessing received data.
    """
    assembler = MergingStreamAssembler(keys)
    extractor = FullHistoryExtractor()
    extractors = {key: extractor for key in keys}

    pipe = FakePipe()

    def pipe_factory(data):
        pipe.init_data = data
        return pipe

    subscriber = DataSubscriber(assembler, pipe_factory, extractors)
    return subscriber, pipe


def verify_output_has_time_coord_and_accumulates(
    backend,
    workflow_id: WorkflowId,
    source_names: list[str],
    config,
    output_name: str,
) -> None:
    """
    Helper to verify that an output has time coord and accumulates properly.

    This encapsulates the common test logic for verifying:
    - Output has time coordinate and dimension
    - Multiple updates arrive
    - History accumulates (size increases or stays same)
    - Times are monotonically increasing

    Parameters
    ----------
    backend:
        DashboardBackend instance
    workflow_id:
        WorkflowId for the workflow to test
    source_names:
        List of source names to start workflow for
    config:
        Configuration for the workflow (or None)
    output_name:
        Name of the output to verify (e.g., 'current', 'delta')
    """
    # Start workflow
    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=source_names,
        config=config,
    )

    assert len(job_ids) == 1
    job_id = job_ids[0]

    # Wait for initial data
    wait_for_job_data(backend, job_ids, timeout=10.0)

    # Create subscriber with FullHistoryExtractor
    result_key = ResultKey(
        workflow_id=workflow_id, job_id=job_id, output_name=output_name
    )
    subscriber, pipe = create_subscriber_with_full_history_extractor(keys={result_key})

    # Register subscriber and trigger update
    backend.data_service.register_subscriber(subscriber)
    backend.update()

    # Verify pipe was initialized with data
    assert pipe.init_data is not None, "Pipe should be initialized with data"
    assert result_key in pipe.init_data, f"Expected '{output_name}' output in pipe data"

    # Verify the data has a 'time' coordinate and dimension
    first_data = pipe.init_data[result_key]
    assert isinstance(first_data, sc.DataArray), "Expected DataArray"
    assert (
        'time' in first_data.coords
    ), f"Expected 'time' coordinate in '{output_name}' output"
    assert 'time' in first_data.dims, "Expected 'time' dimension for history buffer"

    # Get initial size
    initial_size = first_data.sizes['time']

    # If we have multiple points initially, verify they're monotonic
    if initial_size > 1:
        times = first_data.coords['time'].values
        time_diffs = times[1:] - times[:-1]
        assert (time_diffs > 0).all(), "Expected monotonically increasing initial times"

    # Wait for more data to accumulate
    time.sleep(2.0)
    backend.update()

    # Verify we received updates
    assert len(pipe.sent_data) > 0, "Expected updates after initial data"

    # Get latest accumulated data
    latest_data = pipe.sent_data[-1][result_key]
    assert isinstance(latest_data, sc.DataArray)
    assert 'time' in latest_data.dims

    # Verify history is accumulating
    new_size = latest_data.sizes['time']
    assert (
        new_size >= initial_size
    ), f"Expected history to accumulate: {initial_size} -> {new_size}"

    # If we got new data points, verify times are monotonically increasing
    if new_size > initial_size:
        new_times = latest_data.coords['time'].values
        time_diffs = new_times[1:] - new_times[:-1]
        assert (
            time_diffs > 0
        ).all(), (
            "Expected monotonically increasing times, but found non-increasing values"
        )


@pytest.mark.integration
@pytest.mark.services('monitor')
def test_monitor_current_output_with_full_history_extractor(
    integration_env: IntegrationEnv,
) -> None:
    """
    Test monitor 'current' output with history extractor.

    Verifies time coordinate, accumulation, and monotonic times.
    """
    verify_output_has_time_coord_and_accumulates(
        backend=integration_env.backend,
        workflow_id=WorkflowId(
            instrument='dummy',
            namespace='monitor_data',
            name='monitor_histogram',
            version=1,
        ),
        source_names=['monitor1'],
        config=MonitorDataParams(),
        output_name='current',
    )


@pytest.mark.integration
@pytest.mark.services('detector')
def test_detector_current_output_with_full_history_extractor(
    integration_env: IntegrationEnv,
) -> None:
    """
    Test detector 'current' output with history extractor.

    Verifies time coordinate, accumulation, and monotonic times.
    """
    verify_output_has_time_coord_and_accumulates(
        backend=integration_env.backend,
        workflow_id=WorkflowId(
            instrument='dummy',
            namespace='detector_data',
            name='panel_0_xy',
            version=1,
        ),
        source_names=['panel_0'],
        config=DetectorViewParams(),
        output_name='current',
    )


@pytest.mark.integration
@pytest.mark.services('monitor')
def test_monitor_cumulative_output_does_not_have_time_coord(
    integration_env: IntegrationEnv,
) -> None:
    """
    Test that monitor 'cumulative' output does NOT have a time coordinate.

    This test verifies the contrast: only 'current' outputs should have time coords,
    not 'cumulative' outputs.
    """
    backend = integration_env.backend

    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )
    source_names = ['monitor1']

    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=source_names,
        config=MonitorDataParams(),
    )

    assert len(job_ids) == 1
    job_id = job_ids[0]

    wait_for_job_data(backend, job_ids, timeout=10.0)

    # Get cumulative output directly from job_data
    job_data = backend.job_service.job_data[job_id.job_number]
    assert 'monitor1' in job_data
    source_data = job_data['monitor1']
    assert 'cumulative' in source_data

    cumulative_data = source_data['cumulative']
    assert isinstance(cumulative_data, sc.DataArray)
    # Cumulative should NOT have time coordinate
    assert (
        'time' not in cumulative_data.coords
    ), "'cumulative' output should not have time coordinate"


@pytest.mark.integration
@pytest.mark.services('timeseries')
def test_timeseries_delta_output_with_full_history_extractor(
    integration_env: IntegrationEnv,
) -> None:
    """
    Test timeseries 'delta' output with history extractor.

    Verifies time coordinate, accumulation, and monotonic times.
    Timeseries may publish multiple time points at once.
    """
    verify_output_has_time_coord_and_accumulates(
        backend=integration_env.backend,
        workflow_id=WorkflowId(
            instrument='dummy',
            namespace='timeseries',
            name='timeseries_data',
            version=1,
        ),
        source_names=['motion1'],
        config=None,
        output_name='delta',
    )
