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


@pytest.mark.integration
@pytest.mark.services('monitor')
def test_monitor_current_output_with_full_history_extractor(
    integration_env: IntegrationEnv,
) -> None:
    """
    Test that monitor 'current' output has time coord and works with history extractor.

    The monitor workflow produces 'current' and 'cumulative' outputs. The 'current'
    output should have a 'time' coordinate, making it compatible with history buffers.
    This test verifies that multiple updates arrive and accumulate in the buffer.
    """
    backend = integration_env.backend

    # Start monitor histogram workflow
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

    assert len(job_ids) == 1, f"Expected 1 job, got {len(job_ids)}"
    job_id = job_ids[0]

    # Wait for initial data
    wait_for_job_data(backend, job_ids, timeout=10.0)

    # Create subscriber with FullHistoryExtractor for 'current' output
    result_key = ResultKey(
        workflow_id=workflow_id, job_id=job_id, output_name='current'
    )
    subscriber, pipe = create_subscriber_with_full_history_extractor(keys={result_key})

    # Register subscriber with data service
    backend.data_service.register_subscriber(subscriber)

    # Trigger backend update to ensure subscriber gets notified
    backend.update()

    # Verify pipe was initialized and received data
    assert pipe.init_data is not None, "Pipe should be initialized with data"
    assert result_key in pipe.init_data, "Expected 'current' output in pipe data"

    # Verify the data has a 'time' coordinate
    first_data = pipe.init_data[result_key]
    assert isinstance(first_data, sc.DataArray), "Expected DataArray"
    assert 'time' in first_data.coords, "Expected 'time' coordinate in 'current' output"
    assert 'time' in first_data.dims, "Expected 'time' dimension for history buffer"

    # Get initial time and size
    initial_size = first_data.sizes['time']

    # Wait for more data to accumulate (multiple updates)
    time.sleep(2.0)  # Wait for a few more data points
    backend.update()

    # Verify we received new data via pipe.sent_data (subsequent updates)
    assert len(pipe.sent_data) > 0, "Expected updates after initial data"

    # Get latest accumulated data
    latest_data = pipe.sent_data[-1][result_key]
    assert isinstance(latest_data, sc.DataArray)
    assert 'time' in latest_data.dims

    # Verify history is accumulating
    new_size = latest_data.sizes['time']
    assert new_size >= initial_size, (
        f"Expected history to accumulate, but size decreased: "
        f"{initial_size} -> {new_size}"
    )

    # If we got new data points, verify times are monotonically increasing
    if new_size > initial_size:
        new_times = latest_data.coords['time'].values
        # Check that times are strictly increasing
        time_diffs = new_times[1:] - new_times[:-1]
        assert (
            time_diffs > 0
        ).all(), (
            "Expected monotonically increasing times, but found non-increasing values"
        )


@pytest.mark.integration
@pytest.mark.services('detector')
def test_detector_current_output_with_full_history_extractor(
    integration_env: IntegrationEnv,
) -> None:
    """
    Test that detector 'current' output has time coord and works with history extractor.

    The detector view workflow produces 'current' and 'cumulative' outputs. The
    'current' output should have a 'time' coordinate and accumulate over time.
    """
    backend = integration_env.backend

    # Start detector view workflow for panel_0
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='detector_data',
        name='panel_0_xy',
        version=1,
    )
    source_names = ['panel_0']

    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=source_names,
        config=DetectorViewParams(),
    )

    assert len(job_ids) == 1
    job_id = job_ids[0]

    # Wait for initial data
    wait_for_job_data(backend, job_ids, timeout=10.0)

    # Create subscriber with FullHistoryExtractor for 'current' output
    result_key = ResultKey(
        workflow_id=workflow_id, job_id=job_id, output_name='current'
    )
    subscriber, pipe = create_subscriber_with_full_history_extractor(keys={result_key})

    backend.data_service.register_subscriber(subscriber)
    backend.update()

    # Verify data was received
    assert pipe.init_data is not None
    assert result_key in pipe.init_data

    first_data = pipe.init_data[result_key]
    assert isinstance(first_data, sc.DataArray)
    assert 'time' in first_data.coords, "Expected 'time' coordinate in 'current' output"
    assert 'time' in first_data.dims, "Expected 'time' dimension for history buffer"

    initial_size = first_data.sizes['time']

    # Wait for more data to accumulate
    time.sleep(2.0)
    backend.update()

    # Verify we received updates
    assert len(pipe.sent_data) > 0, "Expected updates after initial data"

    latest_data = pipe.sent_data[-1][result_key]
    new_size = latest_data.sizes['time']

    # Verify history is accumulating
    assert (
        new_size >= initial_size
    ), f"Expected history to accumulate: {initial_size} -> {new_size}"

    # If we got new data, verify times are monotonically increasing
    if new_size > initial_size:
        new_times = latest_data.coords['time'].values
        time_diffs = new_times[1:] - new_times[:-1]
        assert (time_diffs > 0).all(), "Expected monotonically increasing times"


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
    Test that timeseries 'delta' output has time coord and works with history extractor.

    The timeseries workflow produces a 'delta' output which behaves like 'current'
    from monitor/detector workflows. The main difference is that timeseries may
    publish multiple time points at once. The 'delta' output should have a 'time'
    coordinate and work with history extractors.
    """
    backend = integration_env.backend

    # Start timeseries workflow
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='timeseries',
        name='timeseries_data',
        version=1,
    )
    source_names = ['motion1']

    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=source_names,
        config=None,
    )

    assert len(job_ids) == 1
    job_id = job_ids[0]

    wait_for_job_data(backend, job_ids, timeout=10.0)

    # Create subscriber with FullHistoryExtractor for 'delta' output
    result_key = ResultKey(workflow_id=workflow_id, job_id=job_id, output_name='delta')
    subscriber, pipe = create_subscriber_with_full_history_extractor(keys={result_key})

    backend.data_service.register_subscriber(subscriber)
    backend.update()

    # Verify pipe was initialized
    assert pipe.init_data is not None
    assert result_key in pipe.init_data

    # Verify delta has time coordinate and dimension
    first_data = pipe.init_data[result_key]
    assert isinstance(first_data, sc.DataArray)
    assert 'time' in first_data.coords, "Delta output should have time coordinate"
    assert 'time' in first_data.dims, "Delta output should have time dimension"

    # Timeseries may publish multiple time points at once
    initial_size = first_data.sizes['time']
    assert initial_size >= 1, "Expected at least one time point"

    # If we have multiple points, verify times are monotonically increasing
    if initial_size > 1:
        times = first_data.coords['time'].values
        time_diffs = times[1:] - times[:-1]
        assert (time_diffs > 0).all(), "Expected monotonically increasing times"

    # Wait for more data
    time.sleep(2.0)
    backend.update()

    # Verify we received updates
    if len(pipe.sent_data) > 0:
        latest_data = pipe.sent_data[-1][result_key]
        new_size = latest_data.sizes['time']

        # Verify history is accumulating
        assert (
            new_size >= initial_size
        ), f"Expected history to accumulate: {initial_size} -> {new_size}"

        # Verify times are monotonically increasing across the full history
        if new_size > 1:
            all_times = latest_data.coords['time'].values
            time_diffs = all_times[1:] - all_times[:-1]
            assert (time_diffs > 0).all(), "Expected monotonically increasing times"
