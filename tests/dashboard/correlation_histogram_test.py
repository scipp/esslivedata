# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for CorrelationHistogramController."""

import uuid

import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.correlation_histogram import (
    CorrelationHistogramController,
    CorrelationHistogramProcessor,
    EdgesWithUnit,
)
from ess.livedata.dashboard.data_service import DataService


def make_timeseries_data(values: list[float], times: list[float]) -> sc.DataArray:
    """Create a 1D timeseries DataArray with time dimension.

    Note: This is a convenience for tests, but in practice individual timeseries
    points arrive as 0D scalars (see make_timeseries_point).
    """
    return sc.DataArray(
        sc.array(dims=['time'], values=values, unit='counts'),
        coords={'time': sc.array(dims=['time'], values=times, unit='s')},
    )


def make_timeseries_point(value: float, time: float) -> sc.DataArray:
    """Create a single 0D timeseries point with time coordinate.

    This mimics how real timeseries data arrives - as individual 0D scalars
    with a time coord (no time dimension yet).
    """
    return sc.DataArray(
        sc.scalar(value, unit='counts'),
        coords={'time': sc.scalar(time, unit='s')},
    )


def make_non_timeseries_data(value: float) -> sc.DataArray:
    """Create a 0D scalar DataArray without time dimension."""
    return sc.DataArray(
        sc.scalar(value, unit='counts'),
        coords={'x': sc.scalar(10.0, unit='m')},
    )


def make_result_key(source_name: str) -> ResultKey:
    """Create a ResultKey for testing."""
    return ResultKey(
        workflow_id=WorkflowId(
            instrument='test', namespace='test', name='workflow', version=1
        ),
        job_id=JobId(source_name=source_name, job_number=uuid.uuid4()),
        output_name=None,
    )


class TestCorrelationHistogramController:
    """Tests for CorrelationHistogramController with buffered DataService."""

    def test_get_timeseries_with_individual_0d_points(self):
        """Test identifying timeseries from individual 0D points.

        In practice, timeseries data arrives as individual 0D scalars with time
        coords (no time dimension). The time dimension is only added when the
        history extractor concatenates them. This test verifies get_timeseries()
        correctly identifies such data.
        """
        data_service = DataService[ResultKey, sc.DataArray]()
        controller = CorrelationHistogramController(data_service)

        # Add individual 0D timeseries points (realistic scenario)
        ts_key = make_result_key('timeseries_stream')
        data_service[ts_key] = make_timeseries_point(1.0, 0.0)
        data_service[ts_key] = make_timeseries_point(2.0, 1.0)
        data_service[ts_key] = make_timeseries_point(3.0, 2.0)

        # Add non-timeseries data
        non_ts_key = make_result_key('scalar_data')
        data_service[non_ts_key] = make_non_timeseries_data(42.0)

        # get_timeseries should identify the timeseries
        timeseries_keys = controller.get_timeseries()

        assert ts_key in timeseries_keys, "Failed to identify 0D point timeseries"
        assert (
            non_ts_key not in timeseries_keys
        ), "Incorrectly identified scalar as timeseries"
        assert len(timeseries_keys) == 1

    def test_get_timeseries_identifies_buffered_timeseries(self):
        """Test identifying timeseries from buffered DataService.

        This test verifies that get_timeseries correctly identifies timeseries
        when DataService uses buffers and LatestValueExtractor returns 0D
        scalars.
        """
        # Create DataService (uses buffers internally)
        data_service = DataService[ResultKey, sc.DataArray]()
        controller = CorrelationHistogramController(data_service)

        # Add timeseries data - stored in buffers
        ts_key1 = make_result_key('timeseries_1')
        ts_key2 = make_result_key('timeseries_2')
        data_service[ts_key1] = make_timeseries_data([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
        data_service[ts_key2] = make_timeseries_data([10.0, 20.0], [0.0, 1.0])

        # Add non-timeseries data
        non_ts_key = make_result_key('scalar_data')
        data_service[non_ts_key] = make_non_timeseries_data(42.0)

        # Get timeseries - this should find the timeseries keys
        timeseries_keys = controller.get_timeseries()

        # Should identify both timeseries, but NOT the scalar
        assert ts_key1 in timeseries_keys, "Failed to identify timeseries_1"
        assert ts_key2 in timeseries_keys, "Failed to identify timeseries_2"
        assert (
            non_ts_key not in timeseries_keys
        ), "Incorrectly identified scalar as timeseries"
        assert len(timeseries_keys) == 2

    def test_processor_receives_full_history_from_0d_points(self):
        """Test processor receives concatenated history from individual 0D points.

        Realistic scenario: Individual 0D timeseries points arrive after subscriber
        registration. The buffer accumulates history only after a FullHistoryExtractor
        subscriber is registered, then concatenates subsequent 0D points into 1D.
        """
        data_service = DataService[ResultKey, sc.DataArray]()
        controller = CorrelationHistogramController(data_service)

        # Add initial 0D points (before subscriber registration)
        data_key = make_result_key('data_stream')
        coord_key = make_result_key('coord_stream')

        data_service[data_key] = make_timeseries_point(1.0, 0.0)
        data_service[coord_key] = make_timeseries_point(10.0, 0.0)

        # Track what data the processor receives
        received_data = []

        def result_callback(_: sc.DataArray) -> None:
            """Callback to capture processor results."""

        # Create processor with edges for binning
        edges = EdgesWithUnit(start=5.0, stop=35.0, num_bins=3, unit='counts')

        processor = CorrelationHistogramProcessor(
            data_key=data_key,
            coord_keys=[coord_key],
            edges_params=[edges],
            normalize=False,
            result_callback=result_callback,
        )

        # Monkey-patch processor.send to capture received data
        original_send = processor.send

        def capturing_send(data: dict[ResultKey, sc.DataArray]) -> None:
            received_data.append(data)
            original_send(data)

        processor.send = capturing_send

        # Register subscriber with FullHistoryExtractor - buffer starts accumulating now
        items = {
            data_key: data_service[data_key],
            coord_key: data_service[coord_key],
        }
        controller.add_correlation_processor(processor, items)

        # Processor triggered immediately with existing data (1 point each)
        assert len(received_data) == 1, "Processor should be triggered on registration"
        received = received_data[0]
        assert received[data_key].dims == ('time',)
        assert received[data_key].sizes['time'] == 1, "Initial data: 1 point"

        # Now add more 0D points - use transaction to batch updates
        with data_service.transaction():
            data_service[data_key] = make_timeseries_point(2.0, 1.0)
            data_service[coord_key] = make_timeseries_point(20.0, 1.0)

        # Processor should receive accumulated history (2 points)
        assert len(received_data) == 2, "Processor triggered on new data"
        received = received_data[1]
        assert received[data_key].sizes['time'] == 2, "Should have 2 time points"
        assert received[coord_key].sizes['time'] == 2

        # Add another point
        with data_service.transaction():
            data_service[data_key] = make_timeseries_point(3.0, 2.0)
            data_service[coord_key] = make_timeseries_point(30.0, 2.0)

        # Processor should receive full accumulated history (3 points)
        assert len(received_data) == 3, "Processor triggered again"
        latest_received = received_data[-1]
        assert latest_received[data_key].sizes['time'] == 3, "Should have 3 time points"
        assert latest_received[coord_key].sizes['time'] == 3

    def test_processor_receives_full_history_not_latest(self):
        """Test processor receives full timeseries history via extractors.

        The processor needs complete history to compute correlation histograms.
        This test verifies that FullHistoryExtractor is used for subscribers.
        """
        # Create DataService and controller
        data_service = DataService[ResultKey, sc.DataArray]()
        controller = CorrelationHistogramController(data_service)

        # Add initial timeseries data
        data_key = make_result_key('data_stream')
        coord_key = make_result_key('coord_stream')

        data_service[data_key] = make_timeseries_data([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
        data_service[coord_key] = make_timeseries_data(
            [10.0, 20.0, 30.0], [0.0, 1.0, 2.0]
        )

        # Track what data the processor receives
        received_data = []

        def result_callback(_: sc.DataArray) -> None:
            """Callback to capture processor results."""

        # Create processor with edges for binning
        edges = EdgesWithUnit(start=5.0, stop=35.0, num_bins=3, unit='counts')

        processor = CorrelationHistogramProcessor(
            data_key=data_key,
            coord_keys=[coord_key],
            edges_params=[edges],
            normalize=False,
            result_callback=result_callback,
        )

        # Monkey-patch processor.send to capture received data
        original_send = processor.send

        def capturing_send(data: dict[ResultKey, sc.DataArray]) -> None:
            received_data.append(data)
            original_send(data)

        processor.send = capturing_send

        # Add processor - this should register subscriber with FullHistoryExtractor
        items = {
            data_key: data_service[data_key],
            coord_key: data_service[coord_key],
        }
        controller.add_correlation_processor(processor, items)

        # Processor should have been triggered immediately with existing data
        assert len(received_data) == 1, "Processor should be triggered on registration"

        # Verify that processor received FULL history, not just latest value
        received = received_data[0]
        assert data_key in received
        assert coord_key in received

        # Check data dimensions - should be 1D timeseries, not 0D scalar
        assert received[data_key].dims == (
            'time',
        ), f"Expected 1D timeseries with time dim, got {received[data_key].dims}"
        assert received[data_key].sizes['time'] == 3, "Should have all 3 time points"

        assert received[coord_key].dims == ('time',)
        assert received[coord_key].sizes['time'] == 3

        # Add more data and verify processor gets updated history
        data_service[data_key] = make_timeseries_data([4.0], [3.0])
        data_service[coord_key] = make_timeseries_data([40.0], [3.0])

        # Should have received second update
        assert len(received_data) >= 2, "Processor should be triggered on data update"
        latest_received = received_data[-1]

        # Latest update should also include full history (now 4 points)
        assert latest_received[data_key].sizes['time'] == 4
        assert latest_received[coord_key].sizes['time'] == 4
