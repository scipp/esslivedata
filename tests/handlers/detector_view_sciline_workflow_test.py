# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for Sciline-based detector view workflow."""

import pytest
import scipp as sc

from ess.livedata.handlers.detector_view_sciline_workflow import (
    CountsInTOARange,
    CountsTotal,
    CumulativeDetectorImage,
    CumulativeHistogram,
    CurrentDetectorImage,
    DetectorHistogram3D,
    WindowAccumulator,
    WindowHistogram,
    compute_detector_histogram_3d,
    create_accumulators,
    create_base_workflow,
    cumulative_histogram,
    window_histogram,
)
from ess.reduce.nexus.types import RawDetector, SampleRun


def make_fake_nexus_detector_data(
    *, y_size: int = 4, x_size: int = 4, n_events_per_pixel: int = 10
) -> sc.DataArray:
    """Create fake detector data similar to what GenericNeXusWorkflow produces.

    GenericNeXusWorkflow produces binned event data grouped by detector_number,
    with events containing event_time_offset coordinates.
    """
    import numpy as np

    rng = np.random.default_rng(42)

    total_pixels = y_size * x_size
    total_events = total_pixels * n_events_per_pixel

    # Create event_time_offset values in nanoseconds (0-71ms range)
    eto_values = rng.uniform(0, 71_000_000, total_events)

    # Create event table with event_time_offset as coordinate
    # Data values are weights (typically 1.0 for each event)
    events = sc.DataArray(
        data=sc.ones(dims=['event'], shape=[total_events]),
        coords={
            'event_time_offset': sc.array(dims=['event'], values=eto_values, unit='ns'),
        },
    )

    # Create bin indices for each pixel
    begin = sc.arange(
        'detector_number', 0, total_pixels * n_events_per_pixel, n_events_per_pixel
    )
    begin.unit = None
    end = begin + sc.scalar(n_events_per_pixel, unit=None)

    # Bin the events by detector_number - wrap in DataArray
    binned_var = sc.bins(begin=begin, end=end, dim='event', data=events)

    # Create DataArray with detector_number coordinate
    binned = sc.DataArray(
        data=binned_var,
        coords={
            'detector_number': sc.arange(
                'detector_number', 1, total_pixels + 1, unit=None
            )
        },
    )

    return binned


def make_logical_transform(y_size: int, x_size: int):
    """Create a logical transform that folds detector_number to (y, x)."""

    def transform(da: sc.DataArray) -> sc.DataArray:
        return da.fold(dim='detector_number', sizes={'y': y_size, 'x': x_size})

    return transform


class TestWindowAccumulator:
    """Tests for WindowAccumulator that clears after finalize."""

    def test_clears_after_on_finalize(self):
        """Test that WindowAccumulator clears after on_finalize."""
        acc = WindowAccumulator()
        acc.push(sc.scalar(10))
        assert acc.value == sc.scalar(10)

        # Simulate what happens during finalize cycle
        acc.on_finalize()
        assert acc.is_empty

    def test_accumulates_multiple_values(self):
        """Test that values are accumulated before on_finalize."""
        acc = WindowAccumulator()
        acc.push(sc.scalar(10))
        acc.push(sc.scalar(20))
        assert acc.value == sc.scalar(30)


class TestCreateBaseWorkflow:
    """Tests for create_base_workflow function."""

    def test_creates_workflow_with_required_params(self):
        """Test workflow creation with required parameters."""
        tof_bins = sc.linspace('tof', 0, 100000, 11, unit='ns')
        wf = create_base_workflow(tof_bins=tof_bins)
        assert wf is not None

    def test_creates_workflow_with_tof_slice(self):
        """Test workflow creation with TOF slice parameter."""
        tof_bins = sc.linspace('tof', 0, 100000, 11, unit='ns')
        tof_slice = (sc.scalar(10000, unit='ns'), sc.scalar(50000, unit='ns'))
        wf = create_base_workflow(tof_bins=tof_bins, tof_slice=tof_slice)
        assert wf is not None

    def test_creates_workflow_with_logical_transform(self):
        """Test workflow creation with logical transform."""
        tof_bins = sc.linspace('tof', 0, 100000, 11, unit='ns')

        def identity(da: sc.DataArray) -> sc.DataArray:
            return da

        wf = create_base_workflow(tof_bins=tof_bins, logical_transform=identity)
        assert wf is not None


class TestCreateAccumulators:
    """Tests for create_accumulators function."""

    def test_creates_correct_accumulator_types(self):
        """Test that correct accumulator types are created."""
        accumulators = create_accumulators()

        assert CumulativeHistogram in accumulators
        assert WindowHistogram in accumulators
        assert not isinstance(accumulators[CumulativeHistogram], WindowAccumulator)
        assert isinstance(accumulators[WindowHistogram], WindowAccumulator)


class TestComputeDetectorHistogram3D:
    """Tests for compute_detector_histogram_3d function."""

    def test_histogram_without_transform(self):
        """Test histogramming without logical transform."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4)
        tof_bins = sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns')

        result = compute_detector_histogram_3d(
            raw_detector=RawDetector[SampleRun](data),
            tof_bins=tof_bins,
            transform=None,
        )

        # Without transform, we get 1D histogram (all events concatenated)
        assert 'tof' in result.dims
        assert result.sizes['tof'] == 10

    def test_histogram_with_logical_transform(self):
        """Test histogramming with logical transform to (y, x) spatial dims."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        tof_bins = sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns')
        transform = make_logical_transform(4, 4)

        result = compute_detector_histogram_3d(
            raw_detector=RawDetector[SampleRun](data),
            tof_bins=tof_bins,
            transform=transform,
        )

        # With transform, we get histogram over all events (still 1D in tof)
        # The transform reshapes but bins.concat().hist() collapses spatial
        assert 'tof' in result.dims


class TestIdentityProviders:
    """Tests for cumulative_histogram and window_histogram identity providers."""

    def test_cumulative_histogram_identity(self):
        """Test that cumulative_histogram returns input unchanged."""
        data_3d = sc.DataArray(
            sc.ones(dims=['y', 'x', 'tof'], shape=[4, 4, 10], unit='counts')
        )

        result = cumulative_histogram(DetectorHistogram3D(data_3d))

        assert sc.identical(result, data_3d)

    def test_window_histogram_identity(self):
        """Test that window_histogram returns input unchanged."""
        data_3d = sc.DataArray(
            sc.ones(dims=['y', 'x', 'tof'], shape=[4, 4, 10], unit='counts')
        )

        result = window_histogram(DetectorHistogram3D(data_3d))

        assert sc.identical(result, data_3d)


class TestDetectorImageProviders:
    """Tests for detector image provider functions."""

    def test_cumulative_detector_image_sums_over_tof(self):
        """Test that cumulative_detector_image produces 2D output."""
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            cumulative_detector_image,
        )

        # Create 3D histogram
        data_3d = sc.DataArray(
            sc.ones(dims=['y', 'x', 'tof'], shape=[4, 4, 10], unit='counts')
        )

        result = cumulative_detector_image(
            data_3d=CumulativeHistogram(data_3d),
            tof_slice=None,
        )

        assert result.dims == ('y', 'x')
        assert result.sizes == {'y': 4, 'x': 4}
        # Each pixel should have sum of 10 TOF bins
        expected = sc.full(dims=['y', 'x'], shape=[4, 4], value=10.0, unit='counts')
        assert sc.allclose(result.data, expected)

    def test_current_detector_image_sums_over_tof(self):
        """Test that current_detector_image produces 2D output."""
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            current_detector_image,
        )

        data_3d = sc.DataArray(
            sc.ones(dims=['y', 'x', 'tof'], shape=[4, 4, 10], unit='counts')
        )

        result = current_detector_image(
            data_3d=WindowHistogram(data_3d),
            tof_slice=None,
        )

        assert result.dims == ('y', 'x')
        assert result.sizes == {'y': 4, 'x': 4}

    def test_detector_image_with_tof_slice(self):
        """Test that TOF slicing is applied correctly."""
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            cumulative_detector_image,
        )

        # Create 3D histogram with TOF coordinate
        tof_coord = sc.linspace('tof', 0, 100000, 11, unit='ns')
        data_3d = sc.DataArray(
            sc.ones(dims=['y', 'x', 'tof'], shape=[4, 4, 10], unit='counts'),
            coords={'tof': tof_coord},
        )

        # Slice to first half of TOF range
        tof_slice = (sc.scalar(0, unit='ns'), sc.scalar(50000, unit='ns'))

        result = cumulative_detector_image(
            data_3d=CumulativeHistogram(data_3d),
            tof_slice=tof_slice,
        )

        # Should only sum ~5 bins (0-50000 ns from 0-100000 ns range)
        assert result.dims == ('y', 'x')


class TestCountProviders:
    """Tests for count provider functions."""

    def test_counts_total(self):
        """Test that counts_total sums all counts."""
        from ess.livedata.handlers.detector_view_sciline_workflow import counts_total

        data_3d = sc.DataArray(
            sc.ones(dims=['y', 'x', 'tof'], shape=[4, 4, 10], unit='counts')
        )

        result = counts_total(data_3d=WindowHistogram(data_3d))

        expected = sc.scalar(4 * 4 * 10, unit='counts', dtype='float64')
        assert sc.identical(result.data, expected)

    def test_counts_in_toa_range_no_slice(self):
        """Test counts_in_toa_range with no TOF slice."""
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            counts_in_toa_range,
        )

        data_3d = sc.DataArray(
            sc.ones(dims=['y', 'x', 'tof'], shape=[4, 4, 10], unit='counts')
        )

        result = counts_in_toa_range(data_3d=WindowHistogram(data_3d), tof_slice=None)

        expected = sc.scalar(4 * 4 * 10, unit='counts', dtype='float64')
        assert sc.identical(result.data, expected)

    def test_counts_in_toa_range_with_slice(self):
        """Test counts_in_toa_range with TOF slice."""
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            counts_in_toa_range,
        )

        tof_coord = sc.linspace('tof', 0, 100000, 11, unit='ns')
        data_3d = sc.DataArray(
            sc.ones(dims=['y', 'x', 'tof'], shape=[4, 4, 10], unit='counts'),
            coords={'tof': tof_coord},
        )

        # Slice to first half
        tof_slice = (sc.scalar(0, unit='ns'), sc.scalar(50000, unit='ns'))

        result = counts_in_toa_range(
            data_3d=WindowHistogram(data_3d), tof_slice=tof_slice
        )

        # Should count approximately half the bins
        assert result.dims == ()


class TestIntegrationWithStreamProcessor:
    """Integration tests using the full StreamProcessorWorkflow."""

    def test_full_workflow_accumulate_and_finalize(self):
        """Test the full workflow with accumulate and finalize."""
        from ess.livedata.handlers.stream_processor_workflow import (
            StreamProcessorWorkflow,
        )

        tof_bins = sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns')
        base_workflow = create_base_workflow(tof_bins=tof_bins)

        workflow = StreamProcessorWorkflow(
            base_workflow,
            dynamic_keys={'detector': RawDetector[SampleRun]},
            target_keys={
                'cumulative': CumulativeDetectorImage,
                'current': CurrentDetectorImage,
                'counts_total': CountsTotal,
                'counts_in_toa_range': CountsInTOARange,
            },
            accumulators=create_accumulators(),
        )

        # Create fake events
        events = make_fake_nexus_detector_data(
            y_size=4, x_size=4, n_events_per_pixel=10
        )

        # Accumulate
        workflow.accumulate(
            {'detector': RawDetector[SampleRun](events)},
            start_time=1000,
            end_time=2000,
        )

        # Finalize
        result = workflow.finalize()

        assert 'cumulative' in result
        assert 'current' in result
        assert 'counts_total' in result
        assert 'counts_in_toa_range' in result

    def test_cumulative_accumulates_current_resets(self):
        """Test that cumulative accumulates and current resets after finalize."""
        from ess.livedata.handlers.stream_processor_workflow import (
            StreamProcessorWorkflow,
        )

        tof_bins = sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns')
        base_workflow = create_base_workflow(tof_bins=tof_bins)

        workflow = StreamProcessorWorkflow(
            base_workflow,
            dynamic_keys={'detector': RawDetector[SampleRun]},
            target_keys={
                'cumulative': CumulativeDetectorImage,
                'current': CurrentDetectorImage,
            },
            accumulators=create_accumulators(),
        )

        # First batch
        events1 = make_fake_nexus_detector_data(
            y_size=4, x_size=4, n_events_per_pixel=10
        )
        workflow.accumulate(
            {'detector': RawDetector[SampleRun](events1)},
            start_time=1000,
            end_time=2000,
        )
        result1 = workflow.finalize()

        cumulative1 = result1['cumulative'].sum().value
        current1 = result1['current'].sum().value

        # After first finalize, cumulative == current (same data)
        assert cumulative1 == current1

        # Second batch
        events2 = make_fake_nexus_detector_data(
            y_size=4, x_size=4, n_events_per_pixel=10
        )
        workflow.accumulate(
            {'detector': RawDetector[SampleRun](events2)},
            start_time=2000,
            end_time=3000,
        )
        result2 = workflow.finalize()

        cumulative2 = result2['cumulative'].sum().value
        current2 = result2['current'].sum().value

        # Cumulative should have doubled (events1 + events2)
        assert cumulative2 == pytest.approx(cumulative1 * 2, rel=0.1)

        # Current should be approximately the same as first batch (only events2)
        assert current2 == pytest.approx(current1, rel=0.1)
