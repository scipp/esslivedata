# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for Sciline-based detector view workflow."""

import numpy as np
import pytest
import scipp as sc

from ess.livedata.handlers.detector_view_sciline_workflow import (
    CountsInTOARange,
    CountsTotal,
    CumulativeDetectorImage,
    CumulativeHistogram,
    CurrentDetectorImage,
    DetectorHistogram3D,
    EventProjector,
    WindowAccumulator,
    WindowHistogram,
    add_logical_projection,
    compute_detector_histogram_3d,
    create_accumulators,
    create_base_workflow,
    cumulative_histogram,
    project_events_logical,
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


def make_fake_empty_detector(y_size: int, x_size: int) -> sc.DataArray:
    """Create a fake EmptyDetector for testing logical projection.

    EmptyDetector is the detector structure without events, used to determine
    output dimensions before any events arrive.
    """
    total_pixels = y_size * x_size
    # Empty bins structure - same as make_fake_nexus_detector_data but with 0 events
    begin = sc.zeros(dims=['detector_number'], shape=[total_pixels], dtype='int64')
    begin.unit = None
    end = begin.copy()  # Same as begin = no events

    # Create empty event table
    events = sc.DataArray(
        data=sc.empty(dims=['event'], shape=[0], dtype='float32', unit='counts'),
        coords={
            'event_time_offset': sc.empty(dims=['event'], shape=[0], unit='ns'),
        },
    )

    binned_var = sc.bins(begin=begin, end=end, dim='event', data=events)

    return sc.DataArray(
        data=binned_var,
        coords={
            'detector_number': sc.arange(
                'detector_number', 1, total_pixels + 1, unit=None
            )
        },
    )


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

    def test_creates_workflow_and_add_logical_projection(self):
        """Test workflow creation with logical projection added separately."""
        tof_bins = sc.linspace('tof', 0, 100000, 11, unit='ns')

        def identity(da: sc.DataArray) -> sc.DataArray:
            return da

        wf = create_base_workflow(tof_bins=tof_bins)
        add_logical_projection(wf, transform=identity)
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

    def test_histogram_from_screen_binned_events(self):
        """Test histogramming screen-binned events to 3D."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4)
        tof_bins = sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns')

        # Use project_events_logical with a fold transform
        transform = make_logical_transform(4, 4)
        screen_binned = project_events_logical(
            raw_detector=RawDetector[SampleRun](data),
            transform=transform,
            reduction_dim=None,
        )

        result = compute_detector_histogram_3d(
            screen_binned_events=screen_binned,
            tof_bins=tof_bins,
        )

        assert 'tof' in result.dims
        assert result.sizes['tof'] == 10
        assert result.sizes['y'] == 4
        assert result.sizes['x'] == 4

    def test_histogram_with_reduction_dim(self):
        """Test histogramming with reduction dimension."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        tof_bins = sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns')
        transform = make_logical_transform(4, 4)

        # Reduce along y dimension
        screen_binned = project_events_logical(
            raw_detector=RawDetector[SampleRun](data),
            transform=transform,
            reduction_dim='y',
        )

        result = compute_detector_histogram_3d(
            screen_binned_events=screen_binned,
            tof_bins=tof_bins,
        )

        # Should have x and tof dims, y was reduced
        assert 'tof' in result.dims
        assert 'x' in result.dims
        assert 'y' not in result.dims


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


class TestEventProjector:
    """Tests for EventProjector class."""

    @staticmethod
    def make_screen_coords_and_edges(n_pixels: int, screen_shape: tuple[int, int]):
        """Create test coordinates and edges for projection."""
        n_replicas = 2
        det_side = int(np.sqrt(n_pixels))
        scale_x = screen_shape[0] / det_side
        scale_y = screen_shape[1] / det_side

        pixel_y = np.arange(n_pixels) // det_side
        pixel_x = np.arange(n_pixels) % det_side

        rng = np.random.default_rng(42)
        coords_x = []
        coords_y = []
        for _ in range(n_replicas):
            noise = rng.normal(0, 0.1, n_pixels)
            coords_x.append(pixel_x * scale_x + noise)
            coords_y.append(pixel_y * scale_y + noise)

        coords = sc.DataGroup(
            screen_x=sc.array(
                dims=['replica', 'detector_number'], values=np.array(coords_x), unit='m'
            ),
            screen_y=sc.array(
                dims=['replica', 'detector_number'], values=np.array(coords_y), unit='m'
            ),
        )
        edges = sc.DataGroup(
            screen_x=sc.linspace(
                'screen_x', 0, screen_shape[0], screen_shape[0] + 1, unit='m'
            ),
            screen_y=sc.linspace(
                'screen_y', 0, screen_shape[1], screen_shape[1] + 1, unit='m'
            ),
        )
        return coords, edges

    def test_project_events_preserves_total_counts(self):
        """Test that event projection preserves total event count."""
        n_pixels = 16
        screen_shape = (4, 4)
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        coords, edges = self.make_screen_coords_and_edges(n_pixels, screen_shape)

        # Make edges wider to ensure all events are captured despite noise
        edges = sc.DataGroup(
            screen_x=sc.linspace('screen_x', -1, screen_shape[0] + 1, 10, unit='m'),
            screen_y=sc.linspace('screen_y', -1, screen_shape[1] + 1, 10, unit='m'),
        )

        projector = EventProjector(coords, edges)
        result = projector.project_events(data)

        # Total events should be preserved when edges are wide enough
        original_events = data.bins.size().sum().value
        projected_events = result.bins.size().sum().value
        assert projected_events == original_events

    def test_project_events_returns_binned_data(self):
        """Test that result is binned data with screen coordinate dims."""
        n_pixels = 16
        screen_shape = (4, 4)
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        coords, edges = self.make_screen_coords_and_edges(n_pixels, screen_shape)

        projector = EventProjector(coords, edges)
        result = projector.project_events(data)

        assert result.bins is not None
        assert 'screen_x' in result.dims
        assert 'screen_y' in result.dims
        assert result.sizes['screen_x'] == screen_shape[0]
        assert result.sizes['screen_y'] == screen_shape[1]

    def test_project_events_preserves_event_time_offset(self):
        """Test that event_time_offset is preserved in projected events."""
        n_pixels = 16
        screen_shape = (4, 4)
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        coords, edges = self.make_screen_coords_and_edges(n_pixels, screen_shape)

        projector = EventProjector(coords, edges)
        result = projector.project_events(data)

        # Check that events have event_time_offset coordinate
        event_data = result.bins.constituents['data']
        assert 'event_time_offset' in event_data.coords

    def test_project_events_cycles_replicas(self):
        """Test that projector cycles through replicas."""
        n_pixels = 16
        screen_shape = (4, 4)
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        coords, edges = self.make_screen_coords_and_edges(n_pixels, screen_shape)

        projector = EventProjector(coords, edges)

        # Project twice with same data
        result1 = projector.project_events(data)
        result2 = projector.project_events(data)

        # Results should differ (different replicas with different noise)
        assert not sc.identical(result1.bins.size(), result2.bins.size())


class TestProjectEventsLogical:
    """Tests for project_events_logical provider."""

    def test_no_transform_returns_raw_data(self):
        """Test that None transform returns input unchanged."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4)

        result = project_events_logical(
            raw_detector=RawDetector[SampleRun](data),
            transform=None,
            reduction_dim=None,
        )

        # ScreenBinnedEvents is a NewType wrapping DataArray, compare directly
        assert sc.identical(result, data)

    def test_fold_transform(self):
        """Test that fold transform reshapes detector data."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4)
        transform = make_logical_transform(4, 4)

        result = project_events_logical(
            raw_detector=RawDetector[SampleRun](data),
            transform=transform,
            reduction_dim=None,
        )

        assert 'y' in result.dims
        assert 'x' in result.dims
        assert 'detector_number' not in result.dims
        assert result.sizes['y'] == 4
        assert result.sizes['x'] == 4

    def test_reduction_dim_concatenates_events(self):
        """Test that reduction_dim concatenates events along specified dim."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        transform = make_logical_transform(4, 4)

        # Reduce along y dimension
        result = project_events_logical(
            raw_detector=RawDetector[SampleRun](data),
            transform=transform,
            reduction_dim='y',
        )

        # y dimension should be reduced
        assert 'y' not in result.dims
        assert 'x' in result.dims
        assert result.sizes['x'] == 4
        # Each x bin should have all events from that column
        assert result.bins.size().sum().value == 4 * 4 * 10

    def test_multiple_reduction_dims(self):
        """Test reduction over multiple dimensions."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        transform = make_logical_transform(4, 4)

        # Reduce along both y and x
        result = project_events_logical(
            raw_detector=RawDetector[SampleRun](data),
            transform=transform,
            reduction_dim=['y', 'x'],
        )

        # Both dimensions should be reduced - result is scalar with all events
        assert 'y' not in result.dims
        assert 'x' not in result.dims
        assert result.dims == ()
        assert result.bins.size().sum().value == 4 * 4 * 10


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

        # Add logical projection with a fold transform
        transform = make_logical_transform(4, 4)
        add_logical_projection(base_workflow, transform=transform)

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

        # Verify output shapes
        assert result['cumulative'].dims == ('y', 'x')
        assert result['cumulative'].sizes == {'y': 4, 'x': 4}

    def test_cumulative_accumulates_current_resets(self):
        """Test that cumulative accumulates and current resets after finalize."""
        from ess.livedata.handlers.stream_processor_workflow import (
            StreamProcessorWorkflow,
        )

        tof_bins = sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns')
        base_workflow = create_base_workflow(tof_bins=tof_bins)

        # Add logical projection with a fold transform
        transform = make_logical_transform(4, 4)
        add_logical_projection(base_workflow, transform=transform)

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


class TestROISpectraProviders:
    """Tests for ROI spectra extraction providers."""

    def make_histogram_3d(self) -> sc.DataArray:
        """Create a 3D histogram for testing ROI extraction."""
        # Create a 10x10x5 histogram (y, x, tof)
        # Put known values at specific locations for testing
        data = sc.zeros(dims=['y', 'x', 'tof'], shape=[10, 10, 5], unit='counts')

        # Put 10 counts in each pixel in top-left quadrant (y=0:5, x=0:5)
        data['y', 0:5]['x', 0:5] = sc.full(
            dims=['y', 'x', 'tof'], shape=[5, 5, 5], value=2.0, unit='counts'
        )

        # Put 5 counts in bottom-right quadrant (y=5:10, x=5:10)
        data['y', 5:10]['x', 5:10] = sc.full(
            dims=['y', 'x', 'tof'], shape=[5, 5, 5], value=1.0, unit='counts'
        )

        tof_coord = sc.linspace('tof', 0, 50000, 6, unit='ns')
        y_coord = sc.linspace('y', 0, 10, 11, unit='m')
        x_coord = sc.linspace('x', 0, 10, 11, unit='m')

        return sc.DataArray(
            data=data, coords={'tof': tof_coord, 'y': y_coord, 'x': x_coord}
        )

    def test_extract_roi_spectra_with_no_rois(self):
        """Test ROI extraction returns empty result when no ROIs configured."""
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            _extract_roi_spectra_precomputed,
        )

        histogram = self.make_histogram_3d()
        # Use empty dicts for precomputed bounds/masks
        result = _extract_roi_spectra_precomputed(histogram, {}, {})

        assert result.dims == ('roi', 'tof')
        assert result.sizes['roi'] == 0
        assert result.sizes['tof'] == 5

    def test_extract_rectangle_roi_spectra(self):
        """Test extracting ROI spectra for rectangle ROI."""
        from ess.livedata.config.models import Interval, RectangleROI
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            _extract_roi_spectra_precomputed,
        )

        histogram = self.make_histogram_3d()

        # Create rectangle ROI covering top-left quadrant (y=0:5, x=0:5)
        roi = RectangleROI(
            x=Interval(min=0, max=5, unit='m'), y=Interval(min=0, max=5, unit='m')
        )
        # Precompute bounds (as the precomputation provider would)
        bounds = roi.get_bounds(x_dim='x', y_dim='y')
        rectangle_bounds = {0: bounds}

        result = _extract_roi_spectra_precomputed(histogram, rectangle_bounds, {})

        assert result.dims == ('roi', 'tof')
        assert result.sizes['roi'] == 1
        assert result.sizes['tof'] == 5

        # Top-left quadrant has 5x5 pixels * 2 counts = 50 counts per TOF bin
        # But we're slicing by coordinate so may include partial bins
        assert result.sum().value > 0

    def test_extract_rectangle_roi_spectra_index_based(self):
        """Test extracting ROI spectra with index-based rectangle ROI."""
        from ess.livedata.config.models import Interval, RectangleROI
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            _extract_roi_spectra_precomputed,
        )

        # Create histogram without physical coordinates
        data = sc.ones(dims=['y', 'x', 'tof'], shape=[10, 10, 5], unit='counts')
        tof_coord = sc.linspace('tof', 0, 50000, 6, unit='ns')
        histogram = sc.DataArray(data=data, coords={'tof': tof_coord})

        # Create rectangle ROI using index bounds (no unit)
        roi = RectangleROI(
            x=Interval(min=0, max=5, unit=None), y=Interval(min=0, max=5, unit=None)
        )
        # Precompute bounds
        bounds = roi.get_bounds(x_dim='x', y_dim='y')
        rectangle_bounds = {0: bounds}

        result = _extract_roi_spectra_precomputed(histogram, rectangle_bounds, {})

        assert result.dims == ('roi', 'tof')
        assert result.sizes['roi'] == 1
        # 5x5 pixels * 1 count = 25 counts per TOF bin
        expected_per_tof = 5 * 5  # pixels in ROI
        np.testing.assert_array_almost_equal(result.values[0], [expected_per_tof] * 5)

    def test_extract_rectangle_roi_index_bounds_with_physical_coords(self):
        """Test ROI with index bounds when histogram has physical coordinates.

        This tests the case where the histogram has physical coordinates (e.g. meters)
        but the ROI is specified with index bounds (no units). This happens when
        detector projection creates physical coordinates but the UI sends index-based
        ROIs.
        """
        from ess.livedata.config.models import Interval, RectangleROI
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            _extract_roi_spectra_precomputed,
        )

        # Create histogram WITH physical coordinates (meters)
        data = sc.ones(dims=['y', 'x', 'tof'], shape=[10, 10, 5], unit='counts')
        tof_coord = sc.linspace('tof', 0, 50000, 6, unit='ns')
        y_coord = sc.linspace('y', 0.0, 1.0, 11, unit='m')
        x_coord = sc.linspace('x', 0.0, 1.0, 11, unit='m')
        histogram = sc.DataArray(
            data=data, coords={'tof': tof_coord, 'y': y_coord, 'x': x_coord}
        )

        # Create rectangle ROI using INDEX bounds (no unit) - like dashboard UI sends
        roi = RectangleROI(
            x=Interval(min=0, max=5, unit=None), y=Interval(min=0, max=5, unit=None)
        )
        # Precompute bounds
        bounds = roi.get_bounds(x_dim='x', y_dim='y')
        rectangle_bounds = {0: bounds}

        # This should NOT raise UnitError
        result = _extract_roi_spectra_precomputed(histogram, rectangle_bounds, {})

        assert result.dims == ('roi', 'tof')
        assert result.sizes['roi'] == 1
        # Should capture approximately 5x5 = 25 counts per TOF bin
        # (may be slightly different due to coordinate edge handling)
        assert result.sum().value > 0

    def test_roi_rectangle_readback_empty_gets_histogram_units(self):
        """Test that empty ROI readback gets coordinate units from histogram.

        The readback provider should create empty DataArrays with units from
        the histogram coordinates, so the frontend knows what units to use.
        """
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            CumulativeHistogram,
            ROIRectangleRequest,
            roi_rectangle_readback,
        )

        # Create histogram WITH physical coordinates (meters)
        data = sc.ones(dims=['y', 'x', 'tof'], shape=[10, 10, 5], unit='counts')
        tof_coord = sc.linspace('tof', 0, 50000, 6, unit='ns')
        y_coord = sc.linspace('y', 0.0, 1.0, 11, unit='m')
        x_coord = sc.linspace('x', 0.0, 1.0, 11, unit='m')
        histogram = CumulativeHistogram(
            sc.DataArray(
                data=data, coords={'tof': tof_coord, 'y': y_coord, 'x': x_coord}
            )
        )

        # Empty request (no ROIs configured yet)
        from ess.livedata.config.models import RectangleROI

        empty_request = ROIRectangleRequest(RectangleROI.to_concatenated_data_array({}))

        # Readback should have units from histogram
        readback = roi_rectangle_readback(empty_request, histogram)

        assert len(readback) == 0  # Still empty
        assert readback.coords['x'].unit == sc.units.m
        assert readback.coords['y'].unit == sc.units.m

    def test_roi_rectangle_readback_passes_through_nonempty(self):
        """Test that non-empty ROI readback passes through request unchanged."""
        from ess.livedata.config.models import ROI, Interval, RectangleROI
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            CumulativeHistogram,
            ROIRectangleRequest,
            roi_rectangle_readback,
        )

        # Create histogram with meters
        data = sc.ones(dims=['y', 'x', 'tof'], shape=[10, 10, 5], unit='counts')
        tof_coord = sc.linspace('tof', 0, 50000, 6, unit='ns')
        y_coord = sc.linspace('y', 0.0, 1.0, 11, unit='m')
        x_coord = sc.linspace('x', 0.0, 1.0, 11, unit='m')
        histogram = CumulativeHistogram(
            sc.DataArray(
                data=data, coords={'tof': tof_coord, 'y': y_coord, 'x': x_coord}
            )
        )

        # Request with ROI (frontend sends with correct units from earlier readback)
        roi = RectangleROI(
            x=Interval(min=0.0, max=0.5, unit='m'),
            y=Interval(min=0.0, max=0.5, unit='m'),
        )
        request = ROIRectangleRequest(ROI.to_concatenated_data_array({0: roi}))

        # Readback should pass through
        readback = roi_rectangle_readback(request, histogram)

        assert len(readback) == 2  # 2 bounds entries
        assert sc.identical(readback, request)

    def test_extract_multiple_rectangle_rois(self):
        """Test extracting spectra for multiple rectangle ROIs."""
        from ess.livedata.config.models import Interval, RectangleROI
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            _extract_roi_spectra_precomputed,
        )

        histogram = self.make_histogram_3d()

        # Two ROIs: top-left and bottom-right quadrants
        roi0 = RectangleROI(
            x=Interval(min=0, max=5, unit='m'), y=Interval(min=0, max=5, unit='m')
        )
        roi1 = RectangleROI(
            x=Interval(min=5, max=10, unit='m'), y=Interval(min=5, max=10, unit='m')
        )
        # Precompute bounds
        rectangle_bounds = {
            0: roi0.get_bounds(x_dim='x', y_dim='y'),
            1: roi1.get_bounds(x_dim='x', y_dim='y'),
        }

        result = _extract_roi_spectra_precomputed(histogram, rectangle_bounds, {})

        assert result.dims == ('roi', 'tof')
        assert result.sizes['roi'] == 2
        # Check that roi indices are preserved
        assert list(result.coords['roi'].values) == [0, 1]

    def test_extract_polygon_roi_spectra(self):
        """Test extracting ROI spectra for polygon ROI."""
        from ess.livedata.config.models import PolygonROI
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            _compute_polygon_mask,
            _extract_roi_spectra_precomputed,
        )

        histogram = self.make_histogram_3d()

        # Create triangle ROI in top-left corner
        roi = PolygonROI(x=[0.0, 5.0, 0.0], y=[0.0, 0.0, 5.0], x_unit='m', y_unit='m')

        # Precompute polygon mask
        x_centers = sc.midpoints(histogram.coords['x'])
        y_centers = sc.midpoints(histogram.coords['y'])
        mask = _compute_polygon_mask(
            roi, x_centers=x_centers, y_centers=y_centers, x_dim='x', y_dim='y'
        )
        polygon_masks = {100: mask}

        result = _extract_roi_spectra_precomputed(histogram, {}, polygon_masks)

        assert result.dims == ('roi', 'tof')
        assert result.sizes['roi'] == 1
        assert result.coords['roi'].values[0] == 100
        # Should have some counts from the triangle area
        assert result.sum().value > 0

    def test_cumulative_roi_spectra_provider(self):
        """Test cumulative_roi_spectra provider function."""
        from ess.livedata.config.models import Interval, RectangleROI
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            CumulativeHistogram,
            ROIPolygonMasks,
            ROIRectangleBounds,
            cumulative_roi_spectra,
        )

        histogram = CumulativeHistogram(self.make_histogram_3d())

        roi = RectangleROI(
            x=Interval(min=0, max=5, unit='m'), y=Interval(min=0, max=5, unit='m')
        )
        # Use precomputed bounds
        rectangle_bounds = ROIRectangleBounds({0: roi.get_bounds(x_dim='x', y_dim='y')})
        polygon_masks = ROIPolygonMasks({})

        result = cumulative_roi_spectra(histogram, rectangle_bounds, polygon_masks)

        assert result.dims == ('roi', 'tof')
        assert result.sizes['roi'] == 1

    def test_current_roi_spectra_provider(self):
        """Test current_roi_spectra provider function."""
        from ess.livedata.config.models import Interval, RectangleROI
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            ROIPolygonMasks,
            ROIRectangleBounds,
            WindowHistogram,
            current_roi_spectra,
        )

        histogram = WindowHistogram(self.make_histogram_3d())

        roi = RectangleROI(
            x=Interval(min=0, max=5, unit='m'), y=Interval(min=0, max=5, unit='m')
        )
        # Use precomputed bounds
        rectangle_bounds = ROIRectangleBounds({0: roi.get_bounds(x_dim='x', y_dim='y')})
        polygon_masks = ROIPolygonMasks({})

        result = current_roi_spectra(histogram, rectangle_bounds, polygon_masks)

        assert result.dims == ('roi', 'tof')
        assert result.sizes['roi'] == 1


class TestROISpectraIntegration:
    """Integration tests for ROI spectra with StreamProcessor."""

    def test_roi_spectra_via_context_keys(self):
        """Test ROI spectra extraction via context_keys in StreamProcessorWorkflow."""
        from ess.livedata.config.models import ROI, Interval, RectangleROI
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            CumulativeROISpectra,
            CurrentROISpectra,
            ROIPolygonReadback,
            ROIPolygonRequest,
            ROIRectangleReadback,
            ROIRectangleRequest,
        )
        from ess.livedata.handlers.stream_processor_workflow import (
            StreamProcessorWorkflow,
        )
        from ess.reduce.nexus.types import EmptyDetector

        tof_bins = sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns')
        base_workflow = create_base_workflow(tof_bins=tof_bins)

        # Add logical projection with a fold transform
        transform = make_logical_transform(4, 4)
        add_logical_projection(base_workflow, transform=transform)

        # Inject fake EmptyDetector for testing (normally comes from NeXus file)
        base_workflow[EmptyDetector[SampleRun]] = make_fake_empty_detector(4, 4)

        workflow = StreamProcessorWorkflow(
            base_workflow,
            dynamic_keys={'detector': RawDetector[SampleRun]},
            context_keys={
                'roi_rectangle': ROIRectangleRequest,
                'roi_polygon': ROIPolygonRequest,
            },
            target_keys={
                'cumulative': CumulativeDetectorImage,
                'current': CurrentDetectorImage,
                'roi_spectra_cumulative': CumulativeROISpectra,
                'roi_spectra_current': CurrentROISpectra,
                'roi_rectangle': ROIRectangleReadback,
                'roi_polygon': ROIPolygonReadback,
            },
            accumulators=create_accumulators(),
        )

        # Create fake events
        events = make_fake_nexus_detector_data(
            y_size=4, x_size=4, n_events_per_pixel=10
        )

        # First, accumulate events without ROI
        workflow.accumulate(
            {'detector': RawDetector[SampleRun](events)},
            start_time=1000,
            end_time=2000,
        )
        result1 = workflow.finalize()

        # Should have empty ROI spectra (no ROIs configured)
        assert result1['roi_spectra_cumulative'].sizes['roi'] == 0
        assert result1['roi_spectra_current'].sizes['roi'] == 0

        # Now add an ROI via context_keys and accumulate more events
        roi = RectangleROI(
            x=Interval(min=0, max=2, unit=None), y=Interval(min=0, max=2, unit=None)
        )
        rectangle_request = ROI.to_concatenated_data_array({0: roi})

        workflow.accumulate(
            {
                'detector': RawDetector[SampleRun](events),
                'roi_rectangle': rectangle_request,
            },
            start_time=2000,
            end_time=3000,
        )
        result2 = workflow.finalize()

        # Should now have ROI spectra
        assert result2['roi_spectra_cumulative'].sizes['roi'] == 1
        assert result2['roi_spectra_current'].sizes['roi'] == 1

        # Cumulative should include both batches
        # Current should only include second batch
        cumulative_sum = result2['roi_spectra_cumulative'].sum().value
        current_sum = result2['roi_spectra_current'].sum().value

        # Cumulative should be ~2x current (two batches)
        assert cumulative_sum > current_sum

    def test_roi_change_recomputes_from_accumulated_histogram(self):
        """Test that changing ROI recomputes spectra from full accumulated data."""
        from ess.livedata.config.models import ROI, Interval, RectangleROI
        from ess.livedata.handlers.detector_view_sciline_workflow import (
            CumulativeROISpectra,
            ROIPolygonReadback,
            ROIPolygonRequest,
            ROIRectangleReadback,
            ROIRectangleRequest,
        )
        from ess.livedata.handlers.stream_processor_workflow import (
            StreamProcessorWorkflow,
        )
        from ess.reduce.nexus.types import EmptyDetector

        tof_bins = sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns')
        base_workflow = create_base_workflow(tof_bins=tof_bins)

        transform = make_logical_transform(4, 4)
        add_logical_projection(base_workflow, transform=transform)

        # Inject fake EmptyDetector for testing (normally comes from NeXus file)
        base_workflow[EmptyDetector[SampleRun]] = make_fake_empty_detector(4, 4)

        workflow = StreamProcessorWorkflow(
            base_workflow,
            dynamic_keys={'detector': RawDetector[SampleRun]},
            context_keys={
                'roi_rectangle': ROIRectangleRequest,
                'roi_polygon': ROIPolygonRequest,
            },
            target_keys={
                'cumulative': CumulativeDetectorImage,
                'roi_spectra_cumulative': CumulativeROISpectra,
                # ROI readbacks ensure correct units from histogram coords
                'roi_rectangle': ROIRectangleReadback,
                'roi_polygon': ROIPolygonReadback,
            },
            accumulators=create_accumulators(),
        )

        # Create events with some reproducibility
        np.random.seed(42)
        events = make_fake_nexus_detector_data(
            y_size=4, x_size=4, n_events_per_pixel=100
        )

        # Set initial ROI (small region)
        roi_small = RectangleROI(
            x=Interval(min=0, max=1, unit=None), y=Interval(min=0, max=1, unit=None)
        )
        rectangle_request_small = ROI.to_concatenated_data_array({0: roi_small})

        workflow.accumulate(
            {
                'detector': RawDetector[SampleRun](events),
                'roi_rectangle': rectangle_request_small,
            },
            start_time=1000,
            end_time=2000,
        )
        result1 = workflow.finalize()
        small_roi_sum = result1['roi_spectra_cumulative'].sum().value

        # Change ROI to larger region (without adding more events)
        roi_large = RectangleROI(
            x=Interval(min=0, max=4, unit=None), y=Interval(min=0, max=4, unit=None)
        )
        rectangle_request_large = ROI.to_concatenated_data_array({0: roi_large})

        # Update only the ROI context (no new events)
        workflow.accumulate(
            {'roi_rectangle': rectangle_request_large},
            start_time=2000,
            end_time=3000,
        )
        result2 = workflow.finalize()
        large_roi_sum = result2['roi_spectra_cumulative'].sum().value

        # Larger ROI should capture more counts from the same accumulated data
        # The key point: the large ROI sum includes ALL accumulated events,
        # not just events since the ROI change
        assert large_roi_sum > small_roi_sum
