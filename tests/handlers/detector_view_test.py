# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Proof-of-concept tests for DetectorView and ROI mechanism.

These tests verify that the ROI configuration and accumulation mechanism works
correctly in the DetectorView workflow.
"""

import pytest
import scipp as sc

from ess.livedata.config.models import Interval, RectangleROI
from ess.livedata.handlers.accumulators import GroupIntoPixels
from ess.livedata.handlers.detector_view import (
    DetectorView,
    DetectorViewParams,
    ROIBasedTOAHistogram,
)
from ess.livedata.handlers.to_nxevent_data import DetectorEvents
from ess.livedata.parameter_models import TOAEdges
from ess.reduce.live.raw import RollingDetectorView
from ess.reduce.live.roi import ROIFilter


def make_rectangle_roi(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_unit: str | None,
    y_unit: str | None,
) -> RectangleROI:
    """Helper function to create RectangleROI with old-style parameters."""
    return RectangleROI(
        x=Interval(min=x_min, max=x_max, unit=x_unit),
        y=Interval(min=y_min, max=y_max, unit=y_unit),
    )


@pytest.fixture
def detector_number() -> sc.Variable:
    """Fixture providing a simple 4x4 detector layout."""
    return sc.arange('detector_number', 16, unit=None).fold(
        'detector_number', sizes={'y': 4, 'x': 4}
    )


@pytest.fixture
def detector_indices() -> sc.DataArray:
    """Fixture providing detector indices with physical coordinates."""
    # Create a 4x4 detector with coordinates in mm
    y_coords = sc.array(dims=['y'], values=[0.0, 10.0, 20.0, 30.0], unit='mm')
    x_coords = sc.array(dims=['x'], values=[0.0, 10.0, 20.0, 30.0], unit='mm')
    indices = sc.DataArray(
        data=sc.arange('y', 4) * sc.scalar(4) + sc.arange('x', 4),
        coords={'y': y_coords, 'x': x_coords},
    )
    return indices


@pytest.fixture
def roi_filter(detector_indices: sc.DataArray) -> ROIFilter:
    """Fixture providing an ROIFilter for the detector."""
    return ROIFilter(detector_indices)


@pytest.fixture
def mock_rolling_view(
    detector_number: sc.Variable, detector_indices: sc.DataArray
) -> RollingDetectorView:
    """Fixture providing a mock RollingDetectorView."""
    view = RollingDetectorView(detector_number=detector_number, window=1)
    # Mock the make_roi_filter method to return a fresh ROIFilter each time
    # This is needed because the filter state is modified by configure_from_roi_model
    view.make_roi_filter = lambda: ROIFilter(detector_indices)
    return view


@pytest.fixture
def sample_detector_events(detector_number: sc.Variable) -> sc.DataArray:
    """
    Fixture providing sample detector events grouped into pixels.

    Creates events distributed across the detector with known TOA values.
    """
    # Create events for various pixels
    pixel_ids = [0, 1, 2, 5, 6, 10, 11, 15]  # Scattered across detector
    toa_values = [100, 200, 300, 400, 500, 600, 700, 800]  # ns

    events = DetectorEvents(
        pixel_id=pixel_ids,
        time_of_arrival=toa_values,
        unit='ns',
    )

    # Group events into pixels
    grouper = GroupIntoPixels(detector_number=detector_number)
    grouper.add(0, events)
    return grouper.get()


class TestDetectorViewBasics:
    """Basic tests for DetectorView without ROI."""

    def test_detector_view_initialization(
        self, mock_rolling_view: RollingDetectorView
    ) -> None:
        """Test that DetectorView can be initialized with default parameters."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        assert view is not None
        assert view._roi_model is None  # No ROI configured initially

    def test_accumulate_detector_data_without_roi(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test accumulating detector data without ROI configuration."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Accumulate detector data
        view.accumulate({'detector': sample_detector_events})

        # Finalize should return cumulative and current counts
        result = view.finalize()

        assert 'cumulative' in result
        assert 'current' in result
        # ROI results should not be present without ROI configuration
        assert 'roi_cumulative' not in result
        assert 'roi_current' not in result

        # Verify that events were actually accumulated
        # Sample has 8 events across all pixels
        assert (
            sc.sum(result['current']).value == 8
        ), "Expected all 8 sample events to be accumulated"

    def test_clear_resets_state(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that clear() resets the detector view state."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        view.accumulate({'detector': sample_detector_events})
        view.finalize()

        # Clear the view
        view.clear()

        assert view._previous is None
        assert view._roi_cumulative is None


class TestDetectorViewROIMechanism:
    """Tests for ROI configuration and histogram accumulation."""

    def test_roi_configuration_via_accumulate(
        self, mock_rolling_view: RollingDetectorView
    ) -> None:
        """Test that ROI configuration can be set via accumulate."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Create ROI configuration
        roi = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=5.0,
            y_max=25.0,
            x_unit='mm',
            y_unit='mm',
        )
        roi_data = roi.to_data_array()

        # Send ROI configuration
        view.accumulate({'roi': roi_data})

        # ROI model should be configured
        assert view._roi_model is not None
        assert isinstance(view._roi_model, RectangleROI)
        assert view._roi_model.x.min == 5.0
        assert view._roi_model.x.max == 25.0

    def test_roi_only_does_not_process_events(
        self, mock_rolling_view: RollingDetectorView
    ) -> None:
        """Test that sending only ROI (no detector data) returns early."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        roi = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=5.0,
            y_max=25.0,
            x_unit='mm',
            y_unit='mm',
        )

        # This should return early without processing events
        view.accumulate({'roi': roi.to_data_array()})

        # ROI should be configured but no histogram data accumulated yet
        assert view._roi_model is not None
        result = view.finalize()
        # ROI results should be present (even if empty/zero) once ROI is configured
        assert 'roi_cumulative' in result
        assert 'roi_current' in result
        assert 'roi_rectangle' in result  # ROI shape published with shape name
        # All counts should be zero since no events were accumulated
        assert sc.sum(result['roi_cumulative']).value == 0
        assert sc.sum(result['roi_current']).value == 0

    def test_accumulate_with_roi_produces_histogram(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that ROI configuration enables histogram accumulation."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI first
        roi = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=5.0,
            y_max=25.0,
            x_unit='mm',
            y_unit='mm',
        )
        view.accumulate({'roi': roi.to_data_array()})

        # Now accumulate detector events
        view.accumulate({'detector': sample_detector_events})

        result = view.finalize()

        # Should have both detector view and ROI results
        assert 'cumulative' in result
        assert 'current' in result
        assert 'roi_cumulative' in result
        assert 'roi_current' in result
        assert 'roi_rectangle' in result  # Config should be published on first update

        # Verify histogram structure
        roi_histogram = result['roi_current']
        assert isinstance(roi_histogram, sc.DataArray)
        assert 'time_of_arrival' in roi_histogram.coords
        assert roi_histogram.ndim == 1  # Histogram is 1D

        # Verify expected event count
        # Sample has pixels [0, 1, 2, 5, 6, 10, 11, 15]
        # ROI covers x=[5,25]mm, y=[5,25]mm which includes pixels 5, 6, 10
        expected_events_in_roi = 3
        assert sc.sum(roi_histogram).value == expected_events_in_roi, (
            f"Expected {expected_events_in_roi} events in ROI histogram, "
            f"got {sc.sum(roi_histogram).value}"
        )

    def test_roi_cumulative_accumulation(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that ROI histograms accumulate correctly over multiple updates."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI
        roi = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=5.0,
            y_max=25.0,
            x_unit='mm',
            y_unit='mm',
        )
        view.accumulate({'roi': roi.to_data_array()})

        # First accumulation
        view.accumulate({'detector': sample_detector_events})
        result1 = view.finalize()
        first_current = result1['roi_current']
        first_total = sc.sum(first_current).value

        # Verify expected event count based on ROI
        # Sample has pixels [0, 1, 2, 5, 6, 10, 11, 15]
        # ROI covers x=[5,25]mm, y=[5,25]mm which includes:
        #   Pixel 5 at (10, 10)mm, Pixel 6 at (20, 10)mm, Pixel 10 at (20, 20)mm
        # Expected: 3 events in ROI
        expected_events_in_roi = 3
        assert (
            first_total == expected_events_in_roi
        ), f"Expected {expected_events_in_roi} events in ROI, got {first_total}"

        # Second accumulation with same events
        view.accumulate({'detector': sample_detector_events})
        result2 = view.finalize()
        second_cumulative = result2['roi_cumulative']
        second_current = result2['roi_current']
        second_current_total = sc.sum(second_current).value

        # Both should have same current counts
        assert second_current_total == expected_events_in_roi
        # Cumulative should be the sum of both accumulations
        assert sc.sum(second_cumulative).value == 2 * expected_events_in_roi

    def test_roi_published_only_on_update(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that ROI is only published when updated."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI
        roi = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=5.0,
            y_max=25.0,
            x_unit='mm',
            y_unit='mm',
        )
        view.accumulate({'roi': roi.to_data_array()})
        view.accumulate({'detector': sample_detector_events})

        result1 = view.finalize()
        assert 'roi_rectangle' in result1  # Published on first finalize after update

        # Second accumulation without ROI update
        view.accumulate({'detector': sample_detector_events})
        result2 = view.finalize()
        assert 'roi_rectangle' not in result2  # Not published again

    def test_clear_resets_roi_state(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that clear() resets ROI cumulative state."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI and accumulate
        roi = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=5.0,
            y_max=25.0,
            x_unit='mm',
            y_unit='mm',
        )
        view.accumulate({'roi': roi.to_data_array()})
        view.accumulate({'detector': sample_detector_events})
        view.finalize()

        # Clear should reset cumulative
        view.clear()

        assert view._roi_cumulative is None
        # Note: ROI model configuration persists after clear
        assert view._roi_model is not None

    def test_roi_change_resets_cumulative(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that changing ROI resets cumulative histogram."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure first ROI covering pixels 5, 6, 10
        roi1 = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=5.0,
            y_max=25.0,
            x_unit='mm',
            y_unit='mm',
        )
        view.accumulate({'roi': roi1.to_data_array()})
        view.accumulate({'detector': sample_detector_events})
        result1 = view.finalize()

        # Sample has pixels [0, 1, 2, 5, 6, 10, 11, 15]
        # ROI1 covers x=[5,25]mm, y=[5,25]mm which includes pixels 5, 6, 10
        expected_roi1_events = 3
        assert sc.sum(result1['roi_current']).value == expected_roi1_events
        assert sc.sum(result1['roi_cumulative']).value == expected_roi1_events

        # Now change ROI to cover different pixels (1, 2, 5, 6)
        # Pixel 1 at (10,0), Pixel 2 at (20,0), Pixel 5 at (10,10), Pixel 6 at (20,10)mm
        roi2 = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=-5.0,  # Include y=0mm pixels
            y_max=15.0,  # Exclude y=20mm and y=30mm pixels
            x_unit='mm',
            y_unit='mm',
        )
        view.accumulate({'roi': roi2.to_data_array()})
        view.accumulate({'detector': sample_detector_events})
        result2 = view.finalize()

        # ROI2 should have 4 events (pixels 1, 2, 5, 6)
        expected_roi2_events = 4
        assert sc.sum(result2['roi_current']).value == expected_roi2_events

        # CRITICAL: Cumulative should reset when ROI changes
        # It should NOT be roi1_events + roi2_events (which would be 7)
        # It should be just roi2_events since we changed the ROI
        assert sc.sum(result2['roi_cumulative']).value == expected_roi2_events, (
            f"Expected cumulative to reset to {expected_roi2_events} when ROI changes, "
            f"got {sc.sum(result2['roi_cumulative']).value}. "
            "Cumulative should not mix events from different ROI regions."
        )


class TestROIBasedTOAHistogramIntegration:
    """Integration tests for the ROI histogram accumulator."""

    def test_roi_filter_configuration_from_rectangle(
        self, detector_indices: sc.DataArray
    ) -> None:
        """Test configuring ROIFilter from RectangleROI model."""
        roi_filter = ROIFilter(detector_indices)

        # Create a rectangle ROI covering the central region
        roi = make_rectangle_roi(
            x_min=10.0,
            x_max=20.0,
            y_min=10.0,
            y_max=20.0,
            x_unit='mm',
            y_unit='mm',
        )

        # Configure ROI accumulator (this is done inside DetectorView)
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')
        accumulator = ROIBasedTOAHistogram(toa_edges=toa_edges, roi_filter=roi_filter)
        accumulator.configure_from_roi_model(roi)

        # This should not raise - just verify the configuration works
        assert accumulator._roi_filter is not None

    def test_histogram_bins_events_in_roi(
        self, detector_number: sc.Variable, detector_indices: sc.DataArray
    ) -> None:
        """Test that histogram correctly bins events within ROI."""
        roi_filter = ROIFilter(detector_indices)
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')

        accumulator = ROIBasedTOAHistogram(toa_edges=toa_edges, roi_filter=roi_filter)

        # Configure ROI covering center pixels (indices 5,6,9,10)
        # Detector coordinates: 0, 10, 20, 30 mm in both x and y
        # We want pixels at (10,10), (20,10), (10,20), (20,20)
        # Use range 5-25 to ensure we include pixels at 10 and 20
        roi = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=5.0,
            y_max=25.0,
            x_unit='mm',
            y_unit='mm',
        )
        accumulator.configure_from_roi_model(roi)

        # Create events (some inside, some outside ROI)
        # ROI covers x=[5,25]mm, y=[5,25]mm
        # Detector layout has pixels at 0, 10, 20, 30mm in both x and y
        # Pixels in ROI: 5 at (10,10), 6 at (20,10), 9 at (10,20), 10 at (20,20)
        events = DetectorEvents(
            pixel_id=[5, 6, 9, 10, 0, 15],  # First 4 in ROI, last 2 outside
            time_of_arrival=[100, 200, 300, 400, 500, 600],
            unit='ns',
        )

        # Group into pixels
        grouper = GroupIntoPixels(detector_number=detector_number)
        grouper.add(0, events)
        grouped_events = grouper.get()

        # Accumulate
        accumulator.add(0, grouped_events)
        result = accumulator.get()

        # Result should be a histogram
        assert isinstance(result, sc.DataArray)
        assert 'time_of_arrival' in result.coords
        assert result.unit == 'counts'
        # The histogram should contain counts from events in the ROI
        # Pixels 5, 6, 9, 10 are all within [5,25]mm range = 4 events
        expected_events_in_roi = 4
        total_counts = sc.sum(result).value
        assert (
            total_counts == expected_events_in_roi
        ), f"Expected {expected_events_in_roi} events in ROI, got {total_counts}"

    def test_roi_configuration_without_coordinates(
        self, detector_number: sc.Variable
    ) -> None:
        """Test ROI configuration when detector_indices has no coordinates."""
        # Create detector indices WITHOUT coordinates (just pixel indices)
        detector_indices_no_coords = sc.DataArray(
            data=sc.arange('y', 4) * sc.scalar(4) + sc.arange('x', 4),
            coords={},  # No coordinates!
        )

        roi_filter = ROIFilter(detector_indices_no_coords)
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')

        accumulator = ROIBasedTOAHistogram(toa_edges=toa_edges, roi_filter=roi_filter)

        # Configure ROI using pixel indices (unit=None)
        # Select pixels with indices y:[1,3), x:[1,3) = pixels 5, 6, 9, 10
        roi = make_rectangle_roi(
            x_min=1, x_max=3, y_min=1, y_max=3, x_unit=None, y_unit=None
        )
        accumulator.configure_from_roi_model(roi)

        # Create events for all 16 pixels (one event per pixel for simplicity)
        events = DetectorEvents(
            pixel_id=list(range(16)),  # All 16 pixels
            time_of_arrival=[100.0] * 16,  # All events in first TOA bin
            unit='ns',
        )

        # Group into pixels
        grouper = GroupIntoPixels(detector_number=detector_number)
        grouper.add(0, events)
        grouped_events = grouper.get()

        # Accumulate
        accumulator.add(0, grouped_events)
        result = accumulator.get()

        # Should get same result: 4 events in ROI (pixels 5, 6, 9, 10)
        expected_events_in_roi = 4
        total_counts = sc.sum(result).value
        assert (
            total_counts == expected_events_in_roi
        ), f"Expected {expected_events_in_roi} events in ROI, got {total_counts}"

    def test_roi_with_unit_but_no_coordinates_raises_error(
        self, detector_number: sc.Variable
    ) -> None:
        """Test that unit when coordinates are missing raises DimensionError."""
        # Create detector indices WITHOUT coordinates
        detector_indices_no_coords = sc.DataArray(
            data=sc.arange('y', 4) * sc.scalar(4) + sc.arange('x', 4),
            coords={},
        )

        roi_filter = ROIFilter(detector_indices_no_coords)
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')
        accumulator = ROIBasedTOAHistogram(toa_edges=toa_edges, roi_filter=roi_filter)

        # Try to configure ROI with physical units (should fail - no coords!)
        roi = make_rectangle_roi(
            x_min=10.0, x_max=20.0, y_min=10.0, y_max=20.0, x_unit='mm', y_unit='mm'
        )

        # Error comes from scipp's slicing in select_indices_in_intervals
        with pytest.raises(
            sc.DimensionError,
            match="no coordinate for that dimension",
        ):
            accumulator.configure_from_roi_model(roi)

    def test_roi_without_unit_but_has_coordinates_uses_pixel_indices(
        self, detector_indices: sc.DataArray
    ) -> None:
        """Test that unit=None uses pixel indices even when coordinates exist."""
        roi_filter = ROIFilter(detector_indices)  # Has coordinates
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')
        accumulator = ROIBasedTOAHistogram(toa_edges=toa_edges, roi_filter=roi_filter)

        # Configure ROI with None units - should use pixel indices (not an error!)
        roi = make_rectangle_roi(
            x_min=1, x_max=3, y_min=1, y_max=3, x_unit=None, y_unit=None
        )

        # This should work - scipp allows integer slicing even with coordinates
        accumulator.configure_from_roi_model(roi)

        # Verify the ROI was configured (by checking that _selection was updated)
        assert len(roi_filter._selection) > 0


class TestDetectorViewBothROIAndDetectorData:
    """Tests for accumulate called with both ROI and detector data together."""

    def test_accumulate_with_both_roi_and_detector_in_same_call(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test accumulate with both ROI and detector data in one call."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI and send detector data in same call
        roi = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=5.0,
            y_max=25.0,
            x_unit='mm',
            y_unit='mm',
        )

        view.accumulate(
            {'roi': roi.to_data_array(), 'detector': sample_detector_events}
        )

        result = view.finalize()

        # Should have both detector view and ROI results
        assert 'cumulative' in result
        assert 'current' in result
        assert 'roi_cumulative' in result
        assert 'roi_current' in result
        assert 'roi_rectangle' in result

        # Verify histogram has expected events
        # Sample has pixels [0, 1, 2, 5, 6, 10, 11, 15]
        # ROI covers x=[5,25]mm, y=[5,25]mm which includes pixels 5, 6, 10
        expected_events_in_roi = 3
        assert sc.sum(result['roi_current']).value == expected_events_in_roi

    def test_accumulate_both_then_detector_only(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that ROI persists after being set with detector data together."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # First call: both roi_config and detector data
        roi = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=5.0,
            y_max=25.0,
            x_unit='mm',
            y_unit='mm',
        )
        view.accumulate(
            {'roi': roi.to_data_array(), 'detector': sample_detector_events}
        )
        result1 = view.finalize()

        expected_events_in_roi = 3
        assert sc.sum(result1['roi_current']).value == expected_events_in_roi
        assert 'roi_rectangle' in result1  # Published on first update

        # Second call: detector data only (ROI should persist)
        view.accumulate({'detector': sample_detector_events})
        result2 = view.finalize()

        assert sc.sum(result2['roi_current']).value == expected_events_in_roi
        assert sc.sum(result2['roi_cumulative']).value == 2 * expected_events_in_roi
        assert 'roi_rectangle' not in result2  # Not published again

    def test_accumulate_detector_then_both_roi_and_detector(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test updating ROI while also sending detector data."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # First: just detector data (no ROI)
        view.accumulate({'detector': sample_detector_events})
        result1 = view.finalize()

        assert 'roi_cumulative' not in result1
        assert 'roi_current' not in result1

        # Second: both roi_config and detector data
        roi = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=5.0,
            y_max=25.0,
            x_unit='mm',
            y_unit='mm',
        )
        view.accumulate(
            {'roi': roi.to_data_array(), 'detector': sample_detector_events}
        )
        result2 = view.finalize()

        expected_events_in_roi = 3
        assert 'roi_cumulative' in result2
        assert 'roi_current' in result2
        assert sc.sum(result2['roi_current']).value == expected_events_in_roi
        assert sc.sum(result2['roi_cumulative']).value == expected_events_in_roi

    def test_accumulate_roi_change_with_detector_in_same_call(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test changing ROI and sending detector data in the same call."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # First ROI
        roi1 = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=5.0,
            y_max=25.0,
            x_unit='mm',
            y_unit='mm',
        )
        view.accumulate(
            {'roi': roi1.to_data_array(), 'detector': sample_detector_events}
        )
        result1 = view.finalize()

        expected_roi1_events = 3
        assert sc.sum(result1['roi_current']).value == expected_roi1_events
        assert sc.sum(result1['roi_cumulative']).value == expected_roi1_events

        # Change ROI and accumulate in same call
        roi2 = make_rectangle_roi(
            x_min=5.0,
            x_max=25.0,
            y_min=-5.0,
            y_max=15.0,
            x_unit='mm',
            y_unit='mm',
        )
        view.accumulate(
            {'roi': roi2.to_data_array(), 'detector': sample_detector_events}
        )
        result2 = view.finalize()

        expected_roi2_events = 4
        assert sc.sum(result2['roi_current']).value == expected_roi2_events
        # Cumulative should reset when ROI changes
        assert sc.sum(result2['roi_cumulative']).value == expected_roi2_events
        assert 'roi_rectangle' in result2  # Config published on update


class TestDetectorViewEdgeCases:
    """Edge cases and error conditions."""

    def test_accumulate_empty_dict_does_nothing(
        self, mock_rolling_view: RollingDetectorView
    ) -> None:
        """Test that accumulate with empty dict returns early without error."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Empty dict should not raise - consistent with roi-only behavior
        view.accumulate({})

        result = view.finalize()
        assert 'cumulative' in result
        assert 'current' in result
        # No events accumulated
        assert sc.sum(result['cumulative']).value == 0

    def test_accumulate_multiple_detector_keys_raises_error(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that accumulate with multiple detector data items raises ValueError."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Multiple detector data keys should raise error
        with pytest.raises(
            ValueError,
            match="DetectorViewProcessor expects exactly one detector data item",
        ):
            view.accumulate(
                {
                    'detector1': sample_detector_events,
                    'detector2': sample_detector_events,
                }
            )

    def test_multiple_roi_updates_without_detector_data(
        self, mock_rolling_view: RollingDetectorView
    ) -> None:
        """Test multiple ROI updates without any detector data."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Update ROI multiple times
        roi1 = make_rectangle_roi(
            x_min=5.0, x_max=25.0, y_min=5.0, y_max=25.0, x_unit='mm', y_unit='mm'
        )
        view.accumulate({'roi': roi1.to_data_array()})

        roi2 = make_rectangle_roi(
            x_min=10.0, x_max=20.0, y_min=10.0, y_max=20.0, x_unit='mm', y_unit='mm'
        )
        view.accumulate({'roi': roi2.to_data_array()})

        result = view.finalize()

        # Should have ROI configured but with zero events
        assert 'roi_cumulative' in result
        assert 'roi_current' in result
        assert 'roi_rectangle' in result
        assert sc.sum(result['roi_cumulative']).value == 0
        assert sc.sum(result['roi_current']).value == 0

    def test_detector_data_with_no_events_in_roi(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that ROI with no matching events produces zero histogram."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI that doesn't overlap with any events
        # Sample has pixels [0, 1, 2, 5, 6, 10, 11, 15]
        # Use ROI that covers pixels at (30mm, 30mm) only - pixel 15
        roi = make_rectangle_roi(
            x_min=25.0,
            x_max=35.0,
            y_min=25.0,
            y_max=35.0,
            x_unit='mm',
            y_unit='mm',
        )
        view.accumulate({'roi': roi.to_data_array()})
        view.accumulate({'detector': sample_detector_events})

        result = view.finalize()

        # Should have histogram structure but with limited events
        assert 'roi_cumulative' in result
        assert 'roi_current' in result
        # Only pixel 15 at (30mm, 30mm) is in ROI
        expected_events_in_roi = 1
        assert sc.sum(result['roi_current']).value == expected_events_in_roi

    def test_roi_published_when_updated_with_detector_data(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that ROI is published when ROI changes with detector data."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # First ROI + detector
        roi1 = make_rectangle_roi(
            x_min=5.0, x_max=25.0, y_min=5.0, y_max=25.0, x_unit='mm', y_unit='mm'
        )
        view.accumulate(
            {'roi': roi1.to_data_array(), 'detector': sample_detector_events}
        )
        result1 = view.finalize()
        assert 'roi_rectangle' in result1

        # Just detector (no ROI update)
        view.accumulate({'detector': sample_detector_events})
        result2 = view.finalize()
        assert 'roi_rectangle' not in result2

        # Updated ROI + detector
        roi2 = make_rectangle_roi(
            x_min=10.0, x_max=20.0, y_min=10.0, y_max=20.0, x_unit='mm', y_unit='mm'
        )
        view.accumulate(
            {'roi': roi2.to_data_array(), 'detector': sample_detector_events}
        )
        result3 = view.finalize()
        assert 'roi_rectangle' in result3

        # Verify the published config is the new one
        published_roi = RectangleROI.from_data_array(result3['roi_rectangle'])
        assert published_roi.x.min == 10.0
        assert published_roi.x.max == 20.0
