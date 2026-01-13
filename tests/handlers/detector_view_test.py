# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Proof-of-concept tests for DetectorView and ROI mechanism.

These tests verify that the ROI configuration and accumulation mechanism works
correctly in the DetectorView workflow.
"""

import pytest
import scipp as sc

from ess.livedata.config.models import Interval, PolygonROI, RectangleROI
from ess.livedata.handlers.accumulators import GroupIntoPixels
from ess.livedata.handlers.detector_view import (
    DetectorView,
    DetectorViewParams,
)
from ess.livedata.handlers.to_nxevent_data import DetectorEvents
from ess.livedata.parameter_models import TOAEdges
from ess.reduce.live.raw import RollingDetectorView
from ess.reduce.live.roi import ROIFilter

# Detector layout constants
# 4x4 detector grid with 10mm spacing (coordinates in mm):
#   Row 0 (y=0):  [0: (0,0),   1: (10,0),  2: (20,0),  3: (30,0)]
#   Row 1 (y=10): [4: (0,10),  5: (10,10), 6: (20,10), 7: (30,10)]
#   Row 2 (y=20): [8: (0,20),  9: (10,20), 10:(20,20), 11:(30,20)]
#   Row 3 (y=30): [12:(0,30),  13:(10,30), 14:(20,30), 15:(30,30)]
#
# Sample events are at pixels: [0, 1, 2, 5, 6, 10, 11, 15]

# Expected event counts for various ROI configurations
TOTAL_SAMPLE_EVENTS = 8  # All events in sample_detector_events
STANDARD_ROI_EVENTS = 3  # Pixels 5, 6, 10 in x=[5,25]mm, y=[5,25]mm
WIDE_ROI_EVENTS = 4  # Pixels 1, 2, 5, 6 in x=[5,25]mm, y=[-5,15]mm
CORNER_ROI_EVENTS = 1  # Pixel 15 only in x=[25,35]mm, y=[25,35]mm


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


def roi_to_accumulate_data(
    roi: RectangleROI, index: int = 0
) -> dict[str, sc.DataArray]:
    """Convert ROI to accumulate data format."""
    return {'roi_rectangle': RectangleROI.to_concatenated_data_array({index: roi})}


def assert_has_detector_view_results(result: dict) -> None:
    """Assert result contains standard detector view outputs."""
    assert 'cumulative' in result
    assert 'current' in result


def assert_has_roi_results(result: dict, roi_index: int = 0) -> None:
    """Assert result contains stacked ROI histogram outputs with given index."""
    assert 'roi_spectra_current' in result
    assert 'roi_spectra_cumulative' in result
    roi_indices = list(result['roi_spectra_current'].coords['roi'].values)
    assert roi_index in roi_indices, f"ROI index {roi_index} not in {roi_indices}"


def assert_roi_config_published(result: dict) -> None:
    """Assert ROI configuration was published in result."""
    assert 'roi_rectangle' in result


def assert_roi_event_count(
    result: dict, expected: int, roi_index: int = 0, view: str = 'current'
) -> None:
    """Assert ROI histogram has expected event count using stacked outputs."""
    stacked_key = f'roi_spectra_{view}'
    stacked = result[stacked_key]
    roi_indices = list(stacked.coords['roi'].values)
    position = roi_indices.index(roi_index)
    actual = sc.sum(stacked['roi', position]).value
    assert (
        actual == expected
    ), f"Expected {expected} events for ROI {roi_index}, got {actual}"


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


@pytest.fixture
def standard_roi() -> RectangleROI:
    """Standard ROI covering pixels 5, 6, 10 (x=[5,25]mm, y=[5,25]mm)."""
    return make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')


@pytest.fixture
def wide_roi() -> RectangleROI:
    """Wide ROI covering pixels 1, 2, 5, 6 (x=[5,25]mm, y=[-5,15]mm)."""
    return make_rectangle_roi(5.0, 25.0, -5.0, 15.0, 'mm', 'mm')


@pytest.fixture
def corner_roi() -> RectangleROI:
    """Corner ROI covering only pixel 15 (x=[25,35]mm, y=[25,35]mm)."""
    return make_rectangle_roi(25.0, 35.0, 25.0, 35.0, 'mm', 'mm')


@pytest.fixture
def standard_toa_edges() -> TOAEdges:
    """Standard TOA edges for histogram tests (0-1000ns, 10 bins)."""
    return TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')


class TestDetectorViewBasics:
    """Basic tests for DetectorView without ROI."""

    def test_detector_view_initialization(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that DetectorView can be initialized with default parameters."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        assert view is not None
        # Accumulate some detector data before finalize
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        # Verify stacked outputs are present but empty when no ROI configured
        result = view.finalize()
        assert 'cumulative' in result
        assert 'current' in result
        # Stacked outputs are always present (empty when no ROIs)
        assert 'roi_spectra_current' in result
        assert 'roi_spectra_cumulative' in result
        assert result['roi_spectra_current'].sizes['roi'] == 0
        assert result['roi_spectra_cumulative'].sizes['roi'] == 0

    def test_current_has_time_coord(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that 'current' result has time coord from first accumulate call."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Accumulate with specific start_time
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result = view.finalize()

        # Verify time coord is present on current
        assert 'time' in result['current'].coords
        assert result['current'].coords['time'].value == 1000
        assert result['current'].coords['time'].unit == 'ns'
        # cumulative should not have time coord
        assert 'time' not in result['cumulative'].coords

    def test_time_coord_tracks_first_accumulate(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that time coord uses first accumulate start_time, not later ones."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # First accumulate with start_time=1000
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        # Second accumulate with different start_time
        view.accumulate(
            {'detector': sample_detector_events}, start_time=3000, end_time=4000
        )

        result = view.finalize()

        # Time should be from first accumulate
        assert result['current'].coords['time'].value == 1000

        # After finalize, next accumulate should track new start_time
        view.accumulate(
            {'detector': sample_detector_events}, start_time=5000, end_time=6000
        )
        result2 = view.finalize()
        assert result2['current'].coords['time'].value == 5000

    def test_current_has_start_end_time_coords(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that 'current' result has start_time and end_time coords.

        Delta outputs like 'current' need their own time bounds that represent
        the period since the last finalize, not the entire job duration.
        """
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Accumulate with specific time range
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result = view.finalize()

        # Verify start_time and end_time coords are present on current
        assert 'start_time' in result['current'].coords
        assert 'end_time' in result['current'].coords
        assert result['current'].coords['start_time'].value == 1000
        assert result['current'].coords['end_time'].value == 2000
        assert result['current'].coords['start_time'].unit == 'ns'
        assert result['current'].coords['end_time'].unit == 'ns'

        # cumulative should not have start_time or end_time coords
        # (they will be added by Job.get() with job-level times)
        assert 'start_time' not in result['cumulative'].coords
        assert 'end_time' not in result['cumulative'].coords

    def test_delta_outputs_track_time_since_last_finalize(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that delta outputs track time since last finalize, not job start.

        This is critical for showing correct time bounds when the dashboard
        displays the period used to compute delta outputs like 'current'.
        """
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # First period: accumulate and finalize
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result1 = view.finalize()
        assert result1['current'].coords['start_time'].value == 1000
        assert result1['current'].coords['end_time'].value == 2000

        # Second period: new time range
        view.accumulate(
            {'detector': sample_detector_events}, start_time=3000, end_time=4000
        )
        result2 = view.finalize()

        # Delta output should reflect second period's time range, NOT job start
        assert result2['current'].coords['start_time'].value == 3000
        assert result2['current'].coords['end_time'].value == 4000

    def test_finalize_without_accumulate_raises(
        self, mock_rolling_view: RollingDetectorView
    ) -> None:
        """Test that finalize raises if called without accumulate."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        with pytest.raises(
            RuntimeError,
            match="finalize called without any detector data accumulated",
        ):
            view.finalize()

    def test_accumulate_detector_data_without_roi(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test accumulating detector data without ROI configuration."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Accumulate detector data
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )

        # Finalize should return cumulative and current counts
        result = view.finalize()

        assert_has_detector_view_results(result)
        # Stacked ROI outputs are always present but empty without ROI configuration
        assert result['roi_spectra_current'].sizes['roi'] == 0
        assert result['roi_spectra_cumulative'].sizes['roi'] == 0

        # Verify that events were actually accumulated
        assert sc.sum(result['current']).value == TOTAL_SAMPLE_EVENTS

    def test_clear_resets_state(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that clear() resets the detector view state."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result1 = view.finalize()
        assert sc.sum(result1['current']).value == TOTAL_SAMPLE_EVENTS

        # Clear the view
        view.clear()

        # After clear, accumulate new data then finalize should return zero counts
        view.accumulate(
            {'detector': sample_detector_events}, start_time=2000, end_time=3000
        )
        result2 = view.finalize()
        assert sc.sum(result2['current']).value == TOTAL_SAMPLE_EVENTS
        assert sc.sum(result2['cumulative']).value == TOTAL_SAMPLE_EVENTS


class TestDetectorViewROIMechanism:
    """Tests for ROI configuration and histogram accumulation."""

    def test_roi_configuration_via_accumulate(
        self,
        mock_rolling_view: RollingDetectorView,
        standard_roi: RectangleROI,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that ROI configuration can be set via accumulate."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Send ROI configuration
        view.accumulate(
            roi_to_accumulate_data(standard_roi), start_time=1000, end_time=2000
        )
        # Add detector data before finalize
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )

        # Verify ROI configuration is active via finalize output
        result = view.finalize()
        # ROI streams should be present (current, cumulative, and concatenated config)
        assert_has_roi_results(result)
        assert_roi_config_published(result)
        # Verify the echoed ROI config matches what we sent
        concatenated_roi_data = result['roi_rectangle']
        assert 'roi_index' in concatenated_roi_data.coords
        # Parse back to individual ROIs
        from ess.livedata.config.models import ROI

        echoed_rois = ROI.from_concatenated_data_array(concatenated_roi_data)
        assert 0 in echoed_rois
        echoed_roi = echoed_rois[0]
        assert echoed_roi.x.min == 5.0
        assert echoed_roi.x.max == 25.0
        assert echoed_roi.y.min == 5.0
        assert echoed_roi.y.max == 25.0

    def test_roi_only_does_not_process_events(
        self,
        mock_rolling_view: RollingDetectorView,
        standard_roi: RectangleROI,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that ROI configuration persists when detector data arrives."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Send ROI configuration without detector data
        view.accumulate(
            roi_to_accumulate_data(standard_roi), start_time=1000, end_time=2000
        )

        # Now send detector data
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )

        # ROI should be configured and histogram data accumulated
        result = view.finalize()
        # ROI results should be present
        assert_has_roi_results(result)
        assert_roi_config_published(result)
        # Counts should match events within ROI
        assert_roi_event_count(result, STANDARD_ROI_EVENTS)

    def test_accumulate_with_roi_produces_histogram(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        standard_roi: RectangleROI,
        standard_toa_edges: TOAEdges,
    ) -> None:
        """Test that ROI configuration enables histogram accumulation."""
        params = DetectorViewParams(toa_edges=standard_toa_edges)
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI first
        view.accumulate(
            roi_to_accumulate_data(standard_roi), start_time=1000, end_time=2000
        )

        # Now accumulate detector events
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )

        result = view.finalize()

        # Should have both detector view and ROI results
        assert_has_detector_view_results(result)
        assert_has_roi_results(result)
        assert_roi_config_published(result)

        # Verify stacked histogram structure
        stacked = result['roi_spectra_current']
        assert isinstance(stacked, sc.DataArray)
        assert 'time_of_arrival' in stacked.coords
        assert stacked.ndim == 2  # Stacked is 2D: roi x time_of_arrival
        assert stacked.sizes['roi'] == 1  # One ROI

        # Verify expected event count
        assert_roi_event_count(result, STANDARD_ROI_EVENTS)

    def test_roi_current_has_time_coord(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        standard_roi: RectangleROI,
        standard_toa_edges: TOAEdges,
    ) -> None:
        """Test that ROI current results have time coord matching detector view."""
        params = DetectorViewParams(toa_edges=standard_toa_edges)
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI
        view.accumulate(
            roi_to_accumulate_data(standard_roi), start_time=1000, end_time=2000
        )

        # Accumulate detector events
        view.accumulate(
            {'detector': sample_detector_events}, start_time=2500, end_time=3000
        )

        result = view.finalize()

        # Verify time coord on stacked ROI current
        assert 'time' in result['roi_spectra_current'].coords
        assert result['roi_spectra_current'].coords['time'].value == 2500
        assert result['roi_spectra_current'].coords['time'].unit == 'ns'

        # Verify it matches detector view current time coord
        assert (
            result['roi_spectra_current'].coords['time']
            == result['current'].coords['time']
        )

        # ROI cumulative should not have time coord
        assert 'time' not in result['roi_spectra_cumulative'].coords

    def test_roi_current_has_start_end_time_coords(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        standard_roi: RectangleROI,
        standard_toa_edges: TOAEdges,
    ) -> None:
        """Test that ROI current results have start_time and end_time coords."""
        params = DetectorViewParams(toa_edges=standard_toa_edges)
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI
        view.accumulate(
            roi_to_accumulate_data(standard_roi), start_time=1000, end_time=2000
        )

        # Accumulate detector events
        view.accumulate(
            {'detector': sample_detector_events}, start_time=2500, end_time=3000
        )

        result = view.finalize()

        # Verify start_time and end_time coords on ROI current
        assert 'start_time' in result['roi_spectra_current'].coords
        assert 'end_time' in result['roi_spectra_current'].coords
        assert result['roi_spectra_current'].coords['start_time'].value == 2500
        assert result['roi_spectra_current'].coords['end_time'].value == 3000

        # Verify they match detector view current time coords
        assert (
            result['roi_spectra_current'].coords['start_time']
            == result['current'].coords['start_time']
        )
        assert (
            result['roi_spectra_current'].coords['end_time']
            == result['current'].coords['end_time']
        )

        # ROI cumulative should not have start_time/end_time coords
        assert 'start_time' not in result['roi_spectra_cumulative'].coords
        assert 'end_time' not in result['roi_spectra_cumulative'].coords

    def test_roi_cumulative_accumulation(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        standard_roi: RectangleROI,
        standard_toa_edges: TOAEdges,
    ) -> None:
        """Test that ROI histograms accumulate correctly over multiple updates."""
        params = DetectorViewParams(toa_edges=standard_toa_edges)
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI
        view.accumulate(
            roi_to_accumulate_data(standard_roi), start_time=1000, end_time=2000
        )

        # First accumulation
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result1 = view.finalize()
        assert_roi_event_count(result1, STANDARD_ROI_EVENTS)

        # Second accumulation with same events
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result2 = view.finalize()

        # Both should have same current counts
        assert_roi_event_count(result2, STANDARD_ROI_EVENTS, view='current')
        # Cumulative should be the sum of both accumulations
        assert_roi_event_count(result2, 2 * STANDARD_ROI_EVENTS, view='cumulative')

    def test_roi_published_only_on_update(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        standard_roi: RectangleROI,
    ) -> None:
        """Test that ROI is only published when updated."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI
        view.accumulate(
            roi_to_accumulate_data(standard_roi), start_time=1000, end_time=2000
        )
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )

        result1 = view.finalize()
        assert_roi_config_published(result1)

        # Second accumulation without ROI update
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result2 = view.finalize()
        assert 'roi_rectangle' not in result2  # Not published again

    def test_clear_resets_roi_state(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        standard_roi: RectangleROI,
    ) -> None:
        """Test that clear() resets ROI cumulative state."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI and accumulate
        view.accumulate(
            roi_to_accumulate_data(standard_roi), start_time=1000, end_time=2000
        )
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result1 = view.finalize()

        # Verify we accumulated some events
        assert sc.sum(result1['roi_spectra_cumulative']).value > 0

        # Clear should reset cumulative
        view.clear()

        # After clear, accumulate new data
        view.accumulate(
            {'detector': sample_detector_events}, start_time=2000, end_time=3000
        )
        result2 = view.finalize()
        assert_has_roi_results(result2)  # ROI config still active
        # Cumulative should be reset (only contains events from after clear)
        assert sc.sum(result2['roi_spectra_cumulative']).value == STANDARD_ROI_EVENTS
        assert sc.sum(result2['roi_spectra_current']).value == STANDARD_ROI_EVENTS

    def test_roi_change_resets_cumulative(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        standard_roi: RectangleROI,
        wide_roi: RectangleROI,
        standard_toa_edges: TOAEdges,
    ) -> None:
        """Test that changing ROI resets cumulative histogram."""
        params = DetectorViewParams(toa_edges=standard_toa_edges)
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure first ROI covering pixels 5, 6, 10
        view.accumulate(
            roi_to_accumulate_data(standard_roi), start_time=1000, end_time=2000
        )
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result1 = view.finalize()

        assert_roi_event_count(result1, STANDARD_ROI_EVENTS, view='current')
        assert_roi_event_count(result1, STANDARD_ROI_EVENTS, view='cumulative')

        # Now change ROI to cover different pixels (1, 2, 5, 6)
        view.accumulate(
            roi_to_accumulate_data(wide_roi), start_time=1000, end_time=2000
        )
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result2 = view.finalize()

        assert_roi_event_count(result2, WIDE_ROI_EVENTS, view='current')

        # CRITICAL: Cumulative should reset when ROI changes
        # It should NOT be roi1_events + roi2_events (which would be 7)
        # It should be just roi2_events since we changed the ROI
        assert sc.sum(result2['roi_spectra_cumulative']).value == WIDE_ROI_EVENTS, (
            f"Expected cumulative to reset to {WIDE_ROI_EVENTS} when ROI changes, "
            f"got {sc.sum(result2['roi_spectra_cumulative']).value}. "
            "Cumulative should not mix events from different ROI regions."
        )


class TestDetectorViewBothROIAndDetectorData:
    """Tests for accumulate called with both ROI and detector data together."""

    def test_accumulate_with_both_roi_and_detector_in_same_call(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        standard_roi: RectangleROI,
        standard_toa_edges: TOAEdges,
    ) -> None:
        """Test accumulate with both ROI and detector data in one call."""
        params = DetectorViewParams(toa_edges=standard_toa_edges)
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI and send detector data in same call
        view.accumulate(
            {
                **roi_to_accumulate_data(standard_roi),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )

        result = view.finalize()

        # Should have both detector view and ROI results
        assert_has_detector_view_results(result)
        assert_has_roi_results(result)
        assert_roi_config_published(result)

        # Verify histogram has expected events
        assert_roi_event_count(result, STANDARD_ROI_EVENTS)

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
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array({0: roi}),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )
        result1 = view.finalize()

        expected_events_in_roi = 3
        assert sc.sum(result1['roi_spectra_current']).value == expected_events_in_roi
        assert 'roi_rectangle' in result1  # Published on first update

        # Second call: detector data only (ROI should persist)
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result2 = view.finalize()

        assert sc.sum(result2['roi_spectra_current']).value == expected_events_in_roi
        assert (
            sc.sum(result2['roi_spectra_cumulative']).value
            == 2 * expected_events_in_roi
        )
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
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result1 = view.finalize()

        # Stacked outputs are always present but empty without ROI
        assert result1['roi_spectra_current'].sizes['roi'] == 0
        assert result1['roi_spectra_cumulative'].sizes['roi'] == 0

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
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array({0: roi}),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )
        result2 = view.finalize()

        expected_events_in_roi = 3
        assert result2['roi_spectra_current'].sizes['roi'] == 1
        assert result2['roi_spectra_cumulative'].sizes['roi'] == 1
        assert sc.sum(result2['roi_spectra_current']).value == expected_events_in_roi
        assert sc.sum(result2['roi_spectra_cumulative']).value == expected_events_in_roi

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
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array({0: roi1}),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )
        result1 = view.finalize()

        expected_roi1_events = 3
        assert sc.sum(result1['roi_spectra_current']).value == expected_roi1_events
        assert sc.sum(result1['roi_spectra_cumulative']).value == expected_roi1_events

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
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array({0: roi2}),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )
        result2 = view.finalize()

        expected_roi2_events = 4
        assert sc.sum(result2['roi_spectra_current']).value == expected_roi2_events
        # Cumulative should reset when ROI changes
        assert sc.sum(result2['roi_spectra_cumulative']).value == expected_roi2_events
        assert 'roi_rectangle' in result2  # Config published on update


class TestDetectorViewEdgeCases:
    """Edge cases and error conditions."""

    def test_accumulate_empty_dict_does_nothing(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that accumulate with empty dict returns early without error."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Empty dict should not raise - consistent with roi-only behavior
        view.accumulate({}, start_time=1000, end_time=2000)

        # Add detector data before finalize
        view.accumulate(
            {'detector': sample_detector_events}, start_time=2000, end_time=3000
        )

        result = view.finalize()
        assert 'cumulative' in result
        assert 'current' in result
        # Events from sample_detector_events
        assert sc.sum(result['cumulative']).value == TOTAL_SAMPLE_EVENTS

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
                },
                start_time=1000,
                end_time=2000,
            )

    def test_multiple_roi_updates_without_detector_data(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test multiple ROI updates followed by detector data."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Update ROI multiple times
        roi1 = make_rectangle_roi(
            x_min=5.0, x_max=25.0, y_min=5.0, y_max=25.0, x_unit='mm', y_unit='mm'
        )
        view.accumulate(
            {'roi_rectangle': RectangleROI.to_concatenated_data_array({0: roi1})},
            start_time=1000,
            end_time=2000,
        )

        roi2 = make_rectangle_roi(
            x_min=10.0, x_max=20.0, y_min=10.0, y_max=20.0, x_unit='mm', y_unit='mm'
        )
        view.accumulate(
            {'roi_rectangle': RectangleROI.to_concatenated_data_array({0: roi2})},
            start_time=1000,
            end_time=2000,
        )

        # Now add detector data
        view.accumulate(
            {'detector': sample_detector_events}, start_time=2000, end_time=3000
        )

        result = view.finalize()

        # Should have ROI configured with accumulated events
        assert result['roi_spectra_cumulative'].sizes['roi'] == 1
        assert result['roi_spectra_current'].sizes['roi'] == 1
        assert 'roi_rectangle' in result
        # roi2 only covers pixel 10 (single event at x=20mm, y=20mm)
        expected_events = 1
        assert sc.sum(result['roi_spectra_cumulative']).value == expected_events
        assert sc.sum(result['roi_spectra_current']).value == expected_events

    def test_detector_data_with_no_events_in_roi(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        corner_roi: RectangleROI,
        standard_toa_edges: TOAEdges,
    ) -> None:
        """Test that ROI with no matching events produces zero histogram."""
        params = DetectorViewParams(toa_edges=standard_toa_edges)
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Configure ROI that covers only pixel 15 at (30mm, 30mm)
        view.accumulate(
            roi_to_accumulate_data(corner_roi), start_time=1000, end_time=2000
        )
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )

        result = view.finalize()

        # Should have histogram structure with only pixel 15 in ROI
        assert_has_roi_results(result)
        assert_roi_event_count(result, CORNER_ROI_EVENTS)

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
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array({0: roi1}),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )
        result1 = view.finalize()
        assert 'roi_rectangle' in result1

        # Just detector (no ROI update)
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result2 = view.finalize()
        assert 'roi_rectangle' not in result2

        # Updated ROI + detector
        roi2 = make_rectangle_roi(
            x_min=10.0, x_max=20.0, y_min=10.0, y_max=20.0, x_unit='mm', y_unit='mm'
        )
        view.accumulate(
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array({0: roi2}),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )
        result3 = view.finalize()
        assert 'roi_rectangle' in result3

        # Verify the published config is the new one
        from ess.livedata.config.models import ROI

        published_rois = ROI.from_concatenated_data_array(result3['roi_rectangle'])
        assert 0 in published_rois
        published_roi = published_rois[0]
        assert published_roi.x.min == 10.0
        assert published_roi.x.max == 20.0

    def test_roi_deletion_is_published(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that ROI deletion triggers publication and clears all ROIs.

        When any ROI is deleted, all ROIs are cleared to maintain consistent
        accumulation periods. This is critical since ROI spectra are overlaid
        on the same plot.
        """
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Create two ROIs
        roi0 = make_rectangle_roi(
            x_min=5.0, x_max=25.0, y_min=5.0, y_max=25.0, x_unit='mm', y_unit='mm'
        )
        roi1 = make_rectangle_roi(
            x_min=25.0, x_max=35.0, y_min=25.0, y_max=35.0, x_unit='mm', y_unit='mm'
        )
        view.accumulate(
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array(
                    {0: roi0, 1: roi1}
                )
            },
            start_time=1000,
            end_time=2000,
        )
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result1 = view.finalize()

        # Both ROIs should be published
        assert 'roi_rectangle' in result1
        from ess.livedata.config.models import ROI

        published_rois = ROI.from_concatenated_data_array(result1['roi_rectangle'])
        assert 0 in published_rois
        assert 1 in published_rois

        # Accumulate more data to build up cumulative
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result_no_change = view.finalize()
        assert 'roi_rectangle' not in result_no_change
        # ROI 0 should have accumulated events from both rounds
        assert_roi_event_count(result_no_change, 2 * 3, roi_index=0, view='cumulative')

        # Now delete ROI 1, keeping only ROI 0
        view.accumulate(
            {'roi_rectangle': RectangleROI.to_concatenated_data_array({0: roi0})},
            start_time=1000,
            end_time=2000,
        )
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result2 = view.finalize()

        # ROI config should be published to signal deletion
        assert (
            'roi_rectangle' in result2
        ), "ROI deletion should trigger publication of updated ROI set"

        # Verify only ROI 0 is present
        published_rois = ROI.from_concatenated_data_array(result2['roi_rectangle'])
        assert 0 in published_rois
        assert 1 not in published_rois, "Deleted ROI should not be in published set"

        # CRITICAL: ROI 0 should be cleared (reset to fresh accumulation)
        # This ensures all visible ROIs have consistent accumulation periods
        assert_roi_event_count(result2, 3, roi_index=0, view='cumulative')

    def test_roi_deletion_with_index_renumbering_clears_all(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        standard_roi: RectangleROI,
        corner_roi: RectangleROI,
    ) -> None:
        """Test that ROI deletion with renumbering clears all ROIs.

        This tests the specific scenario: if you have ROI 0 and ROI 1, and delete
        ROI 0 in the UI, the frontend renumbers ROI 1 -> ROI 0. The backend must
        clear all ROIs to avoid misleading overlaid spectra with different
        accumulation periods.
        """
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Create two ROIs with different geometries
        # ROI 0: standard_roi (pixels 5, 6, 10) -> 3 events
        # ROI 1: corner_roi (pixel 15) -> 1 event
        view.accumulate(
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array(
                    {0: standard_roi, 1: corner_roi}
                )
            },
            start_time=1000,
            end_time=2000,
        )

        # Accumulate events twice to build up cumulative
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        view.finalize()
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result = view.finalize()

        # Verify both ROIs have accumulated
        assert_roi_event_count(
            result, 2 * 3, roi_index=0, view='cumulative'
        )  # 6 events
        assert_roi_event_count(
            result, 2 * 1, roi_index=1, view='cumulative'
        )  # 2 events

        # Now simulate deleting ROI 0 in UI: ROI 1 gets renumbered to ROI 0
        # From backend perspective: index 0 changes from standard_roi to corner_roi
        view.accumulate(
            {'roi_rectangle': RectangleROI.to_concatenated_data_array({0: corner_roi})},
            start_time=1000,
            end_time=2000,
        )
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result2 = view.finalize()

        # CRITICAL: The renumbered ROI should be cleared
        # Even though it's the "same" ROI geometrically (corner_roi),
        # it appeared at a different index, so we clear all ROIs
        assert_roi_event_count(result2, 1, roi_index=0, view='cumulative')

    def test_unchanged_roi_resend_unnecessarily_resets_cumulative(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that resending identical ROI config preserves cumulative.

        When the exact same ROI configuration is sent again (same indices,
        same geometries), cumulative should be preserved.
        """
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Create ROI
        roi0 = make_rectangle_roi(
            x_min=5.0, x_max=25.0, y_min=5.0, y_max=25.0, x_unit='mm', y_unit='mm'
        )
        view.accumulate(
            {'roi_rectangle': RectangleROI.to_concatenated_data_array({0: roi0})},
            start_time=1000,
            end_time=2000,
        )

        # Accumulate data twice to build up cumulative
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        view.finalize()
        expected_events = 3  # pixels 5, 6, 10 in ROI

        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result2 = view.finalize()

        # Cumulative should have doubled
        assert_roi_event_count(
            result2, 2 * expected_events, roi_index=0, view='cumulative'
        )

        # Now resend the SAME ROI configuration (no actual change)
        view.accumulate(
            {'roi_rectangle': RectangleROI.to_concatenated_data_array({0: roi0})},
            start_time=1000,
            end_time=2000,
        )
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result3 = view.finalize()

        # BUG: Current implementation resets cumulative even though ROI didn't change!
        # Cumulative is reset to just the new events instead of continuing to accumulate
        stacked = result3['roi_spectra_cumulative']
        roi_indices = list(stacked.coords['roi'].values)
        position = roi_indices.index(0)
        cumulative_after_resend = sc.sum(stacked['roi', position]).value

        # This assertion will FAIL with current implementation, exposing the bug
        assert cumulative_after_resend == 3 * expected_events, (
            f"Expected cumulative to continue accumulating (3 * {expected_events} = "
            f"{3 * expected_events}), but got {cumulative_after_resend}. "
            "Resending identical ROI config should not reset cumulative histogram!"
        )


class TestDetectorViewRatemeter:
    """Tests for ratemeter (event count) functionality."""

    def test_counts_output_in_finalize(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that finalize outputs counts_total and counts_in_toa_range."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result = view.finalize()

        assert 'counts_total' in result
        assert 'counts_in_toa_range' in result
        assert result['counts_total'].unit == 'counts'
        assert result['counts_in_toa_range'].unit == 'counts'

    def test_counts_have_start_end_time_coords(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that counts outputs have start_time and end_time coords."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result = view.finalize()

        # Verify start_time and end_time coords on counts outputs
        for key in ['counts_total', 'counts_in_toa_range']:
            assert 'start_time' in result[key].coords, f"Missing start_time on {key}"
            assert 'end_time' in result[key].coords, f"Missing end_time on {key}"
            assert result[key].coords['start_time'].value == 1000
            assert result[key].coords['end_time'].value == 2000
            assert result[key].coords['start_time'].unit == 'ns'
            assert result[key].coords['end_time'].unit == 'ns'

    def test_counts_match_total_events_without_toa_filter(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that counts equal total events when TOA filter is disabled."""
        params = DetectorViewParams()  # TOA range disabled by default
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result = view.finalize()

        assert result['counts_total'].value == TOTAL_SAMPLE_EVENTS
        assert result['counts_in_toa_range'].value == TOTAL_SAMPLE_EVENTS

    def test_counts_filtered_by_toa_range(
        self,
        mock_rolling_view: RollingDetectorView,
        detector_number: sc.Variable,
    ) -> None:
        """Test that counts_in_toa_range respects TOA filter."""
        from ess.livedata.parameter_models import TOARange

        # Enable TOA range filter for 200-600 ns (should capture 4 of 8 events)
        params = DetectorViewParams(
            toa_range=TOARange(enabled=True, start=200.0, stop=600.0, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Create events with known TOA values: 100, 200, 300, 400, 500, 600, 700, 800
        # Filter 200-600 should include: 200, 300, 400, 500 (4 events)
        pixel_ids = [0, 1, 2, 5, 6, 10, 11, 15]
        toa_values = [100, 200, 300, 400, 500, 600, 700, 800]

        events = DetectorEvents(
            pixel_id=pixel_ids,
            time_of_arrival=toa_values,
            unit='ns',
        )
        grouper = GroupIntoPixels(detector_number=detector_number)
        grouper.add(0, events)
        detector_events = grouper.get()

        view.accumulate({'detector': detector_events}, start_time=1000, end_time=2000)
        result = view.finalize()

        assert result['counts_total'].value == 8
        # TOA range [200, 600) should include events at 200, 300, 400, 500
        assert result['counts_in_toa_range'].value == 4

    def test_counts_reset_after_finalize(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that counts are reset after each finalize call."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # First accumulation
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result1 = view.finalize()
        assert result1['counts_total'].value == TOTAL_SAMPLE_EVENTS

        # Second accumulation - counts should be fresh, not cumulative
        view.accumulate(
            {'detector': sample_detector_events}, start_time=2000, end_time=3000
        )
        result2 = view.finalize()
        assert result2['counts_total'].value == TOTAL_SAMPLE_EVENTS

    def test_counts_accumulate_within_finalize_period(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that multiple accumulate calls sum counts before finalize."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Two accumulate calls before finalize
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        view.accumulate(
            {'detector': sample_detector_events}, start_time=2000, end_time=3000
        )
        result = view.finalize()

        # Should have sum of both accumulations
        assert result['counts_total'].value == 2 * TOTAL_SAMPLE_EVENTS

    def test_counts_reset_by_clear(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that clear() resets accumulated counts."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Accumulate some events (don't finalize yet)
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        assert view._counts_total == TOTAL_SAMPLE_EVENTS

        # Clear should reset counts
        view.clear()
        assert view._counts_total == 0
        assert view._counts_in_toa_range == 0

        # Accumulate new events
        view.accumulate(
            {'detector': sample_detector_events}, start_time=2000, end_time=3000
        )
        result = view.finalize()

        # Should only have events from after clear
        assert result['counts_total'].value == TOTAL_SAMPLE_EVENTS


class TestDetectorViewPolygonROI:
    """Tests for polygon ROI support in DetectorView."""

    def test_polygon_roi_processes_correctly(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that polygon ROI is processed and produces histogram results."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Polygon covering center region: same area as standard rectangle ROI
        polygon_roi = PolygonROI(
            x=[5.0, 25.0, 25.0, 5.0],
            y=[5.0, 5.0, 25.0, 25.0],
            x_unit='mm',
            y_unit='mm',
        )

        # Polygon uses index 4 (first polygon slot after 4 rectangles)
        view.accumulate(
            {
                'roi_polygon': PolygonROI.to_concatenated_data_array({4: polygon_roi}),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )

        result = view.finalize()

        # Should have detector view results
        assert 'cumulative' in result
        assert 'current' in result

        # Should have polygon ROI results at index 4
        assert_has_roi_results(result, roi_index=4)

        # Should publish polygon readback
        assert 'roi_polygon' in result

        # Should have expected events (same as rectangle: pixels 5, 6, 10)
        assert_roi_event_count(result, STANDARD_ROI_EVENTS, roi_index=4)

    def test_polygon_roi_readback_contains_correct_data(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that polygon ROI readback contains the polygon data."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        polygon_roi = PolygonROI(
            x=[5.0, 25.0, 25.0, 5.0],
            y=[5.0, 5.0, 25.0, 25.0],
            x_unit='mm',
            y_unit='mm',
        )

        view.accumulate(
            {
                'roi_polygon': PolygonROI.to_concatenated_data_array({4: polygon_roi}),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )

        result = view.finalize()

        # Verify readback can be deserialized back to ROI
        readback = result['roi_polygon']
        rois = PolygonROI.from_concatenated_data_array(readback)
        assert 4 in rois
        assert rois[4].x == polygon_roi.x
        assert rois[4].y == polygon_roi.y

    def test_polygon_roi_cumulative_accumulation(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that polygon ROI accumulates events across multiple periods."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        polygon_roi = PolygonROI(
            x=[5.0, 25.0, 25.0, 5.0],
            y=[5.0, 5.0, 25.0, 25.0],
            x_unit='mm',
            y_unit='mm',
        )

        # First accumulation with ROI config
        view.accumulate(
            {
                'roi_polygon': PolygonROI.to_concatenated_data_array({4: polygon_roi}),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )
        result1 = view.finalize()

        assert_roi_event_count(
            result1, STANDARD_ROI_EVENTS, roi_index=4, view='current'
        )
        assert_roi_event_count(
            result1, STANDARD_ROI_EVENTS, roi_index=4, view='cumulative'
        )

        # Second accumulation without ROI config (should persist)
        view.accumulate(
            {'detector': sample_detector_events},
            start_time=2000,
            end_time=3000,
        )
        result2 = view.finalize()

        assert_roi_event_count(
            result2, STANDARD_ROI_EVENTS, roi_index=4, view='current'
        )
        assert_roi_event_count(
            result2, 2 * STANDARD_ROI_EVENTS, roi_index=4, view='cumulative'
        )
        # Readback not published on unchanged ROI
        assert 'roi_polygon' not in result2


class TestDetectorViewMultipleGeometries:
    """Tests for multiple ROI geometry types in DetectorView."""

    def test_rectangle_and_polygon_independent(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that rectangle and polygon ROIs can coexist independently."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Create both rectangle and polygon ROIs
        rect_roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        polygon_roi = PolygonROI(
            x=[5.0, 25.0, 25.0, 5.0],
            y=[5.0, 5.0, 25.0, 25.0],
            x_unit='mm',
            y_unit='mm',
        )

        # Send both geometry types together
        view.accumulate(
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array({0: rect_roi}),
                'roi_polygon': PolygonROI.to_concatenated_data_array({4: polygon_roi}),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )

        result = view.finalize()

        # Both should produce results
        assert_has_roi_results(result, roi_index=0)  # Rectangle at index 0
        assert_has_roi_results(result, roi_index=4)  # Polygon at index 4

        # Both should have same expected events (same shape)
        assert_roi_event_count(result, STANDARD_ROI_EVENTS, roi_index=0)
        assert_roi_event_count(result, STANDARD_ROI_EVENTS, roi_index=4)

        # Both readbacks should be published
        assert 'roi_rectangle' in result
        assert 'roi_polygon' in result

    def test_updating_polygon_does_not_affect_rectangle(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that updating polygon ROI does not reset rectangle ROI state."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Setup: create rectangle ROI and accumulate data
        rect_roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        view.accumulate(
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array({0: rect_roi}),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )
        result1 = view.finalize()

        assert_roi_event_count(
            result1, STANDARD_ROI_EVENTS, roi_index=0, view='cumulative'
        )

        # Now add a polygon ROI (rectangle should keep its state)
        polygon_roi = PolygonROI(
            x=[5.0, 25.0, 25.0, 5.0],
            y=[5.0, 5.0, 25.0, 25.0],
            x_unit='mm',
            y_unit='mm',
        )
        view.accumulate(
            {
                'roi_polygon': PolygonROI.to_concatenated_data_array({4: polygon_roi}),
                'detector': sample_detector_events,
            },
            start_time=2000,
            end_time=3000,
        )
        result2 = view.finalize()

        # Rectangle cumulative should have continued accumulating
        assert_roi_event_count(
            result2, 2 * STANDARD_ROI_EVENTS, roi_index=0, view='cumulative'
        )

        # Polygon should have only first period
        assert_roi_event_count(
            result2, STANDARD_ROI_EVENTS, roi_index=4, view='cumulative'
        )

        # Only polygon readback should be published (rectangle unchanged)
        assert 'roi_polygon' in result2
        assert 'roi_rectangle' not in result2

    def test_updating_rectangle_does_not_affect_polygon(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that updating rectangle ROI does not reset polygon ROI state."""
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=1000.0, num_bins=10, unit='ns')
        )
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Setup: create polygon ROI and accumulate data
        polygon_roi = PolygonROI(
            x=[5.0, 25.0, 25.0, 5.0],
            y=[5.0, 5.0, 25.0, 25.0],
            x_unit='mm',
            y_unit='mm',
        )
        view.accumulate(
            {
                'roi_polygon': PolygonROI.to_concatenated_data_array({4: polygon_roi}),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )
        result1 = view.finalize()

        assert_roi_event_count(
            result1, STANDARD_ROI_EVENTS, roi_index=4, view='cumulative'
        )

        # Now add a rectangle ROI (polygon should keep its state)
        rect_roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        view.accumulate(
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array({0: rect_roi}),
                'detector': sample_detector_events,
            },
            start_time=2000,
            end_time=3000,
        )
        result2 = view.finalize()

        # Polygon cumulative should have continued accumulating
        assert_roi_event_count(
            result2, 2 * STANDARD_ROI_EVENTS, roi_index=4, view='cumulative'
        )

        # Rectangle should have only first period
        assert_roi_event_count(
            result2, STANDARD_ROI_EVENTS, roi_index=0, view='cumulative'
        )

        # Only rectangle readback should be published (polygon unchanged)
        assert 'roi_rectangle' in result2
        assert 'roi_polygon' not in result2


class TestDetectorViewStackedROISpectra:
    """Tests for stacked 2D ROI spectra outputs."""

    def test_stacked_spectra_roi_coordinate(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        standard_toa_edges: TOAEdges,
    ) -> None:
        """Test that roi coordinate correctly identifies ROI indices."""
        params = DetectorViewParams(toa_edges=standard_toa_edges)
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Create two ROIs at indices 0 and 3 (sparse, not contiguous)
        roi0 = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        roi3 = make_rectangle_roi(25.0, 35.0, 25.0, 35.0, 'mm', 'mm')

        view.accumulate(
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array(
                    {0: roi0, 3: roi3}
                )
            },
            start_time=1000,
            end_time=2000,
        )
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result = view.finalize()

        stacked = result['roi_spectra_current']

        # roi coordinate should identify actual ROI indices
        assert 'roi' in stacked.coords
        roi_indices = stacked.coords['roi'].values
        assert list(roi_indices) == [0, 3]  # Sorted order

        # roi should have no unit (just an index)
        assert stacked.coords['roi'].unit is None

    def test_stacked_spectra_sorted_by_index(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        standard_toa_edges: TOAEdges,
    ) -> None:
        """Test that ROIs are stacked in sorted order by index."""
        params = DetectorViewParams(toa_edges=standard_toa_edges)
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Create ROIs in non-sorted order (index 3 before index 1)
        roi3 = make_rectangle_roi(25.0, 35.0, 25.0, 35.0, 'mm', 'mm')  # pixel 15 only
        roi1 = make_rectangle_roi(5.0, 15.0, -5.0, 5.0, 'mm', 'mm')  # pixel 1 only

        # Send in non-sorted order
        view.accumulate(
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array(
                    {3: roi3, 1: roi1}
                )
            },
            start_time=1000,
            end_time=2000,
        )
        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result = view.finalize()

        stacked = result['roi_spectra_current']
        roi_indices = list(stacked.coords['roi'].values)

        # Should be sorted: [1, 3] not [3, 1]
        assert roi_indices == [1, 3]

        # Verify data matches: position 0 should be ROI 1, position 1 should be ROI 3
        # ROI 1 covers pixel 1 (1 event), ROI 3 covers pixel 15 (1 event)
        assert sc.sum(stacked['roi', 0]).value == 1  # ROI 1
        assert sc.sum(stacked['roi', 1]).value == 1  # ROI 3

    def test_stacked_spectra_empty_when_no_rois(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        standard_toa_edges: TOAEdges,
    ) -> None:
        """Test that stacked spectra are empty 2D arrays when no ROIs configured."""
        params = DetectorViewParams(toa_edges=standard_toa_edges)
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result = view.finalize()

        # Should have empty stacked outputs
        stacked_current = result['roi_spectra_current']
        stacked_cumulative = result['roi_spectra_cumulative']

        # Should be 2D with 0 ROIs
        assert stacked_current.dims == ('roi', 'time_of_arrival')
        assert stacked_current.sizes['roi'] == 0
        assert stacked_current.sizes['time_of_arrival'] == standard_toa_edges.num_bins

        assert stacked_cumulative.dims == ('roi', 'time_of_arrival')
        assert stacked_cumulative.sizes['roi'] == 0
        assert (
            stacked_cumulative.sizes['time_of_arrival'] == standard_toa_edges.num_bins
        )

        # Should have toa coordinate even when empty
        assert 'time_of_arrival' in stacked_current.coords
        assert 'time_of_arrival' in stacked_cumulative.coords

        # Empty roi coordinate
        assert 'roi' in stacked_current.coords
        assert len(stacked_current.coords['roi'].values) == 0

        # Current should still have time coord
        assert 'time' in stacked_current.coords
        assert 'time' not in stacked_cumulative.coords

    def test_stacked_spectra_with_multiple_geometry_types(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
        standard_toa_edges: TOAEdges,
    ) -> None:
        """Test stacked spectra with both rectangle and polygon ROIs."""
        params = DetectorViewParams(toa_edges=standard_toa_edges)
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Rectangle at index 0, polygon at index 4
        rect_roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        polygon_roi = PolygonROI(
            x=[25.0, 35.0, 35.0, 25.0],
            y=[25.0, 25.0, 35.0, 35.0],
            x_unit='mm',
            y_unit='mm',
        )

        view.accumulate(
            {
                'roi_rectangle': RectangleROI.to_concatenated_data_array({0: rect_roi}),
                'roi_polygon': PolygonROI.to_concatenated_data_array({4: polygon_roi}),
                'detector': sample_detector_events,
            },
            start_time=1000,
            end_time=2000,
        )
        result = view.finalize()

        stacked = result['roi_spectra_current']

        # Should have both ROIs stacked
        assert stacked.sizes['roi'] == 2

        # Indices should be sorted: [0, 4]
        roi_indices = list(stacked.coords['roi'].values)
        assert roi_indices == [0, 4]

        # Verify counts: ROI 0 (rectangle) has 3 events, ROI 4 (polygon) has 1 event
        assert sc.sum(stacked['roi', 0]).value == STANDARD_ROI_EVENTS
        assert sc.sum(stacked['roi', 1]).value == CORNER_ROI_EVENTS


class TestDetectorViewROISupportDisabled:
    """Tests for DetectorView with roi_support=False."""

    def test_finalize_without_roi_readback_when_roi_support_disabled(
        self,
        mock_rolling_view: RollingDetectorView,
        sample_detector_events: sc.DataArray,
    ) -> None:
        """Test that ROI readbacks are not published when roi_support=False."""
        params = DetectorViewParams()
        view = DetectorView(
            params=params, detector_view=mock_rolling_view, roi_support=False
        )

        view.accumulate(
            {'detector': sample_detector_events}, start_time=1000, end_time=2000
        )
        result = view.finalize()

        # Should have detector view results
        assert 'cumulative' in result
        assert 'current' in result

        # Should NOT have ROI readbacks
        assert 'roi_rectangle' not in result
        assert 'roi_polygon' not in result

        # Should still have roi spectra output (empty)
        assert 'roi_spectra_current' in result
        assert 'roi_spectra_cumulative' in result

    def test_roi_support_disabled_does_not_call_get_detector_coord_units(
        self,
    ) -> None:
        """Test that 1D views don't crash when roi_support=False.

        This tests the fix for the bug where _get_detector_coord_units was called
        even for views that don't support ROIs and have non-2D output.
        """
        from ess.reduce.live import raw

        # Create a 1D view (simulating strip_view)
        detector_number = sc.arange('strip', 5, dtype='int64')
        detector_view = raw.RollingDetectorView.with_logical_view(
            detector_number=detector_number,
            window=1,
            transform=lambda da: da,  # Identity transform
        )

        params = DetectorViewParams()
        view = DetectorView(
            params=params, detector_view=detector_view, roi_support=False
        )

        # Directly add counts to the view (simpler than creating binned events)
        view._view.add_counts([0, 1, 2, 3, 4])
        view._current_start_time = 1000
        view._counts_total = 5
        view._counts_in_toa_range = 5

        # This should NOT raise even though cumulative has 1 dimension
        result = view.finalize()

        assert 'cumulative' in result
        assert len(result['cumulative'].dims) == 1
        # No ROI readbacks should be published
        assert 'roi_rectangle' not in result
        assert 'roi_polygon' not in result
