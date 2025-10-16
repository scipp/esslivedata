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
    return {'roi': RectangleROI.to_concatenated_data_array({index: roi})}


def assert_has_detector_view_results(result: dict) -> None:
    """Assert result contains standard detector view outputs."""
    assert 'cumulative' in result
    assert 'current' in result


def assert_has_roi_results(result: dict, roi_index: int = 0) -> None:
    """Assert result contains ROI histogram outputs for given index."""
    assert f'roi_cumulative_{roi_index}' in result
    assert f'roi_current_{roi_index}' in result


def assert_roi_config_published(result: dict) -> None:
    """Assert ROI configuration was published in result."""
    assert 'roi_rectangle' in result


def assert_roi_event_count(
    result: dict, expected: int, roi_index: int = 0, view: str = 'current'
) -> None:
    """Assert ROI histogram has expected event count."""
    key = f'roi_{view}_{roi_index}'
    actual = sc.sum(result[key]).value
    assert actual == expected, f"Expected {expected} events in {key}, got {actual}"


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
        self, mock_rolling_view: RollingDetectorView
    ) -> None:
        """Test that DetectorView can be initialized with default parameters."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        assert view is not None
        # Verify no ROI results are present when no ROI configured
        result = view.finalize()
        assert 'cumulative' in result
        assert 'current' in result
        assert not any(key.startswith('roi_') for key in result)

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

        assert_has_detector_view_results(result)
        # ROI results should not be present without ROI configuration
        assert 'roi_cumulative_0' not in result
        assert 'roi_current_0' not in result

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

        view.accumulate({'detector': sample_detector_events})
        result1 = view.finalize()
        assert sc.sum(result1['current']).value == TOTAL_SAMPLE_EVENTS

        # Clear the view
        view.clear()

        # After clear, finalize should return zero counts
        result2 = view.finalize()
        assert sc.sum(result2['current']).value == 0
        assert sc.sum(result2['cumulative']).value == 0


class TestDetectorViewROIMechanism:
    """Tests for ROI configuration and histogram accumulation."""

    def test_roi_configuration_via_accumulate(
        self, mock_rolling_view: RollingDetectorView, standard_roi: RectangleROI
    ) -> None:
        """Test that ROI configuration can be set via accumulate."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Send ROI configuration
        view.accumulate(roi_to_accumulate_data(standard_roi))

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
        self, mock_rolling_view: RollingDetectorView, standard_roi: RectangleROI
    ) -> None:
        """Test that sending only ROI (no detector data) produces empty histograms."""
        params = DetectorViewParams()
        view = DetectorView(params=params, detector_view=mock_rolling_view)

        # Send ROI configuration without detector data
        view.accumulate(roi_to_accumulate_data(standard_roi))

        # ROI should be configured but no histogram data accumulated yet
        result = view.finalize()
        # ROI results should be present (even if empty/zero) once ROI is configured
        assert_has_roi_results(result)
        assert_roi_config_published(result)
        # All counts should be zero since no events were accumulated
        assert sc.sum(result['roi_cumulative_0']).value == 0
        assert sc.sum(result['roi_current_0']).value == 0

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
        view.accumulate(roi_to_accumulate_data(standard_roi))

        # Now accumulate detector events
        view.accumulate({'detector': sample_detector_events})

        result = view.finalize()

        # Should have both detector view and ROI results
        assert_has_detector_view_results(result)
        assert_has_roi_results(result)
        assert_roi_config_published(result)

        # Verify histogram structure
        roi_histogram = result['roi_current_0']
        assert isinstance(roi_histogram, sc.DataArray)
        assert 'time_of_arrival' in roi_histogram.coords
        assert roi_histogram.ndim == 1  # Histogram is 1D

        # Verify expected event count
        assert_roi_event_count(result, STANDARD_ROI_EVENTS)

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
        view.accumulate(roi_to_accumulate_data(standard_roi))

        # First accumulation
        view.accumulate({'detector': sample_detector_events})
        result1 = view.finalize()
        assert_roi_event_count(result1, STANDARD_ROI_EVENTS)

        # Second accumulation with same events
        view.accumulate({'detector': sample_detector_events})
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
        view.accumulate(roi_to_accumulate_data(standard_roi))
        view.accumulate({'detector': sample_detector_events})

        result1 = view.finalize()
        assert_roi_config_published(result1)

        # Second accumulation without ROI update
        view.accumulate({'detector': sample_detector_events})
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
        view.accumulate(roi_to_accumulate_data(standard_roi))
        view.accumulate({'detector': sample_detector_events})
        result1 = view.finalize()

        # Verify we accumulated some events
        assert sc.sum(result1['roi_cumulative_0']).value > 0

        # Clear should reset cumulative
        view.clear()

        # After clear, ROI cumulative should be reset to zero
        result2 = view.finalize()
        assert_has_roi_results(result2)  # ROI config still active
        assert sc.sum(result2['roi_cumulative_0']).value == 0
        assert sc.sum(result2['roi_current_0']).value == 0

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
        view.accumulate(roi_to_accumulate_data(standard_roi))
        view.accumulate({'detector': sample_detector_events})
        result1 = view.finalize()

        assert_roi_event_count(result1, STANDARD_ROI_EVENTS, view='current')
        assert_roi_event_count(result1, STANDARD_ROI_EVENTS, view='cumulative')

        # Now change ROI to cover different pixels (1, 2, 5, 6)
        view.accumulate(roi_to_accumulate_data(wide_roi))
        view.accumulate({'detector': sample_detector_events})
        result2 = view.finalize()

        assert_roi_event_count(result2, WIDE_ROI_EVENTS, view='current')

        # CRITICAL: Cumulative should reset when ROI changes
        # It should NOT be roi1_events + roi2_events (which would be 7)
        # It should be just roi2_events since we changed the ROI
        assert sc.sum(result2['roi_cumulative_0']).value == WIDE_ROI_EVENTS, (
            f"Expected cumulative to reset to {WIDE_ROI_EVENTS} when ROI changes, "
            f"got {sc.sum(result2['roi_cumulative_0']).value}. "
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
            }
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
                'roi': RectangleROI.to_concatenated_data_array({0: roi}),
                'detector': sample_detector_events,
            }
        )
        result1 = view.finalize()

        expected_events_in_roi = 3
        assert sc.sum(result1['roi_current_0']).value == expected_events_in_roi
        assert 'roi_rectangle' in result1  # Published on first update

        # Second call: detector data only (ROI should persist)
        view.accumulate({'detector': sample_detector_events})
        result2 = view.finalize()

        assert sc.sum(result2['roi_current_0']).value == expected_events_in_roi
        assert sc.sum(result2['roi_cumulative_0']).value == 2 * expected_events_in_roi
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
            {
                'roi': RectangleROI.to_concatenated_data_array({0: roi}),
                'detector': sample_detector_events,
            }
        )
        result2 = view.finalize()

        expected_events_in_roi = 3
        assert 'roi_cumulative_0' in result2
        assert 'roi_current_0' in result2
        assert sc.sum(result2['roi_current_0']).value == expected_events_in_roi
        assert sc.sum(result2['roi_cumulative_0']).value == expected_events_in_roi

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
                'roi': RectangleROI.to_concatenated_data_array({0: roi1}),
                'detector': sample_detector_events,
            }
        )
        result1 = view.finalize()

        expected_roi1_events = 3
        assert sc.sum(result1['roi_current_0']).value == expected_roi1_events
        assert sc.sum(result1['roi_cumulative_0']).value == expected_roi1_events

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
                'roi': RectangleROI.to_concatenated_data_array({0: roi2}),
                'detector': sample_detector_events,
            }
        )
        result2 = view.finalize()

        expected_roi2_events = 4
        assert sc.sum(result2['roi_current_0']).value == expected_roi2_events
        # Cumulative should reset when ROI changes
        assert sc.sum(result2['roi_cumulative_0']).value == expected_roi2_events
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
        view.accumulate({'roi': RectangleROI.to_concatenated_data_array({0: roi1})})

        roi2 = make_rectangle_roi(
            x_min=10.0, x_max=20.0, y_min=10.0, y_max=20.0, x_unit='mm', y_unit='mm'
        )
        view.accumulate({'roi': RectangleROI.to_concatenated_data_array({0: roi2})})

        result = view.finalize()

        # Should have ROI configured but with zero events
        assert 'roi_cumulative_0' in result
        assert 'roi_current_0' in result
        assert 'roi_rectangle' in result
        assert sc.sum(result['roi_cumulative_0']).value == 0
        assert sc.sum(result['roi_current_0']).value == 0

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
        view.accumulate(roi_to_accumulate_data(corner_roi))
        view.accumulate({'detector': sample_detector_events})

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
                'roi': RectangleROI.to_concatenated_data_array({0: roi1}),
                'detector': sample_detector_events,
            }
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
            {
                'roi': RectangleROI.to_concatenated_data_array({0: roi2}),
                'detector': sample_detector_events,
            }
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
