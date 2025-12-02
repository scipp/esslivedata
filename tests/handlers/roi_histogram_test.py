# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unit tests for the ROIHistogram class."""

import pytest
import scipp as sc

from ess.livedata.config.models import EllipseROI, Interval, PolygonROI, RectangleROI
from ess.livedata.handlers.accumulators import GroupIntoPixels
from ess.livedata.handlers.roi_histogram import ROIHistogram
from ess.livedata.handlers.to_nxevent_data import DetectorEvents
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
def toa_linspace() -> sc.Variable:
    """Linspace version of standard TOA edges for direct use."""
    return sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')


@pytest.fixture
def make_grouped_events(detector_number: sc.Variable):
    """Factory fixture to create grouped detector events."""

    def _make(pixel_ids: list[int], toa_values: list[int]) -> sc.DataArray:
        events = DetectorEvents(
            pixel_id=pixel_ids, time_of_arrival=toa_values, unit='ns'
        )
        grouper = GroupIntoPixels(detector_number=detector_number)
        grouper.add(0, events)
        return grouper.get()

    return _make


class TestROIHistogram:
    """Unit tests for the ROIHistogram class."""

    def test_initialization_with_model(
        self,
        detector_indices: sc.DataArray,
        toa_linspace: sc.Variable,
    ) -> None:
        """Test that ROIHistogram can be initialized with a model."""
        standard_roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        roi_filter = ROIFilter(detector_indices)

        roi_histogram = ROIHistogram(
            toa_edges=toa_linspace, roi_filter=roi_filter, model=standard_roi
        )

        assert roi_histogram.model == standard_roi
        assert roi_histogram.cumulative is None

    def test_cumulative_accumulation_across_multiple_periods(
        self,
        detector_indices: sc.DataArray,
        toa_linspace: sc.Variable,
        make_grouped_events,
    ) -> None:
        """Test that cumulative correctly sums across multiple get_delta() calls."""
        standard_roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        roi_filter = ROIFilter(detector_indices)

        roi_histogram = ROIHistogram(
            toa_edges=toa_linspace, roi_filter=roi_filter, model=standard_roi
        )

        # First period: 4 events
        grouped1 = make_grouped_events([5, 6, 9, 10], [100, 200, 300, 400])

        roi_histogram.add_data(grouped1)
        delta1 = roi_histogram.get_delta()

        assert sc.sum(delta1).value == 4
        assert sc.sum(roi_histogram.cumulative).value == 4

        # Second period: 3 events
        grouped2 = make_grouped_events([5, 6, 9], [150, 250, 350])

        roi_histogram.add_data(grouped2)
        delta2 = roi_histogram.get_delta()

        assert sc.sum(delta2).value == 3
        assert sc.sum(roi_histogram.cumulative).value == 7  # Accumulated!

    def test_clear_resets_all_state(
        self, detector_number: sc.Variable, detector_indices: sc.DataArray
    ) -> None:
        """Test that clear() resets cumulative and chunks but preserves config."""
        roi = make_rectangle_roi(
            x_min=5.0, x_max=25.0, y_min=5.0, y_max=25.0, x_unit='mm', y_unit='mm'
        )
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')
        roi_filter = ROIFilter(detector_indices)

        roi_histogram = ROIHistogram(
            toa_edges=toa_edges, roi_filter=roi_filter, model=roi
        )

        # Accumulate some data
        events = DetectorEvents(
            pixel_id=[5, 6, 9, 10],
            time_of_arrival=[100, 200, 300, 400],
            unit='ns',
        )
        grouper = GroupIntoPixels(detector_number=detector_number)
        grouper.add(0, events)
        grouped = grouper.get()

        roi_histogram.add_data(grouped)
        roi_histogram.get_delta()
        assert sc.sum(roi_histogram.cumulative).value == 4

        # Clear should reset everything
        roi_histogram.clear()

        assert roi_histogram.cumulative is None
        # Model should be preserved
        assert roi_histogram.model == roi

    def test_empty_histogram_when_no_data(self, detector_indices: sc.DataArray) -> None:
        """Test that get_delta returns empty histogram when no data accumulated."""
        roi = make_rectangle_roi(
            x_min=5.0, x_max=25.0, y_min=5.0, y_max=25.0, x_unit='mm', y_unit='mm'
        )
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')
        roi_filter = ROIFilter(detector_indices)

        roi_histogram = ROIHistogram(
            toa_edges=toa_edges, roi_filter=roi_filter, model=roi
        )

        # Get without adding data
        delta = roi_histogram.get_delta()

        assert isinstance(delta, sc.DataArray)
        assert 'time_of_arrival' in delta.coords
        assert sc.sum(delta).value == 0
        assert sc.sum(roi_histogram.cumulative).value == 0

    def test_polygon_roi_is_supported(self, detector_indices: sc.DataArray) -> None:
        """Test that PolygonROI is supported."""
        polygon_roi = PolygonROI(
            x=[5.0, 25.0, 25.0, 5.0],
            y=[5.0, 5.0, 25.0, 25.0],
            x_unit='mm',
            y_unit='mm',
        )
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')
        roi_filter = ROIFilter(detector_indices)

        roi_histogram = ROIHistogram(
            toa_edges=toa_edges, roi_filter=roi_filter, model=polygon_roi
        )
        assert roi_histogram.model == polygon_roi

    def test_polygon_roi_with_pixel_indices_is_supported(
        self, detector_indices: sc.DataArray
    ) -> None:
        """Test that PolygonROI with pixel indices (no units) is supported."""
        polygon_roi = PolygonROI(
            x=[5.0, 25.0, 25.0, 5.0],
            y=[5.0, 5.0, 25.0, 25.0],
            x_unit=None,
            y_unit=None,
        )
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')
        roi_filter = ROIFilter(detector_indices)

        roi_histogram = ROIHistogram(
            toa_edges=toa_edges, roi_filter=roi_filter, model=polygon_roi
        )
        assert roi_histogram.model == polygon_roi

    def test_ellipse_roi_raises_value_error(
        self, detector_indices: sc.DataArray
    ) -> None:
        """Test that EllipseROI raises ValueError (not yet supported)."""
        ellipse_roi = EllipseROI(
            center_x=15.0,
            center_y=15.0,
            radius_x=10.0,
            radius_y=10.0,
            rotation=0.0,
            unit='mm',
        )
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')
        roi_filter = ROIFilter(detector_indices)

        with pytest.raises(ValueError, match='Unsupported ROI type: EllipseROI'):
            ROIHistogram(toa_edges=toa_edges, roi_filter=roi_filter, model=ellipse_roi)

    def test_toa_edges_unit_conversion_from_microseconds(
        self,
        detector_indices: sc.DataArray,
        make_grouped_events,
    ) -> None:
        """Test that TOA edges in microseconds are converted correctly to ns."""
        roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        roi_filter = ROIFilter(detector_indices)

        # TOA edges in microseconds
        toa_edges_us = sc.linspace('time_of_arrival', 0, 1, num=11, unit='us')
        roi_histogram = ROIHistogram(
            toa_edges=toa_edges_us, roi_filter=roi_filter, model=roi
        )

        # Add events with TOA in range [0, 1000 ns] = [0, 1 us]
        grouped = make_grouped_events([5, 6], [100, 900])
        roi_histogram.add_data(grouped)
        delta = roi_histogram.get_delta()

        # Should have 2 events
        assert sc.sum(delta).value == 2
        # Edges should be in original unit (us)
        assert delta.coords['time_of_arrival'].unit == 'us'

    def test_histogram_bins_match_toa_values(
        self,
        detector_indices: sc.DataArray,
        make_grouped_events,
    ) -> None:
        """Test that events end up in the correct TOA bins."""
        roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        roi_filter = ROIFilter(detector_indices)

        # Create edges with clear bin boundaries: [0-100), [100-200), [200-300) ns
        toa_edges = sc.array(
            dims=['time_of_arrival'], values=[0, 100, 200, 300], unit='ns'
        )
        roi_histogram = ROIHistogram(
            toa_edges=toa_edges, roi_filter=roi_filter, model=roi
        )

        # Add events: 1 in first bin, 2 in second bin, 1 in third bin
        grouped = make_grouped_events(
            [5, 6, 9, 10],  # All within ROI
            [50, 150, 150, 250],  # TOA values
        )
        roi_histogram.add_data(grouped)
        delta = roi_histogram.get_delta()

        # Check bin counts
        assert delta.values[0] == 1  # [0-100) ns
        assert delta.values[1] == 2  # [100-200) ns
        assert delta.values[2] == 1  # [200-300) ns

    def test_events_outside_roi_are_filtered(
        self,
        detector_indices: sc.DataArray,
        make_grouped_events,
    ) -> None:
        """Test that events outside ROI boundaries don't contribute to histogram."""
        # Small ROI covering only center pixels (x: 10-20, y: 10-20)
        roi = make_rectangle_roi(10.0, 20.0, 10.0, 20.0, 'mm', 'mm')
        roi_filter = ROIFilter(detector_indices)
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')

        roi_histogram = ROIHistogram(
            toa_edges=toa_edges, roi_filter=roi_filter, model=roi
        )

        # Pixels: 0 (0,0), 5 (10,10), 10 (20,20), 15 (30,30)
        # Only pixel 5 (10,10) is inside ROI [10-20, 10-20]
        grouped = make_grouped_events(
            [0, 5, 10, 15],  # Mix of inside and outside ROI
            [100, 200, 300, 400],
        )
        roi_histogram.add_data(grouped)
        delta = roi_histogram.get_delta()

        # Only 1 event should be counted (pixel 5 at 10,10)
        assert sc.sum(delta).value == 1

    def test_consecutive_add_data_without_get_delta(
        self,
        detector_indices: sc.DataArray,
        make_grouped_events,
    ) -> None:
        """Test that multiple add_data calls accumulate correctly before get_delta."""
        roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        roi_filter = ROIFilter(detector_indices)
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')

        roi_histogram = ROIHistogram(
            toa_edges=toa_edges, roi_filter=roi_filter, model=roi
        )

        # Add data three times without calling get_delta
        grouped1 = make_grouped_events([5, 6], [100, 200])
        grouped2 = make_grouped_events([9, 10], [300, 400])
        grouped3 = make_grouped_events([5], [500])

        roi_histogram.add_data(grouped1)
        roi_histogram.add_data(grouped2)
        roi_histogram.add_data(grouped3)

        # All 5 events should be in the delta
        delta = roi_histogram.get_delta()
        assert sc.sum(delta).value == 5

    def test_multiple_get_delta_without_add_data(
        self,
        detector_indices: sc.DataArray,
        make_grouped_events,
    ) -> None:
        """Test that consecutive get_delta calls return empty histograms after first."""
        roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        roi_filter = ROIFilter(detector_indices)
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')

        roi_histogram = ROIHistogram(
            toa_edges=toa_edges, roi_filter=roi_filter, model=roi
        )

        # Add some data and get delta
        grouped = make_grouped_events([5, 6], [100, 200])
        roi_histogram.add_data(grouped)
        delta1 = roi_histogram.get_delta()
        assert sc.sum(delta1).value == 2

        # Subsequent get_delta calls should return empty histograms
        delta2 = roi_histogram.get_delta()
        assert sc.sum(delta2).value == 0
        delta3 = roi_histogram.get_delta()
        assert sc.sum(delta3).value == 0

        # But cumulative should still have the original data
        assert sc.sum(roi_histogram.cumulative).value == 2

    def test_add_data_after_clear(
        self,
        detector_indices: sc.DataArray,
        make_grouped_events,
    ) -> None:
        """Test that adding data after clear works correctly."""
        roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        roi_filter = ROIFilter(detector_indices)
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')

        roi_histogram = ROIHistogram(
            toa_edges=toa_edges, roi_filter=roi_filter, model=roi
        )

        # Add data, get delta, clear
        grouped1 = make_grouped_events([5, 6], [100, 200])
        roi_histogram.add_data(grouped1)
        roi_histogram.get_delta()
        assert sc.sum(roi_histogram.cumulative).value == 2

        roi_histogram.clear()
        assert roi_histogram.cumulative is None

        # Add new data after clear
        grouped2 = make_grouped_events([9, 10], [300, 400])
        roi_histogram.add_data(grouped2)
        delta = roi_histogram.get_delta()

        # Should only have the new data
        assert sc.sum(delta).value == 2
        assert sc.sum(roi_histogram.cumulative).value == 2

    def test_histogram_coordinates_and_units(
        self,
        detector_indices: sc.DataArray,
    ) -> None:
        """Test that output histogram has correct coordinates and units."""
        roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        roi_filter = ROIFilter(detector_indices)
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')

        roi_histogram = ROIHistogram(
            toa_edges=toa_edges, roi_filter=roi_filter, model=roi
        )

        delta = roi_histogram.get_delta()

        # Check structure
        assert 'time_of_arrival' in delta.coords
        assert delta.data.unit == 'counts'
        assert delta.coords['time_of_arrival'].unit == 'ns'

        # Check that edges match input
        assert sc.identical(delta.coords['time_of_arrival'], toa_edges)

        # Check number of bins
        assert len(delta.data) == len(toa_edges) - 1

    def test_model_property_getter(
        self,
        detector_indices: sc.DataArray,
    ) -> None:
        """Test the model property returns the ROI model."""
        roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        roi_filter = ROIFilter(detector_indices)
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')

        roi_histogram = ROIHistogram(
            toa_edges=toa_edges, roi_filter=roi_filter, model=roi
        )

        # Model property should return the same object
        assert roi_histogram.model is roi

    def test_cumulative_property_returns_none_initially(
        self,
        detector_indices: sc.DataArray,
    ) -> None:
        """Test that cumulative property returns None before first get_delta."""
        roi = make_rectangle_roi(5.0, 25.0, 5.0, 25.0, 'mm', 'mm')
        roi_filter = ROIFilter(detector_indices)
        toa_edges = sc.linspace('time_of_arrival', 0, 1000, num=11, unit='ns')

        roi_histogram = ROIHistogram(
            toa_edges=toa_edges, roi_filter=roi_filter, model=roi
        )

        assert roi_histogram.cumulative is None
