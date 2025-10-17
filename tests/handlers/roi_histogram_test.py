# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unit tests for the ROIHistogram class."""

import pytest
import scipp as sc

from ess.livedata.config.models import Interval, RectangleROI
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
