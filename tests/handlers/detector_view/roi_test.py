# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for ROI (Region of Interest) extraction in detector view workflow.

These tests verify the end-to-end behavior of ROI extraction:
given screen metadata, histogram, and ROI configuration, the correct
spectra are extracted. Tests are decoupled from internal provider
structure to allow implementation flexibility.
"""

from __future__ import annotations

import numpy as np
import scipp as sc

from ess.livedata.config.models import Interval, PolygonROI, RectangleROI
from ess.livedata.handlers.detector_view.roi import (
    precompute_roi_polygon_masks,
    precompute_roi_rectangle_bounds,
    roi_spectra,
)
from ess.livedata.handlers.detector_view.types import (
    ROIPolygonRequest,
    ROIRectangleRequest,
    ScreenMetadata,
)


def extract_roi_spectra(
    screen_metadata: ScreenMetadata,
    histogram: sc.DataArray,
    rectangle_request: sc.DataArray | None = None,
    polygon_request: sc.DataArray | None = None,
) -> sc.DataArray:
    """
    Test helper: full ROI extraction pipeline.

    This wraps the internal ROI providers to test end-to-end behavior
    without coupling tests to the specific provider structure.

    Parameters
    ----------
    screen_metadata:
        Screen metadata with coordinates and sizes.
    histogram:
        3D histogram (y, x, spectral).
    rectangle_request:
        Rectangle ROI request as concatenated DataArray, or None.
    polygon_request:
        Polygon ROI request as concatenated DataArray, or None.

    Returns
    -------
    :
        ROI spectra with dims (roi, spectral).
    """
    rect_req = (
        ROIRectangleRequest(rectangle_request)
        if rectangle_request is not None
        else None
    )
    poly_req = (
        ROIPolygonRequest(polygon_request) if polygon_request is not None else None
    )

    bounds = precompute_roi_rectangle_bounds(screen_metadata, rect_req)
    masks = precompute_roi_polygon_masks(screen_metadata, poly_req)
    return roi_spectra(histogram, bounds, masks)


def make_screen_metadata_from_edges(
    y_range: tuple[float, float] = (0.0, 10.0),
    x_range: tuple[float, float] = (0.0, 10.0),
    n_bins: int = 10,
    unit: str = 'm',
) -> ScreenMetadata:
    """Create screen metadata from bin edges, storing centers as projectors do."""
    y_edges = sc.linspace('y', y_range[0], y_range[1], n_bins + 1, unit=unit)
    x_edges = sc.linspace('x', x_range[0], x_range[1], n_bins + 1, unit=unit)
    return ScreenMetadata(
        coords={'y': sc.midpoints(y_edges), 'x': sc.midpoints(x_edges)},
        sizes={'y': n_bins, 'x': n_bins},
    )


def make_screen_metadata_with_centers(
    y_range: tuple[float, float] = (0.5, 9.5),
    x_range: tuple[float, float] = (0.5, 9.5),
    n_bins: int = 10,
    unit: str = 'm',
) -> ScreenMetadata:
    """Create screen metadata with bin center coordinates."""
    y_centers = sc.linspace('y', y_range[0], y_range[1], n_bins, unit=unit)
    x_centers = sc.linspace('x', x_range[0], x_range[1], n_bins, unit=unit)
    return ScreenMetadata(
        coords={'y': y_centers, 'x': x_centers},
        sizes={'y': n_bins, 'x': n_bins},
    )


def make_screen_metadata_with_none_coords(sizes: dict[str, int]) -> ScreenMetadata:
    """Create screen metadata with None coordinates (logical view)."""
    return ScreenMetadata(coords=dict.fromkeys(sizes, None), sizes=sizes)


def make_uniform_histogram(
    shape: tuple[int, int, int] = (10, 10, 3),
    value: float = 1.0,
    with_coords: bool = True,
) -> sc.DataArray:
    """Create a uniform histogram for testing."""
    y_size, x_size, tof_size = shape
    data = np.full(shape, value)
    coords = {'tof': sc.linspace('tof', 0, 30000, tof_size + 1, unit='ns')}
    if with_coords:
        coords['y'] = sc.linspace('y', 0.0, 10.0, y_size + 1, unit='m')
        coords['x'] = sc.linspace('x', 0.0, 10.0, x_size + 1, unit='m')
    return sc.DataArray(
        data=sc.array(dims=['y', 'x', 'tof'], values=data, unit='counts'),
        coords=coords,
    )


class TestRectangleROIExtraction:
    """Tests for rectangle ROI spectra extraction."""

    def test_rectangle_with_physical_coords(self):
        """Rectangle ROI with physical coordinates extracts correct counts."""
        metadata = make_screen_metadata_from_edges()
        histogram = make_uniform_histogram(value=1.0)

        # Rectangle covering indices 2-4 in both dims (3x3 = 9 pixels)
        # With edges 0-10 over 10 bins, each bin is 1m wide
        # Bins 2,3,4 have centers at 2.5, 3.5, 4.5
        roi = RectangleROI(
            x=Interval(min=2.0, max=5.0, unit='m'),
            y=Interval(min=2.0, max=5.0, unit='m'),
        )
        request = RectangleROI.to_concatenated_data_array({0: roi})

        result = extract_roi_spectra(metadata, histogram, rectangle_request=request)

        assert result.dims == ('roi', 'tof')
        assert result.sizes['roi'] == 1
        # 3x3 = 9 pixels, each with 1 count per TOF bin
        np.testing.assert_array_almost_equal(result.values[0], [9, 9, 9])

    def test_rectangle_with_index_bounds(self):
        """Rectangle ROI with index bounds (no units) works."""
        metadata = make_screen_metadata_from_edges()
        histogram = make_uniform_histogram(value=1.0)

        # Index-based rectangle: indices 0-4 in both dims (5x5 = 25 pixels)
        roi = RectangleROI(
            x=Interval(min=0, max=5, unit=None),
            y=Interval(min=0, max=5, unit=None),
        )
        request = RectangleROI.to_concatenated_data_array({0: roi})

        result = extract_roi_spectra(metadata, histogram, rectangle_request=request)

        assert result.sizes['roi'] == 1
        # 5x5 = 25 pixels
        np.testing.assert_array_almost_equal(result.values[0], [25, 25, 25])

    def test_rectangle_with_none_coords_projector(self):
        """Rectangle ROI works with projector that has None coordinates."""
        metadata = make_screen_metadata_with_none_coords({'y': 10, 'x': 10})
        histogram = make_uniform_histogram(with_coords=False)

        # Index-based rectangle
        roi = RectangleROI(
            x=Interval(min=2, max=5, unit=None),
            y=Interval(min=2, max=5, unit=None),
        )
        request = RectangleROI.to_concatenated_data_array({0: roi})

        result = extract_roi_spectra(metadata, histogram, rectangle_request=request)

        assert result.sizes['roi'] == 1
        # 3x3 = 9 pixels
        np.testing.assert_array_almost_equal(result.values[0], [9, 9, 9])

    def test_multiple_rectangles(self):
        """Multiple rectangle ROIs are extracted correctly."""
        metadata = make_screen_metadata_from_edges()
        histogram = make_uniform_histogram(value=1.0)

        roi0 = RectangleROI(
            x=Interval(min=0, max=2, unit='m'),
            y=Interval(min=0, max=2, unit='m'),
        )
        roi1 = RectangleROI(
            x=Interval(min=5, max=10, unit='m'),
            y=Interval(min=5, max=10, unit='m'),
        )
        request = RectangleROI.to_concatenated_data_array({0: roi0, 1: roi1})

        result = extract_roi_spectra(metadata, histogram, rectangle_request=request)

        assert result.sizes['roi'] == 2
        assert list(result.coords['roi'].values) == [0, 1]
        # ROI 0: 2x2 = 4 pixels, ROI 1: 5x5 = 25 pixels
        np.testing.assert_array_almost_equal(result.values[0], [4, 4, 4])
        np.testing.assert_array_almost_equal(result.values[1], [25, 25, 25])


class TestPolygonROIExtraction:
    """Tests for polygon ROI spectra extraction."""

    def test_polygon_with_bin_edges(self):
        """Polygon ROI with bin edge coordinates extracts correct counts."""
        metadata = make_screen_metadata_from_edges()
        histogram = make_uniform_histogram(value=1.0)

        # Square polygon covering bins 2-4 in both dims (3x3 = 9 pixels)
        # Polygon vertices at (2,2), (5,2), (5,5), (2,5)
        # Bin centers at 2.5, 3.5, 4.5 are inside
        roi = PolygonROI(
            x=[2.0, 5.0, 5.0, 2.0],
            y=[2.0, 2.0, 5.0, 5.0],
            x_unit='m',
            y_unit='m',
        )
        request = PolygonROI.to_concatenated_data_array({0: roi})

        result = extract_roi_spectra(metadata, histogram, polygon_request=request)

        assert result.dims == ('roi', 'tof')
        assert result.sizes['roi'] == 1
        # 3x3 = 9 pixels inside the square
        np.testing.assert_array_almost_equal(result.values[0], [9, 9, 9])

    def test_polygon_with_bin_centers(self):
        """Polygon ROI works when projector provides bin centers (not edges).

        This tests that the code correctly handles coordinates that are
        already bin centers, without incorrectly calling sc.midpoints().
        """
        metadata = make_screen_metadata_with_centers()
        histogram = make_uniform_histogram(value=1.0)

        # Square polygon: same logic as bin_edges test
        roi = PolygonROI(
            x=[2.0, 5.0, 5.0, 2.0],
            y=[2.0, 2.0, 5.0, 5.0],
            x_unit='m',
            y_unit='m',
        )
        request = PolygonROI.to_concatenated_data_array({0: roi})

        result = extract_roi_spectra(metadata, histogram, polygon_request=request)

        assert result.sizes['roi'] == 1
        # With centers at 0.5, 1.5, ..., 9.5, bins with centers 2.5, 3.5, 4.5
        # are inside the polygon (indices 2, 3, 4) -> 3x3 = 9 pixels
        np.testing.assert_array_almost_equal(result.values[0], [9, 9, 9])

    def test_polygon_with_none_coords(self):
        """Polygon ROI works with None coordinates (logical view).

        When projector returns None for coordinates, index-based polygon
        ROIs should work by synthesizing integer indices.
        """
        metadata = make_screen_metadata_with_none_coords({'y': 10, 'x': 10})
        histogram = make_uniform_histogram(with_coords=False)

        # Index-based square polygon covering indices 2-4
        # Use boundaries at 1.5 and 4.5 to clearly include only indices 2, 3, 4
        # (avoiding ambiguity about points exactly on polygon edges)
        roi = PolygonROI(
            x=[1.5, 4.5, 4.5, 1.5],
            y=[1.5, 1.5, 4.5, 4.5],
            x_unit=None,
            y_unit=None,
        )
        request = PolygonROI.to_concatenated_data_array({0: roi})

        result = extract_roi_spectra(metadata, histogram, polygon_request=request)

        assert result.sizes['roi'] == 1
        # 3x3 = 9 pixels (indices 2, 3, 4 in both dimensions)
        np.testing.assert_array_almost_equal(result.values[0], [9, 9, 9])

    def test_triangle_polygon_extracts_correct_region(self):
        """Triangle polygon extracts only the correct pixels."""
        metadata = make_screen_metadata_from_edges()

        # Histogram where each pixel has value = y_index
        data = np.zeros((10, 10, 3))
        for y in range(10):
            data[y, :, :] = y
        histogram = sc.DataArray(
            data=sc.array(dims=['y', 'x', 'tof'], values=data, unit='counts'),
            coords={
                'tof': sc.linspace('tof', 0, 30000, 4, unit='ns'),
                'y': sc.linspace('y', 0.0, 10.0, 11, unit='m'),
                'x': sc.linspace('x', 0.0, 10.0, 11, unit='m'),
            },
        )

        # Triangle with vertices at (0,0), (10,0), (0,10)
        # This covers the lower-left triangle of the grid
        roi = PolygonROI(
            x=[0.0, 10.0, 0.0],
            y=[0.0, 0.0, 10.0],
            x_unit='m',
            y_unit='m',
        )
        request = PolygonROI.to_concatenated_data_array({0: roi})

        result = extract_roi_spectra(metadata, histogram, polygon_request=request)

        # The triangle covers approximately half the pixels
        # Exact count depends on point-in-polygon for centers on the diagonal
        assert result.sum().value > 0
        # Full grid sum = (0+1+...+9)*10*3 = 45*10*3 = 1350
        # Triangle should cover roughly half, so less than 75% of total
        assert result.sum().value < 1350 * 0.75


class TestMixedROIExtraction:
    """Tests for mixed rectangle and polygon ROI extraction."""

    def test_rectangles_and_polygons_together(self):
        """Both rectangle and polygon ROIs can be extracted together."""
        metadata = make_screen_metadata_from_edges()
        histogram = make_uniform_histogram(value=1.0)

        rect_roi = RectangleROI(
            x=Interval(min=0, max=2, unit='m'),
            y=Interval(min=0, max=2, unit='m'),
        )
        rect_request = RectangleROI.to_concatenated_data_array({0: rect_roi})

        poly_roi = PolygonROI(
            x=[5.0, 10.0, 10.0, 5.0],
            y=[5.0, 5.0, 10.0, 10.0],
            x_unit='m',
            y_unit='m',
        )
        poly_request = PolygonROI.to_concatenated_data_array({100: poly_roi})

        result = extract_roi_spectra(
            metadata,
            histogram,
            rectangle_request=rect_request,
            polygon_request=poly_request,
        )

        assert result.sizes['roi'] == 2
        # Rectangle: 2x2 = 4 pixels, Polygon: 5x5 = 25 pixels
        roi_indices = list(result.coords['roi'].values)
        assert 0 in roi_indices
        assert 100 in roi_indices


class TestEmptyROIRequests:
    """Tests for empty or None ROI requests."""

    def test_empty_roi_requests_return_empty_spectra(self):
        """Empty ROI requests return empty spectra.

        Covers None, empty rectangle, and empty polygon requests.
        """
        metadata = make_screen_metadata_from_edges()
        histogram = make_uniform_histogram()

        # No ROI requests at all
        result_none = extract_roi_spectra(metadata, histogram)
        assert result_none.dims == ('roi', 'tof')
        assert result_none.sizes['roi'] == 0
        assert result_none.sizes['tof'] == 3

        # Empty rectangle request
        empty_rect = RectangleROI.to_concatenated_data_array({})
        result_rect = extract_roi_spectra(
            metadata, histogram, rectangle_request=empty_rect
        )
        assert result_rect.sizes['roi'] == 0

        # Empty polygon request
        empty_poly = PolygonROI.to_concatenated_data_array({})
        result_poly = extract_roi_spectra(
            metadata, histogram, polygon_request=empty_poly
        )
        assert result_poly.sizes['roi'] == 0
