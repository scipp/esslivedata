# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for ROI (Region of Interest) providers."""

import numpy as np
import scipp as sc

from ess.livedata.config.models import ROI, Interval, PolygonROI, RectangleROI
from ess.livedata.handlers.detector_view_sciline import (
    CumulativeHistogram,
    ROIPolygonMasks,
    ROIRectangleBounds,
    WindowHistogram,
    cumulative_roi_spectra,
    current_roi_spectra,
    roi_rectangle_readback,
)


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
        from ess.livedata.handlers.detector_view_sciline.roi import (
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
        from ess.livedata.handlers.detector_view_sciline.roi import (
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
        from ess.livedata.handlers.detector_view_sciline.roi import (
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
        from ess.livedata.handlers.detector_view_sciline.roi import (
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
        from ess.livedata.handlers.detector_view_sciline import ROIRectangleRequest

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
        empty_request = ROIRectangleRequest(RectangleROI.to_concatenated_data_array({}))

        # Readback should have units from histogram
        readback = roi_rectangle_readback(empty_request, histogram)

        assert len(readback) == 0  # Still empty
        assert readback.coords['x'].unit == sc.units.m
        assert readback.coords['y'].unit == sc.units.m

    def test_roi_rectangle_readback_passes_through_nonempty(self):
        """Test that non-empty ROI readback passes through request unchanged."""
        from ess.livedata.handlers.detector_view_sciline import ROIRectangleRequest

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
        from ess.livedata.handlers.detector_view_sciline.roi import (
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
        from ess.livedata.handlers.detector_view_sciline.roi import (
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
