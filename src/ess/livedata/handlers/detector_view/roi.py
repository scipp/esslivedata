# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ROI (Region of Interest) providers for detector view workflow.

This module provides providers for ROI precomputation, spectra extraction,
and readback in the detector view workflow.
"""

from __future__ import annotations

import numpy as np
import scipp as sc

from ess.livedata.config import models

from .types import (
    CumulativeHistogram,
    CumulativeROISpectra,
    CurrentROISpectra,
    ROIPolygonMasks,
    ROIPolygonReadback,
    ROIPolygonRequest,
    ROIRectangleBounds,
    ROIRectangleReadback,
    ROIRectangleRequest,
    ScreenCoordInfo,
    WindowHistogram,
)

# ============================================================================
# Providers - ROI Precomputation
# ============================================================================


def precompute_roi_rectangle_bounds(
    coord_info: ScreenCoordInfo,
    rectangle_request: ROIRectangleRequest,
) -> ROIRectangleBounds:
    """
    Precompute bounds for rectangle ROIs.

    This is computed once when ROI configuration changes, not on every data update.
    The bounds are stored as scipp Variables for label-based slicing.

    Parameters
    ----------
    coord_info:
        Screen coordinate information (dimension names and edges).
    rectangle_request:
        Rectangle ROI configuration.

    Returns
    -------
    :
        Dict mapping ROI index to bounds dict for slicing.
    """
    if rectangle_request is None or len(rectangle_request) == 0:
        return ROIRectangleBounds({})

    y_dim = coord_info.y_dim
    x_dim = coord_info.x_dim

    bounds_dict: dict[int, dict[str, tuple[sc.Variable, sc.Variable]]] = {}
    rois = models.ROI.from_concatenated_data_array(rectangle_request)

    for idx, roi in rois.items():
        if isinstance(roi, models.RectangleROI):
            roi_bounds = roi.get_bounds(x_dim=x_dim, y_dim=y_dim)
            bounds_dict[idx] = roi_bounds

    return ROIRectangleBounds(bounds_dict)


def precompute_roi_polygon_masks(
    coord_info: ScreenCoordInfo,
    polygon_request: ROIPolygonRequest,
) -> ROIPolygonMasks:
    """
    Precompute boolean masks for polygon ROIs.

    This is computed once when ROI configuration changes, not on every data update.
    Masks are True OUTSIDE the polygon (scipp convention: True = excluded from sum).

    Parameters
    ----------
    coord_info:
        Screen coordinate information (dimension names and edges).
    polygon_request:
        Polygon ROI configuration.

    Returns
    -------
    :
        Dict mapping ROI index to 2D mask Variable.
    """
    if polygon_request is None or len(polygon_request) == 0:
        return ROIPolygonMasks({})

    y_dim = coord_info.y_dim
    x_dim = coord_info.x_dim
    y_edges = coord_info.y_edges
    x_edges = coord_info.x_edges

    # Compute bin centers for point-in-polygon test
    x_centers = sc.midpoints(x_edges)
    y_centers = sc.midpoints(y_edges)

    masks_dict: dict[int, sc.Variable] = {}
    rois = models.ROI.from_concatenated_data_array(polygon_request)

    for idx, roi in rois.items():
        if isinstance(roi, models.PolygonROI):
            mask = _compute_polygon_mask(
                roi, x_centers=x_centers, y_centers=y_centers, x_dim=x_dim, y_dim=y_dim
            )
            masks_dict[idx] = mask

    return ROIPolygonMasks(masks_dict)


def _compute_polygon_mask(
    roi: models.PolygonROI,
    *,
    x_centers: sc.Variable,
    y_centers: sc.Variable,
    x_dim: str,
    y_dim: str,
) -> sc.Variable:
    """
    Compute boolean mask for a polygon ROI.

    The mask is True OUTSIDE the polygon (values to exclude in sum).

    Parameters
    ----------
    roi:
        Polygon ROI with vertices.
    x_centers:
        Bin centers for x dimension.
    y_centers:
        Bin centers for y dimension.
    x_dim:
        Name of x dimension.
    y_dim:
        Name of y dimension.

    Returns
    -------
    :
        2D boolean mask Variable with dims (y_dim, x_dim).
    """
    from matplotlib.path import Path

    # Get polygon vertices
    x_vertices = roi.x
    y_vertices = roi.y

    # Convert centers to correct units if needed
    if roi.x_unit is not None:
        x_vals = x_centers.to(unit=roi.x_unit).values
    else:
        x_vals = np.arange(len(x_centers))

    if roi.y_unit is not None:
        y_vals = y_centers.to(unit=roi.y_unit).values
    else:
        y_vals = np.arange(len(y_centers))

    # Create 2D grid of points
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Point-in-polygon test
    polygon_path = Path(list(zip(x_vertices, y_vertices, strict=True)))
    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside_flat = polygon_path.contains_points(points)
    inside_2d = inside_flat.reshape(xx.shape)

    # Return mask as True OUTSIDE polygon (scipp mask convention: True = excluded)
    return sc.array(dims=[y_dim, x_dim], values=~inside_2d)


# ============================================================================
# Providers - ROI Spectra Extraction
# ============================================================================


def _extract_roi_spectra_precomputed(
    histogram_3d: sc.DataArray,
    rectangle_bounds: dict[int, dict[str, tuple[sc.Variable, sc.Variable]]],
    polygon_masks: dict[int, sc.Variable],
) -> sc.DataArray:
    """
    Extract TOF spectra from 3D histogram using precomputed ROI data.

    Parameters
    ----------
    histogram_3d:
        3D histogram with dims (y, x, tof).
    rectangle_bounds:
        Precomputed bounds for rectangle ROIs.
    polygon_masks:
        Precomputed masks for polygon ROIs (True = excluded).

    Returns
    -------
    :
        2D DataArray with dims (roi, tof) containing spectra for each ROI.
        Returns empty DataArray with shape (0, n_tof) if no ROIs configured.
    """
    tof_dim = 'tof' if 'tof' in histogram_3d.dims else histogram_3d.dims[-1]
    tof_coord = histogram_3d.coords[tof_dim]
    n_tof = histogram_3d.sizes[tof_dim]

    # Get spatial dims (all dims except tof)
    spatial_dims = [d for d in histogram_3d.dims if d != tof_dim]
    if len(spatial_dims) != 2:
        raise ValueError(
            f"Expected 2 spatial dims, got {len(spatial_dims)}: {spatial_dims}"
        )
    y_dim, x_dim = spatial_dims

    spectra: list[sc.DataArray] = []
    roi_indices: list[int] = []

    # Process rectangle ROIs using precomputed bounds
    for idx, bounds in rectangle_bounds.items():
        x_low, x_high = bounds[x_dim]
        y_low, y_high = bounds[y_dim]
        sliced = histogram_3d[y_dim, y_low:y_high][x_dim, x_low:x_high]
        spectrum = sliced.sum(dim=[y_dim, x_dim])
        spectra.append(spectrum)
        roi_indices.append(idx)

    # Process polygon ROIs using precomputed masks
    for idx, mask in polygon_masks.items():
        # Add temporary mask to histogram, sum, then remove
        # scipp's sum ignores masked values
        masked = histogram_3d.copy(deep=False)
        masked.masks['_roi_polygon'] = mask
        spectrum = masked.sum(dim=[y_dim, x_dim])
        spectra.append(spectrum)
        roi_indices.append(idx)

    # Build output DataArray
    if not spectra:
        # Return empty DataArray with correct structure
        return sc.DataArray(
            data=sc.zeros(dims=['roi', tof_dim], shape=[0, n_tof], unit='counts'),
            coords={
                'roi': sc.array(dims=['roi'], values=[], dtype='int32'),
                tof_dim: tof_coord,
            },
        )

    # Stack spectra along roi dimension
    stacked = sc.concat(spectra, dim='roi')
    stacked.coords['roi'] = sc.array(dims=['roi'], values=roi_indices, dtype='int32')
    return stacked


def cumulative_roi_spectra(
    data_3d: CumulativeHistogram,
    rectangle_bounds: ROIRectangleBounds,
    polygon_masks: ROIPolygonMasks,
) -> CumulativeROISpectra:
    """
    Extract ROI spectra from cumulative histogram using precomputed ROI data.

    Parameters
    ----------
    data_3d:
        Cumulative 3D histogram (y, x, tof).
    rectangle_bounds:
        Precomputed bounds for rectangle ROIs.
    polygon_masks:
        Precomputed masks for polygon ROIs.

    Returns
    -------
    :
        ROI spectra with dims (roi, tof).
    """
    return CumulativeROISpectra(
        _extract_roi_spectra_precomputed(data_3d, rectangle_bounds, polygon_masks)
    )


def current_roi_spectra(
    data_3d: WindowHistogram,
    rectangle_bounds: ROIRectangleBounds,
    polygon_masks: ROIPolygonMasks,
) -> CurrentROISpectra:
    """
    Extract ROI spectra from current window histogram using precomputed ROI data.

    Parameters
    ----------
    data_3d:
        Current window 3D histogram (y, x, tof).
    rectangle_bounds:
        Precomputed bounds for rectangle ROIs.
    polygon_masks:
        Precomputed masks for polygon ROIs.

    Returns
    -------
    :
        ROI spectra with dims (roi, tof).
    """
    return CurrentROISpectra(
        _extract_roi_spectra_precomputed(data_3d, rectangle_bounds, polygon_masks)
    )


def _get_coord_units_from_histogram(
    histogram: sc.DataArray,
) -> dict[str, sc.Unit | None]:
    """Extract coordinate units from histogram for ROI readback.

    Maps histogram spatial coordinate units to ROI 'x' and 'y' coordinates.
    Assumes histogram dims are (y, x, tof) or similar with last dim being TOF.
    """
    tof_dim = 'tof' if 'tof' in histogram.dims else histogram.dims[-1]
    spatial_dims = [d for d in histogram.dims if d != tof_dim]

    if len(spatial_dims) != 2:
        return {'x': None, 'y': None}

    y_dim, x_dim = spatial_dims

    def get_unit_for_dim(dim: str) -> sc.Unit | None:
        coord = histogram.coords.get(dim)
        if coord is not None:
            return coord.unit
        return None

    return {'x': get_unit_for_dim(x_dim), 'y': get_unit_for_dim(y_dim)}


def roi_rectangle_readback(
    request: ROIRectangleRequest,
    histogram: CumulativeHistogram,
) -> ROIRectangleReadback:
    """
    Produce ROI rectangle readback with correct coordinate units.

    If request has ROIs, returns them unchanged. If empty, creates empty
    DataArray with coordinate units from the histogram so the frontend
    knows what units to use when creating ROIs.

    Parameters
    ----------
    request:
        ROI rectangle request from context.
    histogram:
        Cumulative histogram with coordinate units.

    Returns
    -------
    :
        ROI readback with correct coordinate units.
    """
    if request is not None and len(request) > 0:
        return ROIRectangleReadback(request)

    coord_units = _get_coord_units_from_histogram(histogram)
    return ROIRectangleReadback(
        models.RectangleROI.to_concatenated_data_array({}, coord_units=coord_units)
    )


def roi_polygon_readback(
    request: ROIPolygonRequest,
    histogram: CumulativeHistogram,
) -> ROIPolygonReadback:
    """
    Produce ROI polygon readback with correct coordinate units.

    If request has ROIs, returns them unchanged. If empty, creates empty
    DataArray with coordinate units from the histogram so the frontend
    knows what units to use when creating ROIs.

    Parameters
    ----------
    request:
        ROI polygon request from context.
    histogram:
        Cumulative histogram with coordinate units.

    Returns
    -------
    :
        ROI readback with correct coordinate units.
    """
    if request is not None and len(request) > 0:
        return ROIPolygonReadback(request)

    coord_units = _get_coord_units_from_histogram(histogram)
    return ROIPolygonReadback(
        models.PolygonROI.to_concatenated_data_array({}, coord_units=coord_units)
    )
