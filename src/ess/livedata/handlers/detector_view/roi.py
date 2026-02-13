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
    AccumulatedHistogram,
    AccumulationMode,
    ROIPolygonMasks,
    ROIPolygonReadback,
    ROIPolygonRequest,
    ROIRectangleBounds,
    ROIRectangleReadback,
    ROIRectangleRequest,
    ROISpectra,
    ScreenMetadata,
)


def precompute_roi_rectangle_bounds(
    screen_metadata: ScreenMetadata,
    rectangle_request: ROIRectangleRequest,
) -> ROIRectangleBounds:
    """
    Precompute bounds for rectangle ROIs.

    This is computed once when ROI configuration changes, not on every data update.
    The bounds are stored as scipp Variables for label-based slicing.

    Parameters
    ----------
    screen_metadata:
        Screen metadata with coordinate information.
    rectangle_request:
        Rectangle ROI configuration.

    Returns
    -------
    :
        Dict mapping ROI index to bounds dict for slicing.
    """
    if rectangle_request is None or len(rectangle_request) == 0:
        return ROIRectangleBounds({})

    dims = list(screen_metadata.coords.keys())
    if len(dims) < 2:
        raise ValueError(f"Rectangle ROIs require at least 2 dimensions, got {dims}")
    y_dim, x_dim = dims[0], dims[1]

    bounds_dict: dict[int, dict[str, tuple[sc.Variable, sc.Variable]]] = {}
    rois = models.ROI.from_concatenated_data_array(rectangle_request)

    for idx, roi in rois.items():
        if isinstance(roi, models.RectangleROI):
            roi_bounds = roi.get_bounds(x_dim=x_dim, y_dim=y_dim)
            bounds_dict[idx] = roi_bounds

    return ROIRectangleBounds(bounds_dict)


def precompute_roi_polygon_masks(
    screen_metadata: ScreenMetadata,
    polygon_request: ROIPolygonRequest,
) -> ROIPolygonMasks:
    """
    Precompute boolean masks for polygon ROIs.

    This is computed once when ROI configuration changes, not on every data update.
    Masks are True OUTSIDE the polygon (scipp convention: True = excluded from sum).

    Parameters
    ----------
    screen_metadata:
        Screen metadata with coordinate information.
    polygon_request:
        Polygon ROI configuration.

    Returns
    -------
    :
        Dict mapping ROI index to 2D mask Variable.
    """
    if polygon_request is None or len(polygon_request) == 0:
        return ROIPolygonMasks({})

    screen_coords = screen_metadata.coords
    sizes = screen_metadata.sizes
    dims = list(screen_coords.keys())
    if len(dims) < 2:
        raise ValueError(f"Polygon ROIs require at least 2 dimensions, got {dims}")
    y_dim, x_dim = dims[0], dims[1]

    # ScreenMetadata guarantees bin centers; synthesize indices for logical views (None)
    y_coord = screen_coords[y_dim]
    x_coord = screen_coords[x_dim]
    y_centers = (
        sc.arange(y_dim, sizes[y_dim], dtype='float64') if y_coord is None else y_coord
    )
    x_centers = (
        sc.arange(x_dim, sizes[x_dim], dtype='float64') if x_coord is None else x_coord
    )

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


def roi_spectra(
    histogram: AccumulatedHistogram[AccumulationMode],
    rectangle_bounds: ROIRectangleBounds,
    polygon_masks: ROIPolygonMasks,
) -> ROISpectra[AccumulationMode]:
    """
    Extract ROI spectra from histogram using precomputed ROI data.

    This generic provider works for both accumulation modes:

    - ROISpectra[Cumulative]: Extracted from cumulative histogram
    - ROISpectra[Current]: Extracted from current window histogram

    Parameters
    ----------
    histogram:
        Histogram with screen dims and spectral dim.
    rectangle_bounds:
        Precomputed bounds for rectangle ROIs.
    polygon_masks:
        Precomputed masks for polygon ROIs.

    Returns
    -------
    :
        ROI spectra with dims (roi, spectral).
    """
    spectral_dim = histogram.dims[-1]
    spectral_coord = histogram.coords[spectral_dim]
    n_spectral = histogram.sizes[spectral_dim]

    # Get spatial dims (all dims except spectral)
    spatial_dims = [d for d in histogram.dims if d != spectral_dim]
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
        sliced = histogram[y_dim, y_low:y_high][x_dim, x_low:x_high]
        spectrum = sliced.sum(dim=[y_dim, x_dim])
        spectra.append(spectrum)
        roi_indices.append(idx)

    # Process polygon ROIs using precomputed masks
    for idx, mask in polygon_masks.items():
        # scipp's sum ignores masked values
        masked = histogram.copy(deep=False)
        masked.masks['_roi_polygon'] = mask
        spectrum = masked.sum(dim=[y_dim, x_dim])
        spectra.append(spectrum)
        roi_indices.append(idx)

    # Build output DataArray
    if not spectra:
        return ROISpectra[AccumulationMode](
            sc.DataArray(
                data=sc.zeros(
                    dims=['roi', spectral_dim], shape=[0, n_spectral], unit='counts'
                ),
                coords={
                    'roi': sc.array(dims=['roi'], values=[], dtype='int32'),
                    spectral_dim: spectral_coord,
                },
            )
        )

    # Stack spectra along roi dimension
    stacked = sc.concat(spectra, dim='roi')
    stacked.coords['roi'] = sc.array(dims=['roi'], values=roi_indices, dtype='int32')
    return ROISpectra[AccumulationMode](stacked)


def _get_coord_units_from_screen_metadata(
    screen_metadata: ScreenMetadata,
) -> dict[str, sc.Unit | None]:
    """Extract coordinate units from screen metadata for ROI readback.

    Maps screen coordinate units to ROI 'x' and 'y' coordinates.
    """
    dims = list(screen_metadata.coords.keys())
    if len(dims) < 2:
        return {'x': None, 'y': None}

    y_dim, x_dim = dims[0], dims[1]

    def get_unit(coord: sc.Variable | None) -> sc.Unit | None:
        if coord is not None:
            return coord.unit
        return None

    return {
        'x': get_unit(screen_metadata.coords[x_dim]),
        'y': get_unit(screen_metadata.coords[y_dim]),
    }


def roi_rectangle_readback(
    request: ROIRectangleRequest,
    screen_metadata: ScreenMetadata,
) -> ROIRectangleReadback:
    """
    Produce ROI rectangle readback with correct coordinate units.

    If request has ROIs, returns them unchanged. If empty, creates empty
    DataArray with coordinate units from screen metadata so the frontend
    knows what units to use when creating ROIs.

    Parameters
    ----------
    request:
        ROI rectangle request from context.
    screen_metadata:
        Screen metadata with coordinate units.

    Returns
    -------
    :
        ROI readback with correct coordinate units.
    """
    if request is not None and len(request) > 0:
        return ROIRectangleReadback(request)

    coord_units = _get_coord_units_from_screen_metadata(screen_metadata)
    return ROIRectangleReadback(
        models.RectangleROI.to_concatenated_data_array({}, coord_units=coord_units)
    )


def roi_polygon_readback(
    request: ROIPolygonRequest,
    screen_metadata: ScreenMetadata,
) -> ROIPolygonReadback:
    """
    Produce ROI polygon readback with correct coordinate units.

    If request has ROIs, returns them unchanged. If empty, creates empty
    DataArray with coordinate units from screen metadata so the frontend
    knows what units to use when creating ROIs.

    Parameters
    ----------
    request:
        ROI polygon request from context.
    screen_metadata:
        Screen metadata with coordinate units.

    Returns
    -------
    :
        ROI readback with correct coordinate units.
    """
    if request is not None and len(request) > 0:
        return ROIPolygonReadback(request)

    coord_units = _get_coord_units_from_screen_metadata(screen_metadata)
    return ROIPolygonReadback(
        models.PolygonROI.to_concatenated_data_array({}, coord_units=coord_units)
    )
