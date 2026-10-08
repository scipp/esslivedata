# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ROI (Region of Interest) providers for detector view workflow.

This module provides providers for ROI precomputation, spectra extraction,
per-ROI counts and readback in the detector view workflow.
"""

from __future__ import annotations

import numpy as np
import scipp as sc

from ess.livedata.config import models
from ess.livedata.config.roi_names import DETECTOR_PIXELS_COORD

from .providers import slice_spectral_range
from .types import (
    AccumulatedHistogram,
    AccumulationMode,
    HistogramSlice,
    PixelWeights,
    ROICountsInRange,
    ROIDetectorPixels,
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
    if len(rectangle_request) == 0:
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
    if len(polygon_request) == 0:
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


def _sum_over_rois(
    data: sc.DataArray,
    rectangle_bounds: ROIRectangleBounds,
    polygon_masks: ROIPolygonMasks,
) -> sc.DataArray:
    """
    Sum data over the image pixels of each ROI, keeping all other dims.

    Parameters
    ----------
    data:
        Data whose first two dims are the image dims (y, x), followed by any
        number of extra dims. Rectangle bounds with units slice by label, so
        ``data`` must carry the image coords the bounds refer to.
    rectangle_bounds:
        Precomputed bounds for rectangle ROIs.
    polygon_masks:
        Precomputed masks for polygon ROIs.

    Returns
    -------
    :
        Sums with dims (roi, *extra dims) and an int32 ``roi`` coord. Without
        ROIs the ``roi`` dim has length 0; unit and dtype are those of the sum.
    """
    if data.ndim < 2:
        raise ValueError(f"Expected at least 2 image dims, got {data.dims}")
    y_dim, x_dim = data.dims[:2]

    sums: list[sc.DataArray] = []
    roi_indices: list[int] = []

    # Process rectangle ROIs using precomputed bounds
    for idx, bounds in rectangle_bounds.items():
        x_low, x_high = bounds[x_dim]
        y_low, y_high = bounds[y_dim]
        sliced = data[y_dim, y_low:y_high][x_dim, x_low:x_high]
        sums.append(sliced.sum(dim=[y_dim, x_dim]))
        roi_indices.append(idx)

    # Process polygon ROIs using precomputed masks
    for idx, mask in polygon_masks.items():
        # scipp's sum ignores masked values
        masked = data.copy(deep=False)
        masked.masks['_roi_polygon'] = mask
        sums.append(masked.sum(dim=[y_dim, x_dim]))
        roi_indices.append(idx)

    if sums:
        # Stack sums along roi dimension
        result = sc.concat(sums, dim='roi')
    else:
        # Sum over an empty slice: the extra dims, coords, unit and dtype of a
        # real sum (scipp promotes integer sums).
        template = data[y_dim, 0:0].sum(dim=[y_dim, x_dim])
        result = sc.DataArray(
            sc.zeros(
                sizes={'roi': 0, **template.sizes},
                unit=template.unit,
                dtype=template.dtype,
            ),
            coords=template.coords,
        )
    result.coords['roi'] = sc.array(dims=['roi'], values=roi_indices, dtype='int32')
    return result


def roi_spectra(
    histogram: AccumulatedHistogram[AccumulationMode],
    detector_pixels: ROIDetectorPixels,
    rectangle_bounds: ROIRectangleBounds,
    polygon_masks: ROIPolygonMasks,
) -> ROISpectra[AccumulationMode]:
    """
    Extract ROI spectra from histogram using precomputed ROI data.

    This generic provider works for both accumulation modes:

    - ROISpectra[Cumulative]: Extracted from cumulative histogram
    - ROISpectra[Current]: Extracted from current window histogram

    The spectra carry the number of detector pixels in each ROI as the
    ``detector_pixels`` coord, so they can be shown as counts per detector pixel.

    Parameters
    ----------
    histogram:
        Histogram with screen dims and spectral dim.
    detector_pixels:
        Number of detector pixels per ROI.
    rectangle_bounds:
        Precomputed bounds for rectangle ROIs.
    polygon_masks:
        Precomputed masks for polygon ROIs.

    Returns
    -------
    :
        ROI spectra with dims (roi, spectral).
    """
    spectra = _sum_over_rois(histogram, rectangle_bounds, polygon_masks)
    if not sc.identical(spectra.coords['roi'], detector_pixels.coords['roi']):
        raise ValueError(
            f"ROI spectra and detector pixel counts disagree on the ROIs: "
            f"{spectra.coords['roi'].values} vs "
            f"{detector_pixels.coords['roi'].values}"
        )
    spectra.coords[DETECTOR_PIXELS_COORD] = detector_pixels.data
    return ROISpectra[AccumulationMode](spectra)


def roi_detector_pixels(
    weights: PixelWeights,
    rectangle_bounds: ROIRectangleBounds,
    polygon_masks: ROIPolygonMasks,
) -> ROIDetectorPixels:
    """
    Count the detector pixels inside each ROI.

    Sums the pixel weights (detector pixels per image pixel) over each ROI, with
    the same image-pixel selection as the ROI spectra. Image pixels without
    detector pixels contribute nothing. For geometric projections the weights
    are averaged over the position-noise replicas, so the count need not be an
    integer.

    Parameters
    ----------
    weights:
        Number of detector pixels per image pixel.
    rectangle_bounds:
        Precomputed bounds for rectangle ROIs.
    polygon_masks:
        Precomputed masks for polygon ROIs.

    Returns
    -------
    :
        Detector pixel count per ROI with dims (roi,), as float64.
    """
    return ROIDetectorPixels(
        _sum_over_rois(weights.to(dtype='float64'), rectangle_bounds, polygon_masks)
    )


def roi_counts_in_range(
    spectra: ROISpectra[AccumulationMode],
    histogram_slice: HistogramSlice,
) -> ROICountsInRange[AccumulationMode]:
    """
    Sum the ROI spectra over the active range filter.

    The range filter selects the same spectral bins as for the detector image.

    Parameters
    ----------
    spectra:
        ROI spectra with dims (roi, spectral).
    histogram_slice:
        Optional (low, high) range along the spectral dimension.

    Returns
    -------
    :
        Counts per ROI with dims (roi,), with the ``detector_pixels`` coord of the
        spectra.
    """
    in_range = slice_spectral_range(spectra, histogram_slice)
    return ROICountsInRange[AccumulationMode](in_range.sum(spectra.dims[-1]))


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
    if len(request) > 0:
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
    if len(request) > 0:
        return ROIPolygonReadback(request)

    coord_units = _get_coord_units_from_screen_metadata(screen_metadata)
    return ROIPolygonReadback(
        models.PolygonROI.to_concatenated_data_array({}, coord_units=coord_units)
    )
