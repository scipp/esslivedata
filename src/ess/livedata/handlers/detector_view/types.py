# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Type definitions for detector view workflow.

This module defines all types used in the detector view workflow, including
configuration types, intermediate types, output types, and ROI types.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NewType

import scipp as sc

# ============================================================================
# View Configuration Classes
# ============================================================================


@dataclass(frozen=True, slots=True)
class GeometricViewConfig:
    """
    Configuration for geometric projection views.

    Geometric projections map detector pixel positions to screen coordinates
    using calibrated 3D positions.

    Parameters
    ----------
    projection_type:
        Type of geometric projection.
    resolution:
        Number of bins for each screen dimension (e.g., {'x': 200, 'y': 200}).
    pixel_noise:
        Noise to add to pixel positions. 'cylindrical' uses detector geometry,
        a scalar Variable applies Gaussian noise, None disables noise.
    """

    projection_type: Literal['xy_plane', 'cylinder_mantle_z']
    resolution: dict[str, int]
    pixel_noise: Literal['cylindrical'] | sc.Variable | None = None


@dataclass(frozen=True, slots=True)
class LogicalViewConfig:
    """
    Configuration for logical projection views.

    Logical projections reshape detector data using fold/slice transforms
    without requiring calibrated positions.

    Parameters
    ----------
    transform:
        Callable that reshapes detector data. Signature: (da, source_name) -> da.
        If None, uses identity transform.
    reduction_dim:
        Dimension(s) to merge events over after transform. None means no reduction.
    """

    transform: Callable[[sc.DataArray, str], sc.DataArray] | None = None
    reduction_dim: str | list[str] | None = None


ViewConfig = GeometricViewConfig | LogicalViewConfig
"""Union type for view configuration."""

# Configuration types (set once at workflow creation)
EventCoordName = NewType('EventCoordName', str)
"""Name of the event coordinate to histogram (e.g., 'event_time_offset',
'wavelength')."""

HistogramBins = NewType('HistogramBins', sc.Variable)
"""Bin edges for histogramming the event coordinate."""

HistogramSlice = NewType('HistogramSlice', tuple[sc.Variable, sc.Variable] | None)
"""Optional (low, high) range for slicing output images. None means full range."""

# Logical transform configuration
LogicalTransform = NewType(
    'LogicalTransform',
    type(None) | type(lambda da: da),  # Callable[[sc.DataArray], sc.DataArray] | None
)
"""Callable that transforms detector data to logical coordinates, or None."""

# Reduction dimension for logical views
ReductionDim = NewType('ReductionDim', str | list[str] | None)
"""Dimension(s) to sum over after applying logical transform."""

# Projection type for geometric views
ProjectionType = NewType(
    'ProjectionType',
    str | None,  # Literal['xy_plane', 'cylinder_mantle_z'] | None
)
"""Type of geometric projection to use, or None for logical view."""

# Intermediate types for event projection
ScreenBinnedEvents = NewType('ScreenBinnedEvents', sc.DataArray)
"""Events binned by screen coordinates (screen_y, screen_x) with event
coordinate preserved."""

# Shared intermediate - computed once, then split for accumulation
DetectorHistogram3D = NewType('DetectorHistogram3D', sc.DataArray)
"""3D histogram with dims (y, x, event_coord) - computed once, shared by
accumulators."""

# Accumulated data types - use different types for different accumulator behavior
CumulativeHistogram = NewType('CumulativeHistogram', sc.DataArray)
"""3D histogram accumulated forever (EternalAccumulator)."""

WindowHistogram = NewType('WindowHistogram', sc.DataArray)
"""3D histogram for current window (clears after finalize)."""

# Output types
CumulativeDetectorImage = NewType('CumulativeDetectorImage', sc.DataArray)
"""2D detector image summed over all accumulated data."""

CurrentDetectorImage = NewType('CurrentDetectorImage', sc.DataArray)
"""2D detector image for the current window (since last finalize)."""

CountsTotal = NewType('CountsTotal', sc.DataArray)
"""Total event counts as 0D scalar (from current window)."""

CountsInTOARange = NewType('CountsInTOARange', sc.DataArray)
"""Event counts within configured TOA range as 0D scalar (from current window)."""

# ROI configuration types (context keys - updated less frequently than events)
ROIRectangleRequest = NewType('ROIRectangleRequest', sc.DataArray)
"""ROI rectangle configuration as concatenated DataArray (empty if no ROIs)."""

ROIPolygonRequest = NewType('ROIPolygonRequest', sc.DataArray)
"""ROI polygon configuration as concatenated DataArray (empty if no ROIs)."""

# ROI readback types (outputs - echo request with correct units for frontend)
ROIRectangleReadback = NewType('ROIRectangleReadback', sc.DataArray)
"""ROI rectangle readback with coordinate units from histogram."""

ROIPolygonReadback = NewType('ROIPolygonReadback', sc.DataArray)
"""ROI polygon readback with coordinate units from histogram."""

# ROI output types
ROISpectra = NewType('ROISpectra', sc.DataArray)
"""Spectra for ROIs with dims (roi, spectral_dim).

Computed from histogram, not accumulated.
"""

CumulativeROISpectra = NewType('CumulativeROISpectra', sc.DataArray)
"""ROI spectra extracted from cumulative histogram."""

CurrentROISpectra = NewType('CurrentROISpectra', sc.DataArray)
"""ROI spectra extracted from current window histogram."""


ROIRectangleBounds = NewType('ROIRectangleBounds', dict)
"""Precomputed bounds for rectangle ROIs.

Dict mapping ROI index to bounds dict: {idx: {y_dim: (low, high), x_dim: (low, high)}}.
Empty dict if no rectangles configured.
"""

ROIPolygonMasks = NewType('ROIPolygonMasks', dict)
"""Precomputed boolean masks for polygon ROIs.

Dict mapping ROI index to 2D mask Variable with dims (y_dim, x_dim).
Mask is True OUTSIDE the polygon (values to exclude in sum).
Empty dict if no polygons configured.
"""
