# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Detector view workflow - Sciline-based detector data visualization.

This package implements a modular detector view workflow using Sciline,
providing event-based projection of detector data to screen coordinates while
preserving event coordinate information for flexible histogramming.

The workflow is dimension-agnostic: it can histogram any event coordinate
(e.g., event_time_offset, wavelength) specified via EventCoordName.

Supports two projection modes:
1. Geometric projections (xy_plane, cylinder_mantle_z) using calibrated coordinates
2. Logical views (fold/slice transforms with optional reduction)
"""

from .data_source import (
    DetectorDataSource,
    DetectorNumberSource,
    NeXusDetectorSource,
    create_empty_detector,
)
from .factory import DetectorViewFactory
from .projectors import (
    GeometricProjector,
    LogicalProjector,
    Projector,
    make_geometric_projector,
    make_logical_projector,
)
from .providers import (
    compute_detector_histogram_3d,
    counts_in_range,
    counts_total,
    cumulative_detector_image,
    cumulative_histogram,
    current_detector_image,
    project_events,
    window_histogram,
)
from .roi import (
    cumulative_roi_spectra,
    current_roi_spectra,
    precompute_roi_polygon_masks,
    precompute_roi_rectangle_bounds,
    roi_polygon_readback,
    roi_rectangle_readback,
)
from .types import (
    CountsInTOARange,
    CountsTotal,
    CumulativeDetectorImage,
    CumulativeHistogram,
    CumulativeROISpectra,
    CurrentDetectorImage,
    CurrentROISpectra,
    DetectorHistogram3D,
    EventCoordName,
    GeometricViewConfig,
    HistogramBins,
    HistogramSlice,
    LogicalTransform,
    LogicalViewConfig,
    ProjectionType,
    ReductionDim,
    ROIPolygonMasks,
    ROIPolygonReadback,
    ROIPolygonRequest,
    ROIRectangleBounds,
    ROIRectangleReadback,
    ROIRectangleRequest,
    ROISpectra,
    ScreenBinnedEvents,
    ScreenMetadata,
    ViewConfig,
    WindowHistogram,
)
from .workflow import (
    NoCopyAccumulator,
    NoCopyWindowAccumulator,
    add_geometric_projection,
    add_logical_projection,
    create_accumulators,
    create_base_workflow,
    get_screen_metadata,
)

__all__ = [
    'CountsInTOARange',
    'CountsTotal',
    'CumulativeDetectorImage',
    'CumulativeHistogram',
    'CumulativeROISpectra',
    'CurrentDetectorImage',
    'CurrentROISpectra',
    # Data sources
    'DetectorDataSource',
    'DetectorHistogram3D',
    'DetectorNumberSource',
    # Factory
    'DetectorViewFactory',
    # Types - configuration
    'EventCoordName',
    # Projectors
    'GeometricProjector',
    # View configuration
    'GeometricViewConfig',
    'HistogramBins',
    'HistogramSlice',
    'LogicalProjector',
    'LogicalTransform',
    'LogicalViewConfig',
    'NeXusDetectorSource',
    # Accumulators
    'NoCopyAccumulator',
    'NoCopyWindowAccumulator',
    'ProjectionType',
    'Projector',
    'ROIPolygonMasks',
    'ROIPolygonReadback',
    'ROIPolygonRequest',
    'ROIRectangleBounds',
    'ROIRectangleReadback',
    'ROIRectangleRequest',
    'ROISpectra',
    'ReductionDim',
    'ScreenBinnedEvents',
    'ScreenMetadata',
    'ViewConfig',
    # Workflow
    'WindowHistogram',
    'add_geometric_projection',
    'add_logical_projection',
    'compute_detector_histogram_3d',
    'counts_in_range',
    'counts_total',
    'create_accumulators',
    'create_base_workflow',
    'create_empty_detector',
    'cumulative_detector_image',
    'cumulative_histogram',
    'cumulative_roi_spectra',
    'current_detector_image',
    'current_roi_spectra',
    'get_screen_metadata',
    'make_geometric_projector',
    'make_logical_projector',
    'precompute_roi_polygon_masks',
    # ROI providers
    'precompute_roi_rectangle_bounds',
    'project_events',
    'roi_polygon_readback',
    'roi_rectangle_readback',
    # Providers
    'window_histogram',
]
