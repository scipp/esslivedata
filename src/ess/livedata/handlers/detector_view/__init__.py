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
    compute_pixel_weights,
    counts_in_range,
    counts_total,
    detector_image,
    histogram_3d,
    project_events,
)
from .roi import (
    precompute_roi_polygon_masks,
    precompute_roi_rectangle_bounds,
    roi_polygon_readback,
    roi_rectangle_readback,
    roi_spectra,
)
from .types import (
    AccumulationMode,
    CountsInRange,
    CountsTotal,
    Cumulative,
    Current,
    DetectorHistogram3D,
    DetectorImage,
    EventCoordName,
    GeometricViewConfig,
    Histogram3D,
    HistogramBins,
    HistogramSlice,
    LogicalTransform,
    LogicalViewConfig,
    PixelWeights,
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
    UsePixelWeighting,
    ViewConfig,
)
from .workflow import (
    NoCopyAccumulator,
    NoCopyWindowAccumulator,
    add_geometric_projection,
    add_logical_projection,
    create_base_workflow,
    get_screen_metadata,
)

__all__ = [
    # Accumulation mode types
    'AccumulationMode',
    # Output types
    'CountsInRange',
    'CountsTotal',
    'Cumulative',
    'Current',
    # Data sources
    'DetectorDataSource',
    'DetectorHistogram3D',
    'DetectorImage',
    'DetectorNumberSource',
    # Factory
    'DetectorViewFactory',
    # Types - configuration
    'EventCoordName',
    # Projectors
    'GeometricProjector',
    # View configuration
    'GeometricViewConfig',
    'Histogram3D',
    'HistogramBins',
    'HistogramSlice',
    'LogicalProjector',
    'LogicalTransform',
    'LogicalViewConfig',
    'NeXusDetectorSource',
    # Accumulators
    'NoCopyAccumulator',
    'NoCopyWindowAccumulator',
    # Pixel weighting
    'PixelWeights',
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
    'UsePixelWeighting',
    'ViewConfig',
    # Workflow construction
    'add_geometric_projection',
    'add_logical_projection',
    # Providers
    'compute_detector_histogram_3d',
    'compute_pixel_weights',
    'counts_in_range',
    'counts_total',
    'create_base_workflow',
    'create_empty_detector',
    'detector_image',
    'get_screen_metadata',
    'histogram_3d',
    # Projector factories
    'make_geometric_projector',
    'make_logical_projector',
    # ROI providers
    'precompute_roi_polygon_masks',
    'precompute_roi_rectangle_bounds',
    'project_events',
    'roi_polygon_readback',
    'roi_rectangle_readback',
    'roi_spectra',
]
