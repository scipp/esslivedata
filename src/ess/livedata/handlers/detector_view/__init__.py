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

from .data_source import InstrumentDetectorSource, NeXusDetectorSource
from .factory import DetectorViewFactory
from .types import CoordinateMode, GeometricViewConfig, LogicalViewConfig

__all__ = [
    'CoordinateMode',
    'DetectorViewFactory',
    'GeometricViewConfig',
    'InstrumentDetectorSource',
    'LogicalViewConfig',
    'NeXusDetectorSource',
]
