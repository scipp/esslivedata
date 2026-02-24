# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Plotter registration.

Registers all plotter types with the central plotter registry.
"""

from .correlation_plotter import (
    CorrelationHistogram1dPlotter,
    CorrelationHistogram2dPlotter,
)
from .extractors import FullHistoryExtractor
from .plots import (
    BarsPlotter,
    ImagePlotter,
    LinePlotter,
    Overlay1DPlotter,
)
from .plotter_registry import DataRequirements, plotter_registry
from .roi_readback_plots import (
    PolygonsReadbackPlotter,
    RectanglesReadbackPlotter,
)
from .roi_request_plots import (
    PolygonsRequestPlotter,
    RectanglesRequestPlotter,
)
from .scipp_to_holoviews import _all_coords_evenly_spaced
from .slicer_plotter import SlicerPlotter
from .static_plots import _register_static_plotters

plotter_registry.register_plotter(
    name='image',
    title='Image',
    description='Plot the data as a images.',
    data_requirements=DataRequirements(min_dims=2, max_dims=2),
    factory=ImagePlotter.from_params,
)


plotter_registry.register_plotter(
    name='lines',
    title='Lines',
    description='Plot the data as line plots.',
    data_requirements=DataRequirements(
        min_dims=1, max_dims=1, multiple_datasets=True, deny_coords=['roi_index']
    ),
    factory=LinePlotter.from_params,
)


plotter_registry.register_plotter(
    name='timeseries',
    title='Timeseries',
    description='Plot the temporal evolution of scalar values as line plots.',
    data_requirements=DataRequirements(
        min_dims=0,
        max_dims=0,
        multiple_datasets=True,
        required_extractor=FullHistoryExtractor,
    ),
    factory=LinePlotter.from_params,
)


plotter_registry.register_plotter(
    name='bars',
    title='Bars',
    description='Plot 0D scalar values as bars.',
    data_requirements=DataRequirements(min_dims=0, max_dims=0, multiple_datasets=True),
    factory=BarsPlotter.from_params,
)


plotter_registry.register_plotter(
    name='slicer',
    title='3D Slicer',
    description='Interactively slice through 3D data along one dimension.',
    data_requirements=DataRequirements(
        min_dims=3,
        max_dims=3,
        multiple_datasets=False,
        custom_validators=[_all_coords_evenly_spaced],
    ),
    factory=SlicerPlotter.from_params,
)


plotter_registry.register_plotter(
    name='overlay_1d',
    title='Overlay 1D',
    description=(
        'Slice 2D data along the first dimension and overlay as 1D curves. '
        'Useful for visualizing multiple spectra from a single 2D array '
        '(e.g., ROI spectra stacked along a roi dimension).'
    ),
    data_requirements=DataRequirements(min_dims=2, max_dims=2, multiple_datasets=False),
    factory=Overlay1DPlotter.from_params,
)


plotter_registry.register_plotter(
    name='correlation_histogram_1d',
    title='Correlation Histogram 1D',
    description=(
        'Create a 1D histogram correlating the selected timeseries against another '
        'timeseries axis. Useful for visualizing how data varies with a parameter '
        'like temperature or motor position.'
    ),
    data_requirements=DataRequirements(
        min_dims=0,
        max_dims=0,
        multiple_datasets=True,
        required_extractor=FullHistoryExtractor,
    ),
    factory=CorrelationHistogram1dPlotter.from_params,
)


plotter_registry.register_plotter(
    name='correlation_histogram_2d',
    title='Correlation Histogram 2D',
    description=(
        'Create a 2D histogram correlating the selected timeseries against two '
        'timeseries axes. Useful for visualizing how data varies with two parameters '
        'simultaneously.'
    ),
    data_requirements=DataRequirements(
        min_dims=0,
        max_dims=0,
        multiple_datasets=True,
        required_extractor=FullHistoryExtractor,
    ),
    factory=CorrelationHistogram2dPlotter.from_params,
)


# Register static plotters (rectangles, vlines, hlines)
_register_static_plotters()


# Maps base plotter name -> list of (required_output_name, overlay_plotter_name)
# Each tuple specifies: which workflow output is required, and which plotter to use
#
# Overlay suggestions are chained to enforce ordering:
#   image -> readback -> request
# This ensures the read-only visualization layer is added before the interactive editor.
OVERLAY_PATTERNS: dict[str, list[tuple[str, str]]] = {
    'image': [
        ('roi_rectangle', 'rectangles_readback'),
        ('roi_polygon', 'polygons_readback'),
    ],
    'rectangles_readback': [
        ('roi_rectangle', 'rectangles_request'),
    ],
    'polygons_readback': [
        ('roi_polygon', 'polygons_request'),
    ],
}


# ROI data requirements (shared between readback and request plotters)
_RECTANGLE_ROI_REQUIREMENTS: dict = {
    'min_dims': 1,
    'max_dims': 1,
    'required_coords': ['roi_index', 'x', 'y'],
    'required_dim_names': ['bounds'],
    'multiple_datasets': False,
}
_POLYGON_ROI_REQUIREMENTS: dict = {
    'min_dims': 1,
    'max_dims': 1,
    'required_coords': ['roi_index', 'x', 'y'],
    'required_dim_names': ['vertex'],
    'multiple_datasets': False,
}

# Register ROI rectangle plotters (readback + request)
plotter_registry.register_plotter(
    name='rectangles_readback',
    title='ROI Rectangles (Readback)',
    description='Display ROI rectangles from workflow output. '
    'Each rectangle is colored by its ROI index.',
    data_requirements=DataRequirements(**_RECTANGLE_ROI_REQUIREMENTS),
    factory=RectanglesReadbackPlotter.from_params,
)
plotter_registry.register_plotter(
    name='rectangles_request',
    title='ROI Rectangles (Interactive)',
    description='Draw and edit ROI rectangles interactively. '
    'Publishes ROI updates to backend for processing.',
    data_requirements=DataRequirements(**_RECTANGLE_ROI_REQUIREMENTS),
    factory=RectanglesRequestPlotter.from_params,
)

# Register ROI polygon plotters (readback + request)
plotter_registry.register_plotter(
    name='polygons_readback',
    title='ROI Polygons (Readback)',
    description='Display ROI polygons from workflow output. '
    'Each polygon is colored by its ROI index.',
    data_requirements=DataRequirements(**_POLYGON_ROI_REQUIREMENTS),
    factory=PolygonsReadbackPlotter.from_params,
)
plotter_registry.register_plotter(
    name='polygons_request',
    title='ROI Polygons (Interactive)',
    description='Draw and edit ROI polygons interactively. '
    'Publishes ROI updates to backend for processing.',
    data_requirements=DataRequirements(**_POLYGON_ROI_REQUIREMENTS),
    factory=PolygonsRequestPlotter.from_params,
)
