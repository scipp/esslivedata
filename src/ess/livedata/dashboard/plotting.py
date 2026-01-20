# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Plotter definition and registration."""

import enum
import typing
from collections import UserDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

import pydantic
import scipp as sc

from .correlation_plotter import (
    CorrelationHistogram1dPlotter,
    CorrelationHistogram2dPlotter,
)
from .extractors import FullHistoryExtractor, UpdateExtractor
from .plots import (
    BarsPlotter,
    ImagePlotter,
    LinePlotter,
    Overlay1DPlotter,
    Plotter,
    SlicerPlotter,
)
from .roi_readback_plots import (
    PolygonsReadbackPlotter,
    RectanglesReadbackPlotter,
)
from .roi_request_plots import (
    PolygonsRequestPlotter,
    RectanglesRequestPlotter,
)
from .scipp_to_holoviews import _all_coords_evenly_spaced
from .static_plots import _register_static_plotters


class PlotterCategory(enum.Enum):
    """Category of plotter for filtering in wizard and orchestrator."""

    DATA = "data"  # Requires workflow data (standard plotters)
    STATIC = "static"  # No data required (geometric overlays)


@dataclass
class DataRequirements:
    """Specification for data requirements of a plotter."""

    min_dims: int
    max_dims: int
    required_extractor: type[UpdateExtractor] | None = None
    required_coords: list[str] = field(default_factory=list)
    deny_coords: list[str] = field(default_factory=list)
    required_dim_names: list[str] = field(default_factory=list)
    multiple_datasets: bool = True
    custom_validators: list[Callable[[sc.DataArray], bool]] = field(
        default_factory=list
    )

    def validate_data(self, data: dict[Any, sc.DataArray]) -> bool:
        """Validate that the data meets these requirements."""
        if not data:
            return False

        if not self.multiple_datasets and len(data) > 1:
            return False

        for dataset in data.values():
            if not self._validate_dataset(dataset):
                return False

        return True

    def _validate_dataset(self, dataset: sc.DataArray) -> bool:
        """Validate a single dataset."""
        # Check dimensions
        if dataset.ndim < self.min_dims or dataset.ndim > self.max_dims:
            return False

        # Check required coordinates (must have ALL)
        for coord in self.required_coords:
            if coord not in dataset.coords:
                return False

        # Check denied coordinates (must NOT have ANY)
        for coord in self.deny_coords:
            if coord in dataset.coords:
                return False

        # Check required dimension names (must have ALL)
        for dim_name in self.required_dim_names:
            if dim_name not in dataset.dims:
                return False

        # Run custom validators
        for validator in self.custom_validators:
            if not validator(dataset):
                return False

        return True


@dataclass
class SpecRequirements:
    """Requirements based on workflow spec metadata.

    Allows plotters to declare dependencies on workflow spec features like
    auxiliary sources (e.g., ROI support).
    """

    requires_aux_sources: list[type] = field(default_factory=list)

    def validate_spec(self, aux_sources_type: type | None) -> bool:
        """Check if spec meets these requirements.

        Parameters
        ----------
        aux_sources_type:
            The aux_sources type from the workflow spec, or None if not defined.

        Returns
        -------
        :
            True if the spec meets the requirements, False otherwise.
        """
        if not self.requires_aux_sources:
            return True
        if aux_sources_type is None:
            return False
        # Check if aux_sources_type is one of the required types or a subclass
        return any(
            issubclass(aux_sources_type, required)
            for required in self.requires_aux_sources
        )


class PlotterSpec(pydantic.BaseModel):
    """
    Specification for a plotter.

    This model defines the metadata and a parameters specification. This allows for
    dynamic creation of user interfaces for configuring plots.
    """

    name: str = pydantic.Field(description="Name of the plot type. Used internally.")
    title: str = pydantic.Field(
        description="Title of the plot type. For display in the UI."
    )
    description: str = pydantic.Field(description="Description of the plot type.")
    params: type[pydantic.BaseModel] = pydantic.Field(
        description="Pydantic model defining the parameters for the plot."
    )
    data_requirements: DataRequirements = pydantic.Field(
        description="Requirements the data to be plotted must fulfill."
    )
    spec_requirements: SpecRequirements = pydantic.Field(
        default_factory=SpecRequirements,
        description="Requirements based on workflow spec metadata.",
    )
    category: PlotterCategory = pydantic.Field(
        default=PlotterCategory.DATA,
        description="Category of plotter: DATA (requires workflow) or STATIC (overlay)",
    )


# Type variable for parameter types
P = TypeVar('P', bound=pydantic.BaseModel)


class PlotterFactory(Protocol, Generic[P]):
    def __call__(self, params: P) -> Plotter: ...


@dataclass
class PlotterEntry:
    """Entry combining a plotter specification with its factory."""

    spec: PlotterSpec
    factory: PlotterFactory[Any]  # Use Any since we store different param types


class PlotterRegistry(UserDict[str, PlotterEntry]):
    def register_plotter(
        self,
        name: str,
        title: str,
        description: str,
        data_requirements: DataRequirements,
        factory: PlotterFactory[P],
        spec_requirements: SpecRequirements | None = None,
        category: PlotterCategory = PlotterCategory.DATA,
    ) -> None:
        # Try to get the type hint of the 'params' argument if it exists
        # Use get_type_hints to resolve forward references, in case we used
        # `from __future__ import annotations`.
        type_hints = typing.get_type_hints(factory)
        spec = PlotterSpec(
            name=name,
            title=title,
            description=description,
            params=type_hints['params'],
            data_requirements=data_requirements,
            spec_requirements=spec_requirements or SpecRequirements(),
            category=category,
        )
        self[name] = PlotterEntry(spec=spec, factory=factory)

    def get_compatible_plotters(
        self, data: dict[Any, sc.DataArray]
    ) -> dict[str, PlotterSpec]:
        """Get plotters compatible with the given data.

        Note: This only checks data requirements, not spec requirements.
        Use get_compatible_plotters_with_spec() when workflow spec is available.
        Only returns DATA category plotters.
        """
        return {
            name: entry.spec
            for name, entry in self.items()
            if entry.spec.category == PlotterCategory.DATA
            and entry.spec.data_requirements.validate_data(data)
        }

    def get_compatible_plotters_with_spec(
        self,
        data: dict[Any, sc.DataArray],
        aux_sources_type: type | None,
    ) -> dict[str, PlotterSpec]:
        """Get plotters compatible with both data and workflow spec.

        Only returns DATA category plotters.

        Parameters
        ----------
        data:
            Dictionary mapping keys to DataArrays.
        aux_sources_type:
            The aux_sources type from the workflow spec, or None if not defined.

        Returns
        -------
        :
            Dictionary of compatible plotter names to their specifications.
        """
        return {
            name: entry.spec
            for name, entry in self.items()
            if entry.spec.category == PlotterCategory.DATA
            and entry.spec.data_requirements.validate_data(data)
            and entry.spec.spec_requirements.validate_spec(aux_sources_type)
        }

    def get_specs(self) -> dict[str, PlotterSpec]:
        """Get all plotter specifications for UI display (DATA category only)."""
        return {
            name: entry.spec
            for name, entry in self.items()
            if entry.spec.category == PlotterCategory.DATA
        }

    def get_static_plotters(self) -> dict[str, PlotterSpec]:
        """Get all STATIC category plotters for overlay selection."""
        return {
            name: entry.spec
            for name, entry in self.items()
            if entry.spec.category == PlotterCategory.STATIC
        }

    def get_spec(self, name: str) -> PlotterSpec:
        """Get specification for a specific plotter."""
        return self[name].spec

    def create_plotter(self, name: str, params: pydantic.BaseModel) -> Plotter:
        """Create a plotter instance with the given parameters."""
        return self[name].factory(params)


plotter_registry = PlotterRegistry()


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
