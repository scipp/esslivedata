# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Plotter definition and registration."""

import typing
from collections import UserDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

import pydantic
import scipp as sc

from .plots import ImagePlotter, LinePlotter, Plotter, ROIDetectorPlotter, SlicerPlotter
from .scipp_to_holoviews import _all_coords_evenly_spaced


@dataclass
class DataRequirements:
    """Specification for data requirements of a plotter."""

    min_dims: int
    max_dims: int
    required_coords: list[str] = field(default_factory=list)
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

        # Check required coordinates
        for coord in self.required_coords:
            if coord not in dataset.coords:
                return False

        # Run custom validators
        for validator in self.custom_validators:
            if not validator(dataset):
                return False

        return True


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
        )
        self[name] = PlotterEntry(spec=spec, factory=factory)

    def get_compatible_plotters(
        self, data: dict[Any, sc.DataArray]
    ) -> dict[str, PlotterSpec]:
        """Get plotters compatible with the given data."""
        return {
            name: entry.spec
            for name, entry in self.items()
            if entry.spec.data_requirements.validate_data(data)
        }

    def get_specs(self) -> dict[str, PlotterSpec]:
        """Get all plotter specifications for UI display."""
        return {name: entry.spec for name, entry in self.items()}

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
    data_requirements=DataRequirements(min_dims=1, max_dims=1, multiple_datasets=True),
    factory=LinePlotter.from_params,
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


def _validate_roi_detector_data(data: dict[Any, sc.DataArray]) -> bool:
    """
    Validate data for ROI detector plotter.

    Expects:
    - One or more 2D datasets (detector data)
    - Optionally one 1D dataset with output_name 'roi_spectrum'
    """
    detector_count = 0
    roi_spectrum_count = 0

    for key, da in data.items():
        if hasattr(key, 'output_name') and key.output_name == 'roi_spectrum':
            if da.ndim != 1:
                return False
            roi_spectrum_count += 1
        else:
            if da.ndim != 2:
                return False
            detector_count += 1

    # Must have at least one detector, at most one ROI spectrum
    return detector_count >= 1 and roi_spectrum_count <= 1


plotter_registry.register_plotter(
    name='roi_detector',
    title='ROI Detector',
    description='Plot detector data with interactive ROI selection and spectrum.',
    data_requirements=DataRequirements(
        min_dims=1,
        max_dims=2,
        multiple_datasets=True,
        custom_validators=[_validate_roi_detector_data],
    ),
    factory=ROIDetectorPlotter.from_params,
)
