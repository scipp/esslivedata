# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""
Models for configuration values that can be used to control services via Kafka.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import quote, unquote

import numpy as np
import scipp as sc
from pydantic import BaseModel, Field, model_validator

TimeUnit = Literal['ns', 'us', 'μs', 'ms', 's']


class WeightingMethod(StrEnum):
    """
    Methods for pixel weighting.

    - PIXEL_NUMBER: Weight by the number of detector pixels contributing to each screen
        pixel.
    """

    PIXEL_NUMBER = 'pixel_number'


class PixelWeighting(BaseModel):
    """Setting for pixel weighting."""

    enabled: bool = Field(default=False, description="Enable pixel weighting.")
    method: WeightingMethod = Field(
        default=WeightingMethod.PIXEL_NUMBER, description="Method for pixel weighting."
    )


class TimeModel(BaseModel):
    """Base model for time values with unit."""

    value: float = Field(default=0, description="Time value.")
    unit: TimeUnit = Field(
        default="ns", description="Physical unit for the time value."
    )

    _value_ns: int | None = None

    def model_post_init(self, /, __context: Any) -> None:
        """Perform relatively expensive operations after model initialization."""
        self._value_ns = int(
            sc.scalar(self.value, unit=self.unit).to(unit='ns', dtype='int64').value
        )

    @property
    def value_ns(self) -> int:
        """Time in nanoseconds."""
        return self._value_ns


class UpdateEvery(TimeModel):
    """Setting for the update frequency of the accumulation period."""

    value: float = Field(default=1.0, ge=0.1, description="Time value.")
    unit: TimeUnit = Field(default="s", description="Physical unit for the time value.")


class ConfigKey(BaseModel, frozen=True):
    """
    Model for configuration key structure.

    Configuration keys follow the format 'source_name/service_name/key', where:
    - source_name can be a specific source name or '*' for all sources
    - service_name can be a specific service name or '*' for all services
    - key is the specific configuration parameter name
    """

    source_name: str | None = Field(
        default=None,
        description="Source name, or None for wildcard (*) matching all sources",
    )
    service_name: str | None = Field(
        default=None,
        description="Service name, or None for wildcard (*) matching all services",
    )
    key: str = Field(description="Configuration parameter name/key")

    def __str__(self) -> str:
        """
        Convert the configuration key to its string representation.

        Returns
        -------
        :
            String in the format source_name/service_name/key with '*' for None values.
            Components are URL-encoded to handle special characters like '/'.
        """
        source = '*' if self.source_name is None else quote(self.source_name, safe='')
        service = (
            '*' if self.service_name is None else quote(self.service_name, safe='')
        )
        key_encoded = quote(self.key, safe='')
        return f"{source}/{service}/{key_encoded}"

    @classmethod
    def from_string(cls, key_str: str) -> ConfigKey:
        """
        Create a ConfigKey from its string representation.

        Parameters
        ----------
        key_str:
            String in the format 'source_name/service_name/key'.
            Components should be URL-encoded to handle special characters.

        Returns
        -------
        :
            A ConfigKey instance parsed from the string

        Raises
        ------
        ValueError:
            If the key format is invalid
        """
        parts = key_str.split('/')
        if len(parts) != 3:
            raise ValueError(
                "Invalid key format, expected 'source_name/service_name/key', "
                f"got {key_str}"
            )
        source_name, service_name, key = parts
        # Decode URL-encoded components
        source_name = None if source_name == '*' else unquote(source_name)
        service_name = None if service_name == '*' else unquote(service_name)
        key = unquote(key)
        return cls(source_name=source_name, service_name=service_name, key=key)


class ROIType(StrEnum):
    """Types of Region of Interest (ROI) shapes."""

    RECTANGLE = 'rectangle'
    POLYGON = 'polygon'
    ELLIPSE = 'ellipse'


def _unit_to_str(unit: sc.Unit | None) -> str | None:
    """Convert scipp Unit to string, handling None."""
    return None if unit is None else str(unit)


class ROI(BaseModel, ABC):
    """
    Base class for Region of Interest (ROI) definitions.

    ROIs can be serialized to/from scipp DataArrays which can then be converted
    to da00 format using the existing compat module for Kafka transmission.

    The ROI type is encoded in the DataArray's dimension name: 'bounds' for
    rectangles, 'vertex' for polygons, 'ellipse' for ellipses.
    """

    @abstractmethod
    def to_data_array(self) -> sc.DataArray:
        """
        Convert ROI to scipp DataArray representation.

        Returns
        -------
        :
            DataArray with ROI geometry stored in coordinates.
        """
        ...

    @classmethod
    def from_data_array(cls, da: sc.DataArray) -> ROI:
        """
        Create ROI from scipp DataArray representation.

        Dispatches to the appropriate ROI subclass based on the DataArray's
        dimension name, which encodes the ROI type.

        Parameters
        ----------
        da:
            DataArray with ROI geometry in coordinates.

        Returns
        -------
        :
            ROI instance (Rectangle, Polygon, or Ellipse).
        """
        dim = da.dims[0]
        if dim == 'bounds':
            return RectangleROI._from_data_array(da)
        elif dim == 'vertex':
            return PolygonROI._from_data_array(da)
        elif dim == 'ellipse':
            return EllipseROI._from_data_array(da)
        else:
            raise ValueError(f"Cannot determine ROI type from dimension: {dim}")

    @classmethod
    @abstractmethod
    def _from_data_array(cls, da: sc.DataArray) -> ROI:
        """
        Internal method to create specific ROI subclass from DataArray.

        Subclasses must implement this method.
        """
        ...

    @classmethod
    def to_concatenated_data_array(
        cls,
        rois: dict[int, ROI],
        coord_units: dict[str, sc.Unit | str | None] | None = None,
    ) -> sc.DataArray:
        """
        Convert multiple ROIs to single concatenated DataArray.

        Generic implementation that works for all ROI types by:
        1. Converting each ROI to its individual DataArray
        2. Adding roi_index coordinate to track which elements belong to which ROI
        3. Concatenating along the ROI type's natural dimension

        This enables efficient Kafka transmission and deletion detection on the
        consumer side (missing indices indicate deleted ROIs).

        The ROI type is encoded in the dimension name ('bounds' for rectangles,
        'vertex' for polygons, 'ellipse' for ellipses), not in the DataArray name.

        Parameters
        ----------
        rois:
            Dictionary mapping ROI index to ROI instance. Empty dict returns
            empty DataArray with appropriate structure.
        coord_units:
            Optional dict mapping coordinate names (e.g., 'x', 'y') to units.
            Used to set units on empty DataArray coordinates when rois is empty.
            This enables consumers to know the expected coordinate units even
            before any ROIs exist.

        Returns
        -------
        :
            Concatenated DataArray with roi_index coordinate.
        """
        if not rois:
            template = cls._empty_data_array_template()
            template.coords['roi_index'] = sc.empty(
                dims=[template.dim], shape=[0], dtype='int32'
            )
            if coord_units:
                for coord_name, unit in coord_units.items():
                    if coord_name in template.coords:
                        template.coords[coord_name] = sc.empty(
                            dims=template.coords[coord_name].dims,
                            shape=template.coords[coord_name].shape,
                            unit=unit,
                        )
            return template

        roi_das = []
        for idx in sorted(rois.keys()):
            roi_da = rois[idx].to_data_array()
            roi_da = roi_da.assign_coords(
                roi_index=sc.full(sizes=roi_da.data.sizes, value=idx, dtype='int32')
            )
            roi_das.append(roi_da)

        concat_dim = roi_das[0].dims[0]
        return sc.concat(roi_das, dim=concat_dim)

    @classmethod
    def from_concatenated_data_array(cls, da: sc.DataArray) -> dict[int, ROI]:
        """
        Convert concatenated DataArray back to dict of ROIs.

        Generic implementation that works for all ROI types.

        Parameters
        ----------
        da:
            DataArray with concatenated ROIs and roi_index coordinate.

        Returns
        -------
        :
            Dictionary mapping ROI indices to ROI instances.
            Empty DataArray returns empty dict.
        """
        if len(da) == 0:
            return {}

        roi_indices = np.unique(da.coords['roi_index'].values)
        rois = {}

        for idx in roi_indices:
            mask = da.coords['roi_index'] == idx
            roi_da = da[mask].drop_coords('roi_index')
            rois[int(idx)] = ROI.from_data_array(roi_da)

        return rois

    @classmethod
    @abstractmethod
    def _empty_data_array_template(cls) -> sc.DataArray:
        """
        Return empty DataArray template without roi_index coordinate.

        Each ROI subclass must implement this to define the dimension name
        and coordinate structure. The dimension name encodes the ROI type.

        Returns
        -------
        :
            Empty DataArray with ROI-specific coordinates (without roi_index).
        """
        ...


class Interval(BaseModel):
    """
    An interval with min and max bounds.

    If unit is None, the coordinates are interpreted as integer pixel indices
    (though floats are allowed for sub-pixel precision).
    """

    min: float = Field(description="Minimum coordinate")
    max: float = Field(description="Maximum coordinate")
    unit: str | None = Field(
        default=None, description="Unit for coordinates (None for pixel indices)"
    )

    @model_validator(mode='after')
    def validate_bounds(self) -> Interval:
        """Validate that min < max."""
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        return self

    def to_bounds(self) -> tuple[int, int] | tuple[sc.Variable, sc.Variable]:
        """
        Convert to bounds tuple suitable for ROIFilter.set_roi_from_intervals.

        Returns
        -------
        :
            If unit is None, returns integer tuple for pixel indices.
            Otherwise returns tuple of sc.Variable with physical coordinates.
        """
        if self.unit is not None:
            return (
                sc.scalar(self.min, unit=self.unit),
                sc.scalar(self.max, unit=self.unit),
            )
        else:
            return (int(self.min), int(self.max))


class RectangleROI(ROI):
    """
    Rectangle ROI defined by x and y intervals.

    The rectangle is axis-aligned (not rotated).
    """

    x: Interval = Field(description="X interval")
    y: Interval = Field(description="Y interval")

    def get_bounds(
        self, x_dim: str, y_dim: str
    ) -> dict[str, tuple[int, int] | tuple[sc.Variable, sc.Variable]]:
        """
        Get ROI bounds as a dict suitable for ROIFilter.set_roi_from_intervals.

        Parameters
        ----------
        x_dim:
            Name of the x dimension in the data.
        y_dim:
            Name of the y dimension in the data.

        Returns
        -------
        :
            Dict mapping dimension names to bound tuples.
        """
        return {x_dim: self.x.to_bounds(), y_dim: self.y.to_bounds()}

    def to_data_array(self) -> sc.DataArray:
        """Convert to scipp DataArray with bounds dimension."""
        data = sc.array(dims=['bounds'], values=[1, 1], dtype='int32', unit='')
        coords = {
            'x': sc.array(
                dims=['bounds'], values=[self.x.min, self.x.max], unit=self.x.unit
            ),
            'y': sc.array(
                dims=['bounds'], values=[self.y.min, self.y.max], unit=self.y.unit
            ),
        }
        da = sc.DataArray(data, coords=coords)
        return da

    @classmethod
    def _from_data_array(cls, da: sc.DataArray) -> RectangleROI:
        """Create from scipp DataArray."""
        x_vals = da.coords['x'].values
        y_vals = da.coords['y'].values
        return cls(
            x=Interval(
                min=x_vals[0], max=x_vals[1], unit=_unit_to_str(da.coords['x'].unit)
            ),
            y=Interval(
                min=y_vals[0], max=y_vals[1], unit=_unit_to_str(da.coords['y'].unit)
            ),
        )

    @classmethod
    def _empty_data_array_template(cls) -> sc.DataArray:
        """Return empty DataArray template without roi_index."""
        return sc.DataArray(
            sc.empty(dims=['bounds'], shape=[0], dtype='int32', unit=''),
            coords={
                'x': sc.empty(dims=['bounds'], shape=[0]),
                'y': sc.empty(dims=['bounds'], shape=[0]),
            },
        )


class PolygonROI(ROI):
    """
    Polygon ROI defined by a sequence of vertices.

    The polygon is defined by (x, y) coordinate pairs. The polygon is automatically
    closed (last vertex connects to first).

    If x_unit or y_unit is None, the corresponding coordinates are interpreted as
    pixel indices (floats allowed for sub-pixel precision).
    """

    x: list[float] = Field(description="X coordinates of vertices")
    y: list[float] = Field(description="Y coordinates of vertices")
    x_unit: str | None = Field(
        description="Unit for x coordinates (None for pixel indices)"
    )
    y_unit: str | None = Field(
        description="Unit for y coordinates (None for pixel indices)"
    )

    @model_validator(mode='after')
    def validate_vertices(self) -> PolygonROI:
        """Validate that x and y have the same length and at least 3 vertices."""
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")
        if len(self.x) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        return self

    def to_data_array(self) -> sc.DataArray:
        """Convert to scipp DataArray with vertex dimension."""
        n = len(self.x)
        data = sc.array(dims=['vertex'], values=np.ones(n, dtype=np.int32), unit='')
        coords = {
            'x': sc.array(dims=['vertex'], values=self.x, unit=self.x_unit),
            'y': sc.array(dims=['vertex'], values=self.y, unit=self.y_unit),
        }
        da = sc.DataArray(data, coords=coords)
        return da

    @classmethod
    def _from_data_array(cls, da: sc.DataArray) -> PolygonROI:
        """Create from scipp DataArray."""
        return cls(
            x=da.coords['x'].values.tolist(),
            y=da.coords['y'].values.tolist(),
            x_unit=_unit_to_str(da.coords['x'].unit),
            y_unit=_unit_to_str(da.coords['y'].unit),
        )

    @classmethod
    def _empty_data_array_template(cls) -> sc.DataArray:
        """Return empty DataArray template without roi_index."""
        return sc.DataArray(
            sc.empty(dims=['vertex'], shape=[0], dtype='int32', unit=''),
            coords={
                'x': sc.empty(dims=['vertex'], shape=[0]),
                'y': sc.empty(dims=['vertex'], shape=[0]),
            },
        )


class EllipseROI(ROI):
    """
    Ellipse ROI defined by center, radii, and optional rotation.

    The ellipse can be rotated by specifying a rotation angle in degrees.
    Note: Due to rotation, x and y must have the same unit.

    If unit is None, coordinates are interpreted as pixel indices (floats allowed
    for sub-pixel precision).
    """

    center_x: float = Field(description="X coordinate of center")
    center_y: float = Field(description="Y coordinate of center")
    radius_x: float = Field(description="Radius along x-axis", gt=0)
    radius_y: float = Field(description="Radius along y-axis", gt=0)
    rotation: float = Field(
        default=0.0, description="Rotation angle in degrees (counterclockwise)"
    )
    unit: str | None = Field(
        description="Unit for coordinates (None for pixel indices)"
    )

    def to_data_array(self) -> sc.DataArray:
        """Convert to scipp DataArray with ellipse dimension."""
        data = sc.array(dims=['ellipse'], values=[1, 1], dtype='int32', unit='')
        coords = {
            'center': sc.array(
                dims=['ellipse'], values=[self.center_x, self.center_y], unit=self.unit
            ),
            'radius': sc.array(
                dims=['ellipse'], values=[self.radius_x, self.radius_y], unit=self.unit
            ),
        }
        da = sc.DataArray(data, coords=coords)
        # Add rotation as a scalar coordinate (no dimension)
        da.coords['rotation'] = sc.scalar(self.rotation, unit='deg')
        return da

    @classmethod
    def _from_data_array(cls, da: sc.DataArray) -> EllipseROI:
        """Create from scipp DataArray."""
        center = da.coords['center'].values
        radius = da.coords['radius'].values

        # Handle rotation coordinate (can be scalar or dimensional after concatenation)
        if 'rotation' not in da.coords:
            rotation = 0.0
        else:
            rot_coord = da.coords['rotation']
            if rot_coord.ndim == 0:
                # Scalar rotation (single ellipse)
                rotation = float(rot_coord.value)
            else:
                # Dimensional rotation (from concatenation) - take first value
                rotation = float(rot_coord.values[0])

        return cls(
            center_x=float(center[0]),
            center_y=float(center[1]),
            radius_x=float(radius[0]),
            radius_y=float(radius[1]),
            rotation=rotation,
            unit=_unit_to_str(da.coords['center'].unit),
        )

    @classmethod
    def _empty_data_array_template(cls) -> sc.DataArray:
        """Return empty DataArray template without roi_index."""
        return sc.DataArray(
            sc.empty(dims=['ellipse'], shape=[0], dtype='int32', unit=''),
            coords={
                'center': sc.empty(dims=['ellipse'], shape=[0]),
                'radius': sc.empty(dims=['ellipse'], shape=[0]),
                'rotation': sc.empty(dims=['ellipse'], shape=[0], unit='deg'),
            },
        )
