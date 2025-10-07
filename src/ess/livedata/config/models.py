# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""
Models for configuration values that can be used to control services via Kafka.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal

import numpy as np
import scipp as sc
from pydantic import BaseModel, Field, model_validator

TimeUnit = Literal['ns', 'us', 'μs', 'ms', 's']


class WeightingMethod(str, Enum):
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
            String in the format source_name/service_name/key with '*' for None values
        """
        source = '*' if self.source_name is None else self.source_name
        service = '*' if self.service_name is None else self.service_name
        return f"{source}/{service}/{self.key}"

    @classmethod
    def from_string(cls, key_str: str) -> ConfigKey:
        """
        Create a ConfigKey from its string representation.

        Parameters
        ----------
        key_str:
            String in the format 'source_name/service_name/key'

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
        if source_name == '*':
            source_name = None
        if service_name == '*':
            service_name = None
        return cls(source_name=source_name, service_name=service_name, key=key)


class ROIType(str, Enum):
    """Types of Region of Interest (ROI) shapes."""

    RECTANGLE = 'rectangle'
    POLYGON = 'polygon'
    ELLIPSE = 'ellipse'


class ROI(BaseModel, ABC):
    """
    Base class for Region of Interest (ROI) definitions.

    ROIs can be serialized to/from scipp DataArrays which can then be converted
    to da00 format using the existing compat module for Kafka transmission.

    The ROI type is encoded in the DataArray's name attribute, which maps to the
    'label' field in da00 Variable for the signal.
    """

    @abstractmethod
    def to_data_array(self) -> sc.DataArray:
        """
        Convert ROI to scipp DataArray representation.

        The DataArray name is set to the ROI type to distinguish ROI shapes.

        Returns
        -------
        :
            DataArray with ROI geometry stored in coordinates and type in name.
        """
        ...

    @classmethod
    def from_data_array(cls, da: sc.DataArray) -> ROI:
        """
        Create ROI from scipp DataArray representation.

        Dispatches to the appropriate ROI subclass based on the DataArray name.

        Parameters
        ----------
        da:
            DataArray with ROI geometry in coordinates and type in name.

        Returns
        -------
        :
            ROI instance (Rectangle, Polygon, or Ellipse).
        """
        if da.name is None or da.name == '':
            raise ValueError("DataArray missing name (roi_type)")

        roi_type = str(da.name)

        if roi_type == ROIType.RECTANGLE:
            return RectangleROI._from_data_array(da)
        elif roi_type == ROIType.POLYGON:
            return PolygonROI._from_data_array(da)
        elif roi_type == ROIType.ELLIPSE:
            return EllipseROI._from_data_array(da)
        else:
            raise ValueError(f"Unknown ROI type: {roi_type}")

    @classmethod
    @abstractmethod
    def _from_data_array(cls, da: sc.DataArray) -> ROI:
        """
        Internal method to create specific ROI subclass from DataArray.

        Subclasses must implement this method.
        """
        ...


class RectangleROI(ROI):
    """
    Rectangle ROI defined by x and y bounds.

    The rectangle is axis-aligned (not rotated).
    """

    x_min: float = Field(description="Minimum x coordinate")
    x_max: float = Field(description="Maximum x coordinate")
    y_min: float = Field(description="Minimum y coordinate")
    y_max: float = Field(description="Maximum y coordinate")
    x_unit: str = Field(description="Unit for x coordinates")
    y_unit: str = Field(description="Unit for y coordinates")

    @model_validator(mode='after')
    def validate_bounds(self) -> RectangleROI:
        """Validate that min < max for both dimensions."""
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be < x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be < y_max ({self.y_max})")
        return self

    def to_data_array(self) -> sc.DataArray:
        """Convert to scipp DataArray with bounds dimension."""
        data = sc.array(dims=['bounds'], values=[1, 1], dtype='int32', unit='')
        coords = {
            'x': sc.array(
                dims=['bounds'], values=[self.x_min, self.x_max], unit=self.x_unit
            ),
            'y': sc.array(
                dims=['bounds'], values=[self.y_min, self.y_max], unit=self.y_unit
            ),
        }
        da = sc.DataArray(data, coords=coords, name=ROIType.RECTANGLE)
        return da

    @classmethod
    def _from_data_array(cls, da: sc.DataArray) -> RectangleROI:
        """Create from scipp DataArray."""
        x = da.coords['x'].values
        y = da.coords['y'].values
        return cls(
            x_min=float(x[0]),
            x_max=float(x[1]),
            y_min=float(y[0]),
            y_max=float(y[1]),
            x_unit=str(da.coords['x'].unit),
            y_unit=str(da.coords['y'].unit),
        )


class PolygonROI(ROI):
    """
    Polygon ROI defined by a sequence of vertices.

    The polygon is defined by (x, y) coordinate pairs. The polygon is automatically
    closed (last vertex connects to first).
    """

    x: list[float] = Field(description="X coordinates of vertices")
    y: list[float] = Field(description="Y coordinates of vertices")
    x_unit: str = Field(description="Unit for x coordinates")
    y_unit: str = Field(description="Unit for y coordinates")

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
        da = sc.DataArray(data, coords=coords, name=ROIType.POLYGON)
        return da

    @classmethod
    def _from_data_array(cls, da: sc.DataArray) -> PolygonROI:
        """Create from scipp DataArray."""
        return cls(
            x=da.coords['x'].values.tolist(),
            y=da.coords['y'].values.tolist(),
            x_unit=str(da.coords['x'].unit),
            y_unit=str(da.coords['y'].unit),
        )


class EllipseROI(ROI):
    """
    Ellipse ROI defined by center, radii, and optional rotation.

    The ellipse can be rotated by specifying a rotation angle in degrees.
    Note: Due to rotation, x and y must have the same unit.
    """

    center_x: float = Field(description="X coordinate of center")
    center_y: float = Field(description="Y coordinate of center")
    radius_x: float = Field(description="Radius along x-axis", gt=0)
    radius_y: float = Field(description="Radius along y-axis", gt=0)
    rotation: float = Field(
        default=0.0, description="Rotation angle in degrees (counterclockwise)"
    )
    unit: str = Field(description="Unit for coordinates (must be same for x and y)")

    def to_data_array(self) -> sc.DataArray:
        """Convert to scipp DataArray with dim dimension."""
        data = sc.array(dims=['dim'], values=[1, 1], dtype='int32', unit='')
        coords = {
            'center': sc.array(
                dims=['dim'], values=[self.center_x, self.center_y], unit=self.unit
            ),
            'radius': sc.array(
                dims=['dim'], values=[self.radius_x, self.radius_y], unit=self.unit
            ),
        }
        da = sc.DataArray(data, coords=coords, name=ROIType.ELLIPSE)
        # Add rotation as a scalar coordinate (no dimension)
        da.coords['rotation'] = sc.scalar(self.rotation, unit='deg')
        return da

    @classmethod
    def _from_data_array(cls, da: sc.DataArray) -> EllipseROI:
        """Create from scipp DataArray."""
        center = da.coords['center'].values
        radius = da.coords['radius'].values
        rotation = (
            float(da.coords['rotation'].value) if 'rotation' in da.coords else 0.0
        )
        return cls(
            center_x=float(center[0]),
            center_y=float(center[1]),
            radius_x=float(radius[0]),
            radius_y=float(radius[1]),
            rotation=rotation,
            unit=str(da.coords['center'].unit),
        )
