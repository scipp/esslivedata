# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""
Models for data reduction workflow inputs.

These models are used to define inputs to Sciline workflows. The allow for auto-creating
user interfaces for configuring workflows as well as validation of the inputs on the
frontend.
"""

from abc import ABC
from enum import StrEnum
from pathlib import Path

import scipp as sc
from pydantic import BaseModel, Field, field_validator


class RangeModel(BaseModel, ABC):
    """Base model for range with common fields and validation."""

    start: float = Field(default=0.0, description="Start of the range.")
    stop: float = Field(default=10.0, description="Stop of the range.")

    @field_validator('stop')
    @classmethod
    def stop_must_be_greater_than_start(cls, v, info):
        start = info.data.get('start')
        if start is not None and v <= start:
            raise ValueError('stop must be greater than start')
        return v

    def get_start(self) -> sc.Variable:
        """Get the start of the range as a scipp variable."""
        return sc.scalar(self.start, unit=self.unit.value)

    def get_stop(self) -> sc.Variable:
        """Get the stop of the range as a scipp variable."""
        return sc.scalar(self.stop, unit=self.unit.value)


class Scale(StrEnum):
    """Allowed scales for data reduction."""

    LINEAR = 'linear'
    LOG = 'log'


class EdgesModel(BaseModel, ABC):
    """Base model for edges with common fields and validation."""

    start: float = Field(default=1.0, description="Start of the edges.")
    stop: float = Field(default=10.0, description="Stop of the edges.")
    num_bins: int = Field(default=100, ge=1, le=10000, description="Number of bins.")
    scale: Scale = Field(
        default=Scale.LINEAR,
        description="Scale of the edges, either 'linear' or 'log'.",
    )

    @field_validator('stop')
    @classmethod
    def stop_must_be_greater_than_start(cls, v, info):
        start = info.data.get('start')
        if start is not None and v <= start:
            raise ValueError('stop must be greater than start')
        return v

    @field_validator('start')
    @classmethod
    def start_must_be_positive_if_log(cls, v, info):
        log = info.data.get('log')
        if log and v <= 0:
            raise ValueError('start must be positive if log is True')
        return v


class TimeUnit(StrEnum):
    """Allowed units for time."""

    NS = 'ns'
    US = 'us'
    MICROSECOND = 'μs'
    MS = 'ms'
    S = 's'


class WavelengthUnit(StrEnum):
    """Allowed units for wavelength."""

    ANGSTROM = 'Å'
    NANOMETER = 'nm'


class DspacingUnit(StrEnum):
    """Allowed units for d-spacing."""

    ANGSTROM = 'Å'
    NANOMETER = 'nm'


class AngleUnit(StrEnum):
    """Allowed units for angles."""

    DEGREE = 'deg'
    RADIAN = 'rad'


class QUnit(StrEnum):
    """Allowed units for Q."""

    INVERSE_ANGSTROM = '1/Å'
    INVERSE_NANOMETER = '1/nm'


class Filename(BaseModel):
    value: Path = Field(..., description="Path to the file.")


class WavelengthRange(RangeModel):
    """Model for wavelength range."""

    unit: WavelengthUnit = Field(
        default=WavelengthUnit.ANGSTROM, description="Unit of the wavelength range."
    )


class TOARange(RangeModel):
    """Time of arrival range filter settings."""

    enabled: bool = Field(default=False, description="Enable the range filter.")
    unit: TimeUnit = Field(
        default=TimeUnit.MICROSECOND, description="Unit of the interval bounds."
    )

    @property
    def range(self) -> tuple[sc.Variable, sc.Variable]:
        """Time of arrival range as a tuple of scipp scalars."""
        return (self.get_start(), self.get_stop())


class TOAEdges(EdgesModel):
    """Model for time of arrival edges."""

    unit: TimeUnit = Field(default=TimeUnit.MS, description="Unit of the edges.")

    def get_edges(self) -> sc.Variable:
        """Get the edges as a scipp variable."""
        return make_edges(model=self, dim='time_of_arrival', unit=self.unit.value)


class TOFRange(RangeModel):
    """Time of flight range filter settings."""

    enabled: bool = Field(default=False, description="Enable the range filter.")
    unit: TimeUnit = Field(
        default=TimeUnit.MICROSECOND, description="Unit of the interval bounds."
    )

    @property
    def range(self) -> tuple[sc.Variable, sc.Variable]:
        """Time-of-flight range as a tuple of scipp scalars."""
        return (self.get_start(), self.get_stop())


class TOFEdges(EdgesModel):
    """Model for time of flight edges."""

    unit: TimeUnit = Field(default=TimeUnit.MS, description="Unit of the edges.")

    def get_edges(self) -> sc.Variable:
        """Get the edges as a scipp variable."""
        return make_edges(model=self, dim='tof', unit=self.unit.value)


class WavelengthEdges(EdgesModel):
    """Model for wavelength edges."""

    unit: WavelengthUnit = Field(
        default=WavelengthUnit.ANGSTROM, description="Unit of the edges."
    )

    def get_edges(self) -> sc.Variable:
        """Get the edges as a scipp variable."""
        return make_edges(model=self, dim='wavelength', unit=self.unit.value)


class WavelengthRangeFilter(RangeModel):
    """Wavelength range filter settings for detector view."""

    enabled: bool = Field(default=False, description="Enable the range filter.")
    unit: WavelengthUnit = Field(
        default=WavelengthUnit.ANGSTROM, description="Unit of the range bounds."
    )

    @property
    def range(self) -> tuple[sc.Variable, sc.Variable]:
        """Wavelength range as a tuple of scipp scalars."""
        return (self.get_start(), self.get_stop())


class DspacingEdges(EdgesModel):
    """Model for d-spacing edges."""

    unit: DspacingUnit = Field(
        default=DspacingUnit.ANGSTROM, description="Unit of the edges."
    )

    def get_edges(self) -> sc.Variable:
        """Get the edges as a scipp variable."""
        return make_edges(model=self, dim='dspacing', unit=self.unit.value)


class TwoTheta(EdgesModel):
    """Model for two-theta edges."""

    unit: AngleUnit = Field(default=AngleUnit.DEGREE, description="Unit of the edges.")

    def get_edges(self) -> sc.Variable:
        """Get the edges as a scipp variable."""
        return make_edges(model=self, dim='two_theta', unit=self.unit.value)


class QEdges(EdgesModel):
    """Model for Q edges."""

    unit: QUnit = Field(
        default=QUnit.INVERSE_ANGSTROM, description="Unit of the Q edges."
    )

    def get_edges(self) -> sc.Variable:
        """Get the edges as a scipp variable."""
        return make_edges(model=self, dim='Q', unit=self.unit.value)


class EnergyUnit(StrEnum):
    MILLI_EV = 'meV'
    MICRO_EV = 'μeV'


class EnergyEdges(EdgesModel):
    """Model for energy transfer edges."""

    unit: EnergyUnit = Field(
        default=EnergyUnit.MILLI_EV, description="Unit of the energy transfer edges."
    )

    def get_edges(self) -> sc.Variable:
        """Get the edges as a scipp variable."""
        return make_edges(model=self, dim='ΔE', unit=self.unit.value)


def make_edges(*, model: EdgesModel, dim: str, unit: str) -> sc.Variable:
    """Convert the edges to a scipp variable."""
    op = {Scale.LINEAR: sc.linspace, Scale.LOG: sc.logspace}[model.scale]
    return op(
        dim=dim, start=model.start, stop=model.stop, num=model.num_bins + 1, unit=unit
    )
