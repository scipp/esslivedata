# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""
Models for data reduction workflow inputs.

These models are used to define inputs to Sciline workflows. The allow for auto-creating
user interfaces for configuring workflows as well as validation of the inputs on the
frontend.
"""

import json
import math
from abc import ABC
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import scipp as sc
from pydantic import BaseModel, Field, field_validator, model_validator

# One pulse period of the 14 Hz ESS source, in milliseconds (1000 / 14 ≈ 71.43).
# Used as the default upper edge for time-of-arrival histograms: a single pulse
# period is the natural span for events measured relative to the source pulse.
# Rounded *up* to 2 decimals for a clean default: as the histogram's upper edge
# it must not fall below the true period, or events in the final sliver would be
# silently dropped.
ESS_PULSE_PERIOD_MS = math.ceil(1000.0 / 14 * 100) / 100


def parse_number_list(value: str) -> list[float]:
    """Parse a comma-separated string of numbers into floats.

    A blank (or whitespace-only) string yields an empty list. Raises
    ``ValueError`` if any entry is not a number, so it can directly back a
    pydantic ``field_validator`` for free-text numeric-list inputs — the
    parameter widget has no native list input, so such lists are entered as
    text.

    Example: ``"6.2, 9.8, 13"`` -> ``[6.2, 9.8, 13.0]``.
    """
    value = value.strip()
    if not value:
        return []
    try:
        parsed = json.loads(f"[{value}]")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid number list: {e}") from e
    if any(isinstance(x, bool) or not isinstance(x, int | float) for x in parsed):
        raise ValueError("All entries must be numbers")
    return [float(x) for x in parsed]


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

    @property
    def range(self) -> tuple[sc.Variable, sc.Variable]:
        """The range as a tuple of scipp scalars."""
        return (self.get_start(), self.get_stop())


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

    @model_validator(mode='after')
    def start_must_be_positive_if_log(self):
        if self.scale == Scale.LOG and self.start <= 0:
            raise ValueError("start must be positive when scale is 'log'")
        return self


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


class LengthUnit(StrEnum):
    """Allowed units for length."""

    METER = 'm'
    CENTIMETER = 'cm'
    MILLIMETER = 'mm'


class AngleUnit(StrEnum):
    """Allowed units for angles."""

    DEGREE = 'deg'
    RADIAN = 'rad'


class QUnit(StrEnum):
    """Allowed units for Q."""

    INVERSE_ANGSTROM = '1/Å'
    INVERSE_NANOMETER = '1/nm'


class Angle(BaseModel):
    """Model for an angle value."""

    value: float = Field(default=0.0, description="Angle value.")
    unit: AngleUnit = Field(default=AngleUnit.DEGREE, description="Unit of the angle.")

    def get_value(self) -> sc.Variable:
        """Get the angle as a scipp scalar."""
        return sc.scalar(self.value, unit=self.unit.value)


class Filename(BaseModel):
    value: Path = Field(..., description="Path to the file.")


class WavelengthRange(RangeModel):
    """Model for wavelength range."""

    unit: WavelengthUnit = Field(
        default=WavelengthUnit.ANGSTROM, description="Unit of the wavelength range."
    )


class TOARange(RangeModel):
    """Time of arrival range, by default one pulse period of the ESS source."""

    stop: float = Field(default=ESS_PULSE_PERIOD_MS, description="Stop of the range.")
    unit: TimeUnit = Field(
        default=TimeUnit.MS, description="Unit of the interval bounds."
    )


class TOAEdges(EdgesModel):
    """Model for time of arrival edges."""

    unit: TimeUnit = Field(default=TimeUnit.MS, description="Unit of the edges.")

    def get_edges(self) -> sc.Variable:
        """Get the edges as a scipp variable."""
        return make_edges(model=self, dim='time_of_arrival', unit=self.unit.value)


class WavelengthEdges(EdgesModel):
    """Model for wavelength edges."""

    unit: WavelengthUnit = Field(
        default=WavelengthUnit.ANGSTROM, description="Unit of the edges."
    )

    def get_edges(self) -> sc.Variable:
        """Get the edges as a scipp variable."""
        return make_edges(model=self, dim='wavelength', unit=self.unit.value)


_RANGE_FILTER_DESCRIPTION = (
    "Outputs summed over the spectral axis, such as a detector image or the "
    "total in range, include only the histogram bins that overlap the range. "
    "Without a filter they include all bins."
)


class TOASpectralAxis(BaseModel):
    """Histogram events along time of arrival."""

    kind: Literal['toa'] = Field(
        default='toa',
        title="Time of arrival (TOA)",
        description="Time of arrival (TOA) is the time elapsed since the most "
        "recent source pulse.",
    )
    edges: TOAEdges = Field(
        title="Bin edges",
        description="The default range spans one pulse period of the 14 Hz ESS "
        f"source (0 to {ESS_PULSE_PERIOD_MS} ms); events outside the range are "
        "excluded from the histogram.",
        default=TOAEdges(
            start=0.0, stop=ESS_PULSE_PERIOD_MS, num_bins=100, unit=TimeUnit.MS
        ),
    )
    range_filter: TOARange | None = Field(
        default=None, title="Range filter", description=_RANGE_FILTER_DESCRIPTION
    )


class WavelengthSpectralAxis(BaseModel):
    """Histogram events along wavelength."""

    kind: Literal['wavelength'] = Field(
        default='wavelength',
        title="Wavelength",
        description="Wavelength computed from time of arrival with the "
        "instrument's lookup table.",
    )
    edges: WavelengthEdges = Field(
        title="Bin edges",
        default=WavelengthEdges(
            start=1.0, stop=10.0, num_bins=100, unit=WavelengthUnit.ANGSTROM
        ),
    )
    range_filter: WavelengthRange | None = Field(
        default=None, title="Range filter", description=_RANGE_FILTER_DESCRIPTION
    )


SpectralAxis = Annotated[
    TOASpectralAxis | WavelengthSpectralAxis, Field(discriminator='kind')
]
"""The axis along which monitor and detector-view workflows histogram events."""


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


class ThetaEdges(EdgesModel):
    """Model for theta bin edges."""

    unit: AngleUnit = Field(
        default=AngleUnit.DEGREE, description="Unit of the theta bin edges."
    )

    def get_edges(self) -> sc.Variable:
        """Get the edges as a scipp variable."""
        return make_edges(model=self, dim='theta', unit=self.unit.value)


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
    op = {Scale.LINEAR: sc.linspace, Scale.LOG: sc.geomspace}[model.scale]
    return op(
        dim=dim, start=model.start, stop=model.stop, num=model.num_bins + 1, unit=unit
    )
