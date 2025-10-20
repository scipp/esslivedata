# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""
Models for configuration values that can be used to control services via Kafka.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from urllib.parse import quote, unquote

import scipp as sc
from pydantic import BaseModel, Field

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
