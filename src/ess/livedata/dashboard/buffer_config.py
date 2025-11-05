# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Configuration for buffer strategies in HistoryBufferService."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import scipp as sc

from .buffer_strategy import (
    BufferStrategy,
    FixedSizeCircularBuffer,
    GrowingBuffer,
    TimeWindowBuffer,
)


class BufferStrategyType(Enum):
    """Available buffer strategy types."""

    FIXED_SIZE = "fixed_size"
    GROWING = "growing"
    TIME_WINDOW = "time_window"


@dataclass
class BufferConfig:
    """
    Configuration for a buffer strategy.

    Parameters
    ----------
    strategy_type:
        The type of buffer strategy to use.
    max_memory_mb:
        Maximum memory in megabytes for this buffer.
    max_points:
        Maximum number of points to keep (for size-based strategies).
    time_window:
        Time window to keep (for time-based strategies).
    concat_dim:
        The dimension along which to concatenate data.
    """

    strategy_type: BufferStrategyType
    max_memory_mb: float = 100.0
    max_points: int | None = None
    time_window: sc.Variable | None = None
    concat_dim: str = 'time'

    def create_strategy(self) -> BufferStrategy:
        """
        Create a BufferStrategy instance from this configuration.

        Returns
        -------
        :
            A BufferStrategy instance configured according to this config.
        """
        if self.strategy_type == BufferStrategyType.TIME_WINDOW:
            if self.time_window is None:
                raise ValueError(
                    "time_window must be specified for TIME_WINDOW strategy"
                )
            return TimeWindowBuffer(time_window=self.time_window)

        elif self.strategy_type == BufferStrategyType.FIXED_SIZE:
            if self.max_points is None:
                # Estimate max_points from memory budget
                # Assume ~8 bytes per element (float64)
                max_points = int(self.max_memory_mb * 1024 * 1024 / 8)
            else:
                max_points = self.max_points
            return FixedSizeCircularBuffer(
                max_size=max_points, concat_dim=self.concat_dim
            )

        elif self.strategy_type == BufferStrategyType.GROWING:
            if self.max_points is None:
                # Estimate max_points from memory budget
                max_points = int(self.max_memory_mb * 1024 * 1024 / 8)
            else:
                max_points = self.max_points
            initial_size = min(100, max_points // 10)
            return GrowingBuffer(
                initial_size=initial_size,
                max_size=max_points,
                concat_dim=self.concat_dim,
            )

        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")


class BufferConfigRegistry:
    """
    Registry for type-based default buffer configurations.

    Provides a pluggable mechanism to determine buffer configuration
    based on data characteristics.
    """

    def __init__(self) -> None:
        self._detectors: list[
            tuple[Callable[[sc.DataArray], bool], Callable[[], BufferConfig]]
        ] = []

    def register(
        self,
        detector: Callable[[sc.DataArray], bool],
        config_factory: Callable[[], BufferConfig],
    ) -> None:
        """
        Register a type detector and corresponding config factory.

        Parameters
        ----------
        detector:
            A function that takes a DataArray and returns True if this
            config should be used.
        config_factory:
            A function that returns a BufferConfig for this type.
        """
        self._detectors.append((detector, config_factory))

    def get_config(self, data: sc.DataArray) -> BufferConfig:
        """
        Get the appropriate buffer configuration for the given data.

        Parameters
        ----------
        data:
            The data to analyze.

        Returns
        -------
        :
            A BufferConfig appropriate for this data type.
        """
        for detector, config_factory in self._detectors:
            if detector(data):
                return config_factory()

        # Default fallback
        return self._default_config()

    @staticmethod
    def _default_config() -> BufferConfig:
        """Default buffer configuration."""
        return BufferConfig(
            strategy_type=BufferStrategyType.FIXED_SIZE,
            max_memory_mb=50.0,
            max_points=1000,
        )


# Create a default registry with common type detectors
def _is_timeseries(data: sc.DataArray) -> bool:
    """Detect if data is a timeseries (1D with time dimension)."""
    return 'time' in data.dims and data.ndim == 1 and 'time' in data.coords


def _is_2d_image(data: sc.DataArray) -> bool:
    """Detect if data is a 2D image."""
    return data.ndim == 2 and 'time' not in data.dims


def _is_time_varying_image(data: sc.DataArray) -> bool:
    """Detect if data is a time-varying image (3D with time)."""
    return data.ndim >= 2 and 'time' in data.dims


def _timeseries_config() -> BufferConfig:
    """Configuration for timeseries data."""
    return BufferConfig(
        strategy_type=BufferStrategyType.TIME_WINDOW,
        time_window=sc.scalar(300, unit='s'),  # 5 minutes
        concat_dim='time',
    )


def _image_config() -> BufferConfig:
    """Configuration for 2D images."""
    return BufferConfig(
        strategy_type=BufferStrategyType.FIXED_SIZE,
        max_memory_mb=200.0,
        max_points=100,  # Keep last 100 images
    )


def _time_varying_image_config() -> BufferConfig:
    """Configuration for time-varying images."""
    return BufferConfig(
        strategy_type=BufferStrategyType.FIXED_SIZE,
        max_memory_mb=500.0,
        max_points=50,  # Keep last 50 frames
        concat_dim='time',
    )


# Create and populate default registry
default_registry = BufferConfigRegistry()
default_registry.register(_is_timeseries, _timeseries_config)
default_registry.register(_is_time_varying_image, _time_varying_image_config)
default_registry.register(_is_2d_image, _image_config)
