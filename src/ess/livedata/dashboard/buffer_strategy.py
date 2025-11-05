# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Buffer strategies for managing historical data in HistoryBufferService."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import scipp as sc

T = TypeVar('T', bound=sc.DataArray)


class BufferStrategy(ABC):
    """
    Protocol for buffer management strategies.

    A buffer strategy determines how data is accumulated, stored, and evicted
    when size or time limits are exceeded.
    """

    @abstractmethod
    def append(self, data: sc.DataArray) -> None:
        """
        Append new data to the buffer.

        Parameters
        ----------
        data:
            The data to append. Must be compatible with existing buffered data.
        """

    @abstractmethod
    def get_buffer(self) -> sc.DataArray | None:
        """
        Get the complete buffered data.

        Returns
        -------
        :
            The full buffer as a DataArray, or None if empty.
        """

    @abstractmethod
    def get_window(self, size: int | None = None) -> sc.DataArray | None:
        """
        Get a window of buffered data.

        Parameters
        ----------
        size:
            The number of elements to return from the end of the buffer.
            If None, returns the entire buffer.

        Returns
        -------
        :
            A window of the buffer, or None if empty.
        """

    @abstractmethod
    def estimate_memory(self) -> int:
        """
        Estimate the memory usage of the buffer in bytes.

        Returns
        -------
        :
            Estimated memory usage in bytes.
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the buffer."""


class TimeWindowBuffer(BufferStrategy):
    """
    Buffer strategy that maintains data within a time window.

    Keeps only data where the 'time' coordinate falls within the specified
    window from the most recent timestamp.
    """

    def __init__(self, time_window: sc.Variable) -> None:
        """
        Initialize a time window buffer.

        Parameters
        ----------
        time_window:
            The time window to maintain. Must be a scalar time duration.
            Example: sc.scalar(300, unit='s') for 5 minutes.
        """
        if time_window.ndim != 0:
            raise ValueError("time_window must be a scalar")
        self._time_window = time_window
        self._buffer: sc.DataArray | None = None

    def append(self, data: sc.DataArray) -> None:
        if 'time' not in data.dims:
            raise ValueError("Data must have 'time' dimension for TimeWindowBuffer")

        if self._buffer is None:
            self._buffer = data.copy()
        else:
            # Concatenate along time dimension
            self._buffer = sc.concat([self._buffer, data], dim='time')

        # Evict old data outside time window
        self._evict_old_data()

    def _evict_old_data(self) -> None:
        """Remove data outside the time window."""
        if self._buffer is None or len(self._buffer.coords['time']) == 0:
            return

        latest_time = self._buffer.coords['time'][-1]
        cutoff_time = latest_time - self._time_window

        # Keep only data within the window
        mask = self._buffer.coords['time'] >= cutoff_time
        self._buffer = self._buffer[mask]

    def get_buffer(self) -> sc.DataArray | None:
        return self._buffer.copy() if self._buffer is not None else None

    def get_window(self, size: int | None = None) -> sc.DataArray | None:
        if self._buffer is None:
            return None
        if size is None:
            return self._buffer.copy()
        # Return last 'size' elements along time dimension
        return self._buffer['time', -size:].copy()

    def estimate_memory(self) -> int:
        if self._buffer is None:
            return 0
        # Estimate: number of elements * bytes per element (assume float64 = 8 bytes)
        return (
            self._buffer.sizes['time']
            * 8
            * (
                1
                if self._buffer.ndim == 1
                else self._buffer.values.size // self._buffer.sizes['time']
            )
        )

    def clear(self) -> None:
        self._buffer = None


class FixedSizeCircularBuffer(BufferStrategy):
    """
    Buffer strategy with fixed maximum size using circular indexing.

    When the buffer reaches max_size, new data overwrites the oldest data
    in a circular manner.
    """

    def __init__(self, max_size: int, concat_dim: str = 'time') -> None:
        """
        Initialize a fixed-size circular buffer.

        Parameters
        ----------
        max_size:
            Maximum number of elements to keep along the concat dimension.
        concat_dim:
            The dimension along which to concatenate data.
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._max_size = max_size
        self._concat_dim = concat_dim
        self._buffer: sc.DataArray | None = None
        self._write_index = 0
        self._count = 0  # Total elements written

    def append(self, data: sc.DataArray) -> None:
        if self._concat_dim not in data.dims:
            raise ValueError(
                f"Data must have '{self._concat_dim}' dimension for circular buffer"
            )

        new_size = data.sizes[self._concat_dim]

        if self._buffer is None:
            # First append - initialize buffer
            if new_size <= self._max_size:
                self._buffer = data.copy()
                self._count = new_size
            else:
                # New data is larger than max_size - take only last max_size
                self._buffer = data[self._concat_dim, -self._max_size :].copy()
                self._count = self._max_size
            return

        # Subsequent appends
        if self._count < self._max_size:
            # Buffer not yet full - simple concatenation
            self._buffer = sc.concat([self._buffer, data], dim=self._concat_dim)
            self._count += new_size
            if self._count > self._max_size:
                # Trim to max_size
                self._buffer = self._buffer[self._concat_dim, -self._max_size :]
                self._count = self._max_size
        else:
            # Buffer is full - need circular overwrite
            # For simplicity, concatenate and trim to last max_size
            self._buffer = sc.concat([self._buffer, data], dim=self._concat_dim)
            self._buffer = self._buffer[self._concat_dim, -self._max_size :]
            self._count = self._max_size

    def get_buffer(self) -> sc.DataArray | None:
        return self._buffer.copy() if self._buffer is not None else None

    def get_window(self, size: int | None = None) -> sc.DataArray | None:
        if self._buffer is None:
            return None
        if size is None:
            return self._buffer.copy()
        actual_size = min(size, self._buffer.sizes[self._concat_dim])
        return self._buffer[self._concat_dim, -actual_size:].copy()

    def estimate_memory(self) -> int:
        if self._buffer is None:
            return 0
        # Estimate based on number of elements
        total_elements = self._buffer.values.size
        return total_elements * 8  # Assume float64

    def clear(self) -> None:
        self._buffer = None
        self._write_index = 0
        self._count = 0


class GrowingBuffer(BufferStrategy):
    """
    Buffer strategy that grows dynamically up to a maximum limit.

    Starts with a small buffer and doubles capacity when full, up to max_size.
    When max_size is reached, evicts oldest data.
    """

    def __init__(
        self, initial_size: int = 100, max_size: int = 10000, concat_dim: str = 'time'
    ) -> None:
        """
        Initialize a growing buffer.

        Parameters
        ----------
        initial_size:
            Initial buffer capacity.
        max_size:
            Maximum buffer capacity.
        concat_dim:
            The dimension along which to concatenate data.
        """
        if initial_size <= 0 or max_size <= 0:
            raise ValueError("initial_size and max_size must be positive")
        if initial_size > max_size:
            raise ValueError("initial_size cannot exceed max_size")

        self._initial_size = initial_size
        self._max_size = max_size
        self._concat_dim = concat_dim
        self._buffer: sc.DataArray | None = None
        self._current_capacity = initial_size

    def append(self, data: sc.DataArray) -> None:
        if self._concat_dim not in data.dims:
            raise ValueError(
                f"Data must have '{self._concat_dim}' dimension for growing buffer"
            )

        if self._buffer is None:
            self._buffer = data.copy()
            # Check if initial data exceeds max_size
            if self._buffer.sizes[self._concat_dim] > self._max_size:
                self._buffer = self._buffer[self._concat_dim, -self._max_size :]
            return

        # Concatenate new data
        self._buffer = sc.concat([self._buffer, data], dim=self._concat_dim)
        current_size = self._buffer.sizes[self._concat_dim]

        # Check if we need to grow or evict
        if current_size > self._current_capacity:
            if self._current_capacity < self._max_size:
                # Grow capacity (double it)
                self._current_capacity = min(self._current_capacity * 2, self._max_size)

            # If still over capacity, trim to max_size
            if current_size > self._max_size:
                self._buffer = self._buffer[self._concat_dim, -self._max_size :]

    def get_buffer(self) -> sc.DataArray | None:
        return self._buffer.copy() if self._buffer is not None else None

    def get_window(self, size: int | None = None) -> sc.DataArray | None:
        if self._buffer is None:
            return None
        if size is None:
            return self._buffer.copy()
        actual_size = min(size, self._buffer.sizes[self._concat_dim])
        return self._buffer[self._concat_dim, -actual_size:].copy()

    def estimate_memory(self) -> int:
        if self._buffer is None:
            return 0
        total_elements = self._buffer.values.size
        return total_elements * 8  # Assume float64

    def clear(self) -> None:
        self._buffer = None
        self._current_capacity = self._initial_size
