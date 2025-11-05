# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Low-level storage strategies for buffer data management."""

from __future__ import annotations

from abc import ABC, abstractmethod

import scipp as sc


class StorageStrategy(ABC):
    """
    Low-level storage strategy for buffer data.

    Manages data accumulation and eviction using simple concat-and-trim operations.
    Always maintains contiguous views of stored data.
    """

    @abstractmethod
    def append(self, data: sc.DataArray) -> None:
        """
        Append new data to storage.

        Parameters
        ----------
        data:
            The data to append. Must be compatible with existing stored data.
        """

    @abstractmethod
    def get_all(self) -> sc.DataArray | None:
        """
        Get all stored data.

        Returns
        -------
        :
            The complete stored data as a contiguous DataArray, or None if empty.
        """

    @abstractmethod
    def estimate_memory(self) -> int:
        """
        Estimate memory usage in bytes.

        Returns
        -------
        :
            Estimated memory usage in bytes.
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored data."""


class SlidingWindowStorage(StorageStrategy):
    """
    Fixed-size storage that maintains the most recent data.

    Uses a buffer with 2x capacity and index tracking to avoid repeated
    copying. Only performs data movement when the buffer fills completely.
    """

    def __init__(self, max_size: int, concat_dim: str = 'time') -> None:
        """
        Initialize sliding window storage.

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
        self._start = 0
        self._end = 0

    def _ensure_capacity(self, new_data: sc.DataArray) -> None:
        """Ensure buffer has capacity for new data."""
        if self._buffer is None:
            # Initialize with 2x capacity
            new_size = new_data.sizes[self._concat_dim]
            capacity = max(self._max_size * 2, new_size)

            # Create buffer with 2x capacity
            self._buffer = sc.concat(
                [new_data] + [new_data[self._concat_dim, :1]] * (capacity - new_size),
                dim=self._concat_dim,
            )
            self._end = new_size
            return

        # Check if we need to compact
        buffer_size = self._buffer.sizes[self._concat_dim]
        new_size = new_data.sizes[self._concat_dim]

        if self._end + new_size > buffer_size:
            # Need to make room - keep last max_size elements
            if self._end - self._start > self._max_size:
                self._start = self._end - self._max_size

            # Compact buffer to front
            active_data = self._buffer[self._concat_dim, self._start : self._end]
            self._buffer = sc.concat(
                [active_data]
                + [active_data[self._concat_dim, :1]]
                * (buffer_size - (self._end - self._start)),
                dim=self._concat_dim,
            )
            self._start = 0
            self._end = active_data.sizes[self._concat_dim]

    def append(self, data: sc.DataArray) -> None:
        if self._concat_dim not in data.dims:
            raise ValueError(f"Data must have '{self._concat_dim}' dimension")

        self._ensure_capacity(data)
        assert self._buffer is not None

        # Write data to buffer
        new_size = data.sizes[self._concat_dim]
        self._buffer[self._concat_dim, self._end : self._end + new_size] = data
        self._end += new_size

        # Update start if we exceeded max_size
        if self._end - self._start > self._max_size:
            self._start = self._end - self._max_size

    def get_all(self) -> sc.DataArray | None:
        if self._buffer is None:
            return None
        return self._buffer[self._concat_dim, self._start : self._end].copy()

    def estimate_memory(self) -> int:
        if self._buffer is None:
            return 0
        return self._buffer.values.nbytes

    def clear(self) -> None:
        self._start = 0
        self._end = 0


class GrowingStorage(StorageStrategy):
    """
    Storage that grows by doubling capacity until reaching maximum size.

    Uses index tracking and in-place writes to avoid repeated copying.
    """

    def __init__(
        self, initial_size: int = 100, max_size: int = 10000, concat_dim: str = 'time'
    ) -> None:
        """
        Initialize growing storage.

        Parameters
        ----------
        initial_size:
            Initial capacity.
        max_size:
            Maximum capacity.
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
        self._end = 0

    def _ensure_capacity(self, new_data: sc.DataArray) -> None:
        """Ensure buffer has capacity for new data."""
        new_size = new_data.sizes[self._concat_dim]

        if self._buffer is None:
            # Initialize with initial capacity
            capacity = max(self._initial_size, new_size)
            self._buffer = sc.concat(
                [new_data] + [new_data[self._concat_dim, :1]] * (capacity - new_size),
                dim=self._concat_dim,
            )
            self._end = new_size
            return

        buffer_capacity = self._buffer.sizes[self._concat_dim]

        # Check if we need to grow
        if self._end + new_size > buffer_capacity:
            # Double capacity up to max_size
            new_capacity = min(buffer_capacity * 2, self._max_size)

            if new_capacity > buffer_capacity:
                # Grow the buffer
                active_data = self._buffer[self._concat_dim, : self._end]
                self._buffer = sc.concat(
                    [
                        self._buffer,
                        active_data[self._concat_dim, :1]
                        * (new_capacity - buffer_capacity),
                    ],
                    dim=self._concat_dim,
                )

            # If still not enough room, need to trim old data
            if self._end + new_size > self._max_size:
                # Keep last (max_size - new_size) elements
                keep = self._max_size - new_size
                self._buffer[self._concat_dim, :keep] = self._buffer[
                    self._concat_dim, self._end - keep : self._end
                ]
                self._end = keep

    def append(self, data: sc.DataArray) -> None:
        if self._concat_dim not in data.dims:
            raise ValueError(f"Data must have '{self._concat_dim}' dimension")

        self._ensure_capacity(data)
        assert self._buffer is not None

        # Write data to buffer
        new_size = data.sizes[self._concat_dim]
        self._buffer[self._concat_dim, self._end : self._end + new_size] = data
        self._end += new_size

    def get_all(self) -> sc.DataArray | None:
        if self._buffer is None:
            return None
        return self._buffer[self._concat_dim, : self._end].copy()

    def estimate_memory(self) -> int:
        if self._buffer is None:
            return 0
        return self._buffer.values.nbytes

    def clear(self) -> None:
        self._end = 0
