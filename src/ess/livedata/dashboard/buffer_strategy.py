# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Low-level storage strategies for buffer data management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, TypeVar

import scipp as sc

# Type variable for buffer types
T = TypeVar('T')


class StorageStrategy(ABC):
    """
    Low-level storage strategy for buffer data.

    Manages data accumulation and eviction using pre-allocated buffers with
    in-place writes. This avoids the O(n²) complexity of naive concatenation,
    where each append requires copying all existing data: n appends of size m
    each would be O(n·m²) total. Instead, we pre-allocate with doubling
    capacity and use numpy-level indexing for O(1) appends, achieving
    O(n·m) amortized complexity.

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


class BufferInterface(Protocol[T]):
    """
    Protocol for buffer implementations.

    Defines the minimal interface needed by BufferStorage. Implementations
    handle the details of allocating, writing, shifting, and viewing buffers.
    """

    def allocate(self, template: T, capacity: int) -> T:
        """
        Allocate a new buffer with the given capacity.

        Parameters
        ----------
        template:
            Sample data used to determine buffer structure (dtype, dims, etc.).
        capacity:
            Size along concat dimension.

        Returns
        -------
        :
            Newly allocated buffer.
        """
        ...

    def write_slice(self, buffer: T, start: int, end: int, data: T) -> None:
        """
        Write data to a buffer slice in-place.

        Parameters
        ----------
        buffer:
            Pre-allocated buffer to write into.
        start:
            Start index along concat dimension.
        end:
            End index along concat dimension (exclusive).
        data:
            Data to write. Size must match (end - start).
        """
        ...

    def shift(self, buffer: T, src_start: int, src_end: int, dst_start: int) -> None:
        """
        Shift a section of the buffer to a new position in-place.

        Parameters
        ----------
        buffer:
            Buffer to modify in-place.
        src_start:
            Start of source slice.
        src_end:
            End of source slice (exclusive).
        dst_start:
            Start of destination position.
        """
        ...

    def get_view(self, buffer: T, start: int, end: int) -> T:
        """
        Get a view/copy of a buffer slice.

        Parameters
        ----------
        buffer:
            Buffer to slice.
        start:
            Start index along concat dimension.
        end:
            End index along concat dimension (exclusive).

        Returns
        -------
        :
            View or copy of the buffer slice.
        """
        ...

    def estimate_memory(self, buffer: T) -> int:
        """
        Estimate memory usage of buffer in bytes.

        Parameters
        ----------
        buffer:
            Buffer to estimate.

        Returns
        -------
        :
            Memory usage in bytes.
        """
        ...


class VariableBuffer:
    """
    Simple buffer implementation for sc.Variable.

    Handles the concat dimension but otherwise just uses numpy-level slicing.
    """

    def __init__(self, concat_dim: str = 'time') -> None:
        """
        Initialize Variable buffer implementation.

        Parameters
        ----------
        concat_dim:
            The dimension along which to concatenate data.
        """
        self._concat_dim = concat_dim

    def allocate(self, template: sc.Variable, capacity: int) -> sc.Variable:
        """Allocate a new Variable buffer with given capacity."""
        shape = [
            capacity if dim == self._concat_dim else size
            for dim, size in zip(template.dims, template.shape, strict=True)
        ]
        return sc.zeros(dims=template.dims, shape=shape, dtype=template.dtype)

    def write_slice(
        self, buffer: sc.Variable, start: int, end: int, data: sc.Variable
    ) -> None:
        """Write data to buffer slice in-place."""
        size = end - start
        if data.sizes[self._concat_dim] != size:
            raise ValueError(
                f"Size mismatch: expected {size}, got {data.sizes[self._concat_dim]}"
            )
        buffer.values[start:end] = data.values

    def shift(
        self, buffer: sc.Variable, src_start: int, src_end: int, dst_start: int
    ) -> None:
        """Shift buffer data in-place."""
        size = src_end - src_start
        dst_end = dst_start + size
        buffer.values[dst_start:dst_end] = buffer.values[src_start:src_end]

    def get_view(self, buffer: sc.Variable, start: int, end: int) -> sc.Variable:
        """Get a copy of buffer slice."""
        return buffer[self._concat_dim, start:end].copy()

    def estimate_memory(self, buffer: sc.Variable) -> int:
        """Estimate memory usage in bytes."""
        return buffer.values.nbytes


class BufferStorage(StorageStrategy):
    """
    Unified buffer storage with configurable over-allocation.

    Generic implementation that works with any BufferInterface implementation.
    Handles growth, sliding window, and shift-on-overflow logic without
    knowing the details of the underlying buffer type.

    The overallocation_factor controls the memory/performance trade-off:
    - 2.0x: 100% overhead, 2x write amplification
    - 2.5x: 150% overhead, 1.67x write amplification (recommended)
    - 3.0x: 200% overhead, 1.5x write amplification
    """

    def __init__(
        self,
        max_size: int,
        buffer_impl: BufferInterface,
        initial_capacity: int = 100,
        overallocation_factor: float = 2.5,
    ) -> None:
        """
        Initialize unified buffer storage.

        Parameters
        ----------
        max_size:
            Maximum number of data points to maintain (sliding window size).
        buffer_impl:
            Buffer implementation (e.g., VariableBuffer, DataArrayBuffer).
        initial_capacity:
            Initial buffer allocation.
        overallocation_factor:
            Buffer capacity = max_size * overallocation_factor.
            Must be > 1.0.

        Raises
        ------
        ValueError:
            If parameters are invalid.
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive")
        if overallocation_factor <= 1.0:
            raise ValueError("overallocation_factor must be at least 1.0")

        self._max_size = max_size
        self._buffer_impl = buffer_impl
        self._initial_capacity = initial_capacity
        self._overallocation_factor = overallocation_factor
        self._max_capacity = int(max_size * overallocation_factor)

        self._buffer = None
        self._end = 0
        self._capacity = 0

    def _ensure_capacity(self, data) -> None:
        """Ensure buffer has capacity for new data."""
        # Get size from the data (works for both Variable and DataArray)
        if hasattr(data, 'sizes'):
            # DataArray
            new_size = next(iter(data.sizes.values()))
        else:
            # Variable
            new_size = data.shape[0]

        if self._buffer is None:
            # Initial allocation
            capacity = max(self._initial_capacity, new_size)
            self._buffer = self._buffer_impl.allocate(data, capacity)
            self._capacity = capacity
            self._end = 0
            return

        # Check if we need more capacity
        if self._end + new_size > self._capacity:
            # Try doubling, but cap at max_capacity
            new_capacity = min(self._capacity * 2, self._max_capacity)

            # If we've hit max_capacity and still need room, shift first
            if (
                new_capacity == self._max_capacity
                and self._end + new_size > new_capacity
            ):
                self._shift_to_sliding_window()

            # Grow buffer if still needed and haven't hit max_capacity
            if self._end + new_size > self._capacity < self._max_capacity:
                self._grow_buffer(data, new_capacity)

    def _grow_buffer(self, template, new_capacity: int) -> None:
        """Grow buffer by allocating larger buffer and copying data."""
        if self._buffer is None:
            raise RuntimeError("Cannot grow buffer before initialization")

        # Allocate new larger buffer
        new_buffer = self._buffer_impl.allocate(template, new_capacity)

        # Copy existing data
        self._buffer_impl.write_slice(
            new_buffer,
            0,
            self._end,
            self._buffer_impl.get_view(self._buffer, 0, self._end),
        )

        self._buffer = new_buffer
        self._capacity = new_capacity

    def _shift_to_sliding_window(self) -> None:
        """Shift buffer to maintain sliding window of max_size elements."""
        if self._buffer is None or self._end <= self._max_size:
            return

        # Shift last max_size elements to front
        shift_start = self._end - self._max_size
        self._buffer_impl.shift(
            self._buffer, src_start=shift_start, src_end=self._end, dst_start=0
        )
        self._end = self._max_size

    def append(self, data) -> None:
        """Append new data to storage."""
        self._ensure_capacity(data)
        if self._buffer is None:
            raise RuntimeError("Buffer initialization failed")

        if hasattr(data, 'sizes'):
            new_size = next(iter(data.sizes.values()))
        else:
            new_size = data.shape[0]

        start = self._end
        end = self._end + new_size

        # Write data using buffer implementation
        self._buffer_impl.write_slice(self._buffer, start, end, data)
        self._end = end

        # Only trim if we've hit max_capacity AND exceed max_size
        # During growth phase, keep all data
        if self._capacity >= self._max_capacity and self._end > self._max_size:
            self._shift_to_sliding_window()

    def get_all(self):
        """Get all stored data."""
        if self._buffer is None:
            return None
        return self._buffer_impl.get_view(self._buffer, 0, self._end)

    def estimate_memory(self) -> int:
        """Estimate memory usage in bytes."""
        if self._buffer is None:
            return 0
        return self._buffer_impl.estimate_memory(self._buffer)

    def clear(self) -> None:
        """Clear all stored data."""
        self._buffer = None
        self._end = 0
        self._capacity = 0
