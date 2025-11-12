# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Temporal buffer implementations for BufferManager."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import scipp as sc

T = TypeVar('T')


class BufferProtocol(ABC, Generic[T]):
    """Common interface for all buffer types."""

    @abstractmethod
    def add(self, data: T) -> None:
        """
        Add new data to the buffer.

        Parameters
        ----------
        data:
            New data to add to the buffer.
        """

    @abstractmethod
    def get(self) -> T | None:
        """
        Retrieve current buffer contents.

        Returns
        -------
        :
            Current buffer contents, or None if empty.
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the buffer."""

    @abstractmethod
    def set_required_timespan(self, seconds: float) -> None:
        """
        Set the required timespan for the buffer.

        Parameters
        ----------
        seconds:
            Required timespan in seconds.
        """

    @abstractmethod
    def set_max_memory(self, max_bytes: int) -> None:
        """
        Set the maximum memory usage for the buffer.

        Parameters
        ----------
        max_bytes:
            Maximum memory usage in bytes (approximate).
        """


class SingleValueBuffer(BufferProtocol[T]):
    """
    Buffer that stores only the latest value.

    Used when only LatestValueExtractor is present for efficiency.
    """

    def __init__(self) -> None:
        """Initialize empty single value buffer."""
        self._data: T | None = None
        self._max_memory: int | None = None
        self._required_timespan: float = 0.0

    def add(self, data: T) -> None:
        """Store the latest value, replacing any previous value."""
        self._data = data

    def get(self) -> T | None:
        """Return the stored value."""
        return self._data

    def clear(self) -> None:
        """Clear the stored value."""
        self._data = None

    def set_required_timespan(self, seconds: float) -> None:
        """Set required timespan (unused for SingleValueBuffer)."""
        self._required_timespan = seconds

    def set_max_memory(self, max_bytes: int) -> None:
        """Set max memory (unused for SingleValueBuffer)."""
        self._max_memory = max_bytes


class VariableBuffer:
    """
    Buffer managing sc.Variable data with dynamic sizing along a concat dimension.

    Handles appending data slices, capacity management with lazy expansion,
    and dropping old data from the start.
    """

    def __init__(
        self,
        *,
        data: sc.Variable,
        max_capacity: int,
        concat_dim: str = 'time',
    ) -> None:
        """
        Initialize variable buffer with initial data.

        Parameters
        ----------
        data:
            Initial data to store. Defines buffer structure (dims, dtype, unit).
        max_capacity:
            Maximum allowed size along concat_dim.
        concat_dim:
            Dimension along which to concatenate data.
        """
        self._concat_dim = concat_dim
        self._max_capacity = max_capacity

        # Allocate minimal initial buffer
        initial_size = min(16, max_capacity)
        self._buffer = self._allocate_buffer(data, initial_size)
        self._size = 0

        # Delegate to append
        if not self.append(data):
            raise ValueError(f"Initial data exceeds max_capacity {max_capacity}")

    @property
    def size(self) -> int:
        """Get number of valid elements."""
        return self._size

    @property
    def capacity(self) -> int:
        """Get currently allocated buffer size."""
        return self._buffer.sizes[self._concat_dim]

    @property
    def max_capacity(self) -> int:
        """Get maximum capacity limit."""
        return self._max_capacity

    def append(self, data: sc.Variable) -> bool:
        """
        Append data to the buffer.

        Parameters
        ----------
        data:
            Data to append. May or may not have concat_dim dimension.
            If concat_dim is present, appends all slices along that dimension.
            Otherwise, treats as single slice.

        Returns
        -------
        :
            True if successful, False if would exceed max_capacity.
        """
        # Determine how many elements we're adding
        if self._concat_dim in data.dims:
            n_incoming = data.sizes[self._concat_dim]
        else:
            n_incoming = 1

        # Check max_capacity
        if self._size + n_incoming > self._max_capacity:
            return False

        # Expand to fit all incoming data
        if self._size + n_incoming > self.capacity:
            self._expand_to_fit(self._size + n_incoming)

        # Write data
        if self._concat_dim in data.dims:
            # Thick slice
            end = self._size + n_incoming
            self._buffer[self._concat_dim, self._size : end] = data
        else:
            # Single slice
            self._buffer[self._concat_dim, self._size] = data

        self._size += n_incoming
        return True

    def get(self) -> sc.Variable:
        """
        Get buffer contents up to current size.

        Returns
        -------
        :
            Valid buffer data (0:size).
        """
        return self._buffer[self._concat_dim, : self._size]

    def drop(self, index: int) -> None:
        """
        Drop data from start up to (but not including) index.

        Remaining valid data is moved to the start of the buffer.

        Parameters
        ----------
        index:
            Index from start until which to drop (exclusive).
        """
        if index <= 0:
            return

        if index >= self._size:
            # Dropping everything
            self._size = 0
            return

        # Move remaining data to start
        n_remaining = self._size - index
        self._buffer[self._concat_dim, :n_remaining] = self._buffer[
            self._concat_dim, index : self._size
        ]
        self._size = n_remaining

    def _allocate_buffer(self, template: sc.Variable, size: int) -> sc.Variable:
        """
        Allocate new buffer based on template variable structure.

        Makes concat_dim the outer (first) dimension for efficient contiguous writes.
        If template already has concat_dim, preserves its position.
        """
        if self._concat_dim in template.dims:
            # Template has concat_dim - preserve dimension order
            sizes = template.sizes
            sizes[self._concat_dim] = size
        else:
            # Template doesn't have concat_dim - make it the outer dimension
            sizes = {self._concat_dim: size}
            sizes.update(template.sizes)

        return sc.empty(sizes=sizes, dtype=template.dtype, unit=template.unit)

    def _expand_to_fit(self, min_size: int) -> None:
        """Expand buffer to accommodate at least min_size elements."""
        current_allocated = self._buffer.sizes[self._concat_dim]
        while current_allocated < min_size:
            current_allocated = min(self._max_capacity, current_allocated * 2)

        if current_allocated > self._buffer.sizes[self._concat_dim]:
            new_buffer = self._allocate_buffer(self._buffer, current_allocated)
            new_buffer[self._concat_dim, : self._size] = self._buffer[
                self._concat_dim, : self._size
            ]
            self._buffer = new_buffer


class TemporalBuffer(BufferProtocol[sc.DataArray]):
    """
    Buffer that maintains temporal data with a time dimension.

    Uses VariableBuffer for efficient appending without expensive concat operations.
    Validates that non-time coords and masks remain constant across all added data.
    """

    def __init__(self) -> None:
        """Initialize empty temporal buffer."""
        self._data_buffer: VariableBuffer | None = None
        self._time_buffer: VariableBuffer | None = None
        self._reference: sc.DataArray | None = None
        self._max_memory: int | None = None
        self._required_timespan: float = 0.0

    def add(self, data: sc.DataArray) -> None:
        """
        Add data to the buffer, appending along time dimension.

        Parameters
        ----------
        data:
            New data to add. Must have a 'time' coordinate.

        Raises
        ------
        ValueError
            If data does not have a 'time' coordinate.
        """
        if 'time' not in data.coords:
            raise ValueError("TemporalBuffer requires data with 'time' coordinate")

        # First data or metadata mismatch - initialize/reset buffers
        if self._data_buffer is None or not self._metadata_matches(data):
            self._initialize_buffers(data)
            return

        # Append to existing buffers
        self._data_buffer.append(data.data)
        self._time_buffer.append(data.coords['time'])

    def get(self) -> sc.DataArray | None:
        """Return the complete buffer."""
        if self._data_buffer is None:
            return None

        # Reconstruct DataArray from buffers and reference metadata
        data_var = self._data_buffer.get()
        time_coord = self._time_buffer.get()

        coords = {'time': time_coord}
        coords.update(self._reference.coords)

        masks = dict(self._reference.masks)

        return sc.DataArray(data=data_var, coords=coords, masks=masks)

    def clear(self) -> None:
        """Clear all buffered data."""
        self._data_buffer = None
        self._time_buffer = None
        self._reference = None

    def set_required_timespan(self, seconds: float) -> None:
        """Set the required timespan for the buffer."""
        self._required_timespan = seconds

    def set_max_memory(self, max_bytes: int) -> None:
        """Set the maximum memory usage for the buffer."""
        self._max_memory = max_bytes

    def _initialize_buffers(self, data: sc.DataArray) -> None:
        """Initialize buffers with first data, storing reference metadata."""
        # Store reference as slice at time=0 without time coord
        if 'time' in data.dims:
            self._reference = data['time', 0].drop_coords('time')
        else:
            self._reference = data.drop_coords('time')

        # Create buffers with large capacity
        max_capacity = 100
        self._data_buffer = VariableBuffer(
            data=data.data, max_capacity=max_capacity, concat_dim='time'
        )
        self._time_buffer = VariableBuffer(
            data=data.coords['time'], max_capacity=max_capacity, concat_dim='time'
        )

    def _metadata_matches(self, data: sc.DataArray) -> bool:
        """Check if incoming data's metadata matches stored reference metadata."""
        # Extract comparable slice from incoming data
        if 'time' in data.dims:
            new = data['time', 0]
        else:
            new = data

        # Create template with reference data but incoming metadata
        template = new.assign(self._reference.data).drop_coords('time')

        return sc.identical(self._reference, template)
