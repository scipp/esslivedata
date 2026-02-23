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
    def get_required_timespan(self) -> float:
        """
        Get the required timespan for the buffer.

        Returns
        -------
        :
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

    def get_required_timespan(self) -> float:
        """Get the required timespan."""
        return self._required_timespan

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

        return sc.empty(
            sizes=sizes,
            dtype=template.dtype,
            unit=template.unit,
            with_variances=template.variances is not None,
        )

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

    # Coordinates that should be accumulated per-item rather than stored as scalars
    _ACCUMULATED_COORDS = ('time', 'start_time', 'end_time')

    def __init__(self) -> None:
        """Initialize empty temporal buffer."""
        self._data_buffer: VariableBuffer | None = None
        self._time_buffer: VariableBuffer | None = None
        self._start_time_buffer: VariableBuffer | None = None
        self._end_time_buffer: VariableBuffer | None = None
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
            If data does not have a 'time' coordinate or exceeds buffer capacity.
        """
        if 'time' not in data.coords:
            raise ValueError("TemporalBuffer requires data with 'time' coordinate")

        # First data or metadata mismatch - initialize/reset buffers
        if self._data_buffer is None or not self._metadata_matches(data):
            self._initialize_buffers(data)
            return

        # Try to append to existing buffers
        if not self._data_buffer.append(data.data):
            # Failed - trim old data and retry
            self._trim_to_timespan(data)
            if not self._data_buffer.append(data.data):
                raise ValueError("Data exceeds buffer capacity even after trimming")

        # Time buffer should succeed (buffers kept in sync by trimming)
        if not self._time_buffer.append(data.coords['time']):
            raise RuntimeError("Time buffer append failed unexpectedly")

        # Append start_time and end_time if present
        if self._start_time_buffer is not None and 'start_time' in data.coords:
            self._start_time_buffer.append(data.coords['start_time'])
        if self._end_time_buffer is not None and 'end_time' in data.coords:
            self._end_time_buffer.append(data.coords['end_time'])

    def get(self) -> sc.DataArray | None:
        """Return the complete buffer."""
        if self._data_buffer is None:
            return None

        # Reconstruct DataArray from buffers and reference metadata
        data_var = self._data_buffer.get()
        time_coord = self._time_buffer.get()

        coords = {'time': time_coord}
        coords.update(self._reference.coords)

        # Add accumulated start_time and end_time as 1-D coords
        if self._start_time_buffer is not None:
            coords['start_time'] = self._start_time_buffer.get()
        if self._end_time_buffer is not None:
            coords['end_time'] = self._end_time_buffer.get()

        masks = dict(self._reference.masks)

        return sc.DataArray(data=data_var, coords=coords, masks=masks)

    def clear(self) -> None:
        """Clear all buffered data."""
        self._data_buffer = None
        self._time_buffer = None
        self._start_time_buffer = None
        self._end_time_buffer = None
        self._reference = None

    def set_required_timespan(self, seconds: float) -> None:
        """Set the required timespan for the buffer."""
        self._required_timespan = seconds

    def get_required_timespan(self) -> float:
        """Get the required timespan for the buffer."""
        return self._required_timespan

    def set_max_memory(self, max_bytes: int) -> None:
        """Set the maximum memory usage for the buffer."""
        self._max_memory = max_bytes

    def _initialize_buffers(self, data: sc.DataArray) -> None:
        """Initialize buffers with first data, storing reference metadata."""
        # Store reference as slice at time=0 without accumulated coords
        if 'time' in data.dims:
            ref = data['time', 0]
        else:
            ref = data
        # Drop all coords that will be accumulated (not stored in reference)
        ref = ref.drop_coords(
            [coord for coord in self._ACCUMULATED_COORDS if coord in ref.coords]
        )
        self._reference = ref

        # Calculate max_capacity from memory limit
        if 'time' in data.dims:
            bytes_per_element = data.data.values.nbytes / data.sizes['time']
        else:
            bytes_per_element = data.data.values.nbytes

        if self._max_memory is not None:
            max_capacity = int(self._max_memory / bytes_per_element)
        else:
            max_capacity = 10000  # Default large capacity

        # Create buffers
        self._data_buffer = VariableBuffer(
            data=data.data, max_capacity=max_capacity, concat_dim='time'
        )
        self._time_buffer = VariableBuffer(
            data=data.coords['time'], max_capacity=max_capacity, concat_dim='time'
        )

        # Create buffers for start_time and end_time if present
        if 'start_time' in data.coords:
            self._start_time_buffer = VariableBuffer(
                data=data.coords['start_time'],
                max_capacity=max_capacity,
                concat_dim='time',
            )
        else:
            self._start_time_buffer = None

        if 'end_time' in data.coords:
            self._end_time_buffer = VariableBuffer(
                data=data.coords['end_time'],
                max_capacity=max_capacity,
                concat_dim='time',
            )
        else:
            self._end_time_buffer = None

    def _trim_to_timespan(self, new_data: sc.DataArray) -> None:
        """Trim buffer to keep only data within required timespan."""
        if self._required_timespan < 0:
            return

        if self._required_timespan == 0.0:
            # Keep only the latest value - drop all existing data
            drop_count = self._data_buffer.size
            self._drop_from_all_buffers(drop_count)
            return

        # Get latest time from new data
        if 'time' in new_data.dims:
            latest_time = new_data.coords['time'][-1]
        else:
            latest_time = new_data.coords['time']

        # Calculate cutoff time (convert timespan to match the time coordinate unit)
        timespan = sc.to_unit(
            sc.scalar(self._required_timespan, unit='s'), latest_time.unit
        )
        cutoff = latest_time - timespan

        # Find first index to keep
        time_coord = self._time_buffer.get()
        keep_mask = time_coord >= cutoff

        if not keep_mask.values.any():
            # All data is old, drop everything
            drop_count = self._data_buffer.size
        else:
            # Find first True index
            drop_count = int(keep_mask.values.argmax())

        # Trim all buffers by same amount to keep them in sync
        self._drop_from_all_buffers(drop_count)

    def _drop_from_all_buffers(self, drop_count: int) -> None:
        """Drop data from all buffers to keep them in sync."""
        self._data_buffer.drop(drop_count)
        self._time_buffer.drop(drop_count)
        if self._start_time_buffer is not None:
            self._start_time_buffer.drop(drop_count)
        if self._end_time_buffer is not None:
            self._end_time_buffer.drop(drop_count)

    def _metadata_matches(self, data: sc.DataArray) -> bool:
        """Check if incoming data's metadata matches stored reference metadata."""
        # Extract comparable slice from incoming data
        if 'time' in data.dims:
            new = data['time', 0]
        else:
            new = data

        # Create template with reference data but incoming metadata
        # Drop all accumulated coords for comparison
        template = new.assign(self._reference.data)
        template = template.drop_coords(
            [coord for coord in self._ACCUMULATED_COORDS if coord in template.coords]
        )

        return sc.identical(self._reference, template)
