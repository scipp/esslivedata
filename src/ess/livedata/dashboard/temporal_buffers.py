# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Temporal buffer implementations for BufferManager."""

from __future__ import annotations

import math
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

    When the buffer reaches capacity, it behaves as a ring buffer: oldest data is
    dropped to make room for new data.
    """

    # 0-D coord names that are still treated as per-sample on a single-slice
    # message (where ``time`` dim is absent, so the dim signal can't be used).
    _SCALAR_ACCUMULATED_NAMES = ('time', 'start_time', 'end_time')

    DEFAULT_MAX_MEMORY = 20 * 1024 * 1024  # 20 MB

    def __init__(self) -> None:
        """Initialize empty temporal buffer."""
        self._data_buffer: VariableBuffer | None = None
        self._coord_buffers: dict[str, VariableBuffer] = {}
        self._reference: sc.DataArray | None = None
        self._max_memory: int = self.DEFAULT_MAX_MEMORY
        self._required_timespan: float = 0.0

    def _accumulated_coord_names(self, data: sc.DataArray) -> list[str]:
        """Return coords that vary per time step and must be buffered.

        On a thick slice (``time`` in ``data.dims``) any coord carrying the
        ``time`` dim is per-sample. On a single-slice message all coords are
        0-D, so we fall back to a name convention for the known time-like
        annotations.
        """
        if 'time' in data.dims:
            return [name for name, coord in data.coords.items() if 'time' in coord.dims]
        return [name for name in self._SCALAR_ACCUMULATED_NAMES if name in data.coords]

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
                # Timespan trimming wasn't enough (e.g., required_timespan=inf).
                # Fall back to ring-buffer behavior: drop oldest to make room.
                self._make_room(data)
                if not self._data_buffer.append(data.data):
                    raise ValueError("Data exceeds buffer capacity even after trimming")

        for name, buf in self._coord_buffers.items():
            if not buf.append(data.coords[name]):
                raise RuntimeError(f"Coord buffer {name!r} append failed unexpectedly")

    def get(self) -> sc.DataArray | None:
        """Return the complete buffer."""
        if self._data_buffer is None:
            return None

        coords: dict[str, sc.Variable] = {
            name: buf.get() for name, buf in self._coord_buffers.items()
        }
        coords.update(self._reference.coords)
        masks = dict(self._reference.masks)

        return sc.DataArray(data=self._data_buffer.get(), coords=coords, masks=masks)

    def clear(self) -> None:
        """Clear all buffered data."""
        self._data_buffer = None
        self._coord_buffers = {}
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
        accumulated = self._accumulated_coord_names(data)

        ref = data['time', 0] if 'time' in data.dims else data
        ref = ref.drop_coords([name for name in accumulated if name in ref.coords])
        self._reference = ref

        # Calculate max_capacity from memory limit, accounting for the data buffer
        # plus every accumulated coord buffer.
        bytes_per_element = data.data.values.nbytes
        for name in accumulated:
            bytes_per_element += data.coords[name].values.nbytes
        if 'time' in data.dims:
            bytes_per_element /= data.sizes['time']

        max_capacity = max(1, int(self._max_memory / bytes_per_element))

        self._data_buffer = VariableBuffer(
            data=data.data, max_capacity=max_capacity, concat_dim='time'
        )
        self._coord_buffers = {
            name: VariableBuffer(
                data=data.coords[name],
                max_capacity=max_capacity,
                concat_dim='time',
            )
            for name in accumulated
        }

    def _trim_to_timespan(self, new_data: sc.DataArray) -> None:
        """Trim buffer to keep only data within required timespan."""
        if self._required_timespan < 0 or math.isinf(self._required_timespan):
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
        # datetime64 arithmetic requires int64, not float64
        if latest_time.dtype == sc.DType.datetime64:
            timespan = timespan.astype('int64')
        cutoff = latest_time - timespan

        # Find first index to keep
        time_coord = self._coord_buffers['time'].get()
        keep_mask = time_coord >= cutoff

        if not keep_mask.values.any():
            # All data is old, drop everything
            drop_count = self._data_buffer.size
        else:
            # Find first True index
            drop_count = int(keep_mask.values.argmax())

        # Trim all buffers by same amount to keep them in sync
        self._drop_from_all_buffers(drop_count)

    def _make_room(self, new_data: sc.DataArray) -> None:
        """Drop oldest elements to fit new data.

        Drops at least 10% of capacity to amortize the O(n) copy cost of
        VariableBuffer.drop(), avoiding per-update copies in steady state.
        """
        if 'time' in new_data.dims:
            n_incoming = new_data.sizes['time']
        else:
            n_incoming = 1
        available = self._data_buffer.max_capacity - self._data_buffer.size
        min_drop = n_incoming - available
        if min_drop > 0:
            amortized_drop = max(min_drop, self._data_buffer.max_capacity // 10)
            self._drop_from_all_buffers(amortized_drop)

    def _drop_from_all_buffers(self, drop_count: int) -> None:
        """Drop data from all buffers to keep them in sync."""
        self._data_buffer.drop(drop_count)
        for buf in self._coord_buffers.values():
            buf.drop(drop_count)

    def _metadata_matches(self, data: sc.DataArray) -> bool:
        """Check if incoming data's metadata matches stored reference metadata."""
        new = data['time', 0] if 'time' in data.dims else data
        accumulated = self._accumulated_coord_names(data)

        # A shape change (e.g. the number of ROIs in an ROI-spectra output changes
        # while the job runs) means the buffered series can no longer be extended;
        # the buffer must reset. assign() below would raise on a shape mismatch, so
        # guard against it and report a mismatch instead.
        if new.sizes != self._reference.sizes:
            return False

        template = new.assign(self._reference.data)
        template = template.drop_coords(
            [name for name in accumulated if name in template.coords]
        )

        return sc.identical(self._reference, template)
