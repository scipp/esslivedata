# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Low-level storage strategies for buffer data management."""

from __future__ import annotations

import logging
from typing import Any, Generic, Protocol, TypeVar

import scipp as sc

logger = logging.getLogger(__name__)

# Type variable for buffer types
T = TypeVar('T')


class ScippLike(Protocol):
    """Protocol for objects with scipp-like interface (dims, sizes attributes)."""

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension names."""
        ...

    @property
    def sizes(self) -> dict[str, int]:
        """Dimension sizes."""
        ...

    def __getitem__(self, key: Any) -> Any:
        """Index into data."""
        ...


# Type variable constrained to scipp-like objects
ScippT = TypeVar('ScippT', bound=ScippLike)


class BufferInterface(Protocol[T]):
    """
    Protocol for buffer implementations.

    Defines the minimal interface needed by Buffer. Implementations
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

    def write_slice(self, buffer: T, start: int, data: T) -> None:
        """
        Write data to a buffer slice in-place.

        Parameters
        ----------
        buffer:
            Pre-allocated buffer to write into.
        start:
            Start index along concat dimension.
        data:
            Data to write. Will be written starting at start with size determined
            by get_size(data).
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
        Get a view of a buffer slice.

        The returned view shares memory with the buffer and may be invalidated
        by subsequent buffer operations (growth, shifting). Callers must use
        the view immediately or copy it if needed for later use. Modifications
        to the view will affect the underlying buffer.

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
            View of the buffer slice. Valid only until next buffer operation.
        """
        ...

    def get_size(self, data: T) -> int:
        """
        Get size of data along the relevant dimension.

        Parameters
        ----------
        data:
            Data to measure.

        Returns
        -------
        :
            Size along the relevant dimension.
        """
        ...

    def extract_latest_frame(self, data: T) -> T:
        """
        Extract the latest frame from incoming data, removing concat dimension.

        Handles batched data by taking the last frame along concat_dim.
        If data doesn't have concat_dim, returns as-is.

        Parameters
        ----------
        data:
            Incoming data that may contain multiple frames.

        Returns
        -------
        :
            Single frame without concat dimension.
        """
        ...

    def unwrap_window(self, view: T) -> T:
        """
        Unwrap a size-1 buffer view to a scalar value.

        The view is guaranteed to have exactly 1 element along concat_dim.
        This method removes that dimension to return the underlying data.

        Parameters
        ----------
        view:
            A buffer view with exactly 1 element along concat_dim.

        Returns
        -------
        :
            The unwrapped data without the concat dimension.
        """
        ...

    def get_window_by_duration(self, buffer: T, end: int, duration_seconds: float) -> T:
        """
        Get a window covering approximately the specified time duration.

        Parameters
        ----------
        buffer:
            Buffer to extract from.
        end:
            End index of valid data in buffer (exclusive).
        duration_seconds:
            Approximate time duration in seconds.

        Returns
        -------
        :
            Window of data covering approximately the duration.
        """
        ...


class ScippBuffer(Generic[ScippT]):
    """
    Base class for scipp-based buffer implementations (DataArray, Variable).

    Provides common methods for dimension-based buffers with shared concat
    dimension logic.
    """

    def __init__(self, concat_dim: str = 'time') -> None:
        """
        Initialize scipp buffer implementation.

        Parameters
        ----------
        concat_dim:
            The dimension along which to concatenate data.
        """
        self._concat_dim = concat_dim

    def get_size(self, data: ScippT) -> int:
        """Get size along concatenation dimension."""
        if self._concat_dim not in data.dims:
            # Data doesn't have concat dim - treat as single frame
            return 1
        return data.sizes[self._concat_dim]

    def get_view(self, buffer: ScippT, start: int, end: int) -> ScippT:
        """Get a view of buffer slice."""
        return buffer[self._concat_dim, start:end]

    def extract_latest_frame(self, data: ScippT) -> ScippT:
        """Extract the latest frame from incoming data, removing concat dimension."""
        if self._concat_dim not in data.dims:
            # Data doesn't have concat dim - already a single frame
            return data

        # Extract last frame along concat dimension
        return data[self._concat_dim, -1]

    def unwrap_window(self, view: ScippT) -> ScippT:
        """Unwrap a size-1 buffer view to a scalar value."""
        if self._concat_dim not in view.dims:
            # View doesn't have concat dim - already unwrapped
            return view

        # Extract the single element along concat dimension
        return view[self._concat_dim, 0]

    def get_window_by_duration(
        self, buffer: ScippT, end: int, duration_seconds: float
    ) -> ScippT:
        """
        Get window by time duration (naive implementation).

        Assumes nominal 14 Hz frame rate (ESS).
        """
        # Naive conversion: duration → frame count at 14 Hz
        frame_count = max(1, int(duration_seconds * 14.0))
        start = max(0, end - frame_count)
        return self.get_view(buffer, start, end)


class DataArrayBuffer(ScippBuffer[sc.DataArray], BufferInterface[sc.DataArray]):  # type: ignore[type-arg]
    """
    Buffer implementation for sc.DataArray.

    Handles DataArray complexity including:
    - Data variable allocation
    - Concat dimension coordinates (lazy-allocated when first slice provides them)
    - Non-concat coordinates (preserved from input data)
    - Concat-dependent coordinates (pre-allocated from template)
    - Masks
    """

    def __init__(self, concat_dim: str = 'time') -> None:
        """
        Initialize DataArray buffer implementation.

        Parameters
        ----------
        concat_dim:
            The dimension along which to concatenate data.
        """
        super().__init__(concat_dim)

    def allocate(self, template: sc.DataArray, capacity: int) -> sc.DataArray:
        """Allocate a new DataArray buffer with given capacity."""
        # Determine shape with expanded concat dimension
        if self._concat_dim in template.dims:
            shape = [
                capacity if dim == self._concat_dim else size
                for dim, size in zip(template.dims, template.shape, strict=True)
            ]
            dims = template.dims
        else:
            # Data doesn't have concat dim - add it as first dimension
            dims = (self._concat_dim, *template.dims)
            shape = [capacity, *list(template.shape)]

        # Create zeros array with correct structure
        data_var = sc.zeros(dims=dims, shape=shape, dtype=template.data.dtype)

        # Add non-concat coordinates from template
        # Only add those that don't depend on the concat dimension
        coords = {
            coord_name: coord
            for coord_name, coord in template.coords.items()
            if (coord_name != self._concat_dim and self._concat_dim not in coord.dims)
        }

        buffer_data = sc.DataArray(data=data_var, coords=coords)

        # Pre-allocate coordinates that depend on concat dimension
        for coord_name, coord in template.coords.items():
            if coord_name != self._concat_dim and self._concat_dim in coord.dims:
                # Determine the shape for the coord in the buffer
                if self._concat_dim in template.dims:
                    coord_shape = [
                        capacity if dim == self._concat_dim else template.sizes[dim]
                        for dim in coord.dims
                    ]
                else:
                    # Template didn't have concat dim, coord shouldn't either
                    # Add concat dim to coord
                    coord_shape = [
                        capacity if dim == self._concat_dim else coord.sizes[dim]
                        for dim in coord.dims
                    ]
                buffer_data.coords[coord_name] = sc.zeros(
                    dims=coord.dims,
                    shape=coord_shape,
                    dtype=coord.dtype,
                )

        # Pre-allocate masks
        for mask_name, mask in template.masks.items():
            if self._concat_dim in template.dims:
                mask_shape = [
                    capacity if dim == self._concat_dim else s
                    for dim, s in zip(mask.dims, mask.shape, strict=True)
                ]
                mask_dims = mask.dims
            else:
                # Template didn't have concat dim - add it to mask
                mask_dims = (self._concat_dim, *mask.dims)
                mask_shape = [capacity, *list(mask.shape)]
            buffer_data.masks[mask_name] = sc.zeros(
                dims=mask_dims,
                shape=mask_shape,
                dtype=mask.dtype,
            )

        return buffer_data

    def write_slice(self, buffer: sc.DataArray, start: int, data: sc.DataArray) -> None:
        """Write data to buffer slice in-place."""
        size = self.get_size(data)
        end = start + size

        # Write data using slice notation - works for both cases via broadcasting:
        # - Data with concat_dim: direct assignment
        # - Data without concat_dim: numpy broadcasts to (1, *other_dims)
        # Special case: strings require element-by-element assignment
        if data.data.dtype == sc.DType.string:
            buffer_slice = buffer[self._concat_dim, start:end]
            data_flat = list(data.data.values)
            buffer_flat = buffer_slice.data.values
            for i, val in enumerate(data_flat):
                buffer_flat[i] = val
        else:
            buffer.data.values[start:end] = data.data.values

        # Handle concat dimension coordinate
        if self._concat_dim in data.coords:
            # Data has concat coord - add it to buffer
            if self._concat_dim not in buffer.coords:
                # Need to allocate the coordinate in the buffer first
                buffer.coords[self._concat_dim] = sc.zeros(
                    dims=[self._concat_dim],
                    shape=[buffer.sizes[self._concat_dim]],
                    dtype=data.coords[self._concat_dim].dtype,
                )
            # Copy the coordinate values
            buffer.coords[self._concat_dim].values[start:end] = data.coords[
                self._concat_dim
            ].values

        # Copy concat-dependent coords (only if data has concat_dim)
        for coord_name, coord in data.coords.items():
            if coord_name != self._concat_dim and self._concat_dim in coord.dims:
                buffer.coords[coord_name].values[start:end] = coord.values

        # Copy masks - broadcasting handles concat_dim presence/absence
        for mask_name, mask in data.masks.items():
            buffer.masks[mask_name].values[start:end] = mask.values

    def shift(
        self, buffer: sc.DataArray, src_start: int, src_end: int, dst_start: int
    ) -> None:
        """Shift buffer data in-place."""
        size = src_end - src_start
        dst_end = dst_start + size

        # Shift data
        buffer.data.values[dst_start:dst_end] = buffer.data.values[src_start:src_end]

        # Shift concat dimension coordinate if it exists
        if self._concat_dim in buffer.coords:
            buffer.coords[self._concat_dim].values[dst_start:dst_end] = buffer.coords[
                self._concat_dim
            ].values[src_start:src_end]

        # Shift concat-dependent coords
        for coord_name, coord in buffer.coords.items():
            if coord_name != self._concat_dim and self._concat_dim in coord.dims:
                coord.values[dst_start:dst_end] = coord.values[src_start:src_end]

        # Shift masks
        for mask in buffer.masks.values():
            if self._concat_dim in mask.dims:
                mask.values[dst_start:dst_end] = mask.values[src_start:src_end]


class VariableBuffer(ScippBuffer[sc.Variable], BufferInterface[sc.Variable]):  # type: ignore[type-arg]
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
        super().__init__(concat_dim)

    def allocate(self, template: sc.Variable, capacity: int) -> sc.Variable:
        """Allocate a new Variable buffer with given capacity."""
        if self._concat_dim in template.dims:
            shape = [
                capacity if dim == self._concat_dim else size
                for dim, size in zip(template.dims, template.shape, strict=True)
            ]
            dims = template.dims
        else:
            # Data doesn't have concat dim - add it as first dimension
            dims = (self._concat_dim, *template.dims)
            shape = [capacity, *list(template.shape)]
        return sc.zeros(dims=dims, shape=shape, dtype=template.dtype)

    def write_slice(self, buffer: sc.Variable, start: int, data: sc.Variable) -> None:
        """Write data to buffer slice in-place."""
        size = self.get_size(data)
        end = start + size

        # Use slice notation consistently - numpy broadcasts when needed
        # This works for both:
        # - Data with concat_dim: direct assignment
        # - Data without concat_dim: numpy broadcasts to (1, *other_dims)
        buffer.values[start:end] = data.values

    def shift(
        self, buffer: sc.Variable, src_start: int, src_end: int, dst_start: int
    ) -> None:
        """Shift buffer data in-place."""
        size = src_end - src_start
        dst_end = dst_start + size
        buffer.values[dst_start:dst_end] = buffer.values[src_start:src_end]


class ListBuffer(BufferInterface[list]):
    """Simple list-based buffer for non-scipp types."""

    def __init__(self, concat_dim: str = 'time') -> None:
        """
        Initialize list buffer implementation.

        Parameters
        ----------
        concat_dim:
            Ignored for ListBuffer (kept for interface compatibility).
        """
        self._concat_dim = concat_dim

    def allocate(self, template: Any, capacity: int) -> list:
        """Allocate empty list."""
        return []

    def write_slice(self, buffer: list, start: int, data: Any) -> None:
        """Append data to list."""
        if isinstance(data, list):
            buffer.extend(data)
        else:
            buffer.append(data)

    def shift(self, buffer: list, src_start: int, src_end: int, dst_start: int) -> None:
        """Shift list elements."""
        size = src_end - src_start
        dst_end = dst_start + size
        buffer[dst_start:dst_end] = buffer[src_start:src_end]

    def get_view(self, buffer: list, start: int, end: int) -> list:
        """Get slice of list."""
        return buffer[start:end]

    def get_size(self, data: Any) -> int:
        """Get size of data."""
        if isinstance(data, list):
            return len(data)
        return 1

    def extract_latest_frame(self, data: Any) -> Any:
        """Extract the latest frame from incoming data."""
        if isinstance(data, list) and len(data) > 0:
            return data[-1]
        return data

    def unwrap_window(self, view: list) -> Any:
        """Unwrap a size-1 buffer view to a scalar value."""
        if isinstance(view, list) and len(view) > 0:
            return view[0]
        return view

    def get_window_by_duration(
        self, buffer: list, end: int, duration_seconds: float
    ) -> list:
        """
        Get window by time duration (naive implementation).

        Assumes nominal 14 Hz frame rate for list-based buffers.
        """
        frame_count = max(1, int(duration_seconds * 14.0))
        start = max(0, end - frame_count)
        return buffer[start:end]


class SingleValueStorage(Generic[T]):
    """
    Storage for single values with automatic replacement.

    Optimized storage for when only the latest value is needed (max_size=1).
    Uses simple value replacement instead of complex buffer management.
    """

    def __init__(self, buffer_impl: BufferInterface[T]) -> None:
        """
        Initialize single-value storage.

        Parameters
        ----------
        buffer_impl:
            Buffer implementation for extracting latest frame from incoming data.
        """
        self._buffer_impl = buffer_impl
        self._value: T | None = None

    def append(self, data: T) -> None:
        """Replace stored value with latest frame from incoming data."""
        self._value = self._buffer_impl.extract_latest_frame(data)

    def get_all(self) -> T | None:
        """Get the stored value."""
        return self._value

    def get_window(self, size: int | None = None) -> T | None:
        """Get the stored value (size parameter ignored)."""
        return self._value

    def get_latest(self) -> T | None:
        """Get the stored value."""
        return self._value

    def get_window_by_duration(self, duration_seconds: float) -> T | None:
        """Get the stored value (duration parameter ignored)."""
        return self._value

    def clear(self) -> None:
        """Clear the stored value."""
        self._value = None


class StreamingBuffer(Generic[T]):
    """
    Buffer with automatic growth and sliding window management.

    Handles complex buffer management including growth, shifting, and
    windowing logic for max_size > 1.

    Uses pre-allocated buffers with in-place writes to avoid O(n²) complexity
    of naive concatenation. Pre-allocates with doubling capacity and uses
    numpy-level indexing for O(1) appends, achieving O(n·m) amortized complexity.

    The overallocation_factor controls the memory/performance trade-off:
    - 2.0x: 100% overhead, 2x write amplification
    - 2.5x: 150% overhead, 1.67x write amplification (recommended)
    - 3.0x: 200% overhead, 1.5x write amplification
    """

    def __init__(
        self,
        max_size: int,
        buffer_impl: BufferInterface[T],
        initial_capacity: int = 100,
        overallocation_factor: float = 2.5,
    ) -> None:
        """
        Initialize streaming buffer.

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

    def set_max_size(self, new_max_size: int) -> None:
        """
        Update the maximum buffer size (can only grow, never shrink).

        Parameters
        ----------
        new_max_size:
            New maximum size. If smaller than current max_size, no change is made.
        """
        if new_max_size > self._max_size:
            self._max_size = new_max_size
            self._max_capacity = int(new_max_size * self._overallocation_factor)

    def _ensure_capacity(self, data: T) -> None:
        """Ensure buffer has capacity for new data."""
        new_size = self._buffer_impl.get_size(data)

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

    def _grow_buffer(self, template: T, new_capacity: int) -> None:
        """Grow buffer by allocating larger buffer and copying data."""
        if self._buffer is None:
            raise RuntimeError("Cannot grow buffer before initialization")

        # Allocate new larger buffer
        new_buffer = self._buffer_impl.allocate(template, new_capacity)

        # Copy existing data
        self._buffer_impl.write_slice(
            new_buffer,
            0,
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

    def append(self, data: T) -> None:
        """Append new data to storage."""
        try:
            self._ensure_capacity(data)
            if self._buffer is None:
                raise RuntimeError("Buffer initialization failed")

            new_size = self._buffer_impl.get_size(data)
            start = self._end

            # Write data using buffer implementation
            self._buffer_impl.write_slice(self._buffer, start, data)
            self._end = start + new_size

            # Only trim if we've hit max_capacity AND exceed max_size
            # During growth phase, keep all data
            if self._capacity >= self._max_capacity and self._end > self._max_size:
                self._shift_to_sliding_window()
        except Exception as e:
            # Data is incompatible with existing buffer (shape/dims changed).
            # Clear and reallocate with new structure.
            logger.info(
                "Data structure changed, clearing buffer and reallocating: %s",
                e,
            )
            self.clear()
            # Retry append - will allocate new buffer with correct structure
            self._ensure_capacity(data)
            if self._buffer is None:
                raise RuntimeError("Buffer initialization failed") from e
            new_size = self._buffer_impl.get_size(data)
            self._buffer_impl.write_slice(self._buffer, 0, data)
            self._end = new_size

    def get_all(self) -> T | None:
        """Get all stored data."""
        if self._buffer is None:
            return None
        return self._buffer_impl.get_view(self._buffer, 0, self._end)

    def clear(self) -> None:
        """Clear all stored data."""
        self._buffer = None
        self._end = 0
        self._capacity = 0

    def get_window(self, size: int | None = None) -> T | None:
        """
        Get a window of buffered data from the end.

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
        if self._buffer is None:
            return None
        if size is None:
            return self._buffer_impl.get_view(self._buffer, 0, self._end)

        # Get window from the end
        actual_size = min(size, self._end)
        start = self._end - actual_size
        return self._buffer_impl.get_view(self._buffer, start, self._end)

    def get_latest(self) -> T | None:
        """
        Get the latest single value, unwrapped.

        Returns the most recent data point without the concat dimension,
        ready for use without further processing.

        Returns
        -------
        :
            The latest value without concat dimension, or None if empty.
        """
        if self._buffer is None or self._end == 0:
            return None

        # Get last frame as a size-1 window, then unwrap it
        view = self._buffer_impl.get_view(self._buffer, self._end - 1, self._end)
        return self._buffer_impl.unwrap_window(view)

    def get_window_by_duration(self, duration_seconds: float) -> T | None:
        """
        Get window by time duration.

        Parameters
        ----------
        duration_seconds:
            Approximate time duration in seconds.

        Returns
        -------
        :
            Window of data covering approximately the duration, or None if empty.
        """
        if self._buffer is None:
            return None
        return self._buffer_impl.get_window_by_duration(
            self._buffer, self._end, duration_seconds
        )


class Buffer(Generic[T]):
    """
    Unified buffer interface with automatic mode selection.

    Delegates to SingleValueStorage for max_size=1 (optimized single-value mode)
    or StreamingBuffer for max_size>1 (complex buffer management with growth
    and sliding window).

    Handles transparent transition from single-value to streaming mode when
    max_size is increased via set_max_size().
    """

    def __init__(
        self,
        max_size: int,
        buffer_impl: BufferInterface[T],
        initial_capacity: int = 100,
        overallocation_factor: float = 2.5,
    ) -> None:
        """
        Initialize buffer.

        Parameters
        ----------
        max_size:
            Maximum number of data points to maintain (sliding window size).
        buffer_impl:
            Buffer implementation (e.g., VariableBuffer, DataArrayBuffer).
        initial_capacity:
            Initial buffer allocation (ignored for max_size=1).
        overallocation_factor:
            Buffer capacity = max_size * overallocation_factor (ignored for max_size=1).
            Must be > 1.0.
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self._max_size = max_size
        self._buffer_impl = buffer_impl
        self._initial_capacity = initial_capacity
        self._overallocation_factor = overallocation_factor

        # Create appropriate storage based on max_size
        self._storage = self._create_storage(max_size)

    def _create_storage(
        self, max_size: int
    ) -> SingleValueStorage[T] | StreamingBuffer[T]:
        """
        Create appropriate storage implementation based on max_size.

        Parameters
        ----------
        max_size:
            Maximum number of data points to maintain.

        Returns
        -------
        :
            SingleValueStorage for max_size=1, StreamingBuffer otherwise.
        """
        if max_size == 1:
            return SingleValueStorage(self._buffer_impl)
        else:
            return StreamingBuffer(
                max_size=max_size,
                buffer_impl=self._buffer_impl,
                initial_capacity=self._initial_capacity,
                overallocation_factor=self._overallocation_factor,
            )

    def set_max_size(self, new_max_size: int) -> None:
        """
        Update the maximum buffer size (can only grow, never shrink).

        If transitioning from max_size=1 to max_size>1, switches from
        SingleValueStorage to StreamingBuffer and preserves existing value.

        Parameters
        ----------
        new_max_size:
            New maximum size. If smaller than current max_size, no change is made.
        """
        if new_max_size <= self._max_size:
            return
        # Check if we need to transition from single-value to streaming mode
        if isinstance(self._storage, SingleValueStorage) and new_max_size > 1:
            old_value = self._storage.get_all()
            self._storage = self._create_storage(new_max_size)
            if old_value is not None:
                self._storage.append(old_value)
        elif isinstance(self._storage, StreamingBuffer):
            # Already in streaming mode, just grow
            self._storage.set_max_size(new_max_size)
        self._max_size = new_max_size

    def append(self, data: T) -> None:
        """Append new data to storage."""
        self._storage.append(data)

    def get_all(self) -> T | None:
        """Get all stored data."""
        return self._storage.get_all()

    def clear(self) -> None:
        """Clear all stored data."""
        self._storage.clear()

    def get_window(self, size: int | None = None) -> T | None:
        """
        Get a window of buffered data from the end.

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
        return self._storage.get_window(size)

    def get_latest(self) -> T | None:
        """
        Get the latest single value, unwrapped.

        Returns the most recent data point without the concat dimension,
        ready for use without further processing.

        Returns
        -------
        :
            The latest value without concat dimension, or None if empty.
        """
        return self._storage.get_latest()

    def get_window_by_duration(self, duration_seconds: float) -> T | None:
        """
        Get window by time duration.

        Parameters
        ----------
        duration_seconds:
            Approximate time duration in seconds.

        Returns
        -------
        :
            Window of data covering approximately the duration, or None if empty.
        """
        return self._storage.get_window_by_duration(duration_seconds)


class BufferFactory:
    """
    Factory that creates appropriate buffers based on data type.

    Maintains a registry of type → BufferInterface mappings.
    """

    def __init__(
        self,
        concat_dim: str = "time",
        initial_capacity: int = 100,
        overallocation_factor: float = 2.5,
    ) -> None:
        """
        Initialize buffer factory.

        Parameters
        ----------
        concat_dim:
            The dimension along which to concatenate data.
        initial_capacity:
            Initial buffer allocation.
        overallocation_factor:
            Buffer capacity multiplier.
        """
        self._concat_dim = concat_dim
        self._initial_capacity = initial_capacity
        self._overallocation_factor = overallocation_factor

    def create_buffer(self, template: T, max_size: int) -> Buffer[T]:
        """
        Create buffer appropriate for the data type.

        Parameters
        ----------
        template:
            Sample data used to determine buffer type.
        max_size:
            Maximum number of elements to maintain.

        Returns
        -------
        :
            Configured buffer instance.
        """
        data_type = type(template)

        # Dispatch to appropriate buffer implementation
        if data_type == sc.DataArray:
            buffer_impl = DataArrayBuffer(concat_dim=self._concat_dim)
        elif data_type == sc.Variable:
            buffer_impl = VariableBuffer(concat_dim=self._concat_dim)
        else:
            # Default fallback for simple types (int, str, etc.)
            buffer_impl = ListBuffer(concat_dim=self._concat_dim)

        return Buffer(
            max_size=max_size,
            buffer_impl=buffer_impl,  # type: ignore[arg-type]
            initial_capacity=self._initial_capacity,
            overallocation_factor=self._overallocation_factor,
        )
