# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""High-level buffer interface with unified mode selection."""

from __future__ import annotations

import logging
from typing import Generic, TypeVar

import scipp as sc

from .buffer_strategy import (
    BufferInterface,
    DataArrayBuffer,
    ListBuffer,
    SingleValueStorage,
    StreamingBuffer,
    VariableBuffer,
)

logger = logging.getLogger(__name__)

# Type variable for buffer types
T = TypeVar('T')


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

    def get_frame_count(self) -> int:
        """
        Get the number of frames currently stored.

        Returns
        -------
        :
            Number of frames in buffer.
        """
        return self._storage.get_frame_count()


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
