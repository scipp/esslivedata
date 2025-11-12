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
    grow() is called.
    """

    def __init__(
        self,
        max_size: int,
        buffer_impl: BufferInterface[T],
        initial_capacity: int = 100,
        overallocation_factor: float = 2.5,
        memory_budget_bytes: int | None = None,
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
        memory_budget_bytes:
            Maximum memory budget in bytes. If None, no memory limit.
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self._max_size = max_size
        self._buffer_impl = buffer_impl
        self._initial_capacity = initial_capacity
        self._overallocation_factor = overallocation_factor
        self._memory_budget_bytes = memory_budget_bytes

        # Create appropriate storage based on max_size
        self._storage = self._create_storage(max_size, buffer_impl)

    def _create_storage(
        self, max_size: int, buffer_impl: BufferInterface[T]
    ) -> SingleValueStorage[T] | StreamingBuffer[T]:
        """
        Create appropriate storage implementation based on max_size.

        Parameters
        ----------
        max_size:
            Maximum number of data points to maintain.
        buffer_impl:
            Buffer implementation (only used by StreamingBuffer).

        Returns
        -------
        :
            SingleValueStorage for max_size=1, StreamingBuffer otherwise.
        """
        if max_size == 1:
            return SingleValueStorage()
        else:
            return StreamingBuffer(
                max_size=max_size,
                buffer_impl=buffer_impl,
                initial_capacity=self._initial_capacity,
                overallocation_factor=self._overallocation_factor,
                memory_budget_bytes=self._memory_budget_bytes,
            )

    def can_grow(self) -> bool:
        """
        Check if buffer can grow within memory budget.

        Returns
        -------
        :
            True if buffer can allocate more memory.
        """
        # SingleValueStorage can always transition to StreamingBuffer if budget allows
        if isinstance(self._storage, SingleValueStorage):
            if self._memory_budget_bytes is None:
                return True
            return self._storage.get_memory_usage() < self._memory_budget_bytes

        # StreamingBuffer delegates to its own can_grow
        return self._storage.can_grow()

    def grow(self) -> bool:
        """
        Attempt to grow buffer capacity.

        For SingleValueStorage, transitions to StreamingBuffer.
        For StreamingBuffer, doubles max_size.

        Returns
        -------
        :
            True if growth succeeded, False otherwise.
        """
        if not self.can_grow():
            return False

        # Transition from SingleValueStorage to StreamingBuffer
        if isinstance(self._storage, SingleValueStorage):
            old_value = self._storage.get_all()
            # Start with max_size=2 when transitioning
            new_max_size = 2
            self._storage = self._create_storage(new_max_size, self._buffer_impl)
            if old_value is not None:
                self._storage.append(old_value)
            self._max_size = new_max_size
            return True

        # Already in streaming mode, delegate to storage
        if isinstance(self._storage, StreamingBuffer):
            if self._storage.grow():
                self._max_size = self._storage._max_size
                return True

        return False

    def append(self, data: T) -> None:
        """Append new data to storage."""
        self._storage.append(data)

    def get_all(self) -> T | None:
        """Get all stored data."""
        return self._storage.get_all()

    def clear(self) -> None:
        """Clear all stored data."""
        self._storage.clear()

    def get_memory_usage(self) -> int:
        """
        Get current memory usage in bytes.

        Returns
        -------
        :
            Memory usage in bytes.
        """
        return self._storage.get_memory_usage()


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
        memory_budget_mb: int = 100,
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
        memory_budget_mb:
            Maximum memory budget per buffer in megabytes.
        """
        self._concat_dim = concat_dim
        self._initial_capacity = initial_capacity
        self._overallocation_factor = overallocation_factor
        self._memory_budget_bytes = memory_budget_mb * 1024 * 1024

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
            memory_budget_bytes=self._memory_budget_bytes,
        )
