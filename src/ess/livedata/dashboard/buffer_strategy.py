# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Low-level storage strategies for buffer data management."""

from __future__ import annotations

from abc import ABC, abstractmethod

import scipp as sc


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


class SlidingWindowStorage(StorageStrategy):
    """
    Fixed-size storage that maintains the most recent data.

    Assumes non-concat-dimension coordinates are constant across updates.
    Uses pre-allocated buffer with in-place writes and doubling strategy
    for efficient memory usage.
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
        self._end = 0
        self._capacity = 0

    def _ensure_capacity(self, new_data: sc.DataArray) -> None:
        """Ensure buffer has capacity for new data."""
        new_size = new_data.sizes[self._concat_dim]

        if self._buffer is None:
            # Initial allocation: allocate 2x max_size
            capacity = max(self._max_size * 2, new_size)

            # Create zeros array with correct structure
            data_var = sc.zeros(
                dims=new_data.dims,
                shape=[capacity, *new_data.data.shape[1:]],
                dtype=new_data.data.dtype,
            )

            # Create DataArray with coordinates
            coords = {
                self._concat_dim: sc.array(
                    dims=[self._concat_dim],
                    values=list(range(capacity)),
                    dtype='int64',
                )
            }

            # Add non-concat coordinates from new_data
            # Only add those that don't depend on the concat dimension
            # (those are constant across appends)
            coords.update(
                {
                    coord_name: coord
                    for coord_name, coord in new_data.coords.items()
                    if (
                        coord_name != self._concat_dim
                        and self._concat_dim not in coord.dims
                    )
                }
            )

            buffer_data = sc.DataArray(data=data_var, coords=coords)

            # Pre-allocate coordinates that depend on concat dimension
            for coord_name, coord in new_data.coords.items():
                if coord_name != self._concat_dim and self._concat_dim in coord.dims:
                    # Create zero array with full buffer capacity
                    coord_shape = [
                        capacity if dim == self._concat_dim else new_data.sizes[dim]
                        for dim in coord.dims
                    ]
                    buffer_data.coords[coord_name] = sc.zeros(
                        dims=coord.dims,
                        shape=coord_shape,
                        dtype=coord.dtype,
                    )

            # Copy masks structure for each mask in new_data
            for mask_name in new_data.masks:
                # Create mask with full buffer size
                mask_shape = [
                    capacity if dim == self._concat_dim else s
                    for dim, s in zip(
                        new_data.masks[mask_name].dims,
                        new_data.masks[mask_name].shape,
                        strict=False,
                    )
                ]
                buffer_data.masks[mask_name] = sc.zeros(
                    dims=new_data.masks[mask_name].dims,
                    shape=mask_shape,
                    dtype=new_data.masks[mask_name].dtype,
                )

            self._buffer = buffer_data
            self._capacity = capacity
            self._end = 0
            return

        # Check if we need more capacity
        if self._end + new_size > self._capacity:
            # Double capacity (but don't exceed reasonable bounds)
            new_capacity = min(self._capacity * 2, self._max_size * 4)

            # Trim if we already have more than max_size
            if self._end > self._max_size:
                trim_start = self._end - self._max_size
                self._buffer = self._buffer[self._concat_dim, trim_start:].copy()
                self._end = self._max_size
                self._capacity = self._buffer.sizes[self._concat_dim]

            # Grow buffer if still needed
            if self._end + new_size > self._capacity:
                # Create padding array with correct structure
                padding_size = new_capacity - self._capacity
                data_var = sc.zeros(
                    dims=self._buffer.dims,
                    shape=[padding_size, *self._buffer.data.shape[1:]],
                    dtype=self._buffer.data.dtype,
                )

                # Create DataArray with coordinates for padding
                pad_coords = {
                    self._concat_dim: sc.array(
                        dims=[self._concat_dim],
                        values=list(range(self._capacity, new_capacity)),
                        dtype=self._buffer.coords[self._concat_dim].dtype,
                    )
                }

                pad_coords.update(
                    {
                        coord_name: coord
                        for coord_name, coord in self._buffer.coords.items()
                        if (
                            coord_name != self._concat_dim
                            and self._concat_dim not in coord.dims
                        )
                    }
                )

                padding = sc.DataArray(data=data_var, coords=pad_coords)

                # Pre-allocate concat-dependent coordinates for padding
                for coord_name, coord in self._buffer.coords.items():
                    if (
                        coord_name != self._concat_dim
                        and self._concat_dim in coord.dims
                    ):
                        # Create zero array for padding size
                        coord_shape = [
                            padding_size
                            if dim == self._concat_dim
                            else coord.sizes[dim]
                            for dim in coord.dims
                        ]
                        padding.coords[coord_name] = sc.zeros(
                            dims=coord.dims,
                            shape=coord_shape,
                            dtype=coord.dtype,
                        )

                # Create padding masks
                for mask_name, mask in self._buffer.masks.items():
                    mask_shape = [
                        padding_size if dim == self._concat_dim else s
                        for dim, s in zip(mask.dims, mask.shape, strict=False)
                    ]
                    padding.masks[mask_name] = sc.zeros(
                        dims=mask.dims,
                        shape=mask_shape,
                        dtype=mask.dtype,
                    )

                self._buffer = sc.concat(
                    [self._buffer, padding],
                    dim=self._concat_dim,
                )
                self._capacity = new_capacity

    def append(self, data: sc.DataArray) -> None:
        """Append new data to storage."""
        if self._concat_dim not in data.dims:
            raise ValueError(f"Data must have '{self._concat_dim}' dimension")

        self._ensure_capacity(data)
        if self._buffer is None:
            raise RuntimeError("Buffer initialization failed")

        new_size = data.sizes[self._concat_dim]
        start = self._end
        end = self._end + new_size

        # In-place writes using numpy array access
        self._buffer.data.values[start:end] = data.data.values
        self._buffer.coords[self._concat_dim].values[start:end] = data.coords[
            self._concat_dim
        ].values

        # Copy other dimension-dependent coords and masks
        for coord_name, coord in data.coords.items():
            if coord_name != self._concat_dim and self._concat_dim in coord.dims:
                self._buffer.coords[coord_name].values[start:end] = coord.values

        for mask_name, mask in data.masks.items():
            if self._concat_dim in mask.dims:
                self._buffer.masks[mask_name].values[start:end] = mask.values

        self._end = end

        # Trim if we exceed max_size
        if self._end > self._max_size:
            trim_start = self._end - self._max_size
            self._buffer = self._buffer[self._concat_dim, trim_start:].copy()
            self._end = self._max_size
            self._capacity = self._buffer.sizes[self._concat_dim]

    def get_all(self) -> sc.DataArray | None:
        """Get all stored data."""
        if self._buffer is None:
            return None
        return self._buffer[self._concat_dim, : self._end].copy()

    def estimate_memory(self) -> int:
        """Estimate memory usage in bytes."""
        if self._buffer is None:
            return 0
        return self._buffer.values.nbytes

    def clear(self) -> None:
        """Clear all stored data."""
        self._buffer = None
        self._end = 0
        self._capacity = 0


class GrowingStorage(StorageStrategy):
    """
    Storage that grows by doubling capacity until reaching maximum size.

    Assumes non-concat-dimension coordinates are constant across updates.
    Uses pre-allocated buffer with in-place writes, growing capacity
    as needed up to the maximum limit.
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
        self._capacity = 0

    def _ensure_capacity(self, new_data: sc.DataArray) -> None:
        """Ensure buffer has capacity for new data."""
        new_size = new_data.sizes[self._concat_dim]

        if self._buffer is None:
            # Initial allocation
            capacity = max(self._initial_size, new_size)

            # Create zeros array with correct structure
            data_var = sc.zeros(
                dims=new_data.dims,
                shape=[capacity, *new_data.data.shape[1:]],
                dtype=new_data.data.dtype,
            )

            # Create DataArray with coordinates
            coords = {
                self._concat_dim: sc.array(
                    dims=[self._concat_dim],
                    values=list(range(capacity)),
                    dtype='int64',
                )
            }

            # Add non-concat coordinates from new_data
            # Only add those that don't depend on the concat dimension
            # (those are constant across appends)
            coords.update(
                {
                    coord_name: coord
                    for coord_name, coord in new_data.coords.items()
                    if (
                        coord_name != self._concat_dim
                        and self._concat_dim not in coord.dims
                    )
                }
            )

            buffer_data = sc.DataArray(data=data_var, coords=coords)

            # Pre-allocate coordinates that depend on concat dimension
            for coord_name, coord in new_data.coords.items():
                if coord_name != self._concat_dim and self._concat_dim in coord.dims:
                    # Create zero array with full buffer capacity
                    coord_shape = [
                        capacity if dim == self._concat_dim else new_data.sizes[dim]
                        for dim in coord.dims
                    ]
                    buffer_data.coords[coord_name] = sc.zeros(
                        dims=coord.dims,
                        shape=coord_shape,
                        dtype=coord.dtype,
                    )

            # Copy masks structure for each mask in new_data
            for mask_name in new_data.masks:
                # Create mask with full buffer size
                mask_shape = [
                    capacity if dim == self._concat_dim else s
                    for dim, s in zip(
                        new_data.masks[mask_name].dims,
                        new_data.masks[mask_name].shape,
                        strict=False,
                    )
                ]
                buffer_data.masks[mask_name] = sc.zeros(
                    dims=new_data.masks[mask_name].dims,
                    shape=mask_shape,
                    dtype=new_data.masks[mask_name].dtype,
                )

            self._buffer = buffer_data
            self._capacity = capacity
            self._end = 0
            return

        # Check if we need more capacity
        if self._end + new_size > self._capacity:
            # Double capacity, but cap at max_size
            new_capacity = min(self._capacity * 2, self._max_size)

            # If still doesn't fit and we have data to trim, trim first
            if self._end + new_size > new_capacity and self._end > 0:
                keep_size = new_capacity - new_size
                trim_start = max(0, self._end - keep_size)
                self._buffer = self._buffer[self._concat_dim, trim_start:].copy()
                self._end = self._end - trim_start
                self._capacity = self._buffer.sizes[self._concat_dim]

            # Grow buffer if still needed
            if self._end + new_size > self._capacity:
                # Create padding array with correct structure
                padding_size = new_capacity - self._capacity
                data_var = sc.zeros(
                    dims=self._buffer.dims,
                    shape=[padding_size, *self._buffer.data.shape[1:]],
                    dtype=self._buffer.data.dtype,
                )

                # Create DataArray with coordinates for padding
                pad_coords = {
                    self._concat_dim: sc.array(
                        dims=[self._concat_dim],
                        values=list(range(self._capacity, new_capacity)),
                        dtype=self._buffer.coords[self._concat_dim].dtype,
                    )
                }

                pad_coords.update(
                    {
                        coord_name: coord
                        for coord_name, coord in self._buffer.coords.items()
                        if (
                            coord_name != self._concat_dim
                            and self._concat_dim not in coord.dims
                        )
                    }
                )

                padding = sc.DataArray(data=data_var, coords=pad_coords)

                # Pre-allocate concat-dependent coordinates for padding
                for coord_name, coord in self._buffer.coords.items():
                    if (
                        coord_name != self._concat_dim
                        and self._concat_dim in coord.dims
                    ):
                        # Create zero array for padding size
                        coord_shape = [
                            padding_size
                            if dim == self._concat_dim
                            else coord.sizes[dim]
                            for dim in coord.dims
                        ]
                        padding.coords[coord_name] = sc.zeros(
                            dims=coord.dims,
                            shape=coord_shape,
                            dtype=coord.dtype,
                        )

                # Create padding masks
                for mask_name, mask in self._buffer.masks.items():
                    mask_shape = [
                        padding_size if dim == self._concat_dim else s
                        for dim, s in zip(mask.dims, mask.shape, strict=False)
                    ]
                    padding.masks[mask_name] = sc.zeros(
                        dims=mask.dims,
                        shape=mask_shape,
                        dtype=mask.dtype,
                    )

                self._buffer = sc.concat(
                    [self._buffer, padding],
                    dim=self._concat_dim,
                )
                self._capacity = new_capacity

    def append(self, data: sc.DataArray) -> None:
        """Append new data to storage."""
        if self._concat_dim not in data.dims:
            raise ValueError(f"Data must have '{self._concat_dim}' dimension")

        self._ensure_capacity(data)
        if self._buffer is None:
            raise RuntimeError("Buffer initialization failed")

        new_size = data.sizes[self._concat_dim]
        start = self._end
        end = self._end + new_size

        # In-place writes using numpy array access
        self._buffer.data.values[start:end] = data.data.values
        self._buffer.coords[self._concat_dim].values[start:end] = data.coords[
            self._concat_dim
        ].values

        # Copy other dimension-dependent coords and masks
        for coord_name, coord in data.coords.items():
            if coord_name != self._concat_dim and self._concat_dim in coord.dims:
                self._buffer.coords[coord_name].values[start:end] = coord.values

        for mask_name, mask in data.masks.items():
            if self._concat_dim in mask.dims:
                self._buffer.masks[mask_name].values[start:end] = mask.values

        self._end = end

        # Trim if we exceed max_size
        if self._end > self._max_size:
            trim_start = self._end - self._max_size
            self._buffer = self._buffer[self._concat_dim, trim_start:].copy()
            self._end = self._max_size
            self._capacity = self._buffer.sizes[self._concat_dim]

    def get_all(self) -> sc.DataArray | None:
        """Get all stored data."""
        if self._buffer is None:
            return None
        return self._buffer[self._concat_dim, : self._end].copy()

    def estimate_memory(self) -> int:
        """Estimate memory usage in bytes."""
        if self._buffer is None:
            return 0
        return self._buffer.values.nbytes

    def clear(self) -> None:
        """Clear all stored data."""
        self._buffer = None
        self._end = 0
        self._capacity = 0
