# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Buffer interface on top of storage strategies."""

from __future__ import annotations

import scipp as sc

from .buffer_strategy import StorageStrategy


class Buffer:
    """
    Buffer providing data access operations on top of a storage strategy.

    Wraps a low-level StorageStrategy and provides higher-level operations
    like windowing for use by extractors.
    """

    def __init__(self, strategy: StorageStrategy, concat_dim: str = 'time') -> None:
        """
        Initialize a buffer with the given storage strategy.

        Parameters
        ----------
        strategy:
            The storage strategy to use for data management.
        concat_dim:
            The dimension along which data is concatenated.
        """
        self._strategy = strategy
        self._concat_dim = concat_dim

    def append(self, data: sc.DataArray) -> None:
        """
        Append new data to the buffer.

        Parameters
        ----------
        data:
            The data to append.
        """
        self._strategy.append(data)

    def get_buffer(self) -> sc.DataArray | None:
        """
        Get the complete buffered data.

        Returns
        -------
        :
            The full buffer as a DataArray, or None if empty.
        """
        return self._strategy.get_all()

    def get_window(self, size: int | None = None) -> sc.DataArray | None:
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
        data = self._strategy.get_all()
        if data is None or size is None:
            return data

        current_size = data.sizes[self._concat_dim]
        actual_size = min(size, current_size)
        return data[self._concat_dim, -actual_size:]

    def clear(self) -> None:
        """Clear all data from the buffer."""
        self._strategy.clear()

    @property
    def memory_mb(self) -> float:
        """Get the current memory usage in megabytes."""
        return self._strategy.estimate_memory() / (1024 * 1024)
