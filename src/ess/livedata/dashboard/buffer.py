# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Buffer wrapper with metadata and thread safety."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import scipp as sc

from .buffer_strategy import BufferStrategy


class Buffer:
    """
    Thread-safe wrapper around a BufferStrategy with metadata tracking.

    Provides synchronized access to buffer operations and tracks metadata
    like memory usage and last update time.
    """

    def __init__(self, strategy: BufferStrategy) -> None:
        """
        Initialize a buffer with the given strategy.

        Parameters
        ----------
        strategy:
            The buffer strategy to use for data management.
        """
        self._strategy = strategy
        self._lock = threading.RLock()
        self._last_update: datetime | None = None
        self._total_appends = 0

    def append(self, data: sc.DataArray) -> None:
        """
        Append new data to the buffer (thread-safe).

        Parameters
        ----------
        data:
            The data to append.
        """
        with self._lock:
            self._strategy.append(data)
            self._last_update = datetime.now(UTC)
            self._total_appends += 1

    def get_buffer(self) -> sc.DataArray | None:
        """
        Get the complete buffered data (thread-safe).

        Returns
        -------
        :
            The full buffer as a DataArray, or None if empty.
        """
        with self._lock:
            return self._strategy.get_buffer()

    def get_window(self, size: int | None = None) -> sc.DataArray | None:
        """
        Get a window of buffered data (thread-safe).

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
        with self._lock:
            return self._strategy.get_window(size)

    def estimate_memory(self) -> int:
        """
        Estimate the memory usage of the buffer in bytes (thread-safe).

        Returns
        -------
        :
            Estimated memory usage in bytes.
        """
        with self._lock:
            return self._strategy.estimate_memory()

    def clear(self) -> None:
        """Clear all data from the buffer (thread-safe)."""
        with self._lock:
            self._strategy.clear()
            self._last_update = None
            self._total_appends = 0

    @property
    def last_update(self) -> datetime | None:
        """Get the timestamp of the last append operation."""
        with self._lock:
            return self._last_update

    @property
    def total_appends(self) -> int:
        """Get the total number of append operations."""
        with self._lock:
            return self._total_appends

    @property
    def memory_mb(self) -> float:
        """Get the current memory usage in megabytes."""
        return self.estimate_memory() / (1024 * 1024)
