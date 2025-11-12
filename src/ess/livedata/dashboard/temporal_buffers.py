# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Temporal buffer implementations for BufferManager."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
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


class TemporalBuffer(BufferProtocol['sc.DataArray']):
    """
    Buffer that maintains temporal data with a time dimension.

    Concatenates incoming data along the time dimension and validates
    that data has a 'time' coordinate.
    """

    def __init__(self) -> None:
        """Initialize empty temporal buffer."""
        self._buffer: sc.DataArray | None = None
        self._max_memory: int | None = None
        self._required_timespan: float = 0.0

    def add(self, data: sc.DataArray) -> None:
        """
        Add data to the buffer, concatenating along time dimension.

        Parameters
        ----------
        data:
            New data to add. Must have a 'time' coordinate.

        Raises
        ------
        ValueError
            If data does not have a 'time' coordinate.
        """
        import scipp as sc

        if 'time' not in data.coords:
            raise ValueError("TemporalBuffer requires data with 'time' coordinate")

        if self._buffer is None:
            # First data - ensure it has time dimension
            if 'time' not in data.dims:
                # Single slice - add time dimension
                self._buffer = sc.concat([data], dim='time')
            else:
                # Already has time dimension
                self._buffer = data.copy()
        else:
            # Concatenate with existing buffer
            if 'time' not in data.dims:
                # Single slice - concat will handle adding dimension
                self._buffer = sc.concat([self._buffer, data], dim='time')
            else:
                # Thick slice - concat along existing dimension
                self._buffer = sc.concat([self._buffer, data], dim='time')

    def get(self) -> sc.DataArray | None:
        """Return the complete buffer."""
        return self._buffer

    def clear(self) -> None:
        """Clear all buffered data."""
        self._buffer = None

    def set_required_timespan(self, seconds: float) -> None:
        """Set the required timespan for the buffer."""
        self._required_timespan = seconds

    def set_max_memory(self, max_bytes: int) -> None:
        """Set the maximum memory usage for the buffer."""
        self._max_memory = max_bytes
