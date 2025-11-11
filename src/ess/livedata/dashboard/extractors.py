# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .buffer_strategy import Buffer


class UpdateExtractor(ABC):
    """Extracts a specific view of buffer data."""

    @abstractmethod
    def extract(self, buffer: Buffer) -> Any:
        """
        Extract data from a buffer.

        Parameters
        ----------
        buffer:
            The buffer to extract data from.

        Returns
        -------
        :
            The extracted data, or None if no data available.
        """

    @abstractmethod
    def get_required_size(self) -> int:
        """
        Return the minimum buffer size required by this extractor.

        Returns
        -------
        :
            Required buffer size (1 for latest value, n for window, large for full).
        """


class LatestValueExtractor(UpdateExtractor):
    """Extracts the latest single value, unwrapping the concat dimension."""

    def get_required_size(self) -> int:
        """Latest value only needs buffer size of 1."""
        return 1

    def extract(self, buffer: Buffer) -> Any:
        """Extract the latest value from the buffer, unwrapped."""
        return buffer.get_latest()


class WindowExtractor(UpdateExtractor):
    """Extracts a window from the end of the buffer."""

    def __init__(self, size: int) -> None:
        """
        Initialize window extractor.

        Parameters
        ----------
        size:
            Number of elements to extract from the end of the buffer.
        """
        self._size = size

    @property
    def window_size(self) -> int:
        """Return the window size."""
        return self._size

    def get_required_size(self) -> int:
        """Window extractor requires buffer size equal to window size."""
        return self._size

    def extract(self, buffer: Buffer) -> Any:
        """Extract a window of data from the end of the buffer."""
        return buffer.get_window(self._size)


class FullHistoryExtractor(UpdateExtractor):
    """Extracts the complete buffer history."""

    # Maximum size for full history buffers
    DEFAULT_MAX_SIZE = 10000

    def get_required_size(self) -> int:
        """Full history requires large buffer."""
        return self.DEFAULT_MAX_SIZE

    def extract(self, buffer: Buffer) -> Any:
        """Extract all data from the buffer."""
        return buffer.get_all()
