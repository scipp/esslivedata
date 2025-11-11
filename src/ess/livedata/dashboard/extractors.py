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

    def __init__(self, concat_dim: str = 'time') -> None:
        """
        Initialize latest value extractor.

        Parameters
        ----------
        concat_dim:
            The dimension to unwrap when extracting from scipp objects.
        """
        self._concat_dim = concat_dim

    def get_required_size(self) -> int:
        """Latest value only needs buffer size of 1."""
        return 1

    def extract(self, buffer: Buffer) -> Any:
        """
        Extract the latest value from the buffer.

        For list buffers, returns the last element.
        For scipp DataArray/Variable, unwraps the concat dimension.
        """
        view = buffer.get_window(1)
        if view is None:
            return None

        # Unwrap based on type
        if isinstance(view, list):
            return view[0] if view else None

        # Import scipp only when needed to avoid circular imports
        import scipp as sc

        if isinstance(view, sc.DataArray):
            if self._concat_dim in view.dims:
                # Slice to remove concat dimension
                result = view[self._concat_dim, 0]
                # Drop the now-scalar concat coordinate to restore original structure
                if self._concat_dim in result.coords:
                    result = result.drop_coords(self._concat_dim)
                return result
            return view
        elif isinstance(view, sc.Variable):
            if self._concat_dim in view.dims:
                return view[self._concat_dim, 0]
            return view
        else:
            return view


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
