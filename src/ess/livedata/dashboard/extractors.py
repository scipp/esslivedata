# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .buffer_strategy import Buffer

if TYPE_CHECKING:
    import pydantic

    from ess.livedata.config.workflow_spec import ResultKey

    from .plotting import PlotterSpec


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


class WindowAggregatingExtractor(UpdateExtractor):
    """Extracts a window from the buffer and aggregates over the time dimension."""

    def __init__(
        self,
        window_duration_seconds: float,
        aggregation: str = 'sum',
        concat_dim: str = 'time',
    ) -> None:
        """
        Initialize window aggregating extractor.

        Parameters
        ----------
        window_duration_seconds:
            Time duration to extract from the end of the buffer (seconds).
        aggregation:
            Aggregation method: 'sum', 'mean', 'last', or 'max'.
        concat_dim:
            Name of the dimension to aggregate over.
        """
        self._window_duration_seconds = window_duration_seconds
        self._aggregation = aggregation
        self._concat_dim = concat_dim

    def get_required_size(self) -> int:
        """
        Estimate required buffer size (conservative).

        Assumes maximum 20 Hz frame rate for headroom.
        """
        return max(100, int(self._window_duration_seconds * 20))

    def extract(self, buffer: Buffer) -> Any:
        """Extract a window of data and aggregate over the time dimension."""
        data = buffer.get_window_by_duration(self._window_duration_seconds)

        if data is None:
            return None

        # Check if concat dimension exists in the data
        if not hasattr(data, 'dims') or self._concat_dim not in data.dims:
            # Data doesn't have the expected dimension structure, return as-is
            return data

        # Aggregate over the concat dimension
        if self._aggregation == 'sum':
            return data.sum(self._concat_dim)
        elif self._aggregation == 'mean':
            return data.mean(self._concat_dim)
        elif self._aggregation == 'last':
            # Return the last frame (equivalent to latest)
            return data[self._concat_dim, -1]
        elif self._aggregation == 'max':
            return data.max(self._concat_dim)
        else:
            raise ValueError(f"Unknown aggregation method: {self._aggregation}")


def create_extractors_from_params(
    keys: list[ResultKey],
    params: pydantic.BaseModel,
    spec: PlotterSpec | None = None,
) -> dict[ResultKey, UpdateExtractor]:
    """
    Create extractors based on plotter spec and params window configuration.

    Parameters
    ----------
    keys:
        Result keys to create extractors for.
    params:
        Parameters potentially containing window configuration.
    spec:
        Optional plotter specification. If provided and contains a required
        extractor, that extractor type is used.

    Returns
    -------
    :
        Dictionary mapping result keys to extractor instances.
    """
    # Avoid circular import by importing here
    from .plot_params import WindowMode

    if spec is not None and spec.data_requirements.required_extractor is not None:
        # Plotter requires specific extractor (e.g., TimeSeriesPlotter)
        extractor_type = spec.data_requirements.required_extractor
        return {key: extractor_type() for key in keys}

    # No fixed requirement - check if params have window config
    if hasattr(params, 'window'):
        if params.window.mode == WindowMode.latest:
            return {key: LatestValueExtractor() for key in keys}
        else:  # mode == WindowMode.window
            return {
                key: WindowAggregatingExtractor(
                    window_duration_seconds=params.window.window_duration_seconds,
                    aggregation=params.window.aggregation.value,
                )
                for key in keys
            }

    # Fallback to latest value extractor
    return {key: LatestValueExtractor() for key in keys}
