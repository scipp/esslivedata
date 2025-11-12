# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .buffer import Buffer
from .temporal_requirements import (
    CompleteHistory,
    LatestFrame,
    TemporalRequirement,
    TimeWindow,
)

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
    def get_temporal_requirement(self) -> TemporalRequirement:
        """
        Return the temporal requirement for this extractor.

        Returns
        -------
        :
            Temporal requirement describing needed time coverage.
        """


class LatestValueExtractor(UpdateExtractor):
    """Extracts the latest single value, unwrapping the concat dimension."""

    def get_temporal_requirement(self) -> TemporalRequirement:
        """Latest value only needs the most recent frame."""
        return LatestFrame()

    def extract(self, buffer: Buffer) -> Any:
        """Extract the latest value from the buffer, unwrapped."""
        return buffer.get_latest()


class FullHistoryExtractor(UpdateExtractor):
    """Extracts the complete buffer history."""

    def get_temporal_requirement(self) -> TemporalRequirement:
        """Full history requires all available data."""
        return CompleteHistory()

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

    def get_temporal_requirement(self) -> TemporalRequirement:
        """Requires temporal coverage of specified duration."""
        return TimeWindow(duration_seconds=self._window_duration_seconds)

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
