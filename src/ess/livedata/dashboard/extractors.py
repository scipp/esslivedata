# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    import pydantic

    from ess.livedata.config.workflow_spec import ResultKey

    from .plotting import PlotterSpec

T = TypeVar('T')


class UpdateExtractor(ABC, Generic[T]):
    """Extracts a specific view of buffered data."""

    @abstractmethod
    def extract(self, data: T | None) -> Any:
        """
        Extract data from buffered data.

        Parameters
        ----------
        data:
            The buffered data to extract from, or None if no data available.

        Returns
        -------
        :
            The extracted data, or None if no data available.
        """

    @abstractmethod
    def is_requirement_fulfilled(self, data: T | None) -> bool:
        """
        Check if the extractor's requirements are satisfied by the buffered data.

        Parameters
        ----------
        data:
            The buffered data to check.

        Returns
        -------
        :
            True if requirements are satisfied, False otherwise.
        """

    def get_required_timespan(self) -> float | None:
        """
        Get the required timespan for this extractor.

        Returns
        -------
        :
            Required timespan in seconds, or None if no specific requirement.
        """
        return None


class LatestValueExtractor(UpdateExtractor[T]):
    """Extracts the latest single value, unwrapping the concat dimension."""

    def __init__(self, concat_dim: str = 'time') -> None:
        """
        Initialize latest value extractor.

        Parameters
        ----------
        concat_dim:
            The dimension along which data is concatenated.
        """
        self._concat_dim = concat_dim

    def is_requirement_fulfilled(self, data: T | None) -> bool:
        """Latest value only needs any data."""
        return data is not None

    def extract(self, data: T | None) -> Any:
        """Extract the latest value from the data, unwrapped."""
        if data is None:
            return None

        # Handle list buffers
        if isinstance(data, list) and len(data) > 0:
            return data[-1]

        # Check if data has the concat dimension
        if not hasattr(data, 'dims') or self._concat_dim not in data.dims:
            # Data doesn't have concat dim - already a single frame
            return data

        # Extract last frame along concat dimension
        return data[self._concat_dim, -1]


class FullHistoryExtractor(UpdateExtractor[T]):
    """Extracts the complete buffer history."""

    def get_required_timespan(self) -> float | None:
        """Return infinite timespan to indicate wanting all history."""
        return float('inf')

    def is_requirement_fulfilled(self, data: T | None) -> bool:
        """Full history is never fulfilled - always want more data."""
        return False

    def extract(self, data: T | None) -> Any:
        """Extract all data from the buffer."""
        return data


class WindowAggregatingExtractor(UpdateExtractor[T]):
    """Extracts a window from the buffer and aggregates over the time dimension."""

    def __init__(
        self,
        window_duration_seconds: float,
        aggregation: str = 'auto',
        concat_dim: str = 'time',
    ) -> None:
        """
        Initialize window aggregating extractor.

        Parameters
        ----------
        window_duration_seconds:
            Time duration to extract from the end of the buffer (seconds).
        aggregation:
            Aggregation method: 'auto', 'nansum', 'nanmean', 'sum', 'mean', 'last',
            or 'max'. 'auto' uses 'nansum' if data unit is counts, else 'nanmean'.
        concat_dim:
            Name of the dimension to aggregate over.
        """
        self._window_duration_seconds = window_duration_seconds
        self._aggregation = aggregation
        self._concat_dim = concat_dim

    def get_required_timespan(self) -> float | None:
        """Return the required window duration."""
        return self._window_duration_seconds

    def is_requirement_fulfilled(self, data: T | None) -> bool:
        """Requires temporal coverage of specified duration."""
        if data is None:
            return False

        # Check for time coordinate
        if not hasattr(data, 'coords') or self._concat_dim not in data.coords:
            return False

        # Check if data has concat dimension (indicates multiple frames)
        if not hasattr(data, 'dims') or self._concat_dim not in data.dims:
            # Single frame - no temporal coverage
            return False

        time_coord = data.coords[self._concat_dim]
        if data.sizes[self._concat_dim] < 2:
            # Need at least 2 points to measure coverage
            return False

        # Calculate time span
        time_span = time_coord[-1] - time_coord[0]
        coverage_seconds = float(time_span.to(unit='s').value)
        return coverage_seconds >= self._window_duration_seconds

    def extract(self, data: T | None) -> Any:
        """Extract a window of data and aggregate over the time dimension."""
        if data is None:
            return None

        # Check if concat dimension exists in the data
        if not hasattr(data, 'dims') or self._concat_dim not in data.dims:
            # Data doesn't have the expected dimension structure, return as-is
            return data

        # Extract time window
        if not hasattr(data, 'coords') or self._concat_dim not in data.coords:
            # No time coordinate - can't do time-based windowing, return all data
            windowed_data = data
        else:
            # Calculate cutoff time using scipp's unit handling
            import scipp as sc

            time_coord = data.coords[self._concat_dim]
            latest_time = time_coord[-1]
            duration = sc.scalar(self._window_duration_seconds, unit='s').to(
                unit=time_coord.unit
            )
            windowed_data = data[self._concat_dim, latest_time - duration :]

        # Determine aggregation method
        agg_method = self._aggregation
        if agg_method == 'auto':
            # Use nansum if data is dimensionless (counts), else nanmean
            if hasattr(windowed_data, 'unit') and windowed_data.unit == '1':
                agg_method = 'nansum'
            else:
                agg_method = 'nanmean'

        # Aggregate over the concat dimension
        if agg_method == 'sum':
            return windowed_data.sum(self._concat_dim)
        elif agg_method == 'nansum':
            return windowed_data.nansum(self._concat_dim)
        elif agg_method == 'mean':
            return windowed_data.mean(self._concat_dim)
        elif agg_method == 'nanmean':
            return windowed_data.nanmean(self._concat_dim)
        elif agg_method == 'last':
            # Return the last frame (equivalent to latest)
            return windowed_data[self._concat_dim, -1]
        elif agg_method == 'max':
            return windowed_data.max(self._concat_dim)
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")


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
