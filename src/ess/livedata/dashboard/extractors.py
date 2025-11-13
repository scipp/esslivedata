# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import scipp as sc

from .plot_params import WindowAggregation

if TYPE_CHECKING:
    from ess.livedata.config.workflow_spec import ResultKey

    from .plot_params import WindowParams
    from .plotting import PlotterSpec


class UpdateExtractor(ABC):
    """Extracts a specific view of buffered data."""

    @abstractmethod
    def extract(self, data: sc.DataArray) -> Any:
        """
        Extract data from buffered data.

        Parameters
        ----------
        data:
            The buffered data to extract from.

        Returns
        -------
        :
            The extracted data.
        """

    @abstractmethod
    def get_required_timespan(self) -> float:
        """
        Get the required timespan for this extractor.

        Returns
        -------
        :
            Required timespan in seconds. Return 0.0 for extractors that only
            need the latest value.
        """


class LatestValueExtractor(UpdateExtractor):
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

    def get_required_timespan(self) -> float:
        """Latest value requires zero history."""
        return 0.0

    def extract(self, data: sc.DataArray) -> Any:
        """Extract the latest value from the data, unwrapped."""
        # Check if data has the concat dimension
        if not hasattr(data, 'dims') or self._concat_dim not in data.dims:
            # Data doesn't have concat dim - already a single frame
            return data

        # Extract last frame along concat dimension
        return data[self._concat_dim, -1]


class FullHistoryExtractor(UpdateExtractor):
    """Extracts the complete buffer history."""

    def get_required_timespan(self) -> float:
        """Return infinite timespan to indicate wanting all history."""
        return float('inf')

    def extract(self, data: sc.DataArray) -> Any:
        """Extract all data from the buffer."""
        return data


class WindowAggregatingExtractor(UpdateExtractor):
    """Extracts a window from the buffer and aggregates over the time dimension."""

    def __init__(
        self,
        window_duration_seconds: float,
        aggregation: WindowAggregation = WindowAggregation.auto,
        concat_dim: str = 'time',
    ) -> None:
        """
        Initialize window aggregating extractor.

        Parameters
        ----------
        window_duration_seconds:
            Time duration to extract from the end of the buffer (seconds).
        aggregation:
            Aggregation method. WindowAggregation.auto uses 'nansum' if data unit
            is counts, else 'nanmean'.
        concat_dim:
            Name of the dimension to aggregate over.
        """
        self._window_duration_seconds = window_duration_seconds
        self._aggregation = aggregation
        self._concat_dim = concat_dim
        self._aggregator: Callable[[sc.DataArray, str], sc.DataArray] | None = None

    def get_required_timespan(self) -> float:
        """Return the required window duration."""
        return self._window_duration_seconds

    def extract(self, data: sc.DataArray) -> Any:
        """Extract a window of data and aggregate over the time dimension."""
        # Calculate cutoff time using scipp's unit handling
        time_coord = data.coords[self._concat_dim]
        latest_time = time_coord[-1]
        duration = sc.scalar(self._window_duration_seconds, unit='s').to(
            unit=time_coord.unit
        )
        windowed_data = data[self._concat_dim, latest_time - duration :]

        # Resolve and cache aggregator function on first call
        if self._aggregator is None:
            if self._aggregation == WindowAggregation.auto:
                aggregation = (
                    WindowAggregation.nansum
                    if windowed_data.unit == 'counts'
                    else WindowAggregation.nanmean
                )
            else:
                aggregation = self._aggregation
            aggregators = {
                WindowAggregation.sum: sc.sum,
                WindowAggregation.nansum: sc.nansum,
                WindowAggregation.mean: sc.mean,
                WindowAggregation.nanmean: sc.nanmean,
            }
            self._aggregator = aggregators.get(aggregation)
            if self._aggregator is None:
                raise ValueError(f"Unknown aggregation method: {self._aggregation}")

        return self._aggregator(windowed_data, self._concat_dim)


def create_extractors_from_params(
    keys: list[ResultKey],
    window: WindowParams | None,
    spec: PlotterSpec | None = None,
) -> dict[ResultKey, UpdateExtractor]:
    """
    Create extractors based on plotter spec and window configuration.

    Parameters
    ----------
    keys:
        Result keys to create extractors for.
    window:
        Window parameters for extraction mode and aggregation.
        If None, falls back to LatestValueExtractor.
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

    # No fixed requirement - check if window params provided
    if window is not None:
        if window.mode == WindowMode.latest:
            return {key: LatestValueExtractor() for key in keys}
        else:  # mode == WindowMode.window
            return {
                key: WindowAggregatingExtractor(
                    window_duration_seconds=window.window_duration_seconds,
                    aggregation=window.aggregation,
                )
                for key in keys
            }

    # Fallback to latest value extractor
    return {key: LatestValueExtractor() for key in keys}
