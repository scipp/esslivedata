# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import scipp as sc

from .plot_params import WindowAggregation


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
        return data[self._concat_dim, -1] if self._concat_dim in data.dims else data


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
        self._duration: sc.Variable | None = None

    def get_required_timespan(self) -> float:
        """Return the required window duration."""
        return self._window_duration_seconds

    def extract(self, data: sc.DataArray) -> Any:
        """Extract a window of data and aggregate over the time dimension."""
        # Calculate cutoff time
        time_coord = data.coords[self._concat_dim]
        if self._duration is None:
            duration_scalar = sc.scalar(self._window_duration_seconds, unit='s')
            if time_coord.dtype == sc.DType.datetime64:
                self._duration = duration_scalar.to(unit=time_coord.unit, dtype='int64')
            else:
                self._duration = duration_scalar.to(unit=time_coord.unit)

        # Estimate frame period from median interval to handle timing noise.
        # Shift cutoff by half period to place boundary between frame slots,
        # avoiding inclusion of extra frames due to timing jitter.
        latest_time = time_coord[-1]
        if len(time_coord) > 1:
            intervals = time_coord[1:] - time_coord[:-1]
            median_interval = sc.median(intervals)
            half_median = 0.5 * median_interval
            # datetime64 arithmetic requires int64, not float64
            if time_coord.dtype == sc.DType.datetime64:
                half_median = half_median.astype('int64')
            cutoff_time = latest_time - self._duration + half_median
            # Clamp to ensure at least latest frame included
            # (handles narrow windows where duration < median_interval)
            if cutoff_time > latest_time:
                cutoff_time = latest_time
        else:
            # Single frame: use duration-based cutoff
            cutoff_time = latest_time - self._duration

        # Use label-based slicing with inclusive lower bound. If timestamps were precise
        # we would actually want exclusive lower bound, but since there is jitter anyway
        # the cutoff shift above should handle that well enough.
        windowed_data = data[self._concat_dim, cutoff_time:]

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
