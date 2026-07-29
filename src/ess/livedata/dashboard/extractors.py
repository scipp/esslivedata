# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import scipp as sc

from .plot_params import WindowAggregation
from .time_utils import get_local_timezone_offset_ns


def _extract_time_bounds_as_scalars(data: sc.DataArray) -> dict[str, sc.Variable]:
    """
    Capture the covered time range as the scalar ``(start_time, end_time)`` pair.

    This is the naming boundary between wire and dashboard. On the wire a
    bounded output's upper bound is its ``time``: the instant the value refers
    to, whether that is when a window closed or when a cumulative total was
    observed. Downstream of here the pair is spelled ``(start_time, end_time)``,
    which is what :mod:`ess.livedata.dashboard.plots` reads for the
    freshness/lag readout and for rate normalization.

    Every extractor applies this, so the pair reaches the plotters with the same
    meaning whichever buffer and extractor the data came through. Callers must
    apply it to *pristine* data: ``FullHistoryExtractor`` shifts ``time`` into
    the local timezone for Bokeh's benefit, and bounds taken after that shift
    would misreport wall-clock provenance by the UTC offset.

    The pair is minted whole or not at all, and ``start_time`` is what tells a
    bounded output from a series one. A series output has no bounds to capture:
    its ``time`` is the per-sample axis, so its age is already on the plot, and
    an ``end_time`` minted from it would drive the freshness pill and the lag
    readout off the device's sampling rate rather than the pipeline's latency.

    Coords already scalar (a single message, or a slice out of a buffer) are
    taken as they are; 1-D coords from a ``TemporalBuffer`` are reduced to their
    chronological ends.
    """
    start_time = data.coords.get('start_time')
    time = data.coords.get('time')
    if start_time is None or time is None:
        return {}
    return {
        'start_time': start_time[0] if start_time.ndim == 1 else start_time,
        'end_time': time[-1] if time.ndim == 1 else time,
    }


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
        if self._concat_dim in data.dims:
            # Extract last slice - this also gets the last value from any 1-D coords
            data = data[self._concat_dim, -1]
        return data.assign_coords(**_extract_time_bounds_as_scalars(data))


class FullHistoryExtractor(UpdateExtractor):
    """Extracts the complete buffer history."""

    def __init__(self, concat_dim: str = 'time') -> None:
        """
        Initialize full history extractor.

        Parameters
        ----------
        concat_dim:
            The time dimension name.
        """
        self._concat_dim = concat_dim
        self._time_origin: sc.Variable | None = None

    def get_required_timespan(self) -> float:
        """Return infinite timespan to indicate wanting all history."""
        return float('inf')

    def extract(self, data: sc.DataArray) -> Any:
        """Extract all data from the buffer, converting time to datetime64."""
        # Capture bounds before the timezone shift, which rewrites `time`.
        bounds = _extract_time_bounds_as_scalars(data)
        return self._to_local_datetime(data).assign_coords(**bounds)

    def _to_local_datetime(self, data: sc.DataArray) -> sc.DataArray:
        """Convert int64 time coordinate to datetime64 in local time.

        Bokeh requires datetime64 dtype to render human-readable datetime axes.
        Int64 nanoseconds are displayed as raw numbers (e.g., 1.733e18).

        The datetime values are shifted by the local timezone offset so that
        Bokeh displays them as local time rather than UTC.
        """
        dim = self._concat_dim
        if dim not in data.coords:
            return data
        if self._time_origin is not None:
            return data.assign_coords({dim: self._time_origin + data.coords[dim]})
        coord = data.coords[dim]
        if coord.dtype != sc.DType.int64 or coord.unit not in ('ns', 'us', 'ms', 's'):
            return data
        tz_offset = (
            get_local_timezone_offset_ns().to_scipp().to(unit=coord.unit, dtype='int64')
        )
        self._time_origin = sc.epoch(unit=coord.unit) + tz_offset
        return data.assign_coords({dim: self._time_origin + coord})


class WindowAggregatingExtractor(UpdateExtractor):
    """Extracts a window from the buffer and aggregates over the time dimension.

    The aggregation runs on every ``extract`` call; its cost scales with
    window length x element size and dominates extraction cost for image-sized
    data (milliseconds per call, vs. microseconds for a plain copy). This is
    why extraction is pull-based: it runs only when a displayed consumer
    pulls a ``DataService.snapshot``, not per ingested update.
    """

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

        # Capture time bounds before aggregation (which removes the time dimension)
        time_bounds = _extract_time_bounds_as_scalars(windowed_data)

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

        result = self._aggregator(windowed_data, self._concat_dim)

        # Restore time bounds as scalar coords on the aggregated result
        if time_bounds:
            result = result.assign_coords(**time_bounds)

        return result
