# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import scipp as sc
from ess.reduce import streaming
from ess.reduce.streaming import EternalAccumulator
from streaming_data_types import logdata_f144

from ..core.handler import Accumulator
from .to_nxevent_data import MonitorEvents

__all__ = ["MonitorEvents"]

T = TypeVar('T')


@dataclass
class LogData:
    """
    Dataclass for log data.

    Decouples our handlers from upstream schema changes. This also simplifies handler
    testing since tests do not have to construct a full logdata_f144.LogData object.
    """

    time: int
    value: Any
    variances: Any | None = None

    @staticmethod
    def from_f144(f144: logdata_f144.LogData) -> LogData:
        return LogData(time=f144.timestamp_unix_ns, value=f144.value)


class NullAccumulator(Accumulator[Any, None]):
    def add(self, timestamp: int, data: Any) -> None:
        pass

    def get(self) -> None:
        return None

    def clear(self) -> None:
        pass


class LatestValueHandler(Accumulator[sc.DataArray, sc.DataArray]):
    """
    Handler-style accumulator that keeps only the latest value.

    This implements the handler Accumulator protocol (add/get/clear) for use in
    message handlers. For use with StreamProcessor workflows, use LatestValue instead.

    Unlike Cumulative, this does not add values together - it simply replaces
    the stored value with each new addition. Useful for configuration data like ROI
    where only the current state matters.
    """

    def __init__(self):
        self._latest: sc.DataArray | None = None

    def add(self, timestamp: int, data: sc.DataArray) -> None:
        _ = timestamp
        self._latest = data.copy()

    def get(self) -> sc.DataArray:
        if self._latest is None:
            raise ValueError("No data has been added")
        return self._latest

    def clear(self) -> None:
        self._latest = None


class NoCopyAccumulator(EternalAccumulator):
    """
    Accumulator that skips deepcopy on read for better performance.

    The base EternalAccumulator uses deepcopy in _get_value() to ensure safety.
    This accumulator skips that deepcopy, saving ~30ms per read for a 500MB
    histogram.

    The copy on first push is retained to avoid shared references when the same
    value is pushed to multiple accumulators.

    Use only when downstream consumers do not modify or store references to
    the returned value. This constraint is met in streaming workflows
    where downstream just serializes the data.
    """

    def _get_value(self):
        """Return value directly without deepcopy."""
        return self._value


class NoCopyWindowAccumulator(NoCopyAccumulator):
    """
    Window accumulator without deepcopy that clears after finalize.

    Combines the performance benefits of NoCopyAccumulator with window semantics
    (clearing after each finalize cycle).
    """

    def on_finalize(self) -> None:
        """Clear accumulated value after finalize retrieves it."""
        self.clear()


class LatestValue(streaming.Accumulator[T], Generic[T]):
    """
    Streaming accumulator that keeps only the latest value.

    This implements the ess.reduce.streaming.Accumulator protocol (push/value/clear)
    for use with StreamProcessor workflows. Unlike EternalAccumulator, this does not
    accumulate values - it simply replaces the stored value with each new push.

    Useful for scalar outputs like detector region counts where accumulation
    doesn't make sense.
    """

    def __init__(self) -> None:
        super().__init__()
        self._value: T | None = None

    def _do_push(self, value: T) -> None:
        self._value = value

    def _get_value(self) -> T:
        # is_empty check is handled by the base class value property
        return self._value  # type: ignore[return-value]

    @property
    def is_empty(self) -> bool:
        return self._value is None

    def clear(self) -> None:
        self._value = None


class _CumulativeAccumulationMixin:
    """Mixin providing cumulative data accumulation functionality."""

    def __init__(self, clear_on_get: bool = False):
        self._clear_on_get = clear_on_get
        self._cumulative: sc.DataArray | None = None

    def _add_cumulative(self, data: sc.DataArray) -> None:
        """Add data to the cumulative accumulation."""
        if self._cumulative is None or data.sizes != self._cumulative.sizes:
            self._cumulative = data.copy()
        elif data.ndim == 1 and data.dim in data.coords:
            # Check if coordinate changed (e.g., rebinning)
            if not sc.identical(
                data.coords[data.dim], self._cumulative.coords[data.dim]
            ):
                self._cumulative = data.copy()
            else:
                self._cumulative += data
        else:
            # For multi-dimensional data or data without coordinates, just accumulate
            self._cumulative += data

    def _get_cumulative(self) -> sc.DataArray:
        """Get the current cumulative data."""
        if self._cumulative is None:
            raise ValueError("No data has been added")
        return self._cumulative

    def clear(self) -> None:
        """Clear the cumulative data."""
        self._cumulative = None

    def _compute_result(self, cumulative: sc.DataArray) -> sc.DataArray:
        """Compute the final result from cumulative data. Override in subclasses."""
        return cumulative

    def get(self) -> sc.DataArray:
        """Get the accumulated result, optionally clearing data if configured."""
        cumulative = self._get_cumulative()
        result = self._compute_result(cumulative)
        if self._clear_on_get:
            self.clear()
        return result


class Cumulative(_CumulativeAccumulationMixin, Accumulator[sc.DataArray, sc.DataArray]):
    def __init__(self, config: dict | None = None, clear_on_get: bool = False):
        super().__init__(clear_on_get=clear_on_get)
        self._config = config or {}

    def add(self, timestamp: int, data: sc.DataArray) -> None:
        _ = timestamp
        self._add_cumulative(data)
