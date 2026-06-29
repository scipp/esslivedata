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
from ..core.timestamp import Timestamp
from .to_nxevent_data import MonitorEvents

__all__ = ["MonitorEvents"]

T = TypeVar('T')


@dataclass
class LogData:
    """
    Dataclass for log data.

    Decouples our handlers from upstream schema changes. This also simplifies handler
    testing since tests do not have to construct a full logdata_f144.LogData object.

    ``target`` and ``idle`` are populated only by the device-synthesis path
    (:class:`DeviceSynthesizer`) when a :class:`~..config.stream.Device` declares
    the corresponding substreams. They are ``None`` for plain f144 logs.
    """

    time: int
    value: Any
    variances: Any | None = None
    target: float | None = None
    idle: bool | None = None

    @staticmethod
    def from_f144(f144: logdata_f144.LogData) -> LogData:
        return LogData(time=f144.timestamp_unix_ns, value=f144.value)


class NullAccumulator(Accumulator[Any, None]):
    def add(self, timestamp: Timestamp, data: Any) -> bool:
        return True

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

    is_context = True

    def __init__(self):
        self._latest: sc.DataArray | None = None

    def add(self, timestamp: Timestamp, data: sc.DataArray) -> bool:
        _ = timestamp
        self._latest = data.copy()
        return True

    def get(self) -> sc.DataArray:
        if self._latest is None:
            raise ValueError("No data has been added")
        return self._latest

    def clear(self) -> None:
        self._latest = None


class NoCopyAccumulator(EternalAccumulator):
    """
    Accumulator that skips the deepcopy on read to avoid copy overhead.

    The base EternalAccumulator uses deepcopy in _get_value() to ensure safety.
    This accumulator skips that deepcopy, saving ~30ms per read for a 500MB
    histogram.

    The copy on first push is retained to avoid shared references when the same
    value is pushed to multiple accumulators.

    Use only when downstream consumers do not modify or store references to
    the returned value. This constraint is met in streaming workflows
    where downstream just serializes the data.

    When ``reset_coord`` is given the buffer is discarded whenever incoming data
    carries a different value for that scalar coord, so accumulation restarts in
    the new configuration rather than summing across a geometry change. The
    histogram carries a coord identifying the geometry it was
    computed in (e.g. ``position``/``Ltotal`` for monitors, the detector
    transform for detector views). An absent coord — or ``reset_coord=None`` — is
    a no-op, so behaviour matches a plain cumulative accumulator. The check
    short-circuits before the first push and whenever the coord is unchanged, so
    it adds no measurable overhead.
    """

    def __init__(self, *, reset_coord: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._reset_coord = reset_coord

    def _reset_if_geometry_changed(self, value: Any) -> None:
        coord = self._reset_coord
        if coord is None or self._value is None:
            return
        new = getattr(value, 'coords', {})
        stored = getattr(self._value, 'coords', {})
        if (
            coord in new
            and coord in stored
            and not sc.identical(new[coord], stored[coord])
        ):
            self._value = None

    def _do_push(self, value: sc.DataArray) -> None:
        self._reset_if_geometry_changed(value)
        super()._do_push(value)

    def _get_value(self):
        """Return value directly without deepcopy."""
        return self._value


class _NoCopyWindowAccumulator(NoCopyAccumulator):
    """Window accumulator without deepcopy that clears after finalize.

    Skips the deepcopy on push that the base EternalAccumulator performs. This is
    safe only when paired with exactly one NoCopyAccumulator consuming the same input:
    the NoCopyAccumulator deepcopies on its first push, isolating its buffer, and
    ``+=`` only mutates the left operand (``self._value``), never the right (the input).

    Inherits ``reset_coord`` handling from :class:`NoCopyAccumulator`. The current
    driving loop pushes exactly once per finalize, so the reset never fires here in
    practice; carrying it regardless keeps the window correct (discard a partial
    window straddling a move) should that 1:1 relation ever change.

    Must not be constructed directly — use :func:`make_no_copy_accumulator_pair`.
    """

    def _do_push(self, value: T) -> None:
        self._reset_if_geometry_changed(value)
        if self._value is None:
            self._value = value
        else:
            self._value += value

    def on_finalize(self) -> None:
        """Clear accumulated value after finalize retrieves it."""
        self.clear()


def make_no_copy_accumulator_pair(
    *, reset_coord: str | None = None
) -> tuple[NoCopyAccumulator, _NoCopyWindowAccumulator]:
    """Create a paired cumulative/window accumulator that skips redundant copies.

    The two accumulators are designed to receive the same shared input. The cumulative
    accumulator (NoCopyAccumulator) deepcopies on its first push, isolating its buffer.
    The window accumulator (_NoCopyWindowAccumulator) stores a bare reference and clears
    after each finalize cycle.

    This pairing is what makes the no-copy optimization safe: because exactly one
    NoCopyAccumulator always deepcopies, no consumer ever mutates the shared input.

    Parameters
    ----------
    reset_coord:
        If given, both accumulators reset whenever incoming data carries a
        different value for this scalar coord, rather than summing across a
        geometry change. ``None`` keeps plain cumulative
        behaviour.

    Returns
    -------
    :
        ``(cumulative, window)`` accumulator pair.
    """
    return (
        NoCopyAccumulator(reset_coord=reset_coord),
        _NoCopyWindowAccumulator(reset_coord=reset_coord),
    )


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
        """Add data to the cumulative accumulation.

        Restarts the accumulation from a copy whenever incoming data is
        structurally incompatible with it: a changed shape, a changed set of
        coordinates, or changed coordinate values (e.g. rebinning or an upstream
        reconfiguration). Otherwise the data is summed onto the accumulation.

        ``+=`` already validates compatibility in fused C++, so it is the sole
        check: any structural mismatch raises (``RuntimeError`` covers
        dimension/coordinate/unit/variances errors, ``TypeError`` covers dtype),
        and we restart rather than letting the accumulator get stuck rejecting
        every subsequent batch. A coordinate *present only in the accumulation*
        does not raise (``+=`` keeps it); coordinate presence is assumed stable
        per stream, which holds for the histogram and image sources that use
        this accumulator.
        """
        if self._cumulative is None:
            self._cumulative = data.copy()
            return
        try:
            self._cumulative += data
        except (RuntimeError, TypeError):
            self._cumulative = data.copy()

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

    def add(self, timestamp: Timestamp, data: sc.DataArray) -> bool:
        _ = timestamp
        self._add_cumulative(data)
        return True
