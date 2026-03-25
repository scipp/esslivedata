# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Typed wrappers for nanosecond timestamps and durations.

Both :class:`Timestamp` and :class:`Duration` wrap a plain ``int`` (nanoseconds)
but are *not* ``int`` subclasses.  Arithmetic operators are defined only for
physically meaningful combinations — everything else raises ``TypeError``.
"""

from __future__ import annotations

import datetime
import time
from datetime import timezone
from functools import total_ordering
from typing import Any

_NS_PER_S = 1_000_000_000
_NS_PER_MS = 1_000_000


@total_ordering
class Duration:
    """A duration in nanoseconds."""

    __slots__ = ('_ns',)

    def __init__(self, *, ns: int) -> None:
        self._ns = int(ns)

    @classmethod
    def from_ns(cls, ns: int) -> Duration:
        """Create a duration from a value in nanoseconds."""
        return cls(ns=int(ns))

    @classmethod
    def from_seconds(cls, seconds: float) -> Duration:
        """Create a duration from a value in seconds."""
        return cls(ns=int(seconds * _NS_PER_S))

    @classmethod
    def from_ms(cls, ms: int) -> Duration:
        """Create a duration from a value in milliseconds."""
        return cls(ns=ms * _NS_PER_MS)

    @classmethod
    def from_scipp(cls, var: Any) -> Duration:
        """Create a duration from a scipp scalar with a time unit."""
        return cls(ns=int(var.to(unit='ns', copy=False).value))

    def to_ns(self) -> int:
        """Return the duration as an integer number of nanoseconds."""
        return self._ns

    def to_seconds(self) -> float:
        """Convert to seconds."""
        return self._ns / _NS_PER_S

    def to_scipp(self) -> Any:
        """Convert to a scipp scalar with unit 'ns'."""
        import scipp as sc

        return sc.scalar(self._ns, unit='ns')

    def __bool__(self) -> bool:
        return self._ns != 0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Duration):
            return self._ns == other._ns
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, Duration):
            return self._ns < other._ns
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._ns)

    def __repr__(self) -> str:
        return f"Duration(ns={self._ns})"

    def __neg__(self) -> Duration:
        return Duration(ns=-self._ns)

    # Duration + Duration -> Duration
    # Duration + Timestamp -> Timestamp  (commutative with Timestamp.__add__)
    def __add__(self, other: object) -> Duration | Timestamp:
        if isinstance(other, Timestamp):
            return Timestamp(ns=self._ns + other._ns)
        if isinstance(other, Duration):
            return Duration(ns=self._ns + other._ns)
        return NotImplemented

    __radd__ = __add__

    # Duration - Duration -> Duration
    def __sub__(self, other: object) -> Duration:
        if isinstance(other, Duration):
            return Duration(ns=self._ns - other._ns)
        return NotImplemented

    # Duration * int -> Duration
    def __mul__(self, other: object) -> Duration:
        if isinstance(other, int) and not isinstance(other, (Timestamp, Duration)):
            return Duration(ns=self._ns * other)
        return NotImplemented

    __rmul__ = __mul__

    # Duration // int -> Duration  (scale down)
    # Duration // Duration -> int  (dimensionless ratio)
    def __floordiv__(self, other: object) -> Duration | int:
        if isinstance(other, Duration):
            return self._ns // other._ns
        if isinstance(other, int) and not isinstance(other, (Timestamp, Duration)):
            return Duration(ns=self._ns // other)
        return NotImplemented

    # Duration / Duration -> float  (dimensionless ratio)
    def __truediv__(self, other: object) -> float:
        if isinstance(other, Duration):
            return self._ns / other._ns
        return NotImplemented

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(
            lambda v: cls(ns=v) if not isinstance(v, cls) else v,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda d: d._ns
            ),
        )


@total_ordering
class Timestamp:
    """Nanoseconds since Unix epoch (UTC)."""

    __slots__ = ('_ns',)

    def __init__(self, *, ns: int) -> None:
        self._ns = int(ns)

    @classmethod
    def from_ns(cls, ns: int) -> Timestamp:
        """Create a timestamp from a value in nanoseconds."""
        return cls(ns=int(ns))

    @classmethod
    def now(cls) -> Timestamp:
        """Create a timestamp for the current time."""
        return cls(ns=time.time_ns())

    @classmethod
    def from_seconds(cls, seconds: float) -> Timestamp:
        """Create a timestamp from seconds since the epoch."""
        return cls(ns=int(seconds * _NS_PER_S))

    @classmethod
    def from_ms(cls, ms: int) -> Timestamp:
        """Create a timestamp from milliseconds since the epoch."""
        return cls(ns=ms * _NS_PER_MS)

    @classmethod
    def from_scipp(cls, var: Any) -> Timestamp:
        """Create a timestamp from a scipp scalar with a time unit."""
        return cls(ns=int(var.to(unit='ns', copy=False).value))

    def to_ns(self) -> int:
        """Return the timestamp as an integer number of nanoseconds."""
        return self._ns

    def to_seconds(self) -> float:
        """Convert to seconds since the epoch."""
        return self._ns / _NS_PER_S

    def to_datetime(self, tz: timezone | None = None) -> datetime.datetime:
        """Convert to a timezone-aware datetime.

        Parameters
        ----------
        tz:
            Target timezone.  ``None`` keeps the result in UTC.
        """
        dt = datetime.datetime.fromtimestamp(
            self._ns / _NS_PER_S, tz=datetime.timezone.utc
        )
        return dt.astimezone(tz) if tz is not None else dt

    def to_scipp(self) -> Any:
        """Convert to a scipp scalar with unit 'ns'."""
        import scipp as sc

        return sc.scalar(self._ns, unit='ns')

    def quantize(self, period: Duration) -> Timestamp:
        """Round down to the nearest multiple of *period*."""
        p = period.to_ns()
        return Timestamp(ns=self._ns // p * p)

    def quantize_up(self, period: Duration) -> Timestamp:
        """Round up to the nearest multiple of *period*."""
        p = period.to_ns()
        return Timestamp(ns=-(-self._ns // p) * p)

    def __bool__(self) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Timestamp):
            return self._ns == other._ns
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, Timestamp):
            return self._ns < other._ns
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._ns)

    def __repr__(self) -> str:
        return f"Timestamp(ns={self._ns})"

    # Timestamp - Timestamp -> Duration
    # Timestamp - Duration -> Timestamp
    def __sub__(self, other: object) -> Duration | Timestamp:
        if isinstance(other, Timestamp):
            return Duration(ns=self._ns - other._ns)
        if isinstance(other, Duration):
            return Timestamp(ns=self._ns - other._ns)
        return NotImplemented

    # Timestamp + Duration -> Timestamp
    def __add__(self, other: object) -> Timestamp:
        if isinstance(other, Duration):
            return Timestamp(ns=self._ns + other._ns)
        return NotImplemented

    __radd__ = __add__

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(
            lambda v: cls(ns=v) if not isinstance(v, cls) else v,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda ts: ts._ns
            ),
        )
