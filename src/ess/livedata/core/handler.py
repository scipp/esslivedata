# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import ClassVar, Generic, Protocol, TypeVar

from ..config.instrument import Instrument
from .message import StreamId, Tin, Tout
from .timestamp import Timestamp

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


class Accumulator(Protocol, Generic[T, U]):
    """
    Protocol for an accumulator that accumulates data over time.

    Accumulators are used as preprocessors in the message processing pipeline.
    They accumulate data from multiple messages before passing it to workflows.

    The ``get()`` contract varies by implementation: batch-oriented accumulators
    (e.g., ``ToNXevent_data``, ``Cumulative``) may consume or clear their
    internal state on ``get()``, while context-oriented accumulators (e.g.,
    ``ToNXlog``, ``LatestValueHandler``) retain it. Implementations declare
    which category they belong to via the ``is_context`` class variable.

    Context accumulators (``is_context = True``) are idempotent on ``get()`` —
    calling it multiple times returns the same data without consuming or
    clearing it. This property is used by ``MessagePreprocessor.get_context()``
    to safely read current values for seeding newly activated jobs.
    """

    is_context: ClassVar[bool] = False

    def add(self, timestamp: Timestamp, data: T) -> None: ...

    def get(self) -> U: ...

    def clear(self) -> None: ...


class PreprocessorFactory(Protocol, Generic[Tin, Tout]):
    """
    Factory for creating preprocessors (accumulators) for message streams.

    Preprocessors accumulate and transform messages before they are passed to workflows
    for final processing.
    """

    def make_preprocessor(self, key: StreamId) -> Accumulator | None:
        """Create a preprocessor for the given stream, or None to skip."""
        return None


class JobBasedPreprocessorFactoryBase(PreprocessorFactory[Tin, Tout]):
    """Factory base used by job-based backend services."""

    def __init__(self, *, instrument: Instrument) -> None:
        self._instrument = instrument

    @property
    def instrument(self) -> Instrument:
        return self._instrument
