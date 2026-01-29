# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from ..config.instrument import Instrument
from .message import StreamId, Tin, Tout

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


class Accumulator(Protocol, Generic[T, U]):
    """
    Protocol for an accumulator that accumulates data over time.

    Accumulators are used as preprocessors in the message processing pipeline.
    They accumulate data from multiple messages before passing it to workflows.
    """

    def add(self, timestamp: int, data: T) -> None: ...

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
