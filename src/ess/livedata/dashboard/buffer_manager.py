# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Buffer manager for extractor requirement-based sizing."""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from .buffer import Buffer, BufferFactory

if TYPE_CHECKING:
    from .extractors import UpdateExtractor

logger = logging.getLogger(__name__)

K = TypeVar('K', bound=Hashable)
T = TypeVar('T')


@dataclass
class _BufferState(Generic[T]):
    """Internal state for a managed buffer."""

    buffer: Buffer[T]
    extractors: list[UpdateExtractor] = field(default_factory=list)
    needs_growth: bool = field(default=False)


class BufferManager(Mapping[K, Buffer[T]], Generic[K, T]):
    """
    Manages buffer sizing based on temporal requirements.

    Owns and manages buffers, translating temporal requirements (time-based)
    into spatial sizing decisions (frame counts) by observing actual buffer metrics.

    Implements Mapping interface for read-only dictionary-like access to buffers.
    """

    def __init__(self, buffer_factory: BufferFactory | None = None) -> None:
        """
        Initialize BufferManager.

        Parameters
        ----------
        buffer_factory:
            Factory for creating buffers. If None, uses default factory.
        """
        if buffer_factory is None:
            buffer_factory = BufferFactory()
        self._buffer_factory = buffer_factory
        self._states: dict[K, _BufferState[T]] = {}

    def __getitem__(self, key: K) -> Buffer[T]:
        """Get buffer for a key (Mapping interface)."""
        return self._states[key].buffer

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys (Mapping interface)."""
        return iter(self._states)

    def __len__(self) -> int:
        """Return number of buffers (Mapping interface)."""
        return len(self._states)

    def create_buffer(
        self, key: K, template: T, extractors: list[UpdateExtractor]
    ) -> None:
        """
        Create a buffer sized to satisfy extractor requirements.

        Starts with size 1, will resize adaptively based on observations.

        Parameters
        ----------
        key:
            Key to identify this buffer.
        template:
            Sample data to determine buffer type.
        extractors:
            List of extractors that will use this buffer.
        """
        if key in self._states:
            raise ValueError(f"Buffer with key {key} already exists")

        buffer = self._buffer_factory.create_buffer(template, max_size=1)
        state = _BufferState(buffer=buffer, extractors=list(extractors))
        # Compute initial needs_growth based on whether requirements are fulfilled
        state.needs_growth = self._compute_needs_growth(state)
        self._states[key] = state

    def update_buffer(self, key: K, data: T) -> None:
        """
        Update buffer with new data and apply retention policy.

        Checks requirements and resizes if needed BEFORE appending to prevent
        data loss from premature sliding window shifts.

        Parameters
        ----------
        key:
            Key identifying the buffer to update.
        data:
            New data to append.
        """
        if key not in self._states:
            raise KeyError(f"No buffer found for key {key}")

        state = self._states[key]

        # Check cached flag and grow if needed
        if state.needs_growth:
            state.needs_growth = self._compute_needs_growth(state)
            if state.needs_growth:
                self._grow_buffer(state)

        # Append data - buffer is properly sized
        state.buffer.append(data)

        # Recompute needs_growth after appending to validate requirements
        # with actual data. This catches configuration errors (e.g., TimeWindow
        # without time coordinate)
        if state.needs_growth:
            state.needs_growth = self._compute_needs_growth(state)

    def _compute_needs_growth(self, state: _BufferState[T]) -> bool:
        """
        Compute whether buffer needs to grow to satisfy extractor requirements.

        Returns True if any requirement is unfulfilled AND buffer can grow.

        Parameters
        ----------
        state:
            The buffer state to check.

        Returns
        -------
        :
            True if buffer should grow, False otherwise.
        """
        # Check if buffer can grow within memory budget
        if not state.buffer.can_grow():
            return False

        # Get all buffered data
        data = state.buffer.get_all()

        # Check if any extractor's requirements are unfulfilled
        for extractor in state.extractors:
            if not extractor.is_requirement_fulfilled(data):
                return True

        return False

    def _grow_buffer(self, state: _BufferState[T]) -> None:
        """
        Attempt to grow buffer.

        Parameters
        ----------
        state:
            The buffer state to grow.
        """
        if not state.buffer.grow():
            usage = state.buffer.get_memory_usage()
            logger.warning(
                "Buffer growth failed - at memory budget limit (usage: %d bytes)",
                usage,
            )

    def add_extractor(self, key: K, extractor: UpdateExtractor) -> None:
        """
        Register additional extractor for an existing buffer.

        May trigger immediate resize if needed.

        Parameters
        ----------
        key:
            Key identifying the buffer to add extractor to.
        extractor:
            New extractor that will use this buffer.
        """
        if key not in self._states:
            raise KeyError(f"No buffer found for key {key}")

        state = self._states[key]
        state.extractors.append(extractor)

        # Check if growth needed immediately
        state.needs_growth = self._compute_needs_growth(state)
        if state.needs_growth:
            self._grow_buffer(state)

    def delete_buffer(self, key: K) -> None:
        """
        Delete a buffer and its associated state.

        Parameters
        ----------
        key:
            Key identifying the buffer to delete.
        """
        if key in self._states:
            del self._states[key]
