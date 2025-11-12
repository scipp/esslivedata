# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Buffer manager for temporal requirement-based sizing."""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .buffer_strategy import Buffer, BufferFactory
from .temporal_requirements import (
    CompleteHistory,
    LatestFrame,
    TemporalRequirement,
    TimeWindow,
)

logger = logging.getLogger(__name__)

K = TypeVar('K', bound=Hashable)
T = TypeVar('T')

# Growth parameters
MAX_CAPACITY = 10000  # Upper limit to prevent runaway growth
GROWTH_FACTOR = 2.0  # Double buffer size when growing


@dataclass
class _BufferState(Generic[T]):
    """Internal state for a managed buffer."""

    buffer: Buffer[T]
    requirements: list[TemporalRequirement] = field(default_factory=list)
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
        self, key: K, template: T, requirements: list[TemporalRequirement]
    ) -> None:
        """
        Create a buffer sized to satisfy temporal requirements.

        Starts with size 1, will resize adaptively based on observations.

        Parameters
        ----------
        key:
            Key to identify this buffer.
        template:
            Sample data to determine buffer type.
        requirements:
            List of temporal requirements to satisfy.
        """
        if key in self._states:
            raise ValueError(f"Buffer with key {key} already exists")

        buffer = self._buffer_factory.create_buffer(template, max_size=1)
        state = _BufferState(buffer=buffer, requirements=list(requirements))
        # Compute initial needs_growth based on whether requirements are fulfilled
        state.needs_growth = any(
            not self._is_requirement_fulfilled(req, buffer) for req in requirements
        )
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

        # Check cached flag and resize if needed
        if state.needs_growth:
            state.needs_growth = self._compute_needs_growth(state)
            if state.needs_growth:
                self._resize_buffer(state)

        # Append data - buffer is properly sized
        state.buffer.append(data)

        # Recompute needs_growth after appending to validate requirements
        # with actual data. This catches configuration errors (e.g., TimeWindow
        # without time coordinate)
        if state.needs_growth:
            state.needs_growth = self._compute_needs_growth(state)

    def _compute_needs_growth(self, state: _BufferState[T]) -> bool:
        """
        Compute whether buffer needs to grow to satisfy requirements.

        Returns True if any requirement is unfulfilled AND buffer is not at capacity.

        Parameters
        ----------
        state:
            The buffer state to check.

        Returns
        -------
        :
            True if buffer should grow, False otherwise.
        """
        frame_count = state.buffer.get_frame_count()

        # Already at max capacity - don't grow further
        if frame_count >= MAX_CAPACITY:
            return False

        # Check if any requirement is unfulfilled
        for requirement in state.requirements:
            if not self._is_requirement_fulfilled(requirement, state.buffer):
                return True

        return False

    def _is_requirement_fulfilled(
        self, requirement: TemporalRequirement, buffer: Buffer[T]
    ) -> bool:
        """
        Check if a single requirement is satisfied by current buffer state.

        Parameters
        ----------
        requirement:
            The temporal requirement to check.
        buffer:
            The buffer to check against.

        Returns
        -------
        :
            True if requirement is satisfied, False otherwise.
        """
        if isinstance(requirement, LatestFrame):
            # Buffer always starts with max_size >= 1, sufficient for LatestFrame
            return True

        elif isinstance(requirement, TimeWindow):
            temporal_coverage = buffer.get_temporal_coverage()
            return temporal_coverage >= requirement.duration_seconds

        elif isinstance(requirement, CompleteHistory):
            # Complete history is never fulfilled - always want more data
            # Growth is limited by MAX_CAPACITY check in _compute_needs_growth
            return False

        return True

    def _resize_buffer(self, state: _BufferState[T]) -> None:
        """
        Resize buffer by doubling its size (capped at MAX_CAPACITY).

        Parameters
        ----------
        state:
            The buffer state to resize.
        """
        current_size = state.buffer.get_frame_count()

        # Double the size, capped at maximum
        new_size = min(int(current_size * GROWTH_FACTOR), MAX_CAPACITY)

        logger.debug(
            "Growing buffer from %d to %d frames",
            current_size,
            new_size,
        )
        state.buffer.set_max_size(new_size)

    def add_requirement(self, key: K, requirement: TemporalRequirement) -> None:
        """
        Register additional temporal requirement for an existing buffer.

        May trigger immediate resize if needed.

        Parameters
        ----------
        key:
            Key identifying the buffer to add requirement to.
        requirement:
            New temporal requirement.
        """
        if key not in self._states:
            raise KeyError(f"No buffer found for key {key}")

        state = self._states[key]
        state.requirements.append(requirement)

        # Check if resize needed immediately
        state.needs_growth = self._compute_needs_growth(state)
        if state.needs_growth:
            self._resize_buffer(state)

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
