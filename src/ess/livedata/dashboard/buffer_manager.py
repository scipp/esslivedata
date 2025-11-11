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
    needs_growth: bool = True


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
        self._states[key] = _BufferState(
            buffer=buffer, requirements=list(requirements), needs_growth=True
        )

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

        # Cheap flag check - only validate/resize if growth is still needed
        if state.needs_growth and state.requirements:
            if not self._validate_coverage(key):
                self._resize_buffer(key)
            else:
                # Target coverage reached - disable further checks for efficiency
                state.needs_growth = False

        # Append data - buffer is properly sized
        state.buffer.append(data)

    def _validate_coverage(self, key: K) -> bool:
        """
        Check if buffer currently provides sufficient coverage.

        Parameters
        ----------
        key:
            Key identifying the buffer to validate.

        Returns
        -------
        :
            True if buffer satisfies all requirements, False otherwise.
        """
        state = self._states[key]
        temporal_coverage = state.buffer.get_temporal_coverage()
        frame_count = state.buffer.get_frame_count()

        for requirement in state.requirements:
            if isinstance(requirement, LatestFrame):
                if frame_count < 1:
                    return False
            elif isinstance(requirement, TimeWindow):
                # For temporal requirements, check actual time coverage
                if temporal_coverage is not None:
                    # Need at least 2 frames to calculate temporal coverage
                    if frame_count < 2:
                        return False
                    if temporal_coverage < requirement.duration_seconds:
                        return False
                else:
                    # No time coordinate - use heuristic (assume 10 Hz)
                    # Buffer should have at least duration * 10 frames
                    expected_frames = max(100, int(requirement.duration_seconds * 10))
                    if frame_count < expected_frames:
                        return False
            elif isinstance(requirement, CompleteHistory):
                # For complete history, buffer should grow until MAX_FRAMES
                if frame_count < requirement.MAX_FRAMES:
                    # Not yet at maximum capacity, should resize
                    return False

        return True

    def _resize_buffer(self, key: K) -> None:
        """
        Resize buffer to satisfy requirements.

        Parameters
        ----------
        key:
            Key identifying the buffer to resize.
        """
        state = self._states[key]
        current_size = state.buffer.get_frame_count()
        temporal_coverage = state.buffer.get_temporal_coverage()

        # Calculate new size based on requirements
        new_size = current_size

        for requirement in state.requirements:
            if isinstance(requirement, TimeWindow):
                if temporal_coverage is not None and temporal_coverage > 0:
                    # We have time coverage - calculate needed frames
                    frames_per_second = current_size / temporal_coverage
                    # 20% headroom
                    needed_frames = int(
                        requirement.duration_seconds * frames_per_second * 1.2
                    )
                    new_size = max(new_size, needed_frames)
                else:
                    # No time coverage - use heuristic (assume 10 Hz)
                    target_frames = max(100, int(requirement.duration_seconds * 10))
                    new_size = max(new_size, target_frames)
            elif isinstance(requirement, CompleteHistory):
                # Grow towards max
                new_size = max(new_size, int(current_size * GROWTH_FACTOR))

        # Cap at maximum and ensure we actually grow
        new_size = min(max(new_size, int(current_size * GROWTH_FACTOR)), MAX_CAPACITY)

        if new_size > current_size:
            logger.debug(
                "Resizing buffer %s from %d to %d frames (coverage: %s s)",
                key,
                current_size,
                new_size,
                temporal_coverage,
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
        state.needs_growth = True  # Re-enable growth checks

        # Check if resize needed immediately
        if not self._validate_coverage(key):
            self._resize_buffer(key)

    def get_buffer(self, key: K) -> Buffer[T]:
        """
        Get buffer for a key.

        Parameters
        ----------
        key:
            Key identifying the buffer.

        Returns
        -------
        :
            The buffer for this key.

        Notes
        -----
        Prefer using dictionary access: `buffer_manager[key]` instead of
        `buffer_manager.get_buffer(key)`.
        """
        return self[key]

    def has_buffer(self, key: K) -> bool:
        """
        Check if a buffer exists for a key.

        Parameters
        ----------
        key:
            Key to check.

        Returns
        -------
        :
            True if buffer exists for this key.

        Notes
        -----
        Prefer using membership test: `key in buffer_manager` instead of
        `buffer_manager.has_buffer(key)`.
        """
        return key in self

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
