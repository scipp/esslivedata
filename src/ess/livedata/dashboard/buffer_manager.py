# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Buffer manager for temporal requirement-based sizing."""

from __future__ import annotations

import logging
from typing import Generic, TypeVar

from .buffer_strategy import Buffer, BufferFactory
from .temporal_requirements import (
    CompleteHistory,
    LatestFrame,
    TemporalRequirement,
    TimeWindow,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Growth parameters
INITIAL_CAPACITY = 100  # Conservative default for new buffers
MAX_CAPACITY = 10000  # Upper limit to prevent runaway growth
GROWTH_FACTOR = 2.0  # Double buffer size when growing


class BufferManager(Generic[T]):
    """
    Manages buffer sizing based on temporal requirements.

    Translates temporal requirements (time-based) into spatial sizing decisions
    (frame counts) by observing actual buffer metrics.
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
        self._requirements: dict[int, list[TemporalRequirement]] = {}

    def create_buffer(
        self, template: T, requirements: list[TemporalRequirement]
    ) -> Buffer[T]:
        """
        Create a buffer sized to satisfy temporal requirements.

        Starts with conservative default size, will resize based on observations.

        Parameters
        ----------
        template:
            Sample data to determine buffer type.
        requirements:
            List of temporal requirements to satisfy.

        Returns
        -------
        :
            Newly created buffer.
        """
        initial_size = self._calculate_initial_size(requirements)
        buffer = self._buffer_factory.create_buffer(template, max_size=initial_size)
        self._requirements[id(buffer)] = list(requirements)
        return buffer

    def _calculate_initial_size(self, requirements: list[TemporalRequirement]) -> int:
        """
        Calculate initial buffer size from temporal requirements.

        Uses conservative estimates since actual frame rate is unknown.

        Parameters
        ----------
        requirements:
            List of temporal requirements.

        Returns
        -------
        :
            Initial buffer size in frames.
        """
        max_size = 1

        for requirement in requirements:
            if isinstance(requirement, LatestFrame):
                max_size = max(max_size, 1)
            elif isinstance(requirement, TimeWindow):
                # Conservative: assume 10 Hz for initial allocation
                # Will grow based on actual observations
                estimated_frames = max(
                    INITIAL_CAPACITY, int(requirement.duration_seconds * 10)
                )
                max_size = max(max_size, min(estimated_frames, MAX_CAPACITY))
            elif isinstance(requirement, CompleteHistory):
                max_size = max(max_size, requirement.MAX_FRAMES)

        return max_size

    def update_buffer(self, buffer: Buffer[T], data: T) -> None:
        """
        Update buffer with new data and apply retention policy.

        Appends data, observes metrics, and resizes if needed to meet requirements.

        Parameters
        ----------
        buffer:
            Buffer to update.
        data:
            New data to append.
        """
        # Append data first
        buffer.append(data)

        # Get requirements for this buffer
        buffer_id = id(buffer)
        requirements = self._requirements.get(buffer_id, [])

        if not requirements:
            return

        # Check if buffer meets requirements, resize if needed
        if not self.validate_coverage(buffer, requirements):
            self._resize_buffer(buffer, requirements)

    def validate_coverage(
        self, buffer: Buffer[T], requirements: list[TemporalRequirement]
    ) -> bool:
        """
        Check if buffer currently provides sufficient coverage.

        Parameters
        ----------
        buffer:
            Buffer to validate.
        requirements:
            List of temporal requirements to check.

        Returns
        -------
        :
            True if buffer satisfies all requirements, False otherwise.
        """
        temporal_coverage = buffer.get_temporal_coverage()
        frame_count = buffer.get_frame_count()

        for requirement in requirements:
            if isinstance(requirement, LatestFrame):
                if frame_count < 1:
                    return False
            elif isinstance(requirement, TimeWindow):
                # For temporal requirements, check actual time coverage
                if temporal_coverage is None:
                    # No time coordinate - can't validate temporal requirement yet
                    # Buffer will grow adaptively based on frame count
                    return True
                if temporal_coverage < requirement.duration_seconds:
                    return False
            elif isinstance(requirement, CompleteHistory):
                # For complete history, buffer should grow until MAX_FRAMES
                if frame_count < requirement.MAX_FRAMES:
                    # Not yet at maximum capacity, should resize
                    return False

        return True

    def _resize_buffer(
        self, buffer: Buffer[T], requirements: list[TemporalRequirement]
    ) -> None:
        """
        Resize buffer to satisfy requirements.

        Parameters
        ----------
        buffer:
            Buffer to resize.
        requirements:
            List of temporal requirements to satisfy.
        """
        current_size = buffer.get_frame_count()
        temporal_coverage = buffer.get_temporal_coverage()

        # Calculate new size based on requirements
        new_size = current_size

        for requirement in requirements:
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
                    # No time coverage yet - grow by factor
                    new_size = max(new_size, int(current_size * GROWTH_FACTOR))
            elif isinstance(requirement, CompleteHistory):
                # Grow towards max
                new_size = max(new_size, int(current_size * GROWTH_FACTOR))

        # Cap at maximum and ensure we actually grow
        new_size = min(max(new_size, int(current_size * GROWTH_FACTOR)), MAX_CAPACITY)

        if new_size > current_size:
            logger.debug(
                "Resizing buffer from %d to %d frames (coverage: %s s)",
                current_size,
                new_size,
                temporal_coverage,
            )
            buffer.set_max_size(new_size)

    def add_requirement(
        self, buffer: Buffer[T], requirement: TemporalRequirement
    ) -> None:
        """
        Register additional temporal requirement for an existing buffer.

        May trigger immediate resize if needed.

        Parameters
        ----------
        buffer:
            Buffer to add requirement to.
        requirement:
            New temporal requirement.
        """
        buffer_id = id(buffer)
        if buffer_id not in self._requirements:
            self._requirements[buffer_id] = []

        self._requirements[buffer_id].append(requirement)

        # Check if resize needed immediately
        if not self.validate_coverage(buffer, self._requirements[buffer_id]):
            self._resize_buffer(buffer, self._requirements[buffer_id])
