# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Temporal requirements for buffer management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class TemporalRequirement(ABC):
    """
    Base class for temporal coverage requirements.

    Temporal requirements describe what time-based coverage is needed,
    independent of frame rates or buffer sizing decisions.
    """

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the requirement."""


class LatestFrame(TemporalRequirement):
    """Requirement for only the most recent single data point."""

    def __repr__(self) -> str:
        """String representation."""
        return "LatestFrame()"


@dataclass(frozen=True)
class TimeWindow(TemporalRequirement):
    """
    Requirement for temporal coverage of specified duration.

    Attributes
    ----------
    duration_seconds:
        Time duration in seconds that must be covered by buffered data.
    """

    duration_seconds: float

    def __post_init__(self) -> None:
        """Validate duration."""
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

    def __repr__(self) -> str:
        """String representation."""
        return f"TimeWindow({self.duration_seconds}s)"


class CompleteHistory(TemporalRequirement):
    """
    Requirement for all available history.

    May have practical upper limit for memory constraints.
    """

    # Practical upper limit to prevent unbounded growth
    MAX_FRAMES = 10000

    def __repr__(self) -> str:
        """String representation."""
        return f"CompleteHistory(max={self.MAX_FRAMES})"
