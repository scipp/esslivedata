# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Buffer manager with temporal buffer support."""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from .extractors import LatestValueExtractor
from .temporal_buffers import BufferProtocol, SingleValueBuffer, TemporalBuffer

if TYPE_CHECKING:
    from .extractors import UpdateExtractor

logger = logging.getLogger(__name__)

K = TypeVar('K', bound=Hashable)
T = TypeVar('T')


@dataclass
class _BufferState(Generic[T]):
    """Internal state for a managed buffer."""

    buffer: BufferProtocol[T]
    extractors: list[UpdateExtractor] = field(default_factory=list)


class TemporalBufferManager(Mapping[K, BufferProtocol[T]], Generic[K, T]):
    """
    Manages buffers, switching between SingleValueBuffer and TemporalBuffer.

    Decides buffer type based on extractors:
    - All LatestValueExtractor → SingleValueBuffer (efficient)
    - Otherwise → TemporalBuffer (temporal data with time dimension)

    Implements Mapping interface for read-only dictionary-like access to buffers.
    Use get_buffered_data() for convenient access to buffered data.
    """

    def __init__(self) -> None:
        """Initialize TemporalBufferManager."""
        self._states: dict[K, _BufferState[T]] = {}

    def __getitem__(self, key: K) -> BufferProtocol[T]:
        """Return the buffer for a key."""
        return self._states[key].buffer

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys."""
        return iter(self._states)

    def __len__(self) -> int:
        """Return number of buffers."""
        return len(self._states)

    def get_buffered_data(self, key: K) -> T | None:
        """
        Get data from buffer if available.

        Returns None if buffer doesn't exist or if buffer is empty.
        Never raises KeyError - treats "buffer not found" and "buffer empty"
        the same way for convenience.

        Parameters
        ----------
        key:
            Key identifying the buffer.

        Returns
        -------
        :
            Buffered data, or None if unavailable.
        """
        if key not in self._states:
            return None
        return self._states[key].buffer.get()

    def create_buffer(self, key: K, extractors: list[UpdateExtractor]) -> None:
        """
        Create a buffer with appropriate type based on extractors.

        Parameters
        ----------
        key:
            Key to identify this buffer.
        extractors:
            List of extractors that will use this buffer.
        """
        if key in self._states:
            raise ValueError(f"Buffer with key {key} already exists")

        buffer = self._create_buffer_for_extractors(extractors)
        self._update_buffer_requirements(buffer, extractors)
        state = _BufferState(buffer=buffer, extractors=list(extractors))
        self._states[key] = state

    def update_buffer(self, key: K, data: T) -> None:
        """
        Update buffer with new data.

        Parameters
        ----------
        key:
            Key identifying the buffer to update.
        data:
            New data to add.
        """
        if key not in self._states:
            raise KeyError(f"No buffer found for key {key}")

        state = self._states[key]
        state.buffer.add(data)

    def add_extractor(self, key: K, extractor: UpdateExtractor) -> None:
        """
        Register additional extractor for an existing buffer.

        May trigger buffer type switch with data migration:
        - Single→Temporal: Existing data is copied to the new buffer
        - Temporal→Single: Last time slice is copied to the new buffer
        - Other transitions: Data is discarded

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

        # Check if we need to switch buffer type
        new_buffer = self._create_buffer_for_extractors(state.extractors)
        if not isinstance(new_buffer, type(state.buffer)):
            # Handle data migration for Single->Temporal transition
            if isinstance(state.buffer, SingleValueBuffer) and isinstance(
                new_buffer, TemporalBuffer
            ):
                logger.info(
                    "Switching buffer type from %s to %s for key %s (copying data)",
                    type(state.buffer).__name__,
                    type(new_buffer).__name__,
                    key,
                )
                # Copy existing data to new buffer
                old_data = state.buffer.get()
                if old_data is not None:
                    new_buffer.add(old_data)
            # Handle data migration for Temporal->Single transition
            elif isinstance(state.buffer, TemporalBuffer) and isinstance(
                new_buffer, SingleValueBuffer
            ):
                logger.info(
                    "Switching buffer type from %s to %s for key %s"
                    " (copying last slice)",
                    type(state.buffer).__name__,
                    type(new_buffer).__name__,
                    key,
                )
                # Copy last slice to new buffer
                old_data = state.buffer.get()
                if old_data is not None and 'time' in old_data.dims:
                    last_slice = old_data['time', -1]
                    new_buffer.add(last_slice)
            else:
                logger.info(
                    "Switching buffer type from %s to %s for key %s"
                    " (discarding old data)",
                    type(state.buffer).__name__,
                    type(new_buffer).__name__,
                    key,
                )
            state.buffer = new_buffer

        # Update buffer requirements
        self._update_buffer_requirements(state.buffer, state.extractors)

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

    def _create_buffer_for_extractors(
        self, extractors: list[UpdateExtractor]
    ) -> BufferProtocol[T]:
        """
        Create appropriate buffer type based on extractors.

        If all extractors are LatestValueExtractor, use SingleValueBuffer.
        Otherwise, use TemporalBuffer.

        Parameters
        ----------
        extractors:
            List of extractors that will use the buffer.

        Returns
        -------
        :
            New buffer instance of appropriate type.
        """
        if not extractors:
            # No extractors - default to SingleValueBuffer
            return SingleValueBuffer()

        # Check if all extractors are LatestValueExtractor
        all_latest = all(isinstance(e, LatestValueExtractor) for e in extractors)

        if all_latest:
            return SingleValueBuffer()
        else:
            return TemporalBuffer()  # type: ignore[return-value]

    def _update_buffer_requirements(
        self, buffer: BufferProtocol[T], extractors: list[UpdateExtractor]
    ) -> None:
        """
        Update buffer requirements based on extractors.

        Computes the maximum required timespan from all extractors and sets it
        on the buffer.

        Parameters
        ----------
        buffer:
            The buffer to update.
        extractors:
            List of extractors to gather requirements from.
        """
        # Compute maximum required timespan
        timespans = [
            ts for e in extractors if (ts := e.get_required_timespan()) is not None
        ]
        if timespans:
            max_timespan = max(timespans)
            buffer.set_required_timespan(max_timespan)
            logger.debug(
                "Set buffer required timespan to %.2f seconds (from %d extractors)",
                max_timespan,
                len(extractors),
            )
