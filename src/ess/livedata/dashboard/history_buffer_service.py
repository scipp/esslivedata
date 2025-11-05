# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Service for maintaining historical buffers of DataService data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Generic, TypeVar

import scipp as sc

from .buffer import Buffer
from .buffer_strategy import GrowingStorage, SlidingWindowStorage
from .data_service import DataService

K = TypeVar("K", bound=Hashable)


class UpdateExtractor(ABC):
    """Extracts a specific view of buffer data."""

    @abstractmethod
    def extract(self, buffer: Buffer) -> sc.DataArray | None:
        """
        Extract data from a buffer.

        Parameters
        ----------
        buffer:
            The buffer to extract data from.

        Returns
        -------
        :
            The extracted data, or None if no data available.
        """


class FullHistoryExtractor(UpdateExtractor):
    """Extracts the complete buffer history."""

    def extract(self, buffer: Buffer) -> sc.DataArray | None:
        return buffer.get_buffer()


class WindowExtractor(UpdateExtractor):
    """Extracts a window from the end of the buffer."""

    def __init__(self, size: int | None = None) -> None:
        """
        Initialize window extractor.

        Parameters
        ----------
        size:
            Number of elements to extract from the end of the buffer.
            If None, extracts the entire buffer.
        """
        self._size = size

    def extract(self, buffer: Buffer) -> sc.DataArray | None:
        return buffer.get_window(self._size)


class DeltaExtractor(UpdateExtractor):
    """Extracts only data added since last extraction."""

    def __init__(self) -> None:
        # Track the last size we saw for each buffer
        self._last_sizes: dict[int, int] = {}

    def extract(self, buffer: Buffer) -> sc.DataArray | None:
        # TODO: Implement delta tracking properly
        # For now, just return full buffer
        # Need to track buffer state between calls
        return buffer.get_buffer()


class HistorySubscriber(ABC, Generic[K]):
    """
    Protocol for subscribers to HistoryBufferService.

    Subscribers specify what data they need per key via UpdateExtractors
    and receive batched updates for all relevant keys.
    """

    @property
    def keys(self) -> set[K]:
        """Return the set of buffer keys this subscriber depends on."""
        # TODO How can we avoid rebuilding the set every time DataService calls this?
        return set(self.extractors)

    @property
    @abstractmethod
    def extractors(self) -> dict[K, UpdateExtractor]:
        """
        Return the extractors to use for obtaining buffer data.

        Returns a mapping from key to the extractor to use for that key.
        Keys not in this dict will use a default FullHistoryExtractor.
        """

    @abstractmethod
    def on_update(self, data: dict[K, sc.DataArray]) -> None:
        """
        Called when subscribed buffers are updated.

        Parameters
        ----------
        data:
            Dictionary mapping keys to extracted buffer data.
            Only includes keys that were updated and are in self.keys.
        """


class _InternalDataSubscriber(Generic[K]):
    """Internal subscriber to connect HistoryBufferService to DataService."""

    def __init__(self, buffer_service: HistoryBufferService[K]):
        self._buffer_service = buffer_service

    @property
    def keys(self) -> set[K]:
        """Return the keys currently registered in the buffer service."""
        return self._buffer_service.get_tracked_keys()

    def trigger(self, store: dict[K, sc.DataArray]) -> None:
        """
        Process updates from DataService.

        Parameters
        ----------
        store:
            Dictionary of updated data from DataService.
        """
        self._buffer_service.process_data_service_update(store)


class HistoryBufferService(Generic[K]):
    """
    Service for maintaining historical buffers of data from DataService.

    Each subscriber gets its own set of buffers for the keys it needs.
    """

    def __init__(
        self,
        data_service: DataService[K, sc.DataArray],
    ) -> None:
        """
        Initialize the history buffer service.

        Parameters
        ----------
        data_service:
            The DataService to subscribe to.
        """
        self._data_service = data_service
        # Each subscriber has its own buffers for its keys
        self._buffers: dict[HistorySubscriber[K], dict[K, Buffer]] = {}

        # Subscribe to DataService
        self._internal_subscriber = _InternalDataSubscriber(self)
        self._data_service.register_subscriber(self._internal_subscriber)

    def get_tracked_keys(self) -> set[K]:
        """
        Return all keys that should be tracked from DataService.

        Returns the union of all keys from all registered subscribers.
        """
        all_keys: set[K] = set()
        for subscriber in self._buffers:
            all_keys.update(subscriber.keys)
        return all_keys

    def _create_buffer_for_key(
        self, subscriber: HistorySubscriber[K], key: K, data: sc.DataArray
    ) -> Buffer:
        """
        Create a buffer for a key based on subscriber's extractor requirements.

        Parameters
        ----------
        subscriber:
            The subscriber requesting the buffer.
        key:
            The key for which to create a buffer.
        data:
            Sample data to determine dimension.

        Returns
        -------
        :
            A configured buffer for this key.
        """
        # Get the extractor for this key
        extractor = subscriber.extractors.get(key, FullHistoryExtractor())

        # Determine concat dimension
        concat_dim = "time" if "time" in data.dims else data.dims[0]

        # Create storage based on extractor type
        if isinstance(extractor, WindowExtractor):
            # For window extractors, use sliding window storage
            # Allocate 2x the window size for efficiency
            window_size = extractor._size if extractor._size else 1000
            storage = SlidingWindowStorage(
                max_size=window_size * 2, concat_dim=concat_dim
            )
        elif isinstance(extractor, DeltaExtractor):
            # Delta extractor needs to keep history for delta calculation
            # Use growing storage with reasonable limits
            storage = GrowingStorage(
                initial_size=100, max_size=10000, concat_dim=concat_dim
            )
        else:
            # FullHistoryExtractor or unknown - use growing storage
            storage = GrowingStorage(
                initial_size=100, max_size=10000, concat_dim=concat_dim
            )

        return Buffer(storage, concat_dim=concat_dim)

    def process_data_service_update(self, store: dict[K, sc.DataArray]) -> None:
        """
        Handle updates from DataService.

        Parameters
        ----------
        store:
            Dictionary of updated data from DataService.
        """
        # Append to each subscriber's buffers and collect which subscribers to notify
        subscribers_to_notify: set[HistorySubscriber[K]] = set()

        for subscriber, buffers in self._buffers.items():
            for key, data in store.items():
                if key in subscriber.keys:
                    # Lazy initialize buffer if needed
                    if key not in buffers:
                        buffers[key] = self._create_buffer_for_key(
                            subscriber, key, data
                        )

                    # Append to this subscriber's buffer
                    buffers[key].append(data)
                    subscribers_to_notify.add(subscriber)

        # Notify subscribers
        self._notify_subscribers(subscribers_to_notify, set(store.keys()))

    def _notify_subscribers(
        self, subscribers: set[HistorySubscriber[K]], updated_keys: set[K]
    ) -> None:
        """
        Notify subscribers about buffer updates.

        Parameters
        ----------
        subscribers:
            The set of subscribers that have relevant updates.
        updated_keys:
            The set of keys that were updated.
        """
        for subscriber in subscribers:
            relevant_keys = subscriber.keys & updated_keys
            if not relevant_keys:
                continue

            # Extract data for all relevant keys using per-key extractors
            extractors = subscriber.extractors
            buffers = self._buffers[subscriber]
            extracted_data: dict[K, sc.DataArray] = {}

            for key in relevant_keys:
                buffer = buffers.get(key)
                if buffer is None:
                    continue

                # Use key-specific extractor or default to full history
                extractor = extractors.get(key, FullHistoryExtractor())
                data = extractor.extract(buffer)
                if data is not None:
                    extracted_data[key] = data

            # Call subscriber once with all extracted data
            if extracted_data:
                subscriber.on_update(extracted_data)

    def register_subscriber(self, subscriber: HistorySubscriber[K]) -> None:
        """
        Register a subscriber for buffer updates.

        Parameters
        ----------
        subscriber:
            The subscriber to register.
        """
        if subscriber not in self._buffers:
            self._buffers[subscriber] = {}

    def unregister_subscriber(self, subscriber: HistorySubscriber[K]) -> None:
        """
        Unregister a subscriber.

        Parameters
        ----------
        subscriber:
            The subscriber to unregister.
        """
        if subscriber in self._buffers:
            del self._buffers[subscriber]

    def get_memory_usage(self) -> dict[HistorySubscriber[K], dict[K, float]]:
        """
        Get memory usage for all buffers.

        Returns
        -------
        :
            Nested dictionary mapping subscribers to their buffers' keys
            to memory usage in megabytes.
        """
        return {
            subscriber: {key: buffer.memory_mb for key, buffer in buffers.items()}
            for subscriber, buffers in self._buffers.items()
        }

    def clear_all_buffers(self) -> None:
        """Clear all buffers for all subscribers."""
        for buffers in self._buffers.values():
            for buffer in buffers.values():
                buffer.clear()
