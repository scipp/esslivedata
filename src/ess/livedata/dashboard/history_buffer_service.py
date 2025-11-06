# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Service for maintaining historical buffers of DataService data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from functools import cached_property
from typing import Generic, TypeVar

import scipp as sc

from .buffer_strategy import Buffer, DataArrayBuffer
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
        return buffer.get_all()


class WindowExtractor(UpdateExtractor):
    """Extracts a window from the end of the buffer."""

    def __init__(self, size: int) -> None:
        """
        Initialize window extractor.

        Parameters
        ----------
        size:
            Number of elements to extract from the end of the buffer.
        """
        self._size = size

    @property
    def window_size(self) -> int:
        """Return the window size."""
        return self._size

    def extract(self, buffer: Buffer) -> sc.DataArray | None:
        return buffer.get_window(self._size)


class HistorySubscriber(ABC, Generic[K]):
    """
    Protocol for subscribers to HistoryBufferService.

    Subscribers specify what data they need per key via UpdateExtractors
    and receive batched updates for all relevant keys.
    """

    @cached_property
    def keys(self) -> set[K]:
        """
        Return the set of buffer keys this subscriber depends on.

        Cached after first access. If extractors changes after instantiation,
        the cache will not update automatically.
        """
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
        self._buffer_service.add_data(store)


class HistoryBufferService(Generic[K]):
    """
    Service for maintaining historical buffers of data.

    Data can be added either directly via add_data() or by subscribing to a
    DataService (if provided at initialization).

    Each subscriber gets its own set of buffers for the keys it needs.
    """

    # Maximum size for full history buffers
    DEFAULT_MAX_SIZE = 10000

    def __init__(
        self,
        data_service: DataService[K, sc.DataArray] | None = None,
        concat_dim: str = "time",
    ) -> None:
        """
        Initialize the history buffer service.

        Parameters
        ----------
        data_service:
            The DataService to subscribe to. If None, data must be added
            via add_data() method.
        concat_dim:
            The dimension along which to concatenate data. Defaults to "time".
        """
        self._data_service = data_service
        self._concat_dim = concat_dim
        # Each subscriber has its own buffers for its keys
        self._buffers: dict[HistorySubscriber[K], dict[K, Buffer]] = {}

        # Subscribe to DataService if provided
        if self._data_service is not None:
            self._internal_subscriber = _InternalDataSubscriber(self)
            self._data_service.register_subscriber(self._internal_subscriber)

    def get_tracked_keys(self) -> set[K]:
        """
        Return all keys currently tracked by registered subscribers.

        Returns the union of all keys from all registered subscribers.
        """
        all_keys: set[K] = set()
        for subscriber in self._buffers:
            all_keys.update(subscriber.keys)
        return all_keys

    def _create_buffer_for_key(
        self, subscriber: HistorySubscriber[K], key: K
    ) -> Buffer:
        """
        Create a buffer for a key based on subscriber's extractor requirements.

        Parameters
        ----------
        subscriber:
            The subscriber requesting the buffer.
        key:
            The key for which to create a buffer.

        Returns
        -------
        :
            A configured buffer for this key.
        """
        buffer_impl = DataArrayBuffer(concat_dim=self._concat_dim)
        extractor = subscriber.extractors.get(key, FullHistoryExtractor())

        if isinstance(extractor, WindowExtractor):
            return Buffer(
                max_size=extractor.window_size,
                buffer_impl=buffer_impl,
                concat_dim=self._concat_dim,
            )
        else:
            return Buffer(
                max_size=self.DEFAULT_MAX_SIZE,
                buffer_impl=buffer_impl,
                concat_dim=self._concat_dim,
            )

    def add_data(self, store: dict[K, sc.DataArray]) -> None:
        """
        Add a batch of data to the buffers.

        Appends data to subscriber buffers for relevant keys and notifies
        subscribers with extracted views of the buffered data.

        Parameters
        ----------
        store:
            Dictionary mapping keys to data arrays to buffer.
        """
        # Append to each subscriber's buffers and collect which subscribers to notify
        subscribers_to_notify: set[HistorySubscriber[K]] = set()

        for subscriber, buffers in self._buffers.items():
            for key, data in store.items():
                if key in subscriber.keys:
                    # Lazy initialize buffer if needed
                    if key not in buffers:
                        buffers[key] = self._create_buffer_for_key(subscriber, key)

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
