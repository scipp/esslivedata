# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any, Generic, TypeVar

from .extractors import LatestValueExtractor, UpdateExtractor
from .temporal_buffer_manager import TemporalBufferManager

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


class DataServiceSubscriber(ABC, Generic[K]):
    """Base class for data service subscribers with cached keys and extractors."""

    def __init__(self) -> None:
        """Initialize subscriber and cache keys from extractors."""
        # Cache keys from extractors to avoid repeated computation
        self._keys = set(self.extractors.keys())

    @property
    def keys(self) -> set[K]:
        """Return the set of data keys this subscriber depends on."""
        return self._keys

    @property
    @abstractmethod
    def extractors(self) -> Mapping[K, UpdateExtractor]:
        """
        Return extractors for obtaining data views.

        Returns a mapping from key to the extractor to use for that key.
        """

    @abstractmethod
    def trigger(self, store: dict[K, Any]) -> None:
        """Trigger the subscriber with updated data."""


class DataService(MutableMapping[K, V]):
    """
    A service for managing and retrieving data and derived data.

    New data is set from upstream Kafka topics. Subscribers are typically plots that
    provide a live view of the data.

    Uses buffers internally for storage, but presents a dict-like interface
    that returns the latest value for each key.
    """

    def __init__(
        self,
        buffer_manager: TemporalBufferManager | None = None,
    ) -> None:
        """
        Initialize DataService.

        Parameters
        ----------
        buffer_manager:
            Manager for buffer sizing. If None, creates a new TemporalBufferManager.
        """
        if buffer_manager is None:
            buffer_manager = TemporalBufferManager()
        self._buffer_manager = buffer_manager
        self._default_extractor = LatestValueExtractor()
        self._subscribers: list[DataServiceSubscriber[K]] = []
        self._pending_updates: set[K] = set()
        self._transaction_depth = 0

    @contextmanager
    def transaction(self):
        """Context manager for batching multiple updates."""
        self._transaction_depth += 1
        try:
            yield
        finally:
            # Stay in transaction until notifications are done. This ensures that
            # subscribers that make further updates during their notification do not
            # trigger intermediate notifications.
            if self._transaction_depth == 1:
                self._notify()
            self._transaction_depth -= 1

    @property
    def _in_transaction(self) -> bool:
        return self._transaction_depth > 0

    def _get_extractors(self, key: K) -> list[UpdateExtractor]:
        """
        Collect extractors for a key from all subscribers.

        Examines all subscribers that need this key.

        Parameters
        ----------
        key:
            The key to collect extractors for.

        Returns
        -------
        :
            List of extractors from all subscribers for this key.
        """
        extractors = []

        for subscriber in self._subscribers:
            subscriber_extractors = subscriber.extractors
            if key in subscriber_extractors:
                extractor = subscriber_extractors[key]
                extractors.append(extractor)

        return extractors

    def _build_subscriber_data(
        self, subscriber: DataServiceSubscriber[K]
    ) -> dict[K, Any]:
        """
        Extract data for a subscriber based on its extractors.

        Parameters
        ----------
        subscriber:
            The subscriber to extract data for.

        Returns
        -------
        :
            Dictionary mapping keys to extracted data (None values filtered out).
        """
        subscriber_data = {}

        for key, extractor in subscriber.extractors.items():
            buffered_data = self._buffer_manager.get_buffered_data(key)
            if buffered_data is not None:
                subscriber_data[key] = extractor.extract(buffered_data)

        return subscriber_data

    def register_subscriber(self, subscriber: DataServiceSubscriber[K]) -> None:
        """
        Register a subscriber for updates with extractor-based data access.

        Triggers the subscriber immediately with existing data using its extractors.

        Parameters
        ----------
        subscriber:
            The subscriber to register.
        """
        self._subscribers.append(subscriber)

        # Add extractors for keys this subscriber needs
        for key in subscriber.keys:
            if key in self._buffer_manager:
                extractor = subscriber.extractors[key]
                self._buffer_manager.add_extractor(key, extractor)

        # Trigger immediately with existing data using subscriber's extractors
        existing_data = self._build_subscriber_data(subscriber)
        subscriber.trigger(existing_data)

    def unregister_subscriber(self, subscriber: DataServiceSubscriber[K]) -> bool:
        """
        Unregister a subscriber, stopping it from receiving updates.

        Parameters
        ----------
        subscriber:
            The subscriber to unregister.

        Returns
        -------
        :
            True if the subscriber was found and removed, False otherwise.
        """
        try:
            self._subscribers.remove(subscriber)
            return True
        except ValueError:
            return False

    def _notify_subscribers(self, updated_keys: set[K]) -> None:
        """
        Notify relevant subscribers about data updates.

        Parameters
        ----------
        updated_keys
            The set of data keys that were updated.
        """
        for subscriber in self._subscribers:
            if updated_keys & subscriber.keys:
                subscriber_data = self._build_subscriber_data(subscriber)
                subscriber.trigger(subscriber_data)

    def __getitem__(self, key: K) -> V:
        """Get the latest value for a key."""
        buffered_data = self._buffer_manager.get_buffered_data(key)
        if buffered_data is None:
            raise KeyError(key)
        return self._default_extractor.extract(buffered_data)

    def __setitem__(self, key: K, value: V) -> None:
        """Set a value, storing it in a buffer."""
        if key not in self._buffer_manager:
            extractors = self._get_extractors(key)
            self._buffer_manager.create_buffer(key, extractors)
        self._buffer_manager.update_buffer(key, value)
        self._pending_updates.add(key)
        self._notify_if_not_in_transaction()

    def __delitem__(self, key: K) -> None:
        """Delete a key and its buffer."""
        self._buffer_manager.delete_buffer(key)
        self._pending_updates.add(key)
        self._notify_if_not_in_transaction()

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys."""
        return iter(self._buffer_manager)

    def __len__(self) -> int:
        """Return the number of keys."""
        return len(self._buffer_manager)

    def _notify_if_not_in_transaction(self) -> None:
        """Notify subscribers if not in a transaction."""
        if not self._in_transaction:
            self._notify()

    def _notify(self) -> None:
        # Some updates may have been added while notifying
        while self._pending_updates:
            pending = set(self._pending_updates)
            self._pending_updates.clear()
            self._notify_subscribers(pending)
