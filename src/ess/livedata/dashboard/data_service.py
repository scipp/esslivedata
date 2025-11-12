# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any, Generic, TypeVar

from .buffer_manager import BufferManager
from .buffer_strategy import BufferFactory
from .extractors import LatestValueExtractor, UpdateExtractor
from .temporal_requirements import TemporalRequirement

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


class Subscriber(ABC, Generic[K]):
    """Base class for subscribers with cached keys and extractors."""

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
        buffer_factory: BufferFactory | None = None,
        buffer_manager: BufferManager | None = None,
    ) -> None:
        """
        Initialize DataService.

        Parameters
        ----------
        buffer_factory:
            Factory for creating buffers. If None, uses default factory.
        buffer_manager:
            Manager for buffer sizing. If None, creates one with buffer_factory.
        """
        if buffer_factory is None:
            buffer_factory = BufferFactory()
        if buffer_manager is None:
            buffer_manager = BufferManager(buffer_factory)
        self._buffer_factory = buffer_factory
        self._buffer_manager = buffer_manager
        self._default_extractor = LatestValueExtractor()
        self._subscribers: list[Subscriber[K]] = []
        self._update_callbacks: list[Callable[[set[K]], None]] = []
        self._key_change_subscribers: list[Callable[[set[K], set[K]], None]] = []
        self._pending_updates: set[K] = set()
        self._pending_key_additions: set[K] = set()
        self._pending_key_removals: set[K] = set()
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

    def _get_temporal_requirements(self, key: K) -> list[TemporalRequirement]:
        """
        Collect temporal requirements for a key from all subscribers.

        Examines all subscribers' extractor requirements for this key.

        Parameters
        ----------
        key:
            The key to collect requirements for.

        Returns
        -------
        :
            List of temporal requirements from all subscribers for this key.
        """
        requirements = []

        for subscriber in self._subscribers:
            extractors = subscriber.extractors
            if key in extractors:
                extractor = extractors[key]
                requirements.append(extractor.get_temporal_requirement())

        return requirements

    def _build_subscriber_data(self, subscriber: Subscriber[K]) -> dict[K, Any]:
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
        extractors = subscriber.extractors

        for key in subscriber.keys:
            if key in self._buffer_manager:
                extractor = extractors[key]
                buffer = self._buffer_manager[key]
                data = extractor.extract(buffer)
                if data is not None:
                    subscriber_data[key] = data

        return subscriber_data

    def register_subscriber(self, subscriber: Subscriber[K]) -> None:
        """
        Register a subscriber for updates with extractor-based data access.

        Triggers the subscriber immediately with existing data using its extractors.

        Parameters
        ----------
        subscriber:
            The subscriber to register.
        """
        self._subscribers.append(subscriber)

        # Add requirements for keys this subscriber needs
        for key in subscriber.keys:
            if key in self._buffer_manager:
                extractor = subscriber.extractors[key]
                requirement = extractor.get_temporal_requirement()
                self._buffer_manager.add_requirement(key, requirement)

        # Trigger immediately with existing data using subscriber's extractors
        existing_data = self._build_subscriber_data(subscriber)
        subscriber.trigger(existing_data)

    def register_update_callback(self, callback: Callable[[set[K]], None]) -> None:
        """
        Register a callback for key update notifications.

        Callback receives only the set of updated key names, not the data.
        Use this for infrastructure that needs to know what changed but will
        query data itself.

        Parameters
        ----------
        callback:
            Callable that accepts a set of updated keys.
        """
        self._update_callbacks.append(callback)

    def subscribe_to_changed_keys(
        self, subscriber: Callable[[set[K], set[K]], None]
    ) -> None:
        """
        Register a subscriber for key change updates (additions/removals).

        Parameters
        ----------
        subscriber:
            A callable that accepts two sets: added_keys and removed_keys.
        """
        self._key_change_subscribers.append(subscriber)
        subscriber(set(self._buffer_manager.keys()), set())

    def _notify_subscribers(self, updated_keys: set[K]) -> None:
        """
        Notify relevant subscribers about data updates.

        Parameters
        ----------
        updated_keys
            The set of data keys that were updated.
        """
        # Notify extractor-based subscribers
        for subscriber in self._subscribers:
            if updated_keys & subscriber.keys:
                subscriber_data = self._build_subscriber_data(subscriber)
                if subscriber_data:
                    subscriber.trigger(subscriber_data)

        # Notify update callbacks with just key names
        for callback in self._update_callbacks:
            callback(updated_keys)

    def _notify_key_change_subscribers(self) -> None:
        """Notify subscribers about key changes (additions/removals)."""
        if not self._pending_key_additions and not self._pending_key_removals:
            return

        for subscriber in self._key_change_subscribers:
            subscriber(
                self._pending_key_additions.copy(), self._pending_key_removals.copy()
            )

    def __getitem__(self, key: K) -> V:
        """Get the latest value for a key."""
        if key not in self._buffer_manager:
            raise KeyError(key)
        buffer = self._buffer_manager[key]
        return self._default_extractor.extract(buffer)

    def __setitem__(self, key: K, value: V) -> None:
        """Set a value, storing it in a buffer."""
        if key not in self._buffer_manager:
            self._pending_key_additions.add(key)
            requirements = self._get_temporal_requirements(key)
            self._buffer_manager.create_buffer(key, value, requirements)
        self._buffer_manager.update_buffer(key, value)
        self._pending_updates.add(key)
        self._notify_if_not_in_transaction()

    def __delitem__(self, key: K) -> None:
        """Delete a key and its buffer."""
        self._pending_key_removals.add(key)
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
        self._notify_key_change_subscribers()
        self._pending_key_additions.clear()
        self._pending_key_removals.clear()
