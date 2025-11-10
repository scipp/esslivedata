# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any, Protocol, TypeVar

from .buffer_strategy import Buffer, BufferFactory

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


class SubscriberProtocol(Protocol[K]):
    """Protocol for subscribers with keys and trigger method."""

    @property
    def keys(self) -> set[K]:
        """Return the set of data keys this subscriber depends on."""

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

    def __init__(self, buffer_factory: BufferFactory | None = None) -> None:
        """
        Initialize DataService.

        Parameters
        ----------
        buffer_factory:
            Factory for creating buffers. If None, uses default factory.
        """
        from .history_buffer_service import LatestValueExtractor

        if buffer_factory is None:
            buffer_factory = BufferFactory()
        self._buffer_factory = buffer_factory
        self._buffers: dict[K, Buffer[V]] = {}
        self._extractor = LatestValueExtractor()
        self._subscribers: list[SubscriberProtocol[K] | Callable[[set[K]], None]] = []
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

    def register_subscriber(
        self, subscriber: SubscriberProtocol[K] | Callable[[set[K]], None]
    ) -> None:
        """
        Register a subscriber for updates.

        Parameters
        ----------
        subscriber:
            The subscriber to register. Can be either an object with `keys` property
            and `trigger()` method, or a callable that accepts a set of updated keys.
        """
        self._subscribers.append(subscriber)

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
        subscriber(set(self._buffers.keys()), set())

    def _notify_subscribers(self, updated_keys: set[K]) -> None:
        """
        Notify relevant subscribers about data updates.

        Parameters
        ----------
        updated_keys
            The set of data keys that were updated.
        """
        for subscriber in self._subscribers:
            # Duck-type check: does it have keys and trigger?
            if hasattr(subscriber, 'keys') and hasattr(subscriber, 'trigger'):
                if updated_keys & subscriber.keys:
                    # Pass only the data that the subscriber is interested in
                    subscriber_data = {
                        key: self._extractor.extract(self._buffers[key])
                        for key in subscriber.keys
                        if key in self._buffers
                    }
                    subscriber.trigger(subscriber_data)
            else:
                # Plain callable - gets key names only
                subscriber(updated_keys)

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
        if key not in self._buffers:
            raise KeyError(key)
        return self._extractor.extract(self._buffers[key])

    def __setitem__(self, key: K, value: V) -> None:
        """Set a value, storing it in a buffer."""
        if key not in self._buffers:
            self._pending_key_additions.add(key)
            self._buffers[key] = self._buffer_factory.create_buffer(value, max_size=1)
        else:
            # For size-1 buffers, replace entirely if value changes
            # This allows updating with different-shaped data
            self._buffers[key].clear()
            self._buffers[key] = self._buffer_factory.create_buffer(value, max_size=1)
        self._buffers[key].append(value)
        self._pending_updates.add(key)
        self._notify_if_not_in_transaction()

    def __delitem__(self, key: K) -> None:
        """Delete a key and its buffer."""
        self._pending_key_removals.add(key)
        del self._buffers[key]
        self._pending_updates.add(key)
        self._notify_if_not_in_transaction()

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys."""
        return iter(self._buffers)

    def __len__(self) -> int:
        """Return the number of keys."""
        return len(self._buffers)

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
