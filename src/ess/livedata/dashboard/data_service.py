# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any, Generic, TypeVar

import structlog

from .extractors import LatestValueExtractor, UpdateExtractor
from .temporal_buffer_manager import TemporalBufferManager

logger = structlog.get_logger(__name__)

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

    Thread safety
    -------------
    Mutated from two threads: the background ingestion thread (via
    ``transaction`` / ``__setitem__``) and the UI thread (via
    ``register_subscriber`` / ``unregister_subscriber``). A single ``RLock``
    serializes all access to the internal state (buffer manager, subscriber
    list, pending updates, transaction depth).

    The lock is **never** held while calling ``subscriber.trigger`` (which runs
    arbitrary compute, including Bokeh document mutation). Data handed to a
    subscriber is copied out of the buffers under the lock first
    (see :py:meth:`_build_subscriber_data`), so compute operates on detached
    data and cannot observe a concurrent buffer mutation. Keeping ``trigger``
    outside the lock also means no compute-side lock (the Bokeh document lock,
    or ``ActiveJobRegistry``'s ingestion guard) can deadlock against this one.

    Lock ordering: ``ActiveJobRegistry.ingestion_guard`` -> ``DataService`` lock.
    The ingestion path and ``ActiveJobRegistry.cleanup`` both hold the guard
    when they enter ``transaction``; nothing acquires the guard while holding
    this lock.
    """

    def __init__(self) -> None:
        self._buffer_manager = TemporalBufferManager()
        self._default_extractor = LatestValueExtractor()
        self._subscribers: list[DataServiceSubscriber[K]] = []
        self._pending_updates: set[K] = set()
        self._transaction_depth = 0
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self):
        """Context manager for batching multiple updates."""
        with self._lock:
            self._transaction_depth += 1
        try:
            yield
        finally:
            # Stay in transaction until notifications are done. This ensures that
            # subscribers that make further updates during their notification do not
            # trigger intermediate notifications.
            with self._lock:
                at_root = self._transaction_depth == 1
            if at_root:
                self._notify()
            with self._lock:
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

        Notes
        -----
        Must be called while holding ``self._lock``. Extracted values are
        copied so callers can hand them to ``subscriber.trigger`` after
        releasing the lock: extractors may return views aliasing the buffer's
        backing store, which a concurrent ``update``/``drop`` would mutate.
        """
        subscriber_data = {}

        for key, extractor in subscriber.extractors.items():
            buffered_data = self._buffer_manager.get_buffered_data(key)
            if buffered_data is not None:
                extracted = extractor.extract(buffered_data)
                # Copy to detach from the buffer's backing store (see docstring).
                subscriber_data[key] = (
                    extracted.copy() if extracted is not None else None
                )

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
        with self._lock:
            self._subscribers.append(subscriber)

            # Add extractors for keys this subscriber needs
            for key in subscriber.keys:
                if key in self._buffer_manager:
                    extractor = subscriber.extractors[key]
                    self._buffer_manager.add_extractor(key, extractor)

            # Snapshot existing data using subscriber's extractors
            existing_data = self._build_subscriber_data(subscriber)

        # Trigger outside the lock: compute must not run while holding it.
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
        with self._lock:
            try:
                self._subscribers.remove(subscriber)
                return True
            except ValueError:
                return False

    def _notify_subscribers(self, updated_keys: set[K]) -> None:
        """
        Notify relevant subscribers about data updates.

        The subscriber data is built under the lock (copying it out of the
        buffers), then ``trigger`` is called with the lock released so compute
        never runs while the lock is held.

        Parameters
        ----------
        updated_keys
            The set of data keys that were updated.
        """
        # Iterate over a snapshot: a subscriber's trigger, or a concurrent
        # unregister from the UI thread, may mutate self._subscribers mid-loop.
        # Without the copy the list-index iterator would silently skip the
        # subscriber following a removed one.
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            if updated_keys & subscriber.keys:
                try:
                    with self._lock:
                        subscriber_data = self._build_subscriber_data(subscriber)
                    subscriber.trigger(subscriber_data)
                except Exception:
                    logger.exception("Failed to notify subscriber %s", subscriber)

    def __getitem__(self, key: K) -> V:
        """Get the latest value for a key."""
        with self._lock:
            buffered_data = self._buffer_manager.get_buffered_data(key)
            if buffered_data is None:
                raise KeyError(key)
            return self._default_extractor.extract(buffered_data).copy()

    def __setitem__(self, key: K, value: V) -> None:
        """Set a value, storing it in a buffer."""
        with self._lock:
            if key not in self._buffer_manager:
                extractors = self._get_extractors(key)
                self._buffer_manager.create_buffer(key, extractors)
            self._buffer_manager.update_buffer(key, value)
            self._pending_updates.add(key)
            notify = not self._in_transaction
        if notify:
            self._notify()

    def __delitem__(self, key: K) -> None:
        """Delete a key and its buffer."""
        with self._lock:
            self._buffer_manager.delete_buffer(key)
            self._pending_updates.add(key)
            notify = not self._in_transaction
        if notify:
            self._notify()

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys."""
        with self._lock:
            return iter(list(self._buffer_manager))

    def __len__(self) -> int:
        """Return the number of keys."""
        with self._lock:
            return len(self._buffer_manager)

    def _notify(self) -> None:
        # Some updates may have been added while notifying. Drain pending under
        # the lock; notify (which triggers compute) without it.
        while True:
            with self._lock:
                if not self._pending_updates:
                    return
                pending = set(self._pending_updates)
                self._pending_updates.clear()
            self._notify_subscribers(pending)
