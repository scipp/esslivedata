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
    """Base class for data service subscribers with cached keys and extractors.

    Notifications carry no data: :py:meth:`on_updated` only reports which keys
    changed, so a notification is O(changed keys) regardless of data size.
    Consumers pull extracted data when (and if) they need it, via
    :py:meth:`DataService.snapshot` — typically at frame-flush time, and only
    for consumers that are currently displayed. Extraction can be expensive
    (aggregating extractors reduce over the buffered history on every call,
    which dominates cost for image-sized data), so deferring it to the pull
    is what keeps ingestion cheap.

    The subscriber's extractors still determine buffer type and history
    retention from the moment of registration, independent of when or whether
    the consumer pulls.
    """

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
    def on_updated(self, updated_keys: set[K]) -> None:
        """
        Notify that some of this subscriber's keys changed (or were deleted).

        Called with the intersection of the changed keys and :py:attr:`keys`.
        Must be cheap and must not raise: it runs once per subscriber per
        update batch on the ingestion thread. Pull data via
        :py:meth:`DataService.snapshot` instead of computing here.
        """


class DataService(MutableMapping[K, V]):
    """
    A service for managing and retrieving data and derived data.

    New data is set from upstream Kafka topics. Subscribers are typically plots that
    provide a live view of the data.

    Uses buffers internally for storage, but presents a dict-like interface
    that returns the latest value for each key.

    Notification vs. data flow
    --------------------------
    Notifications are keys-only: writes mark keys as updated and
    ``subscriber.on_updated`` reports the changed keys, nothing more.
    Consumers pull extracted data on their own schedule via
    :py:meth:`snapshot`, which applies the subscriber's extractors and copies
    the result out of the buffers. Every pull observes the buffer state at
    pull time — never a partially applied transaction — so coalescing,
    deferring, or dropping notifications never delivers stale data.

    Thread safety
    -------------
    Mutated from two threads: the background ingestion thread (via
    ``transaction`` / ``__setitem__``) and the UI thread (via
    ``register_subscriber`` / ``unregister_subscriber``). A single ``RLock``
    serializes all access to the internal state (buffer manager, subscriber
    list, pending updates). The lock is transaction-scoped: ``transaction``
    holds it from entry to exit, so transactions are atomic to readers and
    mutually exclusive across threads. Transaction depth is thread-local: it
    defers notifications only for updates made on its own thread, so a write
    on another thread (once it acquires the lock) still notifies immediately.

    The lock is **never** held while calling ``subscriber.on_updated``, so no
    lock a subscriber callback takes can deadlock against this one. Extracted
    values returned by ``snapshot`` are copied under the lock before being
    handed out: extractors may return views aliasing a buffer's backing
    store, which a concurrent ``update``/``drop`` would mutate.

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
        self._local = threading.local()
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self):
        """Context manager for batching multiple updates.

        Holds the lock for the whole transaction: readers and pulls observe
        either none or all of its updates, and transactions on other threads
        are serialized against it. Notifications run after the lock is
        released.
        """
        self._local.transaction_depth = self._transaction_depth + 1
        try:
            with self._lock:
                yield
        finally:
            # Stay in transaction until notifications are done. This ensures that
            # subscribers that make further updates during their notification do not
            # trigger intermediate notifications.
            try:
                if self._transaction_depth == 1:
                    self._notify()
            finally:
                # Always restore depth, even if a notification raised: a stuck
                # depth would silently suppress all future notifications.
                self._local.transaction_depth -= 1

    @property
    def _transaction_depth(self) -> int:
        return getattr(self._local, 'transaction_depth', 0)

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
        copied so callers can use them after releasing the lock: extractors
        may return views aliasing the buffer's backing store, which a
        concurrent ``update``/``drop`` would mutate.
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
        Register a subscriber for update notifications.

        Registers the subscriber's extractors with the buffer manager so
        history retention covers its needs from now on. Does not notify:
        the caller pulls existing data via :py:meth:`snapshot` if it wants
        an initial delivery.

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

    def snapshot(self, subscriber: DataServiceSubscriber[K]) -> dict[K, Any]:
        """
        Extract current data for a subscriber through its extractors.

        The pull side of the notification protocol: call at any time after
        an :py:meth:`DataServiceSubscriber.on_updated` notification (or on
        demand, e.g. when a consumer becomes visible) to obtain the current
        state of the subscriber's keys.

        Parameters
        ----------
        subscriber:
            The subscriber whose extractors to apply.

        Returns
        -------
        :
            Extracted data, detached from the internal buffers.
        """
        with self._lock:
            return self._build_subscriber_data(subscriber)

    def unregister_subscriber(self, subscriber: DataServiceSubscriber[K]) -> bool:
        """
        Unregister a subscriber, stopping it from receiving updates.

        Drops the subscriber's extractors from buffer retention: each affected
        buffer is reconfigured to the surviving subscribers' requirements, so
        retention cannot ratchet up over repeated register/unregister cycles.
        Buffer type never downgrades here (sticky-upward, see
        :py:meth:`TemporalBufferManager.set_extractors`), so buffered history
        survives an unregister/re-register cycle on the same key. The cycle is
        not atomic, though: if a surviving subscriber needs a shorter timespan,
        an append landing between unregister and re-register may trim history
        to that shorter window.

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
            except ValueError:
                return False
            for key in subscriber.keys:
                if key in self._buffer_manager:
                    self._buffer_manager.set_extractors(
                        key, self._get_extractors(key), allow_downgrade=False
                    )
            return True

    def _notify_subscribers(self, updated_keys: set[K]) -> None:
        """
        Notify relevant subscribers which of their keys changed.

        Keys-only: no extraction or copying happens here. ``on_updated`` is
        called with the lock released so no lock a callback takes can nest
        inside ours.

        Parameters
        ----------
        updated_keys
            The set of data keys that were updated.
        """
        # Iterate over a snapshot: a concurrent register/unregister from the
        # UI thread may mutate self._subscribers mid-loop. Without the copy
        # the list-index iterator would silently skip the subscriber
        # following a removed one.
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            keys = updated_keys & subscriber.keys
            if not keys:
                continue
            try:
                subscriber.on_updated(keys)
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
        if not self._in_transaction:
            self._notify()

    def __delitem__(self, key: K) -> None:
        """Delete a key and its buffer."""
        with self._lock:
            self._buffer_manager.delete_buffer(key)
            self._pending_updates.add(key)
        if not self._in_transaction:
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
        # the lock; call subscriber callbacks without it.
        while True:
            with self._lock:
                if not self._pending_updates:
                    return
                pending = set(self._pending_updates)
                self._pending_updates.clear()
            self._notify_subscribers(pending)
