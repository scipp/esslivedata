# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any, Protocol, TypeVar

from .buffer_strategy import Buffer, BufferFactory

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


class UpdateExtractor(ABC):
    """Extracts a specific view of buffer data."""

    @abstractmethod
    def extract(self, buffer: Buffer) -> Any:
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

    @abstractmethod
    def get_required_size(self) -> int:
        """
        Return the minimum buffer size required by this extractor.

        Returns
        -------
        :
            Required buffer size (1 for latest value, n for window, large for full).
        """


class LatestValueExtractor(UpdateExtractor):
    """Extracts the latest single value, unwrapping the concat dimension."""

    def __init__(self, concat_dim: str = 'time') -> None:
        """
        Initialize latest value extractor.

        Parameters
        ----------
        concat_dim:
            The dimension to unwrap when extracting from scipp objects.
        """
        self._concat_dim = concat_dim

    def get_required_size(self) -> int:
        """Latest value only needs buffer size of 1."""
        return 1

    def extract(self, buffer: Buffer) -> Any:
        """
        Extract the latest value from the buffer.

        For list buffers, returns the last element.
        For scipp DataArray/Variable, unwraps the concat dimension.
        """
        view = buffer.get_window(1)
        if view is None:
            return None

        # Unwrap based on type
        if isinstance(view, list):
            return view[0] if view else None

        # Import scipp only when needed to avoid circular imports
        import scipp as sc

        if isinstance(view, sc.DataArray):
            if self._concat_dim in view.dims:
                # Slice to remove concat dimension
                result = view[self._concat_dim, 0]
                # Drop the now-scalar concat coordinate to restore original structure
                if self._concat_dim in result.coords:
                    result = result.drop_coords(self._concat_dim)
                return result
            return view
        elif isinstance(view, sc.Variable):
            if self._concat_dim in view.dims:
                return view[self._concat_dim, 0]
            return view
        else:
            return view


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

    def get_required_size(self) -> int:
        """Window extractor requires buffer size equal to window size."""
        return self._size

    def extract(self, buffer: Buffer) -> Any:
        """Extract a window of data from the end of the buffer."""
        return buffer.get_window(self._size)


class FullHistoryExtractor(UpdateExtractor):
    """Extracts the complete buffer history."""

    # Maximum size for full history buffers
    DEFAULT_MAX_SIZE = 10000

    def get_required_size(self) -> int:
        """Full history requires large buffer."""
        return self.DEFAULT_MAX_SIZE

    def extract(self, buffer: Buffer) -> Any:
        """Extract all data from the buffer."""
        return buffer.get_all()


class SubscriberProtocol(Protocol[K]):
    """Protocol for subscribers with keys and trigger method."""

    @property
    def keys(self) -> set[K]:
        """Return the set of data keys this subscriber depends on."""

    @property
    def extractors(self) -> dict[K, UpdateExtractor]:
        """
        Return extractors for obtaining data views.

        Returns a mapping from key to the extractor to use for that key.
        """

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
        if buffer_factory is None:
            buffer_factory = BufferFactory()
        self._buffer_factory = buffer_factory
        self._buffers: dict[K, Buffer[V]] = {}
        self._default_extractor = LatestValueExtractor()
        self._subscribers: list[SubscriberProtocol[K]] = []
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

    def _get_required_buffer_size(self, key: K) -> int:
        """
        Calculate required buffer size for a key based on all subscribers.

        Examines all subscribers' extractor requirements for this key and returns
        the maximum required size.

        Parameters
        ----------
        key:
            The key to calculate buffer size for.

        Returns
        -------
        :
            Maximum buffer size required by all subscribers for this key.
            Defaults to 1 if no subscribers need this key.
        """
        max_size = 1  # Default: latest value only

        for subscriber in self._subscribers:
            if key in subscriber.keys:
                extractors = subscriber.extractors
                if key in extractors:
                    extractor = extractors[key]
                    max_size = max(max_size, extractor.get_required_size())

        return max_size

    def register_subscriber(self, subscriber: SubscriberProtocol[K]) -> None:
        """
        Register a subscriber for updates with extractor-based data access.

        Parameters
        ----------
        subscriber:
            The subscriber to register. Must implement SubscriberProtocol with
            keys, extractors, and trigger() method.
        """
        self._subscribers.append(subscriber)

        # Update buffer sizes for keys this subscriber needs
        for key in subscriber.keys:
            if key in self._buffers:
                required_size = self._get_required_buffer_size(key)
                # Resize buffer if needed (Buffer handles growth, never shrinks)
                self._buffers[key].set_max_size(required_size)

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
        subscriber(set(self._buffers.keys()), set())

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
                # Extract data using per-key extractors
                subscriber_data = {}
                extractors = subscriber.extractors

                for key in subscriber.keys:
                    if key in self._buffers:
                        # Use subscriber's extractor for this key
                        extractor = extractors.get(key, self._default_extractor)
                        data = extractor.extract(self._buffers[key])
                        if data is not None:
                            subscriber_data[key] = data

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
        if key not in self._buffers:
            raise KeyError(key)
        return self._default_extractor.extract(self._buffers[key])

    def __setitem__(self, key: K, value: V) -> None:
        """Set a value, storing it in a buffer."""
        if key not in self._buffers:
            self._pending_key_additions.add(key)
            # Use dynamic buffer sizing based on subscriber requirements
            required_size = self._get_required_buffer_size(key)
            self._buffers[key] = self._buffer_factory.create_buffer(
                value, max_size=required_size
            )
            self._buffers[key].append(value)
        else:
            try:
                # Try to append to existing buffer
                self._buffers[key].append(value)
            except Exception:
                # Data is incompatible (shape/dims changed) - clear and recreate.
                # Note: This is mainly for buffer mode (max_size > 1). For max_size==1,
                # Buffer uses simple value replacement and won't raise exceptions.
                # Buffer.clear() sets internal buffer to None, so next append
                # will allocate a new buffer using the new value as template.
                self._buffers[key].clear()
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
