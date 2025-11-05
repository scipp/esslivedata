# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Service for maintaining historical buffers of DataService data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

import scipp as sc

from .buffer import Buffer
from .buffer_config import BufferConfig, BufferConfigRegistry, default_registry
from .data_service import DataService

K = TypeVar('K', bound=Hashable)


class BufferViewType(Enum):
    """Types of views available for buffer subscribers."""

    FULL = "full"  # Complete buffer
    DELTA = "delta"  # Only new data since last notification
    WINDOW = "window"  # Specific window/slice


class BufferSubscriber(ABC, Generic[K]):
    """
    Protocol for subscribers to HistoryBufferService.

    Subscribers can configure what data they receive when buffers are updated.
    """

    @property
    @abstractmethod
    def keys(self) -> set[K]:
        """Return the set of buffer keys this subscriber depends on."""

    @abstractmethod
    def buffer_updated(
        self, key: K, data: sc.DataArray | None, view_type: BufferViewType
    ) -> None:
        """
        Called when a subscribed buffer is updated.

        Parameters
        ----------
        key:
            The key of the buffer that was updated.
        data:
            The buffer data according to the configured view type.
        view_type:
            The type of view being provided.
        """


class PipeBase(Protocol):
    """Protocol for downstream pipes that can receive data."""

    def send(self, data: Any) -> None:
        """Send data to the downstream pipe."""


class SimpleBufferSubscriber(BufferSubscriber[K]):
    """
    Simple buffer subscriber that sends data to a pipe.

    Provides configurable views of buffer data.
    """

    def __init__(
        self,
        keys: set[K],
        pipe: PipeBase,
        view_type: BufferViewType = BufferViewType.FULL,
        window_size: int | None = None,
    ) -> None:
        """
        Initialize a simple buffer subscriber.

        Parameters
        ----------
        keys:
            The set of keys to subscribe to.
        pipe:
            The pipe to send data to.
        view_type:
            The type of view to request.
        window_size:
            For WINDOW view type, the size of the window.
        """
        self._keys = keys
        self._pipe = pipe
        self._view_type = view_type
        self._window_size = window_size

    @property
    def keys(self) -> set[K]:
        return self._keys

    @property
    def view_type(self) -> BufferViewType:
        """Get the configured view type."""
        return self._view_type

    @property
    def window_size(self) -> int | None:
        """Get the configured window size."""
        return self._window_size

    def buffer_updated(
        self, key: K, data: sc.DataArray | None, view_type: BufferViewType
    ) -> None:
        if data is not None:
            self._pipe.send({key: data})


class _InternalDataSubscriber(Generic[K]):
    """Internal subscriber to connect HistoryBufferService to DataService."""

    def __init__(self, buffer_service: HistoryBufferService[K]):
        self._buffer_service = buffer_service

    @property
    def keys(self) -> set[K]:
        """Return the keys currently registered in the buffer service."""
        # Include registered buffers, explicit configs, and pending lazy init
        return (
            self._buffer_service.keys
            | set(self._buffer_service._explicit_configs.keys())
            | self._buffer_service._pending_lazy_init
        )

    def trigger(self, store: dict[K, sc.DataArray]) -> None:
        """
        Process updates from DataService.

        Parameters
        ----------
        store:
            Dictionary of updated data from DataService.
        """
        self._buffer_service._process_data_service_update(store)


class HistoryBufferService(Generic[K]):
    """
    Service for maintaining historical buffers of data from DataService.

    Subscribes to DataService updates and maintains configurable time/size-limited
    buffers for specified keys. Provides subscription API for widgets that need
    historical data.
    """

    def __init__(
        self,
        data_service: DataService[K, sc.DataArray],
        config_registry: BufferConfigRegistry | None = None,
    ) -> None:
        """
        Initialize the history buffer service.

        Parameters
        ----------
        data_service:
            The DataService to subscribe to.
        config_registry:
            Registry for determining buffer configurations based on data type.
            If None, uses the default registry.
        """
        self._data_service = data_service
        self._config_registry = config_registry or default_registry
        self._buffers: dict[K, Buffer] = {}
        self._subscribers: list[BufferSubscriber[K]] = []
        self._explicit_configs: dict[K, BufferConfig] = {}
        self._pending_lazy_init: set[K] = set()  # Keys awaiting lazy initialization

        # Subscribe to DataService
        self._internal_subscriber = _InternalDataSubscriber(self)
        self._data_service.register_subscriber(self._internal_subscriber)

    @property
    def keys(self) -> set[K]:
        """Return the set of keys being buffered."""
        return set(self._buffers.keys())

    def register_key(
        self,
        key: K,
        config: BufferConfig | None = None,
        initial_data: sc.DataArray | None = None,
    ) -> None:
        """
        Register a key for buffering.

        Parameters
        ----------
        key:
            The key to buffer.
        config:
            Optional explicit configuration. If None, configuration will be
            determined from the first data received using the config registry.
        initial_data:
            Optional initial data to use for type detection if config is None.
        """
        if key in self._buffers:
            return  # Already registered

        if config is not None:
            self._explicit_configs[key] = config
            strategy = config.create_strategy()
            self._buffers[key] = Buffer(strategy)
            self._pending_lazy_init.discard(key)  # Remove from pending if present
        elif initial_data is not None:
            # Use initial data to determine config
            config = self._config_registry.get_config(initial_data)
            strategy = config.create_strategy()
            self._buffers[key] = Buffer(strategy)
            # Append the initial data
            self._buffers[key].append(initial_data)
            self._pending_lazy_init.discard(key)  # Remove from pending if present
        else:
            # Will be lazy-initialized on first data
            self._pending_lazy_init.add(key)

    def unregister_key(self, key: K) -> None:
        """
        Unregister a key from buffering.

        Parameters
        ----------
        key:
            The key to stop buffering.
        """
        if key in self._buffers:
            del self._buffers[key]
        if key in self._explicit_configs:
            del self._explicit_configs[key]
        self._pending_lazy_init.discard(key)

    def _process_data_service_update(self, store: dict[K, sc.DataArray]) -> None:
        """
        Handle updates from DataService.

        This is called by the internal subscriber when DataService notifies
        of updates to registered keys.

        Parameters
        ----------
        store:
            Dictionary of updated data from DataService.
        """
        # Process all updates
        updated_keys = set()

        for key, data in store.items():
            # Lazy initialization if not yet configured
            if key not in self._buffers:
                if key in self._explicit_configs:
                    config = self._explicit_configs[key]
                else:
                    config = self._config_registry.get_config(data)
                strategy = config.create_strategy()
                self._buffers[key] = Buffer(strategy)
                # Remove from pending lazy init
                self._pending_lazy_init.discard(key)

            # Append to buffer
            self._buffers[key].append(data)
            updated_keys.add(key)

        # Notify subscribers
        self._notify_subscribers(updated_keys)

    def _notify_subscribers(self, updated_keys: set[K]) -> None:
        """
        Notify subscribers about buffer updates.

        Parameters
        ----------
        updated_keys:
            The set of keys that were updated.
        """
        for subscriber in self._subscribers:
            relevant_keys = subscriber.keys & updated_keys
            for key in relevant_keys:
                buffer = self._buffers.get(key)
                if buffer is None:
                    continue

                # Get data according to subscriber's view preference
                if isinstance(subscriber, SimpleBufferSubscriber):
                    view_type = subscriber.view_type
                    if view_type == BufferViewType.FULL:
                        data = buffer.get_buffer()
                    elif view_type == BufferViewType.WINDOW:
                        data = buffer.get_window(subscriber.window_size)
                    else:  # DELTA - for now, just get the full buffer
                        # TODO: Implement delta tracking
                        data = buffer.get_buffer()
                else:
                    # Default to full buffer
                    view_type = BufferViewType.FULL
                    data = buffer.get_buffer()

                subscriber.buffer_updated(key, data, view_type)

    def register_subscriber(self, subscriber: BufferSubscriber[K]) -> None:
        """
        Register a subscriber for buffer updates.

        Parameters
        ----------
        subscriber:
            The subscriber to register.
        """
        self._subscribers.append(subscriber)

    def get_buffer(self, key: K) -> sc.DataArray | None:
        """
        Get the complete buffered data for a key.

        Parameters
        ----------
        key:
            The key to query.

        Returns
        -------
        :
            The buffered data, or None if key is not buffered.
        """
        buffer = self._buffers.get(key)
        return buffer.get_buffer() if buffer else None

    def get_window(self, key: K, size: int | None = None) -> sc.DataArray | None:
        """
        Get a window of buffered data for a key.

        Parameters
        ----------
        key:
            The key to query.
        size:
            The number of elements to return from the end of the buffer.

        Returns
        -------
        :
            The window of buffered data, or None if key is not buffered.
        """
        buffer = self._buffers.get(key)
        return buffer.get_window(size) if buffer else None

    def get_memory_usage(self) -> dict[K, float]:
        """
        Get memory usage for all buffers.

        Returns
        -------
        :
            Dictionary mapping keys to memory usage in megabytes.
        """
        return {key: buffer.memory_mb for key, buffer in self._buffers.items()}

    def clear_buffer(self, key: K) -> None:
        """
        Clear a specific buffer.

        Parameters
        ----------
        key:
            The key of the buffer to clear.
        """
        buffer = self._buffers.get(key)
        if buffer:
            buffer.clear()

    def clear_all_buffers(self) -> None:
        """Clear all buffers."""
        for buffer in self._buffers.values():
            buffer.clear()
