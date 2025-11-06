# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for HistoryBufferService."""

from __future__ import annotations

import scipp as sc

from ess.livedata.dashboard.history_buffer_service import (
    FullHistoryExtractor,
    HistoryBufferService,
    HistorySubscriber,
    UpdateExtractor,
    WindowExtractor,
)


def make_data(
    size: int, extra_dim: str | None = None, extra_size: int | None = None
) -> sc.DataArray:
    """Create test data with proper time coordinate."""
    if extra_dim is None:
        data = sc.DataArray(
            sc.ones(dims=["time"], shape=[size]),
            coords={
                "time": sc.array(dims=["time"], values=list(range(size)), dtype="int64")
            },
        )
    else:
        data = sc.DataArray(
            sc.ones(dims=["time", extra_dim], shape=[size, extra_size or 1]),
            coords={
                "time": sc.array(dims=["time"], values=list(range(size)), dtype="int64")
            },
        )
    return data


class SimpleSubscriber(HistorySubscriber[str]):
    """Test subscriber that collects updates."""

    def __init__(
        self,
        keys: set[str] | None = None,
        extractors: dict[str, UpdateExtractor] | None = None,
    ) -> None:
        """Initialize with optional key set and extractors."""
        self._keys = keys or set()
        self._extractors = extractors or {}
        self._updates: list[dict[str, sc.DataArray]] = []

    @property
    def extractors(self) -> dict[str, UpdateExtractor]:
        """Return extractors."""
        return self._extractors

    @property
    def keys(self) -> set[str]:
        """Return tracked keys."""
        return self._keys

    def on_update(self, data: dict[str, sc.DataArray]) -> None:
        """Collect updates."""
        self._updates.append(data.copy())

    def get_updates(self) -> list[dict[str, sc.DataArray]]:
        """Return all collected updates."""
        return self._updates


class TestHistoryBufferServiceBasic:
    """Test basic HistoryBufferService functionality without DataService."""

    def test_add_data_single_key(self):
        """Test adding data to a single key."""
        service = HistoryBufferService[str](data_service=None)

        subscriber = SimpleSubscriber(
            keys={"data"},
            extractors={"data": FullHistoryExtractor()},
        )
        service.register_subscriber(subscriber)

        # Add data
        data = make_data(5)
        service.add_data({"data": data})

        # Verify subscriber got the update
        assert len(subscriber.get_updates()) == 1
        assert "data" in subscriber.get_updates()[0]
        result = subscriber.get_updates()[0]["data"]
        assert result.sizes["time"] == 5

    def test_add_data_multiple_keys(self):
        """Test adding data to multiple keys."""
        service = HistoryBufferService[str](data_service=None)

        subscriber = SimpleSubscriber(
            keys={"key1", "key2"},
            extractors={},
        )
        service.register_subscriber(subscriber)

        # Add data
        data1 = make_data(3)
        data2 = make_data(3)
        service.add_data({"key1": data1, "key2": data2})

        # Verify both keys received data
        assert len(subscriber.get_updates()) == 1
        update = subscriber.get_updates()[0]
        assert "key1" in update
        assert "key2" in update

    def test_window_extractor(self):
        """Test WindowExtractor limiting returned data."""
        service = HistoryBufferService[str](data_service=None)

        subscriber = SimpleSubscriber(
            keys={"data"},
            extractors={"data": WindowExtractor(size=3)},
        )
        service.register_subscriber(subscriber)

        # Add data in chunks
        for _ in range(3):
            data = make_data(2)
            service.add_data({"data": data})

        # Should have 3 updates (one per add_data call)
        assert len(subscriber.get_updates()) == 3

        # Last update should have limited window
        last_update = subscriber.get_updates()[-1]["data"]
        # Window size is 3, so total across all adds is 6, last window is 3
        assert last_update.sizes["time"] <= 3

    def test_full_history_extractor(self):
        """Test FullHistoryExtractor accumulating all data."""
        service = HistoryBufferService[str](data_service=None)

        subscriber = SimpleSubscriber(
            keys={"data"},
            extractors={"data": FullHistoryExtractor()},
        )
        service.register_subscriber(subscriber)

        # Add data multiple times
        for _ in range(3):
            data = make_data(2)
            service.add_data({"data": data})

        # Each update should have accumulated data
        updates = subscriber.get_updates()
        assert len(updates) == 3
        # First update: 2 items
        assert updates[0]["data"].sizes["time"] == 2
        # Second update: 4 items
        assert updates[1]["data"].sizes["time"] == 4
        # Third update: 6 items
        assert updates[2]["data"].sizes["time"] == 6

    def test_selective_keys(self):
        """Test that subscribers only get keys they care about."""
        service = HistoryBufferService[str](data_service=None)

        subscriber1 = SimpleSubscriber(keys={"key1"})
        subscriber2 = SimpleSubscriber(keys={"key2"})
        service.register_subscriber(subscriber1)
        service.register_subscriber(subscriber2)

        # Add data for both keys
        data = make_data(1)
        service.add_data({"key1": data, "key2": data})

        # Each subscriber should only see their key
        assert "key1" in subscriber1.get_updates()[0]
        assert "key1" not in subscriber2.get_updates()[0]
        assert "key2" in subscriber2.get_updates()[0]
        assert "key2" not in subscriber1.get_updates()[0]

    def test_unregister_subscriber(self):
        """Test unregistering a subscriber."""
        service = HistoryBufferService[str](data_service=None)

        subscriber = SimpleSubscriber(keys={"data"})
        service.register_subscriber(subscriber)

        # Add data
        data = make_data(1)
        service.add_data({"data": data})
        assert len(subscriber.get_updates()) == 1

        # Unregister
        service.unregister_subscriber(subscriber)

        # Add more data - subscriber should not be notified
        service.add_data({"data": data})
        assert len(subscriber.get_updates()) == 1

    def test_no_notification_for_irrelevant_updates(self):
        """Test that subscribers aren't notified for keys they don't care about."""
        service = HistoryBufferService[str](data_service=None)

        subscriber = SimpleSubscriber(keys={"key1"})
        service.register_subscriber(subscriber)

        # Add data for a different key
        data = make_data(1)
        service.add_data({"key2": data})

        # Subscriber should not have been notified
        assert len(subscriber.get_updates()) == 0

    def test_get_tracked_keys(self):
        """Test tracking of all keys across subscribers."""
        service = HistoryBufferService[str](data_service=None)

        subscriber1 = SimpleSubscriber(keys={"key1", "key2"})
        subscriber2 = SimpleSubscriber(keys={"key2", "key3"})
        service.register_subscriber(subscriber1)
        service.register_subscriber(subscriber2)

        tracked = service.get_tracked_keys()
        assert tracked == {"key1", "key2", "key3"}

    def test_memory_usage(self):
        """Test memory tracking."""
        service = HistoryBufferService[str](data_service=None)

        subscriber = SimpleSubscriber(keys={"data"})
        service.register_subscriber(subscriber)

        # Add data
        data = make_data(100)
        service.add_data({"data": data})

        # Check memory usage
        memory_usage = service.get_memory_usage()
        assert subscriber in memory_usage
        assert "data" in memory_usage[subscriber]
        assert memory_usage[subscriber]["data"] > 0

    def test_clear_all_buffers(self):
        """Test clearing all buffers."""
        service = HistoryBufferService[str](data_service=None)

        subscriber = SimpleSubscriber(keys={"data"})
        service.register_subscriber(subscriber)

        # Add data
        data = make_data(10)
        service.add_data({"data": data})

        # Verify data was added
        memory_before = service.get_memory_usage()[subscriber]["data"]
        assert memory_before > 0

        # Clear buffers
        service.clear_all_buffers()

        # Memory should be zero
        memory_after = service.get_memory_usage()[subscriber]["data"]
        assert memory_after == 0

    def test_lazy_buffer_initialization(self):
        """Test that buffers are created lazily for each subscriber."""
        service = HistoryBufferService[str](data_service=None)

        subscriber = SimpleSubscriber(keys={"data"})
        # Initially empty
        memory_usage = service.get_memory_usage()
        assert subscriber not in memory_usage or len(memory_usage[subscriber]) == 0

        service.register_subscriber(subscriber)

        # Add data - buffer should be created
        data = make_data(5)
        service.add_data({"data": data})

        memory_usage = service.get_memory_usage()
        assert memory_usage[subscriber]["data"] > 0

    def test_multiple_subscribers_independent_buffers(self):
        """Test that multiple subscribers maintain independent buffers."""
        service = HistoryBufferService[str](data_service=None)

        subscriber1 = SimpleSubscriber(
            keys={"data"},
            extractors={"data": FullHistoryExtractor()},
        )
        subscriber2 = SimpleSubscriber(
            keys={"data"},
            extractors={"data": WindowExtractor(size=2)},
        )
        service.register_subscriber(subscriber1)
        service.register_subscriber(subscriber2)

        # Add data
        data = make_data(3)
        service.add_data({"data": data})

        # Subscriber 1 gets full history
        assert subscriber1.get_updates()[0]["data"].sizes["time"] == 3
        # Subscriber 2 gets windowed data (size limit is 2)
        assert subscriber2.get_updates()[0]["data"].sizes["time"] == 2

    def test_with_multiple_dimensions(self):
        """Test with multidimensional data."""
        service = HistoryBufferService[str](data_service=None)

        subscriber = SimpleSubscriber(keys={"data"})
        service.register_subscriber(subscriber)

        # Add 2D data
        data = make_data(5, extra_dim="x", extra_size=3)
        service.add_data({"data": data})

        assert len(subscriber.get_updates()) == 1
        result = subscriber.get_updates()[0]["data"]
        assert result.sizes["time"] == 5
        assert result.sizes["x"] == 3
