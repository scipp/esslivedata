# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Benchmarks for DataService LatestFrame extraction with subscriber notifications."""

from __future__ import annotations

from typing import Any

import pytest
import scipp as sc

from ess.livedata.dashboard.data_service import DataService, DataServiceSubscriber
from ess.livedata.dashboard.extractors import LatestValueExtractor


class SimpleSubscriber(DataServiceSubscriber[str]):
    """Simple subscriber that tracks trigger calls."""

    def __init__(self, keys: set[str]) -> None:
        """Initialize subscriber with given keys."""
        self._keys_set = keys
        self.trigger_count = 0
        self.received_updates: list[dict[str, Any]] = []
        super().__init__()

    @property
    def keys(self) -> set[str]:
        """Return the keys this subscriber depends on."""
        return self._keys_set

    @property
    def extractors(self) -> dict[str, LatestValueExtractor]:
        """Return extractors for all keys."""
        return {key: LatestValueExtractor() for key in self._keys_set}

    def trigger(self, store: dict[str, Any]) -> None:
        """Track trigger calls and received updates."""
        self.trigger_count += 1
        self.received_updates.append(store.copy())


class TestDataServiceBenchmark:
    """Benchmarks for DataService with LatestFrame extraction."""

    @pytest.fixture
    def service(self) -> DataService[str, sc.Variable]:
        """Create a fresh DataService for each benchmark."""
        return DataService()

    @pytest.fixture
    def sample_data(self) -> dict[str, sc.Variable]:
        """Create sample data with scipp Variables."""
        return {
            'detector_counts': sc.scalar(100, unit='counts'),
            'monitor_counts': sc.scalar(50, unit='counts'),
            'temperature': sc.scalar(298.15, unit='K'),
        }

    def test_update_multiple_keys_with_subscriber(
        self, benchmark, service: DataService[str, sc.Variable], sample_data
    ):
        """Benchmark updating multiple keys with subscriber watching all."""
        subscriber = SimpleSubscriber(set(sample_data.keys()))
        service.register_subscriber(subscriber)

        def update_multiple_keys_with_subscriber():
            with service.transaction():
                for key, value in sample_data.items():
                    service[key] = value

        benchmark(update_multiple_keys_with_subscriber)
        assert len(subscriber.received_updates) > 0

    def test_update_with_multiple_subscribers_same_key(
        self, benchmark, service: DataService[str, sc.Variable]
    ):
        """Benchmark update with multiple subscribers watching the same key."""
        subscribers = [SimpleSubscriber({'data'}) for _ in range(5)]
        for sub in subscribers:
            service.register_subscriber(sub)

        data = sc.scalar(42, unit='counts')

        def update_with_subscribers():
            service['data'] = data

        benchmark(update_with_subscribers)
        # Each subscriber should have been triggered
        for sub in subscribers:
            assert sub.trigger_count >= 1

    def test_update_many_keys_many_subscribers(
        self, benchmark, service: DataService[str, sc.Variable]
    ):
        """Benchmark updating many keys with many subscribers (one per key)."""
        keys = [f'key_{i}' for i in range(100)]
        # Create one subscriber per key - typical real-world scenario
        subscribers = [SimpleSubscriber({key}) for key in keys]
        for sub in subscribers:
            service.register_subscriber(sub)

        def update_many():
            with service.transaction():
                for i, key in enumerate(keys):
                    service[key] = sc.scalar(i, unit='counts')
            return len(service)

        result = benchmark(update_many)
        assert result == 100
        # Each subscriber should have been triggered once
        for sub in subscribers:
            assert sub.trigger_count >= 1

    def test_extract_from_large_service(
        self, benchmark, service: DataService[str, sc.Variable]
    ):
        """Benchmark extracting values from large service via subscriber."""
        keys = [f'key_{i}' for i in range(1000)]

        # Populate service
        for i, key in enumerate(keys):
            service[key] = sc.scalar(i, unit='counts')

        # Subscribe to first 10 keys
        watched_keys = set(keys[:10])
        subscriber = SimpleSubscriber(watched_keys)
        service.register_subscriber(subscriber)

        # Reset trigger count to avoid counting initialization
        subscriber.trigger_count = 0

        def update_and_extract():
            service[keys[0]] = sc.scalar(999, unit='counts')

        benchmark(update_and_extract)
        # Subscriber should have received the update
        assert subscriber.trigger_count >= 1
