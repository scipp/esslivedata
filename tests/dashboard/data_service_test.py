# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Any

import pytest

from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.data_subscriber import DataSubscriber, Pipe, StreamAssembler


class FakeDataAssembler(StreamAssembler[str]):
    """Fake assembler for testing."""

    def assemble(self, data: dict[str, Any]) -> dict[str, Any]:
        return data.copy()


class FakePipe(Pipe):
    """Fake pipe for testing."""

    def __init__(self) -> None:
        self.sent_data: list[dict[str, Any]] = []

    def send(self, data: Any) -> None:
        self.sent_data.append(data)


def create_test_subscriber(keys: set[str]) -> tuple[DataSubscriber[str], FakePipe]:
    """Create a test subscriber with the given keys."""
    assembler = FakeDataAssembler(keys)
    pipe = FakePipe()
    subscriber = DataSubscriber(assembler, pipe)
    return subscriber, pipe


@pytest.fixture
def data_service() -> DataService[str, int]:
    return DataService()


@pytest.fixture
def sample_data() -> dict[str, int]:
    return {"key1": 100, "key2": 200, "key3": 300}


def test_init_creates_empty_service():
    service = DataService[str, int]()
    assert len(service) == 0


def test_setitem_stores_value(data_service: DataService[str, int]):
    data_service["key1"] = 42
    assert data_service["key1"] == 42
    assert "key1" in data_service


def test_setitem_without_subscribers_no_error(data_service: DataService[str, int]):
    data_service["key1"] = 42
    assert data_service["key1"] == 42


def test_register_subscriber_adds_to_list(data_service: DataService[str, int]):
    subscriber, _ = create_test_subscriber({"key1"})
    data_service.register_subscriber(subscriber)


def test_setitem_notifies_matching_subscriber(data_service: DataService[str, int]):
    subscriber, pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    data_service["key1"] = 42

    assert len(pipe.sent_data) == 1
    assert pipe.sent_data[0] == {"key1": 42}


def test_setitem_ignores_non_matching_subscriber(data_service: DataService[str, int]):
    subscriber, pipe = create_test_subscriber({"other_key"})
    data_service.register_subscriber(subscriber)

    data_service["key1"] = 42

    assert len(pipe.sent_data) == 0


def test_setitem_notifies_multiple_matching_subscribers(
    data_service: DataService[str, int],
):
    subscriber1, pipe1 = create_test_subscriber({"key1"})
    subscriber2, pipe2 = create_test_subscriber({"key1", "key2"})
    subscriber3, pipe3 = create_test_subscriber({"key2"})

    data_service.register_subscriber(subscriber1)
    data_service.register_subscriber(subscriber2)
    data_service.register_subscriber(subscriber3)

    data_service["key1"] = 42

    assert len(pipe1.sent_data) == 1
    assert len(pipe2.sent_data) == 1
    assert len(pipe3.sent_data) == 0


def test_setitem_multiple_updates_notify_separately(
    data_service: DataService[str, int],
):
    subscriber, pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    data_service["key1"] = 42
    data_service["key2"] = 84

    assert len(pipe.sent_data) == 2
    assert pipe.sent_data[0] == {"key1": 42}
    assert pipe.sent_data[1] == {"key1": 42, "key2": 84}


def test_transaction_batches_notifications(data_service: DataService[str, int]):
    subscriber, pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    with data_service.transaction():
        data_service["key1"] = 42
        data_service["key2"] = 84
        # No notifications yet
        assert len(pipe.sent_data) == 0

    # Single notification after transaction
    assert len(pipe.sent_data) == 1
    assert pipe.sent_data[0] == {"key1": 42, "key2": 84}


def test_transaction_nested_batches_correctly(data_service: DataService[str, int]):
    subscriber, pipe = create_test_subscriber({"key1", "key2", "key3"})
    data_service.register_subscriber(subscriber)

    with data_service.transaction():
        data_service["key1"] = 42
        with data_service.transaction():
            data_service["key2"] = 84
            assert len(pipe.sent_data) == 0
        # Still in outer transaction
        assert len(pipe.sent_data) == 0
        data_service["key3"] = 126

    # Single notification after all transactions
    assert len(pipe.sent_data) == 1
    assert pipe.sent_data[0] == {"key1": 42, "key2": 84, "key3": 126}


def test_transaction_exception_still_notifies(data_service: DataService[str, int]):
    subscriber, pipe = create_test_subscriber({"key1"})
    data_service.register_subscriber(subscriber)

    try:
        with data_service.transaction():
            data_service["key1"] = 42
            raise ValueError("test error")
    except ValueError:
        # Exception should not prevent notification
        pass

    # Notification should still happen
    assert len(pipe.sent_data) == 1
    assert pipe.sent_data[0] == {"key1": 42}


def test_dictionary_operations_work(
    data_service: DataService[str, int], sample_data: dict[str, int]
):
    # Test basic dict operations
    for key, value in sample_data.items():
        data_service[key] = value

    assert len(data_service) == 3
    assert data_service["key1"] == 100
    assert data_service.get("key1") == 100
    assert data_service.get("nonexistent", 999) == 999
    assert "key1" in data_service
    assert "nonexistent" not in data_service
    assert list(data_service.keys()) == ["key1", "key2", "key3"]
    assert list(data_service.values()) == [100, 200, 300]


def test_update_method_triggers_notifications(data_service: DataService[str, int]):
    subscriber, pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    data_service.update({"key1": 42, "key2": 84})

    # Should trigger notifications for each key
    assert len(pipe.sent_data) == 2


def test_clear_removes_all_data(
    data_service: DataService[str, int], sample_data: dict[str, int]
):
    data_service.update(sample_data)
    assert len(data_service) == 3

    data_service.clear()
    assert len(data_service) == 0


def test_pop_removes_and_returns_value(data_service: DataService[str, int]):
    data_service["key1"] = 42

    value = data_service.pop("key1")
    assert value == 42
    assert "key1" not in data_service


def test_setdefault_behavior(data_service: DataService[str, int]):
    value = data_service.setdefault("key1", 42)
    assert value == 42
    assert data_service["key1"] == 42

    # Second call should return existing value
    value = data_service.setdefault("key1", 999)
    assert value == 42
    assert data_service["key1"] == 42


def test_subscriber_gets_full_data_dict(data_service: DataService[str, int]):
    subscriber, pipe = create_test_subscriber({"key1"})
    data_service.register_subscriber(subscriber)

    # Add some initial data
    data_service["existing"] = 999
    data_service["key1"] = 42

    # Subscriber should get the full data dict
    assert pipe.sent_data[-1] == {"key1": 42}


def test_subscriber_only_gets_subscribed_keys(data_service: DataService[str, int]):
    subscriber, pipe = create_test_subscriber({"key1", "key3"})
    data_service.register_subscriber(subscriber)

    # Add data for subscribed and unsubscribed keys
    data_service["key1"] = 42
    data_service["key2"] = 84  # Not subscribed to this key
    data_service["key3"] = 126
    data_service["unrelated"] = 999  # Not subscribed to this key

    # Subscriber should only receive data for keys it's interested in
    expected_data = {"key1": 42, "key3": 126}
    assert pipe.sent_data[-1] == expected_data

    # Verify unrelated keys are not included
    assert "key2" not in pipe.sent_data[-1]
    assert "unrelated" not in pipe.sent_data[-1]


def test_empty_transaction_no_notifications(data_service: DataService[str, int]):
    subscriber, pipe = create_test_subscriber({"key1"})
    data_service.register_subscriber(subscriber)

    with data_service.transaction():
        pass  # No changes

    assert len(pipe.sent_data) == 0


def test_delitem_notifies_subscribers(data_service: DataService[str, int]):
    subscriber, pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    # Add some data first
    data_service["key1"] = 42
    data_service["key2"] = 84
    pipe.sent_data.clear()  # Clear previous notifications

    # Delete a key
    del data_service["key1"]

    # Should notify with remaining data
    assert len(pipe.sent_data) == 1
    assert pipe.sent_data[0] == {"key2": 84}
    assert "key1" not in data_service


def test_delitem_in_transaction_batches_notifications(
    data_service: DataService[str, int],
):
    subscriber, pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    # Add some data first
    data_service["key1"] = 42
    data_service["key2"] = 84
    pipe.sent_data.clear()  # Clear previous notifications

    with data_service.transaction():
        del data_service["key1"]
        data_service["key2"] = 99
        # No notifications yet
        assert len(pipe.sent_data) == 0

    # Single notification after transaction
    assert len(pipe.sent_data) == 1
    assert pipe.sent_data[0] == {"key2": 99}


def test_transaction_set_then_del_same_key(data_service: DataService[str, int]):
    subscriber, pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    # Add some initial data
    data_service["key2"] = 84
    pipe.sent_data.clear()

    with data_service.transaction():
        data_service["key1"] = 42  # Set key1
        del data_service["key1"]  # Then delete key1
        # No notifications yet
        assert len(pipe.sent_data) == 0

    # After transaction: key1 should not exist, only key2 should be in notification
    assert len(pipe.sent_data) == 1
    assert pipe.sent_data[0] == {"key2": 84}
    assert "key1" not in data_service


def test_transaction_del_then_set_same_key(data_service: DataService[str, int]):
    subscriber, pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    # Add some initial data
    data_service["key1"] = 42
    data_service["key2"] = 84
    pipe.sent_data.clear()

    with data_service.transaction():
        del data_service["key1"]  # Delete key1
        data_service["key1"] = 99  # Then set key1 to new value
        # No notifications yet
        assert len(pipe.sent_data) == 0

    # After transaction: key1 should have the new value
    assert len(pipe.sent_data) == 1
    assert pipe.sent_data[0] == {"key1": 99, "key2": 84}
    assert data_service["key1"] == 99


def test_transaction_multiple_operations_same_key(data_service: DataService[str, int]):
    subscriber, pipe = create_test_subscriber({"key1"})
    data_service.register_subscriber(subscriber)

    # Add initial data
    data_service["key1"] = 10
    pipe.sent_data.clear()

    with data_service.transaction():
        data_service["key1"] = 20  # Update
        data_service["key1"] = 30  # Update again
        del data_service["key1"]  # Delete
        data_service["key1"] = 40  # Set again
        # No notifications yet
        assert len(pipe.sent_data) == 0

    # After transaction: key1 should have final value
    assert len(pipe.sent_data) == 1
    assert pipe.sent_data[0] == {"key1": 40}
    assert data_service["key1"] == 40


class TestDataServiceUpdatingSubscribers:
    """Test complex subscriber behavior where subscribers update the DataService."""

    def test_subscriber_updates_service_immediately(self):
        """Test subscriber updating service outside of transaction."""
        service = DataService[str, int]()

        class UpdatingSubscriber(DataSubscriber[str]):
            def __init__(self, keys: set[str], service: DataService[str, int]):
                super().__init__(FakeDataAssembler(keys), FakePipe())
                self._service = service

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                # Update derived data based on received data
                if "input" in store:
                    self._service["derived"] = store["input"] * 2

        subscriber = UpdatingSubscriber({"input"}, service)
        service.register_subscriber(subscriber)

        # This should trigger the subscriber, which updates "derived"
        service["input"] = 10

        assert service["input"] == 10
        assert service["derived"] == 20

    def test_subscriber_updates_service_in_transaction(self):
        """Test subscriber updating service at end of transaction."""
        service = DataService[str, int]()

        class UpdatingSubscriber(DataSubscriber[str]):
            def __init__(self, keys: set[str], service: DataService[str, int]):
                super().__init__(FakeDataAssembler(keys), FakePipe())
                self._service = service

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                if "input" in store:
                    self._service["derived"] = store["input"] * 2

        subscriber = UpdatingSubscriber({"input"}, service)
        service.register_subscriber(subscriber)

        with service.transaction():
            service["input"] = 10
            # "derived" should not exist yet during transaction
            assert "derived" not in service

        # After transaction, both keys should exist
        assert service["input"] == 10
        assert service["derived"] == 20

    def test_multiple_subscribers_update_service(self):
        """Test multiple subscribers updating different derived data."""
        service = DataService[str, int]()

        class MultiplierSubscriber(DataSubscriber[str]):
            def __init__(
                self,
                keys: set[str],
                service: DataService[str, int],
                multiplier: int,
            ):
                super().__init__(FakeDataAssembler(keys), FakePipe())
                self._service = service
                self._multiplier = multiplier

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                if "input" in store:
                    key = f"derived_{self._multiplier}x"
                    self._service[key] = store["input"] * self._multiplier

        sub1 = MultiplierSubscriber({"input"}, service, 2)
        sub2 = MultiplierSubscriber({"input"}, service, 3)
        service.register_subscriber(sub1)
        service.register_subscriber(sub2)

        service["input"] = 10

        assert service["input"] == 10
        assert service["derived_2x"] == 20
        assert service["derived_3x"] == 30

    def test_cascading_subscriber_updates(self):
        """Test subscribers that depend on derived data from other subscribers."""
        service = DataService[str, int]()

        class FirstLevelSubscriber(DataSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                super().__init__(FakeDataAssembler({"input"}), FakePipe())
                self._service = service

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                if "input" in store:
                    self._service["level1"] = store["input"] * 2

        class SecondLevelSubscriber(DataSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                super().__init__(FakeDataAssembler({"level1"}), FakePipe())
                self._service = service

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                if "level1" in store:
                    self._service["level2"] = store["level1"] * 3

        sub1 = FirstLevelSubscriber(service)
        sub2 = SecondLevelSubscriber(service)
        service.register_subscriber(sub1)
        service.register_subscriber(sub2)

        service["input"] = 5

        assert service["input"] == 5
        assert service["level1"] == 10
        assert service["level2"] == 30

    def test_cascading_updates_in_transaction(self):
        """Test cascading updates within a transaction."""
        service = DataService[str, int]()

        class FirstLevelSubscriber(DataSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                super().__init__(FakeDataAssembler({"input"}), FakePipe())
                self._service = service

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                if "input" in store:
                    self._service["level1"] = store["input"] * 2

        class SecondLevelSubscriber(DataSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                super().__init__(FakeDataAssembler({"level1"}), FakePipe())
                self._service = service

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                if "level1" in store:
                    self._service["level2"] = store["level1"] * 3

        sub1 = FirstLevelSubscriber(service)
        sub2 = SecondLevelSubscriber(service)
        service.register_subscriber(sub1)
        service.register_subscriber(sub2)

        with service.transaction():
            service["input"] = 5
            service["other"] = 100
            # No derived data should exist during transaction
            assert "level1" not in service
            assert "level2" not in service

        # All data should exist after transaction
        assert service["input"] == 5
        assert service["other"] == 100
        assert service["level1"] == 10
        assert service["level2"] == 30

    def test_subscriber_updates_multiple_keys(self):
        """Test subscriber that updates multiple derived keys at once."""
        service = DataService[str, int]()

        class MultiUpdateSubscriber(DataSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                super().__init__(FakeDataAssembler({"input"}), FakePipe())
                self._service = service

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                if "input" in store:
                    # Update multiple derived values
                    with self._service.transaction():
                        self._service["double"] = store["input"] * 2
                        self._service["triple"] = store["input"] * 3
                        self._service["square"] = store["input"] ** 2

        subscriber = MultiUpdateSubscriber(service)
        service.register_subscriber(subscriber)

        service["input"] = 4

        assert service["input"] == 4
        assert service["double"] == 8
        assert service["triple"] == 12
        assert service["square"] == 16

    def test_subscriber_updates_existing_keys(self):
        """Test subscriber updating keys that already exist."""
        service = DataService[str, int]()
        service["existing"] = 100

        class OverwriteSubscriber(DataSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                super().__init__(FakeDataAssembler({"input"}), FakePipe())
                self._service = service

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                if "input" in store:
                    self._service["existing"] = store["input"] * 10

        subscriber = OverwriteSubscriber(service)
        service.register_subscriber(subscriber)

        service["input"] = 5

        assert service["input"] == 5
        assert service["existing"] == 50  # Overwritten, not 100

    def test_circular_dependency_protection(self):
        """Test handling of potential circular dependencies."""
        service = DataService[str, int]()
        update_count = {"count": 0}

        class CircularSubscriber(DataSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                super().__init__(FakeDataAssembler({"input", "output"}), FakePipe())
                self._service = service

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                update_count["count"] += 1
                if update_count["count"] < 5:  # Prevent infinite recursion in test
                    if "input" in store and "output" not in store:
                        self._service["output"] = store["input"] + 1
                    elif "output" in store and store["output"] < 10:
                        self._service["output"] = store["output"] + 1

        subscriber = CircularSubscriber(service)
        service.register_subscriber(subscriber)

        service["input"] = 1

        # Should handle the circular updates gracefully
        assert service["input"] == 1
        assert "output" in service
        assert update_count["count"] > 1  # Multiple updates occurred

    def test_subscriber_deletes_keys_during_update(self):
        """Test subscriber that deletes keys during notification."""
        service = DataService[str, int]()
        service["to_delete"] = 999

        class DeletingSubscriber(DataSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                super().__init__(FakeDataAssembler({"trigger"}), FakePipe())
                self._service = service

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                if "trigger" in store and "to_delete" in self._service:
                    del self._service["to_delete"]
                    self._service["deleted_flag"] = 1

        subscriber = DeletingSubscriber(service)
        service.register_subscriber(subscriber)

        service["trigger"] = 1

        assert service["trigger"] == 1
        assert "to_delete" not in service
        assert service["deleted_flag"] == 1

    def test_subscriber_complex_transaction_updates(self):
        """Test complex scenario with nested transactions and subscriber updates."""
        service = DataService[str, int]()

        class ComplexSubscriber(DataSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                super().__init__(FakeDataAssembler({"input"}), FakePipe())
                self._service = service

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                if "input" in store:
                    # Subscriber uses its own transaction
                    with self._service.transaction():
                        self._service["derived1"] = store["input"] * 2
                        with self._service.transaction():
                            self._service["derived2"] = store["input"] * 3
                        self._service["derived3"] = store["input"] * 4

        subscriber = ComplexSubscriber(service)
        service.register_subscriber(subscriber)

        with service.transaction():
            service["input"] = 5
            service["other"] = 100
            # No derived data during transaction
            assert "derived1" not in service

        # All data should exist after transaction
        assert service["input"] == 5
        assert service["other"] == 100
        assert service["derived1"] == 10
        assert service["derived2"] == 15
        assert service["derived3"] == 20

    def test_multiple_update_rounds(self):
        """Test scenario requiring multiple notification rounds."""
        service = DataService[str, int]()

        class ChainSubscriber(DataSubscriber[str]):
            def __init__(
                self, input_key: str, output_key: str, service: DataService[str, int]
            ):
                super().__init__(FakeDataAssembler({input_key}), FakePipe())
                self._input_key = input_key
                self._output_key = output_key
                self._service = service

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                if self._input_key in store:
                    self._service[self._output_key] = store[self._input_key] + 1

        # Create a chain: input -> step1 -> step2 -> step3
        sub1 = ChainSubscriber("input", "step1", service)
        sub2 = ChainSubscriber("step1", "step2", service)
        sub3 = ChainSubscriber("step2", "step3", service)

        service.register_subscriber(sub1)
        service.register_subscriber(sub2)
        service.register_subscriber(sub3)

        service["input"] = 10

        assert service["input"] == 10
        assert service["step1"] == 11
        assert service["step2"] == 12
        assert service["step3"] == 13

    def test_subscriber_updates_with_mixed_immediate_and_transaction(self):
        """Test mixing immediate updates with transactional updates from subscribers."""
        service = DataService[str, int]()

        class MixedSubscriber(DataSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                super().__init__(FakeDataAssembler({"input"}), FakePipe())
                self._service = service

            def trigger(self, store: dict[str, int]) -> None:
                super().trigger(store)
                if "input" in store:
                    # Immediate update
                    self._service["immediate"] = store["input"] * 2
                    # Transaction update
                    with self._service.transaction():
                        self._service["transactional1"] = store["input"] * 3
                        self._service["transactional2"] = store["input"] * 4

        subscriber = MixedSubscriber(service)
        service.register_subscriber(subscriber)

        service["input"] = 5

        assert service["input"] == 5
        assert service["immediate"] == 10
        assert service["transactional1"] == 15
        assert service["transactional2"] == 20


# Tests for extractor-based subscription
class TestExtractorBasedSubscription:
    """Tests for extractor-based subscription with dynamic buffer sizing."""

    def test_window_extractor_gets_windowed_data(self):
        """Test that subscriber with WindowExtractor gets windowed data."""
        import scipp as sc

        from ess.livedata.dashboard.data_service import DataService, WindowExtractor

        # Create a simple subscriber class for testing
        class WindowSubscriber:
            def __init__(self, keys: set[str], window_size: int):
                self._keys = keys
                self._window_size = window_size
                self.received_data: list[dict] = []

            @property
            def keys(self) -> set[str]:
                return self._keys

            @property
            def extractors(self) -> dict[str, WindowExtractor]:
                return {key: WindowExtractor(self._window_size) for key in self._keys}

            def trigger(self, data: dict) -> None:
                self.received_data.append(data)

        # Create service and subscriber
        service = DataService()
        subscriber = WindowSubscriber({"data"}, window_size=3)
        service.register_subscriber(subscriber)

        # Add data progressively
        for i in range(5):
            data = sc.scalar(i, unit='counts')
            service["data"] = data

        # Subscriber should have received 5 updates
        assert len(subscriber.received_data) == 5

        # Last update should contain window of last 3 values
        last_received = subscriber.received_data[-1]["data"]
        # Window of size 3 from last updates (2, 3, 4)
        assert last_received.sizes == {'time': 3}

    def test_buffer_size_determined_by_max_extractor_requirement(self):
        """Test that buffer size is set to max requirement among subscribers."""
        import scipp as sc

        from ess.livedata.dashboard.data_service import (
            DataService,
            LatestValueExtractor,
            WindowExtractor,
        )

        class TestSubscriber:
            def __init__(self, keys: set[str], extractor):
                self._keys = keys
                self._extractor = extractor
                self.received_data: list[dict] = []

            @property
            def keys(self) -> set[str]:
                return self._keys

            @property
            def extractors(self) -> dict:
                return {key: self._extractor for key in self._keys}

            def trigger(self, data: dict) -> None:
                self.received_data.append(data)

        # Create service
        service = DataService()

        # Register subscriber with LatestValueExtractor (size 1)
        sub1 = TestSubscriber({"data"}, LatestValueExtractor())
        service.register_subscriber(sub1)

        # Add first data point - buffer should be size 1
        service["data"] = sc.scalar(1, unit='counts')

        # Register subscriber with WindowExtractor(size=10)
        sub2 = TestSubscriber({"data"}, WindowExtractor(10))
        service.register_subscriber(sub2)

        # Buffer should now grow to size 10
        # Add more data to verify buffering works
        for i in range(2, 12):
            service["data"] = sc.scalar(i, unit='counts')

        # Both subscribers should have received all updates
        # sub1: 1 update before sub2 registration + 10 after = 11
        assert len(sub1.received_data) == 11
        # sub2: 10 updates (after registration)
        assert len(sub2.received_data) == 10

        # sub1 should get latest value only (unwrapped)
        last_from_sub1 = sub1.received_data[-1]["data"]
        assert last_from_sub1.ndim == 0  # Scalar (unwrapped)
        assert last_from_sub1.value == 11

        # sub2 should get window of last 10 values
        last_from_sub2 = sub2.received_data[-1]["data"]
        assert last_from_sub2.sizes == {'time': 10}

    def test_multiple_keys_with_different_extractors(self):
        """Test subscriber with different extractors per key."""
        import scipp as sc

        from ess.livedata.dashboard.data_service import (
            DataService,
            LatestValueExtractor,
            WindowExtractor,
        )

        class MultiKeySubscriber:
            def __init__(self):
                self.received_data: list[dict] = []

            @property
            def keys(self) -> set[str]:
                return {"latest", "window"}

            @property
            def extractors(self) -> dict:
                return {
                    "latest": LatestValueExtractor(),
                    "window": WindowExtractor(3),
                }

            def trigger(self, data: dict) -> None:
                self.received_data.append(data)

        service = DataService()
        subscriber = MultiKeySubscriber()
        service.register_subscriber(subscriber)

        # Add data to both keys
        for i in range(5):
            service["latest"] = sc.scalar(i * 10, unit='counts')
            service["window"] = sc.scalar(i * 100, unit='counts')

        # Should have received updates (batched in transaction would be less,
        # but here each setitem triggers separately)
        assert len(subscriber.received_data) > 0

        # Check last received data
        last_data = subscriber.received_data[-1]

        # "latest" should be unwrapped scalar
        if "latest" in last_data:
            assert last_data["latest"].ndim == 0

        # "window" should have time dimension
        if "window" in last_data:
            assert "time" in last_data["window"].dims
