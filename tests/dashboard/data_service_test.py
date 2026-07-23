# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import threading
from collections.abc import Callable

import pytest
import scipp as sc

from ess.livedata.dashboard.data_service import DataService, DataServiceSubscriber
from ess.livedata.dashboard.extractors import LatestValueExtractor


def make_test_data(value: int, time: float = 0.0) -> sc.DataArray:
    """Create a test DataArray with the given value and time coordinate."""
    return sc.DataArray(
        sc.scalar(value, unit='counts'),
        coords={'time': sc.scalar(time, unit='s')},
    )


class FakePipe:
    """Fake pipe for testing."""

    def __init__(self) -> None:
        self.notifications: list[set[str]] = []

    def record_keys(self, updated_keys: set[str]) -> None:
        self.notifications.append(updated_keys)


class SimpleTestSubscriber(DataServiceSubscriber[str]):
    """Simple subscriber for testing DataService behavior."""

    def __init__(self, keys: set[str]) -> None:
        self._keys_set = keys
        self._extractors = {key: LatestValueExtractor() for key in keys}
        self._pipe = FakePipe()
        super().__init__()

    @property
    def extractors(self):
        return self._extractors

    @property
    def pipe(self) -> FakePipe:
        return self._pipe

    def on_updated(self, updated_keys: set[str]) -> None:
        self._pipe.record_keys(updated_keys)


def create_test_subscriber(keys: set[str]) -> tuple[SimpleTestSubscriber, Callable]:
    """
    Create a test subscriber with the given keys.

    Returns the subscriber and a callable to get the pipe.
    """
    subscriber = SimpleTestSubscriber(keys)

    def get_pipe() -> FakePipe:
        """Get the pipe."""
        return subscriber.pipe

    return subscriber, get_pipe


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
    import scipp as sc

    data = sc.DataArray(
        sc.scalar(42, unit='counts'), coords={'time': sc.scalar(0.0, unit='s')}
    )
    data_service["key1"] = data
    retrieved = data_service["key1"]
    assert retrieved.value == 42
    assert "key1" in data_service


def test_setitem_without_subscribers_no_error(data_service: DataService[str, int]):
    import scipp as sc

    data = sc.DataArray(
        sc.scalar(42, unit='counts'), coords={'time': sc.scalar(0.0, unit='s')}
    )
    data_service["key1"] = data
    retrieved = data_service["key1"]
    assert retrieved.value == 42


def test_register_subscriber_adds_to_list(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1"})
    data_service.register_subscriber(subscriber)
    # Registration does not trigger notification
    pipe = get_pipe()
    assert len(pipe.notifications) == 0


def test_unregister_subscriber_removes_from_list(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1"})
    data_service.register_subscriber(subscriber)

    result = data_service.unregister_subscriber(subscriber)

    assert result is True
    # Subscriber should no longer receive updates
    data_service["key1"] = make_test_data(42)
    pipe = get_pipe()
    assert len(pipe.notifications) == 0


def test_unregister_nonexistent_subscriber_returns_false(
    data_service: DataService[str, int],
):
    subscriber, _ = create_test_subscriber({"key1"})
    # Don't register the subscriber

    result = data_service.unregister_subscriber(subscriber)

    assert result is False


def test_unregister_during_notification_does_not_skip_other_subscribers(
    data_service: DataService[str, int],
):
    """A subscriber removed mid-notification must not cause a sibling to be skipped.

    Reproduces the list-index skip: notifying iterates the subscriber list, and a
    notification (or, in production, a concurrent UI-thread teardown) removes a
    subscriber positioned before one not yet visited. A raw list-index iterator
    would then skip the following subscriber.
    """
    triggered: list[str] = []

    class RemovingSubscriber(DataServiceSubscriber[str]):
        def __init__(self, name: str) -> None:
            self.name = name
            self.target: DataServiceSubscriber[str] | None = None
            self._extractors = {"key1": LatestValueExtractor()}
            super().__init__()

        @property
        def extractors(self):
            return self._extractors

        def on_updated(self, updated_keys: set[str]) -> None:
            if "key1" not in updated_keys:
                return
            triggered.append(self.name)
            if self.target is not None:
                data_service.unregister_subscriber(self.target)
                self.target = None

    first = RemovingSubscriber("first")
    middle = RemovingSubscriber("middle")
    last = RemovingSubscriber("last")
    # Registration order fixes iteration order: middle removes the earlier
    # 'first', shifting indices so a naive iterator would skip 'last'.
    middle.target = first
    for sub in (first, middle, last):
        data_service.register_subscriber(sub)

    data_service["key1"] = make_test_data(42)

    assert triggered == ["first", "middle", "last"]


def test_setitem_notifies_matching_subscriber(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    data_service["key1"] = make_test_data(42)

    pipe = get_pipe()
    assert len(pipe.notifications) == 1
    assert pipe.notifications[0] == {"key1"}
    snapshot = data_service.snapshot(subscriber)
    assert snapshot["key1"].value == 42


def test_setitem_ignores_non_matching_subscriber(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"other_key"})
    data_service.register_subscriber(subscriber)

    data_service["key1"] = make_test_data(42)

    pipe = get_pipe()
    assert len(pipe.notifications) == 0


def test_setitem_notifies_multiple_matching_subscribers(
    data_service: DataService[str, int],
):
    subscriber1, get_pipe1 = create_test_subscriber({"key1"})
    subscriber2, get_pipe2 = create_test_subscriber({"key1", "key2"})
    subscriber3, get_pipe3 = create_test_subscriber({"key2"})

    data_service.register_subscriber(subscriber1)
    data_service.register_subscriber(subscriber2)
    data_service.register_subscriber(subscriber3)

    data_service["key1"] = make_test_data(42)

    pipe1, pipe2, pipe3 = get_pipe1(), get_pipe2(), get_pipe3()
    assert len(pipe1.notifications) == 1
    assert len(pipe2.notifications) == 1
    assert len(pipe3.notifications) == 0


def test_setitem_multiple_updates_notify_separately(
    data_service: DataService[str, int],
):
    subscriber, get_pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    data_service["key1"] = make_test_data(42)
    data_service["key2"] = make_test_data(84)

    pipe = get_pipe()
    assert len(pipe.notifications) == 2
    assert pipe.notifications[0] == {"key1"}
    assert pipe.notifications[1] == {"key2"}
    snapshot = data_service.snapshot(subscriber)
    assert snapshot["key1"].value == 42
    assert snapshot["key2"].value == 84


def test_transaction_batches_notifications(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    pipe = get_pipe()
    with data_service.transaction():
        data_service["key1"] = make_test_data(42)
        data_service["key2"] = make_test_data(84)
        # No notifications yet
        assert len(pipe.notifications) == 0

    # Single notification after transaction
    assert len(pipe.notifications) == 1
    assert pipe.notifications[0] == {"key1", "key2"}
    snapshot = data_service.snapshot(subscriber)
    assert snapshot["key1"].value == 42
    assert snapshot["key2"].value == 84


def test_transaction_nested_batches_correctly(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1", "key2", "key3"})
    data_service.register_subscriber(subscriber)

    pipe = get_pipe()
    with data_service.transaction():
        data_service["key1"] = make_test_data(42)
        with data_service.transaction():
            data_service["key2"] = make_test_data(84)
            assert len(pipe.notifications) == 0
        # Still in outer transaction
        assert len(pipe.notifications) == 0
        data_service["key3"] = make_test_data(126)

    # Single notification after all transactions
    assert len(pipe.notifications) == 1
    assert pipe.notifications[0] == {"key1", "key2", "key3"}
    snapshot = data_service.snapshot(subscriber)
    assert snapshot["key1"].value == 42
    assert snapshot["key2"].value == 84
    assert snapshot["key3"].value == 126


def test_transaction_exception_still_notifies(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1"})
    data_service.register_subscriber(subscriber)

    try:
        with data_service.transaction():
            data_service["key1"] = make_test_data(42)
            raise ValueError("test error")
    except ValueError:
        # Exception should not prevent notification
        pass

    # Notification should still happen
    pipe = get_pipe()
    assert len(pipe.notifications) == 1
    assert pipe.notifications[0] == {"key1"}
    snapshot = data_service.snapshot(subscriber)
    assert snapshot["key1"].value == 42


def test_dictionary_operations_work(
    data_service: DataService[str, int],
):
    # Test basic dict operations with proper scipp data
    test_data = {
        "key1": make_test_data(100),
        "key2": make_test_data(200),
        "key3": make_test_data(300),
    }
    for key, value in test_data.items():
        data_service[key] = value

    assert len(data_service) == 3
    assert data_service["key1"].value == 100
    assert data_service.get("key1").value == 100
    assert data_service.get("nonexistent") is None
    assert "key1" in data_service
    assert "nonexistent" not in data_service
    assert list(data_service.keys()) == ["key1", "key2", "key3"]
    assert all(
        v.value == v_expected
        for v, v_expected in zip(data_service.values(), [100, 200, 300], strict=True)
    )


def test_update_method_triggers_notifications(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    data_service.update({"key1": make_test_data(42), "key2": make_test_data(84)})

    # Should trigger notifications for each key
    pipe = get_pipe()
    assert len(pipe.notifications) == 2


def test_clear_removes_all_data(
    data_service: DataService[str, int],
):
    test_data = {
        "key1": make_test_data(100),
        "key2": make_test_data(200),
        "key3": make_test_data(300),
    }
    data_service.update(test_data)
    assert len(data_service) == 3

    data_service.clear()
    assert len(data_service) == 0


def test_pop_removes_and_returns_value(data_service: DataService[str, int]):
    data_service["key1"] = make_test_data(42)

    value = data_service.pop("key1")
    assert value.value == 42
    assert "key1" not in data_service


def test_setdefault_behavior(data_service: DataService[str, int]):
    test_val = make_test_data(42)
    value = data_service.setdefault("key1", test_val)
    assert value.value == 42
    assert data_service["key1"].value == 42

    # Second call should return existing value
    value = data_service.setdefault("key1", make_test_data(999))
    assert value.value == 42
    assert data_service["key1"].value == 42


def test_subscriber_gets_full_data_dict(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1"})
    data_service.register_subscriber(subscriber)

    # Add some initial data
    data_service["existing"] = make_test_data(999)
    data_service["key1"] = make_test_data(42)

    # Subscriber should get notified with matching keys
    pipe = get_pipe()
    assert {"key1"} in pipe.notifications
    snapshot = data_service.snapshot(subscriber)
    assert snapshot["key1"].value == 42


def test_subscriber_only_gets_subscribed_keys(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1", "key3"})
    data_service.register_subscriber(subscriber)

    # Add data for subscribed and unsubscribed keys
    data_service["key1"] = make_test_data(42)
    data_service["key2"] = make_test_data(84)  # Not subscribed to this key
    data_service["key3"] = make_test_data(126)
    data_service["unrelated"] = make_test_data(999)  # Not subscribed to this key

    # Subscriber should only receive notifications for keys it's interested in
    pipe = get_pipe()
    assert {"key1"} in pipe.notifications
    assert {"key3"} in pipe.notifications
    assert {"key2"} not in pipe.notifications
    assert {"unrelated"} not in pipe.notifications

    snapshot = data_service.snapshot(subscriber)
    assert snapshot["key1"].value == 42
    assert snapshot["key3"].value == 126
    assert "key2" not in snapshot
    assert "unrelated" not in snapshot


def test_empty_transaction_no_notifications(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1"})
    data_service.register_subscriber(subscriber)

    pipe = get_pipe()
    with data_service.transaction():
        pass  # No changes

    assert len(pipe.notifications) == 0


def test_delitem_notifies_subscribers(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    # Add some data first
    data_service["key1"] = make_test_data(42)
    data_service["key2"] = make_test_data(84)
    pipe = get_pipe()
    pipe.notifications.clear()  # Clear previous notifications

    # Delete a key
    del data_service["key1"]

    # Should notify about the deleted key
    assert len(pipe.notifications) == 1
    assert pipe.notifications[0] == {"key1"}
    assert "key1" not in data_service
    snapshot = data_service.snapshot(subscriber)
    assert "key1" not in snapshot
    assert snapshot["key2"].value == 84


def test_delitem_in_transaction_batches_notifications(
    data_service: DataService[str, int],
):
    subscriber, get_pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    # Add some data first
    data_service["key1"] = make_test_data(42)
    data_service["key2"] = make_test_data(84)
    pipe = get_pipe()
    pipe.notifications.clear()  # Clear previous notifications

    with data_service.transaction():
        del data_service["key1"]
        data_service["key2"] = make_test_data(99)
        # No notifications yet
        assert len(pipe.notifications) == 0

    # Single notification after transaction
    assert len(pipe.notifications) == 1
    assert pipe.notifications[0] == {"key1", "key2"}
    snapshot = data_service.snapshot(subscriber)
    assert snapshot["key2"].value == 99
    assert "key1" not in snapshot


def test_transaction_set_then_del_same_key(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    # Add some initial data
    data_service["key2"] = make_test_data(84)
    pipe = get_pipe()
    pipe.notifications.clear()

    with data_service.transaction():
        data_service["key1"] = make_test_data(42)  # Set key1
        del data_service["key1"]  # Then delete key1
        # No notifications yet
        assert len(pipe.notifications) == 0

    # After transaction: key1 should not exist, but it was in the pending set
    assert len(pipe.notifications) == 1
    assert pipe.notifications[0] == {"key1"}
    assert "key1" not in data_service
    snapshot = data_service.snapshot(subscriber)
    assert snapshot["key2"].value == 84
    assert "key1" not in snapshot


def test_transaction_del_then_set_same_key(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1", "key2"})
    data_service.register_subscriber(subscriber)

    # Add some initial data
    data_service["key1"] = make_test_data(42)
    data_service["key2"] = make_test_data(84)
    pipe = get_pipe()
    pipe.notifications.clear()

    with data_service.transaction():
        del data_service["key1"]  # Delete key1
        data_service["key1"] = make_test_data(99)  # Then set key1 to new value
        # No notifications yet
        assert len(pipe.notifications) == 0

    # After transaction: key1 should have the new value
    assert len(pipe.notifications) == 1
    assert pipe.notifications[0] == {"key1"}
    snapshot = data_service.snapshot(subscriber)
    assert snapshot["key1"].value == 99
    assert snapshot["key2"].value == 84
    assert data_service["key1"].value == 99


def test_transaction_multiple_operations_same_key(data_service: DataService[str, int]):
    subscriber, get_pipe = create_test_subscriber({"key1"})
    data_service.register_subscriber(subscriber)

    # Add initial data
    data_service["key1"] = make_test_data(10)
    pipe = get_pipe()
    pipe.notifications.clear()

    with data_service.transaction():
        data_service["key1"] = make_test_data(20)  # Update
        data_service["key1"] = make_test_data(30)  # Update again
        del data_service["key1"]  # Delete
        data_service["key1"] = make_test_data(40)  # Set again
        # No notifications yet
        assert len(pipe.notifications) == 0

    # After transaction: key1 should have final value
    assert len(pipe.notifications) == 1
    assert pipe.notifications[0] == {"key1"}
    snapshot = data_service.snapshot(subscriber)
    assert snapshot["key1"].value == 40
    assert data_service["key1"].value == 40


class TestDataServiceUpdatingSubscribers:
    """Test complex subscriber behavior where subscribers update the DataService."""

    def test_subscriber_updates_service_immediately(self):
        """Test subscriber updating service outside of transaction."""
        service = DataService[str, int]()

        class UpdatingSubscriber(DataServiceSubscriber[str]):
            def __init__(self, keys: set[str], service: DataService[str, int]):
                self._keys_set = keys
                self._extractors = {key: LatestValueExtractor() for key in keys}
                self._service = service
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if "input" in updated_keys:
                    snapshot = self._service.snapshot(self)
                    if "input" in snapshot:
                        derived_value = snapshot["input"].value * 2
                        self._service["derived"] = make_test_data(derived_value)

        subscriber = UpdatingSubscriber({"input"}, service)
        service.register_subscriber(subscriber)

        # This should notify the subscriber, which updates "derived"
        service["input"] = make_test_data(10)

        assert service["input"].value == 10
        assert service["derived"].value == 20

    def test_subscriber_updates_service_in_transaction(self):
        """Test subscriber updating service at end of transaction."""
        service = DataService[str, int]()

        class UpdatingSubscriber(DataServiceSubscriber[str]):
            def __init__(self, keys: set[str], service: DataService[str, int]):
                self._keys_set = keys
                self._extractors = {key: LatestValueExtractor() for key in keys}
                self._service = service
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if "input" in updated_keys:
                    snapshot = self._service.snapshot(self)
                    if "input" in snapshot:
                        derived_value = snapshot["input"].value * 2
                        self._service["derived"] = make_test_data(derived_value)

        subscriber = UpdatingSubscriber({"input"}, service)
        service.register_subscriber(subscriber)

        with service.transaction():
            service["input"] = make_test_data(10)
            # "derived" should not exist yet during transaction
            assert "derived" not in service

        # After transaction, both keys should exist
        assert service["input"].value == 10
        assert service["derived"].value == 20

    def test_multiple_subscribers_update_service(self):
        """Test multiple subscribers updating different derived data."""
        service = DataService[str, int]()

        class MultiplierSubscriber(DataServiceSubscriber[str]):
            def __init__(
                self,
                keys: set[str],
                service: DataService[str, int],
                multiplier: int,
            ):
                self._keys_set = keys
                self._extractors = {key: LatestValueExtractor() for key in keys}
                self._service = service
                self._multiplier = multiplier
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if "input" in updated_keys:
                    snapshot = self._service.snapshot(self)
                    if "input" in snapshot:
                        key = f"derived_{self._multiplier}x"
                        derived_value = snapshot["input"].value * self._multiplier
                        self._service[key] = make_test_data(derived_value)

        sub1 = MultiplierSubscriber({"input"}, service, 2)
        sub2 = MultiplierSubscriber({"input"}, service, 3)
        service.register_subscriber(sub1)
        service.register_subscriber(sub2)

        service["input"] = make_test_data(10)

        assert service["input"].value == 10
        assert service["derived_2x"].value == 20
        assert service["derived_3x"].value == 30

    def test_cascading_subscriber_updates(self):
        """Test subscribers that depend on derived data from other subscribers."""
        service = DataService[str, int]()

        class FirstLevelSubscriber(DataServiceSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                self._keys_set = {"input"}
                self._extractors = {
                    key: LatestValueExtractor() for key in self._keys_set
                }
                self._service = service
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if "input" in updated_keys:
                    snapshot = self._service.snapshot(self)
                    if "input" in snapshot:
                        derived_value = snapshot["input"].value * 2
                        self._service["level1"] = make_test_data(derived_value)

        class SecondLevelSubscriber(DataServiceSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                self._keys_set = {"level1"}
                self._extractors = {
                    key: LatestValueExtractor() for key in self._keys_set
                }
                self._service = service
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if "level1" in updated_keys:
                    snapshot = self._service.snapshot(self)
                    if "level1" in snapshot:
                        derived_value = snapshot["level1"].value * 3
                        self._service["level2"] = make_test_data(derived_value)

        sub1 = FirstLevelSubscriber(service)
        sub2 = SecondLevelSubscriber(service)
        service.register_subscriber(sub1)
        service.register_subscriber(sub2)

        service["input"] = make_test_data(5)

        assert service["input"].value == 5
        assert service["level1"].value == 10
        assert service["level2"].value == 30

    def test_cascading_updates_in_transaction(self):
        """Test cascading updates within a transaction."""
        service = DataService[str, int]()

        class FirstLevelSubscriber(DataServiceSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                self._keys_set = {"input"}
                self._extractors = {
                    key: LatestValueExtractor() for key in self._keys_set
                }
                self._service = service
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if "input" in updated_keys:
                    snapshot = self._service.snapshot(self)
                    if "input" in snapshot:
                        derived_value = snapshot["input"].value * 2
                        self._service["level1"] = make_test_data(derived_value)

        class SecondLevelSubscriber(DataServiceSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                self._keys_set = {"level1"}
                self._extractors = {
                    key: LatestValueExtractor() for key in self._keys_set
                }
                self._service = service
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if "level1" in updated_keys:
                    snapshot = self._service.snapshot(self)
                    if "level1" in snapshot:
                        derived_value = snapshot["level1"].value * 3
                        self._service["level2"] = make_test_data(derived_value)

        sub1 = FirstLevelSubscriber(service)
        sub2 = SecondLevelSubscriber(service)
        service.register_subscriber(sub1)
        service.register_subscriber(sub2)

        with service.transaction():
            service["input"] = make_test_data(5)
            service["other"] = make_test_data(100)
            # No derived data should exist during transaction
            assert "level1" not in service
            assert "level2" not in service

        # All data should exist after transaction
        assert service["input"].value == 5
        assert service["other"].value == 100
        assert service["level1"].value == 10
        assert service["level2"].value == 30

    def test_subscriber_updates_multiple_keys(self):
        """Test subscriber that updates multiple derived keys at once."""
        service = DataService[str, int]()

        class MultiUpdateSubscriber(DataServiceSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                self._keys_set = {"input"}
                self._extractors = {
                    key: LatestValueExtractor() for key in self._keys_set
                }
                self._service = service
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if "input" in updated_keys:
                    snapshot = self._service.snapshot(self)
                    if "input" in snapshot:
                        input_value = snapshot["input"].value
                        with self._service.transaction():
                            self._service["double"] = make_test_data(input_value * 2)
                            self._service["triple"] = make_test_data(input_value * 3)
                            self._service["square"] = make_test_data(input_value**2)

        subscriber = MultiUpdateSubscriber(service)
        service.register_subscriber(subscriber)

        service["input"] = make_test_data(4)

        assert service["input"].value == 4
        assert service["double"].value == 8
        assert service["triple"].value == 12
        assert service["square"].value == 16

    def test_subscriber_updates_existing_keys(self):
        """Test subscriber updating keys that already exist."""
        service = DataService[str, int]()
        service["existing"] = make_test_data(100)

        class OverwriteSubscriber(DataServiceSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                self._keys_set = {"input"}
                self._extractors = {
                    key: LatestValueExtractor() for key in self._keys_set
                }
                self._service = service
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if "input" in updated_keys:
                    snapshot = self._service.snapshot(self)
                    if "input" in snapshot:
                        derived_value = snapshot["input"].value * 10
                        self._service["existing"] = make_test_data(derived_value)

        subscriber = OverwriteSubscriber(service)
        service.register_subscriber(subscriber)

        service["input"] = make_test_data(5)

        assert service["input"].value == 5
        assert service["existing"].value == 50  # Overwritten, not 100

    def test_circular_dependency_protection(self):
        """Test handling of potential circular dependencies."""
        service = DataService[str, int]()
        update_count = {"count": 0}

        class CircularSubscriber(DataServiceSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                self._keys_set = {"input", "output"}
                self._extractors = {
                    key: LatestValueExtractor() for key in self._keys_set
                }
                self._service = service
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                update_count["count"] += 1
                if update_count["count"] < 5:  # Prevent infinite recursion in test
                    snapshot = self._service.snapshot(self)
                    if "input" in updated_keys and "output" not in snapshot:
                        derived_value = snapshot["input"].value + 1
                        self._service["output"] = make_test_data(derived_value)
                    elif (
                        "output" in updated_keys
                        and snapshot.get("output", make_test_data(0)).value < 10
                    ):
                        derived_value = snapshot["output"].value + 1
                        self._service["output"] = make_test_data(derived_value)

        subscriber = CircularSubscriber(service)
        service.register_subscriber(subscriber)

        service["input"] = make_test_data(1)

        # Should handle the circular updates gracefully
        assert service["input"].value == 1
        assert "output" in service
        assert update_count["count"] > 1  # Multiple updates occurred

    def test_subscriber_deletes_keys_during_update(self):
        """Test subscriber that deletes keys during notification."""
        service = DataService[str, int]()
        service["to_delete"] = make_test_data(999)

        class DeletingSubscriber(DataServiceSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                self._keys_set = {"trigger"}
                self._extractors = {
                    key: LatestValueExtractor() for key in self._keys_set
                }
                self._service = service
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if "trigger" in updated_keys and "to_delete" in self._service:
                    del self._service["to_delete"]
                    self._service["deleted_flag"] = make_test_data(1)

        subscriber = DeletingSubscriber(service)
        service.register_subscriber(subscriber)

        service["trigger"] = make_test_data(1)

        assert service["trigger"].value == 1
        assert "to_delete" not in service
        assert service["deleted_flag"].value == 1

    def test_subscriber_complex_transaction_updates(self):
        """Test complex scenario with nested transactions and subscriber updates."""
        service = DataService[str, int]()

        class ComplexSubscriber(DataServiceSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                self._keys_set = {"input"}
                self._extractors = {
                    key: LatestValueExtractor() for key in self._keys_set
                }
                self._service = service
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if "input" in updated_keys:
                    snapshot = self._service.snapshot(self)
                    if "input" in snapshot:
                        input_value = snapshot["input"].value
                        with self._service.transaction():
                            self._service["derived1"] = make_test_data(input_value * 2)
                            with self._service.transaction():
                                self._service["derived2"] = make_test_data(
                                    input_value * 3
                                )
                            self._service["derived3"] = make_test_data(input_value * 4)

        subscriber = ComplexSubscriber(service)
        service.register_subscriber(subscriber)

        with service.transaction():
            service["input"] = make_test_data(5)
            service["other"] = make_test_data(100)
            # No derived data during transaction
            assert "derived1" not in service

        # All data should exist after transaction
        assert service["input"].value == 5
        assert service["other"].value == 100
        assert service["derived1"].value == 10
        assert service["derived2"].value == 15
        assert service["derived3"].value == 20

    def test_multiple_update_rounds(self):
        """Test scenario requiring multiple notification rounds."""
        service = DataService[str, int]()

        class ChainSubscriber(DataServiceSubscriber[str]):
            def __init__(
                self, input_key: str, output_key: str, service: DataService[str, int]
            ):
                self._keys_set = {input_key}
                self._extractors = {
                    key: LatestValueExtractor() for key in self._keys_set
                }
                self._input_key = input_key
                self._output_key = output_key
                self._service = service
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if self._input_key in updated_keys:
                    snapshot = self._service.snapshot(self)
                    if self._input_key in snapshot:
                        derived_value = snapshot[self._input_key].value + 1
                        self._service[self._output_key] = make_test_data(derived_value)

        # Create a chain: input -> step1 -> step2 -> step3
        sub1 = ChainSubscriber("input", "step1", service)
        sub2 = ChainSubscriber("step1", "step2", service)
        sub3 = ChainSubscriber("step2", "step3", service)

        service.register_subscriber(sub1)
        service.register_subscriber(sub2)
        service.register_subscriber(sub3)

        service["input"] = make_test_data(10)

        assert service["input"].value == 10
        assert service["step1"].value == 11
        assert service["step2"].value == 12
        assert service["step3"].value == 13

    def test_subscriber_updates_with_mixed_immediate_and_transaction(self):
        """Test mixing immediate updates with transactional updates from subscribers."""
        service = DataService[str, int]()

        class MixedSubscriber(DataServiceSubscriber[str]):
            def __init__(self, service: DataService[str, int]):
                self._keys_set = {"input"}
                self._extractors = {
                    key: LatestValueExtractor() for key in self._keys_set
                }
                self._service = service
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if "input" in updated_keys:
                    snapshot = self._service.snapshot(self)
                    if "input" in snapshot:
                        input_value = snapshot["input"].value
                        # Immediate update
                        self._service["immediate"] = make_test_data(input_value * 2)
                        # Transaction update
                        with self._service.transaction():
                            self._service["transactional1"] = make_test_data(
                                input_value * 3
                            )
                            self._service["transactional2"] = make_test_data(
                                input_value * 4
                            )

        subscriber = MixedSubscriber(service)
        service.register_subscriber(subscriber)

        service["input"] = make_test_data(5)

        assert service["input"].value == 5
        assert service["immediate"].value == 10
        assert service["transactional1"].value == 15
        assert service["transactional2"].value == 20


# Tests for extractor-based subscription
class TestNotifySubscriberErrorIsolation:
    """Test that a failing subscriber does not block other subscribers."""

    def test_failing_trigger_does_not_block_subsequent_subscribers(self):
        service = DataService[str, int]()

        class FailingSubscriber(DataServiceSubscriber[str]):
            def __init__(self) -> None:
                self._extractors = {"key1": LatestValueExtractor()}
                self._armed = False
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if self._armed:
                    raise RuntimeError("subscriber failed")

        failing = FailingSubscriber()
        good, get_pipe = create_test_subscriber({"key1"})

        service.register_subscriber(failing)
        service.register_subscriber(good)
        failing._armed = True

        service["key1"] = make_test_data(42)

        pipe = get_pipe()
        assert len(pipe.notifications) == 1
        snapshot = service.snapshot(good)
        assert snapshot["key1"].value == 42

    def test_failing_extractor_does_not_block_subsequent_subscribers(self):
        service = DataService[str, int]()

        class BrokenExtractor:
            def __init__(self) -> None:
                self._armed = False

            @property
            def buffer_size(self) -> int:
                return 1

            def get_required_timespan(self) -> float:
                return 0.0

            def extract(self, buffered_data):
                if self._armed:
                    raise ValueError("extractor failed")
                return None

        class FailingExtractorSubscriber(DataServiceSubscriber[str]):
            def __init__(self, extractor: BrokenExtractor) -> None:
                self._extractors: dict = {"key1": extractor}
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                pass

        extractor = BrokenExtractor()
        failing = FailingExtractorSubscriber(extractor)
        good, get_pipe = create_test_subscriber({"key1"})

        service.register_subscriber(failing)
        service.register_subscriber(good)
        extractor._armed = True

        service["key1"] = make_test_data(42)

        pipe = get_pipe()
        assert len(pipe.notifications) == 1
        snapshot = service.snapshot(good)
        assert snapshot["key1"].value == 42


class TestExtractorBasedSubscription:
    """Tests for extractor-based subscription with dynamic buffer sizing."""

    def test_buffer_size_determined_by_max_extractor_requirement(self):
        """Test that buffer size is set to max requirement among subscribers."""
        import scipp as sc

        from ess.livedata.dashboard.data_service import DataService
        from ess.livedata.dashboard.extractors import (
            FullHistoryExtractor,
            LatestValueExtractor,
        )

        class TestSubscriber(DataServiceSubscriber[str]):
            def __init__(self, keys: set[str], extractor):
                self._keys_set = keys
                self._extractor = extractor
                self.notification_keys: list[set[str]] = []
                super().__init__()

            @property
            def extractors(self) -> dict:
                return dict.fromkeys(self._keys_set, self._extractor)

            def on_updated(self, updated_keys: set[str]) -> None:
                self.notification_keys.append(updated_keys)

        # Create service
        service = DataService()

        # Register subscriber with LatestValueExtractor (size 1)
        sub1 = TestSubscriber({"data"}, LatestValueExtractor())
        service.register_subscriber(sub1)

        # Add first data point - buffer should be size 1
        service["data"] = sc.DataArray(
            sc.scalar(1, unit='counts'), coords={'time': sc.scalar(0.0, unit='s')}
        )

        # Register subscriber with FullHistoryExtractor (size 10000)
        sub2 = TestSubscriber({"data"}, FullHistoryExtractor())
        service.register_subscriber(sub2)

        # Buffer should now grow to size 10000
        # Add more data to verify buffering works
        for i in range(2, 12):
            service["data"] = sc.DataArray(
                sc.scalar(i, unit='counts'),
                coords={'time': sc.scalar(float(i - 1), unit='s')},
            )

        # Both subscribers should have received all updates
        # sub1: 1 update before sub2 registration + 10 after = 11
        assert len(sub1.notification_keys) == 11
        # sub2: 10 updates after registration = 10
        assert len(sub2.notification_keys) == 10

        # sub1 should get latest value only (unwrapped)
        snapshot1 = service.snapshot(sub1)
        last_from_sub1 = snapshot1["data"]
        assert last_from_sub1.ndim == 0  # Scalar (unwrapped)
        assert last_from_sub1.value == 11

        # sub2 should get all history after it was registered
        # When sub2 registered, buffer switched from SingleValueBuffer to
        # TemporalBuffer, copying the first data point, then receiving 10 more
        # updates = 11 total
        snapshot2 = service.snapshot(sub2)
        last_from_sub2 = snapshot2["data"]
        assert last_from_sub2.sizes == {'time': 11}

    def test_multiple_keys_with_different_extractors(self):
        """Test subscriber with different extractors per key."""
        import scipp as sc

        from ess.livedata.dashboard.data_service import DataService
        from ess.livedata.dashboard.extractors import (
            FullHistoryExtractor,
            LatestValueExtractor,
        )

        class MultiKeySubscriber(DataServiceSubscriber[str]):
            def __init__(self):
                self.notification_keys: list[set[str]] = []
                super().__init__()

            @property
            def extractors(self) -> dict:
                return {
                    "latest": LatestValueExtractor(),
                    "history": FullHistoryExtractor(),
                }

            def on_updated(self, updated_keys: set[str]) -> None:
                self.notification_keys.append(updated_keys)

        service = DataService()
        subscriber = MultiKeySubscriber()
        service.register_subscriber(subscriber)

        # Add data to both keys
        for i in range(5):
            service["latest"] = sc.DataArray(
                sc.scalar(i * 10, unit='counts'),
                coords={'time': sc.scalar(float(i), unit='s')},
            )
            service["history"] = sc.DataArray(
                sc.scalar(i * 100, unit='counts'),
                coords={'time': sc.scalar(float(i), unit='s')},
            )

        # Should have received updates (batched in transaction would be less,
        # but here each setitem triggers separately)
        assert len(subscriber.notification_keys) > 0

        # Check last received data via snapshot
        last_data = service.snapshot(subscriber)

        # "latest" should be unwrapped scalar
        if "latest" in last_data:
            assert last_data["latest"].ndim == 0

        # "history" should return all accumulated values with time dimension
        if "history" in last_data:
            assert "time" in last_data["history"].dims


class TestThreadSafety:
    """Tests for concurrent access from the ingestion and UI threads.

    The background ingestion thread mutates buffers via ``transaction`` /
    ``__setitem__`` while the UI thread (re)configures plots via
    ``register_subscriber`` / ``unregister_subscriber``. A single
    transaction-scoped lock serializes the internal state, and ``on_updated``
    runs outside the lock.
    """

    def test_getitem_returns_data_detached_from_buffer(self):
        service = DataService[str, int]()
        service["key1"] = make_test_data(1, time=1.0)

        first = service["key1"]
        # A later update must not retroactively change an already-returned value.
        service["key1"] = make_test_data(2, time=2.0)

        assert first.value == 1
        # Distinct objects: the returned value is copied, not a buffer view.
        assert service["key1"] is not service["key1"]

    def test_on_updated_runs_without_holding_lock(self):
        """A subscriber's on_updated must not block other threads' locked ops.

        Proves the lock is released around ``on_updated``: while one subscriber is
        mid-notification, another thread can complete a ``__setitem__`` (which needs
        the lock) on an unrelated key.
        """
        service = DataService[str, int]()

        entered = threading.Event()
        release = threading.Event()

        class BlockingSubscriber(DataServiceSubscriber[str]):
            def __init__(self) -> None:
                self._extractors = {"key1": LatestValueExtractor()}
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if updated_keys:  # block only on a real update
                    entered.set()
                    release.wait(timeout=5)

        service.register_subscriber(BlockingSubscriber())

        # Ingestion thread enters on_updated and blocks there.
        ingest = threading.Thread(
            target=lambda: service.__setitem__("key1", make_test_data(1))
        )
        ingest.start()
        assert entered.wait(timeout=5), "on_updated was not entered"

        # While on_updated is blocked, an unrelated locked op must still complete.
        other_done = threading.Event()

        def other() -> None:
            service["key2"] = make_test_data(2)
            other_done.set()

        other_thread = threading.Thread(target=other)
        other_thread.start()
        assert other_done.wait(timeout=5), (
            "lock held during on_updated blocked a writer"
        )

        release.set()
        ingest.join(timeout=5)
        other_thread.join(timeout=5)

    def test_concurrent_ingestion_and_subscriber_churn(self):
        """Stress the race the lock fixes: buffer writes vs. subscriber churn.

        The ingestion thread continuously writes to a key while the UI thread
        registers/unregisters subscribers whose ``FullHistoryExtractor`` forces
        a Single->Temporal buffer swap (``add_extractor``) on the same key.
        Without serialization this corrupts buffer state or raises; with the
        lock it must run cleanly.
        """
        from ess.livedata.dashboard.extractors import FullHistoryExtractor

        service = DataService[str, int]()
        errors: list[BaseException] = []
        stop = threading.Event()

        def ingest() -> None:
            i = 0
            try:
                while not stop.is_set():
                    with service.transaction():
                        service["key1"] = make_test_data(i, time=float(i))
                    i += 1
            except BaseException as exc:
                errors.append(exc)

        class HistorySubscriber(DataServiceSubscriber[str]):
            def __init__(self) -> None:
                self._extractors = {"key1": FullHistoryExtractor()}
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                pass

        def churn() -> None:
            try:
                for _ in range(500):
                    sub = HistorySubscriber()
                    service.register_subscriber(sub)
                    service.unregister_subscriber(sub)
            except BaseException as exc:
                errors.append(exc)
            finally:
                stop.set()

        ingest_thread = threading.Thread(target=ingest)
        churn_thread = threading.Thread(target=churn)
        ingest_thread.start()
        churn_thread.start()
        churn_thread.join(timeout=30)
        ingest_thread.join(timeout=30)

        assert not errors, f"concurrent access raised: {errors[0]!r}"
        assert service["key1"].value >= 0

    def test_write_blocked_by_transaction_notifies_once_it_commits(self):
        """The lock is transaction-scoped, the notify deferral thread-local.

        A write on another thread blocks until an open transaction commits;
        once through, it notifies immediately — the foreign transaction must
        not defer it.
        """
        service = DataService[str, int]()
        triggered: list[int] = []

        class RecordingSubscriber(DataServiceSubscriber[str]):
            def __init__(self) -> None:
                self._extractors = {"key1": LatestValueExtractor()}
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            def on_updated(self, updated_keys: set[str]) -> None:
                if updated_keys:
                    snapshot = service.snapshot(self)
                    if "key1" in snapshot:
                        triggered.append(snapshot["key1"].value)

        service.register_subscriber(RecordingSubscriber())

        in_transaction = threading.Event()
        release = threading.Event()

        def hold_transaction() -> None:
            with service.transaction():
                in_transaction.set()
                release.wait(timeout=5)

        holder = threading.Thread(target=hold_transaction)
        holder.start()
        assert in_transaction.wait(timeout=5)

        write_done = threading.Event()

        def write() -> None:
            service["key1"] = make_test_data(42, time=1.0)
            write_done.set()

        writer = threading.Thread(target=write)
        writer.start()
        assert not write_done.wait(timeout=0.2), (
            "write completed while a transaction was open on another thread"
        )

        release.set()
        assert write_done.wait(timeout=5), "write never unblocked"
        writer.join(timeout=5)
        holder.join(timeout=5)
        assert triggered == [42]

    def test_snapshot_does_not_observe_partial_transaction(self):
        """A pull sees either none or all of a transaction's writes.

        A snapshot racing an open transaction must block until the
        transaction commits, not return a state where only some of its
        writes are visible.
        """
        service = DataService[str, int]()
        subscriber, _ = create_test_subscriber({"key1", "key2"})
        service.register_subscriber(subscriber)

        first_written = threading.Event()
        release = threading.Event()

        def transact() -> None:
            with service.transaction():
                service["key1"] = make_test_data(1, time=1.0)
                first_written.set()
                release.wait(timeout=5)
                service["key2"] = make_test_data(2, time=2.0)

        writer = threading.Thread(target=transact)
        writer.start()
        assert first_written.wait(timeout=5)

        result = {}
        snapshot_done = threading.Event()

        def snapshot() -> None:
            result.update(service.snapshot(subscriber))
            snapshot_done.set()

        reader = threading.Thread(target=snapshot)
        reader.start()
        assert not snapshot_done.wait(timeout=0.2), "snapshot completed mid-transaction"

        release.set()
        assert snapshot_done.wait(timeout=5), "snapshot never unblocked"
        reader.join(timeout=5)
        writer.join(timeout=5)
        assert result["key1"].value == 1
        assert result["key2"].value == 2

    def test_notification_failure_does_not_wedge_transactions(self):
        """An exception escaping notification at transaction exit must not
        leave the service permanently in-transaction, which would silently
        suppress all future notifications."""
        service = DataService[str, int]()
        triggered: list[int] = []

        class ExplodingKeysSubscriber(DataServiceSubscriber[str]):
            explode = False

            def __init__(self) -> None:
                self._extractors = {"key1": LatestValueExtractor()}
                super().__init__()

            @property
            def extractors(self):
                return self._extractors

            @property
            def keys(self):
                if self.explode:
                    raise RuntimeError("boom")
                return super().keys

            def on_updated(self, updated_keys: set[str]) -> None:
                if updated_keys:
                    snapshot = service.snapshot(self)
                    if "key1" in snapshot:
                        triggered.append(snapshot["key1"].value)

        sub = ExplodingKeysSubscriber()
        service.register_subscriber(sub)

        sub.explode = True
        with pytest.raises(RuntimeError), service.transaction():
            service["key1"] = make_test_data(1, time=1.0)

        sub.explode = False
        service["key1"] = make_test_data(2, time=2.0)
        assert triggered == [2]


def test_snapshot_returns_data_after_registration(data_service: DataService[str, int]):
    """Snapshot returns existing data even if subscriber hasn't received updates."""
    data_service["key1"] = make_test_data(42, time=1.0)
    subscriber, _ = create_test_subscriber({"key1"})
    data_service.register_subscriber(subscriber)
    snapshot = data_service.snapshot(subscriber)
    assert snapshot["key1"].value == 42


def test_buffering_continues_after_registration():
    """History accumulates from the moment of registration."""
    from ess.livedata.dashboard.extractors import FullHistoryExtractor

    service = DataService[str, int]()
    subscriber = SimpleTestSubscriber({"key1"})
    # Replace extractor with FullHistoryExtractor
    subscriber._extractors = {"key1": FullHistoryExtractor()}
    service.register_subscriber(subscriber)
    for i in range(3):
        service["key1"] = make_test_data(i, time=float(i))
    snapshot = service.snapshot(subscriber)
    assert snapshot["key1"].sizes["time"] == 3


class TestUnregisterExtractorSymmetry:
    """Unregistering drops a subscriber's extractors from buffer retention.

    Without this symmetry, extractor lists (and the retention they pin) grow
    without bound over repeated register/unregister cycles on a long-lived key.
    """

    @staticmethod
    def _history_subscriber() -> SimpleTestSubscriber:
        from ess.livedata.dashboard.extractors import FullHistoryExtractor

        subscriber = SimpleTestSubscriber({"key1"})
        subscriber._extractors = {"key1": FullHistoryExtractor()}
        return subscriber

    def test_history_preserved_across_unregister_reregister_cycle(self):
        """A plot-edit style unregister/re-register must not wipe history."""
        service = DataService[str, int]()
        sub1 = self._history_subscriber()
        service.register_subscriber(sub1)
        for i in range(3):
            service["key1"] = make_test_data(i, time=float(i))

        service.unregister_subscriber(sub1)
        sub2 = self._history_subscriber()
        service.register_subscriber(sub2)

        snapshot = service.snapshot(sub2)
        assert snapshot["key1"].sizes["time"] == 3

    def test_unregister_releases_retention_requirement(self):
        """A departed subscriber's timespan no longer pins the buffer."""
        from ess.livedata.dashboard.temporal_buffers import TemporalBuffer

        service = DataService[str, int]()
        latest = SimpleTestSubscriber({"key1"})
        service.register_subscriber(latest)
        service["key1"] = make_test_data(0, time=0.0)

        for _ in range(20):
            history = self._history_subscriber()
            service.register_subscriber(history)
            assert service._buffer_manager["key1"].get_required_timespan() == float(
                "inf"
            )
            service.unregister_subscriber(history)
            # Retention shrinks to the surviving LatestValueExtractor; the
            # buffer stays temporal (sticky-upward). Extractor count is the
            # leak observable, unreachable through the public API.
            buffer = service._buffer_manager["key1"]
            assert buffer.get_required_timespan() == 0.0
            assert isinstance(buffer, TemporalBuffer)
            assert len(service._buffer_manager._states["key1"].extractors) == 1


class TestClearKeysAndStamps:
    """Generation-clear and provenance stamps (#1042 slice i)."""

    def test_clear_keys_empties_buffers_and_notifies(self):
        service = DataService[str, int]()
        subscriber, get_pipe = create_test_subscriber({"key1", "key2"})
        service.register_subscriber(subscriber)
        service["key1"] = make_test_data(1)
        service["key2"] = make_test_data(2)
        service["other"] = make_test_data(3)

        service.clear_keys(["key1", "key2", "missing"])

        snapshot = service.snapshot(subscriber)
        assert snapshot == {}
        assert service["other"].value == 3
        assert get_pipe().notifications[-1] == {"key1", "key2"}

    def test_cleared_keys_are_absent_from_the_mapping_view(self):
        """Every iterated key must be readable: dict(service) after a clear.

        Buffers survive clear_keys (extractors, retention), but a key whose
        read raises must not be yielded by iteration or counted by len.
        """
        service = DataService[str, int]()
        service["key1"] = make_test_data(1)
        service["key2"] = make_test_data(2)

        service.clear_keys(["key1"])

        assert set(service) == {"key2"}
        assert len(service) == 1
        assert "key1" not in service
        assert {k: v.value for k, v in dict(service).items()} == {"key2": 2}

        service["key1"] = make_test_data(3)
        assert set(service) == {"key1", "key2"}
        assert len(service) == 2

    def test_clear_keys_preserves_subscriber_registration(self):
        """Data arriving after a clear reaches the still-registered subscriber."""
        service = DataService[str, int]()
        subscriber, _get_pipe = create_test_subscriber({"key1"})
        service.register_subscriber(subscriber)
        service["key1"] = make_test_data(1)
        service.clear_keys(["key1"])

        service["key1"] = make_test_data(42)

        assert service.snapshot(subscriber)["key1"].value == 42

    def test_stamps_follow_data_lifecycle(self):
        service = DataService[str, int]()
        subscriber, _ = create_test_subscriber({"key1", "key2"})
        service.register_subscriber(subscriber)
        service.set_item("key1", make_test_data(1), stamp="gen-a")
        service.set_item("key2", make_test_data(2), stamp="gen-b")

        data, stamps = service.snapshot_with_stamps(subscriber)
        assert stamps == {"key1": "gen-a", "key2": "gen-b"}
        assert data["key1"].value == 1

        service.clear_keys(["key1"])
        service.set_item("key1", make_test_data(3), stamp="gen-c")
        _, stamps = service.snapshot_with_stamps(subscriber)
        assert stamps == {"key1": "gen-c", "key2": "gen-b"}

    def test_plain_setitem_records_no_stamp(self):
        service = DataService[str, int]()
        subscriber, _ = create_test_subscriber({"key1"})
        service.register_subscriber(subscriber)
        service["key1"] = make_test_data(1)

        _, stamps = service.snapshot_with_stamps(subscriber)
        assert stamps == {"key1": None}
