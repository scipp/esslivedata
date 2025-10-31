# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pydantic
import pytest

from ess.livedata.config.schema_registry import FakeSchemaRegistry
from ess.livedata.dashboard.config_service import ConfigService
from ess.livedata.dashboard.message_bridge import FakeMessageBridge
from ess.livedata.dashboard.schema_validator import (
    PydanticSchemaValidator,
    SchemaValidator,
)
from ess.livedata.parameter_models import TOARange


# Test models for more comprehensive testing
class SimpleConfig(pydantic.BaseModel):
    value: int
    name: str = "default"


class ComplexConfig(pydantic.BaseModel):
    items: list[str]
    metadata: dict[str, int]
    threshold: float


# Simple fake callback for testing
class FakeCallback:
    def __init__(self) -> None:
        self.called = False
        self.call_count = 0
        self.call_args: list = []
        self.data = None

    def __call__(self, data) -> None:
        self.called = True
        self.call_count += 1
        self.call_args.append(data)
        self.data = data

    def reset(self) -> None:
        self.called = False
        self.call_count = 0
        self.call_args.clear()
        self.data = None


class FailingCallback:
    def __init__(self, exception: Exception | None = None) -> None:
        self.exception = exception or ValueError("Callback failed")
        self.call_count = 0

    def __call__(self, data) -> None:
        self.call_count += 1
        raise self.exception


class MockGUIComponent:
    """
    Mock GUI component that simulates bidirectional parameter binding.

    1. Updates from config service update the component's values
    2. Changes to component values echo back to the config service
    3. Widget change events (simulated by _trigger_change_event) would normally
       cause infinite loops without proper cycle prevention
    """

    def __init__(self, config_key: str) -> None:
        self.config_key = config_key
        self.enabled = True
        self.low = 0.0
        self.high = 10000.0
        self.unit = 'us'
        self._config_service: ConfigService | None = None
        self._update_count = 0
        self._change_events_enabled = True

    def subscribe_to_config(self, config_service: ConfigService) -> None:
        """Subscribe to config service updates."""
        self._config_service = config_service
        config_service.subscribe(self.config_key, self._on_config_update)

    def _on_config_update(self, data: TOARange) -> None:
        """Handle updates from config service."""
        self._update_count += 1
        # Simulate widget value changes that would normally trigger change events
        old_low = self.low
        self.enabled = data.enabled
        self.low = data.start
        self.high = data.stop
        self.unit = data.unit

        # Simulate widget change event that would trigger in real widgets
        # This is what could cause infinite loops without proper cycle prevention
        if self._change_events_enabled and old_low != self.low:
            self._trigger_change_event('low', self.low)

    def _trigger_change_event(self, param_name: str, value) -> None:
        """Simulate a widget change event that would echo back to config service."""
        if self._config_service:
            config = TOARange(
                enabled=self.enabled, start=self.low, stop=self.high, unit=self.unit
            )
            self._config_service.update_config(self.config_key, config)

    def set_low(self, value: float) -> None:
        """Simulate user changing the low value in GUI."""
        old_low = self.low
        self.low = value
        # User changes always trigger change events
        if old_low != self.low:
            self._trigger_change_event('low', self.low)


@pytest.fixture
def config_key() -> str:
    return "toa_range"


@pytest.fixture
def simple_key() -> str:
    return "simple_config"


@pytest.fixture
def complex_key() -> str:
    return "complex_config"


@pytest.fixture
def schemas(config_key: str, simple_key: str, complex_key: str) -> SchemaValidator:
    registry = FakeSchemaRegistry(
        {config_key: TOARange, simple_key: SimpleConfig, complex_key: ComplexConfig}
    )
    return PydanticSchemaValidator(schema_registry=registry)


@pytest.fixture
def service(schemas: SchemaValidator) -> ConfigService:
    return ConfigService(schema_validator=schemas)


@pytest.fixture
def service_with_bridge(
    schemas: SchemaValidator,
) -> tuple[ConfigService, FakeMessageBridge]:
    bridge = FakeMessageBridge()
    return ConfigService(schema_validator=schemas, message_bridge=bridge), bridge


class TestConfigService:
    def test_subscriber(
        self,
        service_with_bridge: tuple[ConfigService, FakeMessageBridge],
        config_key: str,
    ) -> None:
        service, bridge = service_with_bridge
        callback = FakeCallback()
        service.subscribe(key=config_key, callback=callback)

        # Create a config update and add it to the bridge
        config_data = {'enabled': False, 'start': 1000.0, 'stop': 2000.0, 'unit': 'us'}
        bridge.add_incoming_message((config_key, config_data))

        # Process the message
        service.process_incoming_messages()

        assert callback.called is True
        toa_range = callback.data
        assert toa_range.enabled is False
        assert toa_range.start == 1000.0
        assert toa_range.stop == 2000.0
        assert toa_range.unit == 'us'

    def test_bidirectional_param_binding_no_infinite_cycle(
        self,
        service_with_bridge: tuple[ConfigService, FakeMessageBridge],
        config_key: str,
    ) -> None:
        """Test that bidirectional param binding doesn't cause infinite cycles."""
        service, bridge = service_with_bridge

        # Create mock GUI component that simulates TOARangeParam behavior
        gui_component = MockGUIComponent(config_key)
        gui_component.subscribe_to_config(service)

        # Init bridge with a single message
        bridge.add_incoming_message(
            (
                config_key,
                {'enabled': True, 'start': 1000.0, 'stop': 2000.0, 'unit': 'us'},
            )
        )

        # Simulate GUI change (should publish to bridge)
        gui_component.set_low(1500.0)
        assert len(bridge.get_published_messages()) == 1

        # Process the external message. Should NOT trigger another update to the bridge
        # The key insight: _disable_updates() in process_incoming_messages prevents
        # the widget change event from triggering another update_config call
        service.process_incoming_messages()
        assert gui_component.low == 1000.0  # Updated from external message
        published = list(bridge.get_published_messages())
        assert len(published) == 1  # Still only the original user change

        # Simulate our own updating making it back to the bridge
        bridge.clear()
        bridge.add_incoming_message(published[0])

        # Process our own update. Should NOT trigger another update to the bridge
        service.process_incoming_messages()
        assert gui_component.low == 1500.0  # Updated from our own echo
        assert len(bridge.get_published_messages()) == 0  # No new publications

    def test_get_nonexistent_key_returns_default(self, service: ConfigService) -> None:
        """Test getting a non-existent key returns the default value."""
        assert service.get_config("nonexistent") is None
        assert service.get_config("nonexistent", "default_value") == "default_value"
        assert service.get_config("nonexistent", 42) == 42

    def test_get_existing_key_returns_value(
        self, service: ConfigService, simple_key: str
    ) -> None:
        """Test getting an existing key returns the stored value."""
        config = SimpleConfig(value=123, name="test")
        service._config[simple_key] = config

        result = service.get_config(simple_key)
        assert result == config
        assert result.value == 123
        assert result.name == "test"

    def test_update_config_valid_data(
        self, service: ConfigService, simple_key: str
    ) -> None:
        """Test updating config with valid pydantic model."""
        config = SimpleConfig(value=456, name="updated")
        service.update_config(simple_key, config)

        assert service.get_config(simple_key) == config

    def test_update_config_unregistered_schema_raises_error(
        self, service: ConfigService
    ) -> None:
        """Test updating config with unregistered schema raises ValueError."""
        config = SimpleConfig(value=123)
        with pytest.raises(ValueError, match="No schema registered"):
            service.update_config("unregistered_key", config)

    def test_update_config_wrong_schema_type_raises_error(
        self, service: ConfigService, simple_key: str
    ) -> None:
        """Test updating config with wrong schema type raises ValueError."""
        config = TOARange(enabled=True, low=100.0, high=200.0, unit="us")
        with pytest.raises(ValueError, match="No schema registered"):
            service.update_config(simple_key, config)

    def test_update_config_publishes_to_bridge(
        self,
        service_with_bridge: tuple[ConfigService, FakeMessageBridge],
        simple_key: str,
    ) -> None:
        """Test that config updates are published to the message bridge."""
        service, bridge = service_with_bridge
        config = SimpleConfig(value=789, name="bridge_test")

        service.update_config(simple_key, config)

        published = bridge.get_published_messages()
        assert len(published) == 1
        assert published[0][0] == simple_key
        assert published[0][1] == {"value": 789, "name": "bridge_test"}

    def test_subscribe_immediate_callback_with_existing_data(
        self, service: ConfigService, simple_key: str
    ) -> None:
        """Test that subscribe immediately calls callback if data exists."""
        config = SimpleConfig(value=100, name="immediate")
        service._config[simple_key] = config

        callback = FakeCallback()
        service.subscribe(simple_key, callback)

        assert callback.called is True
        assert callback.data == config

    def test_subscribe_no_immediate_callback_without_data(
        self, service: ConfigService, simple_key: str
    ) -> None:
        """Test that subscribe doesn't call callback if no data exists."""
        callback = FakeCallback()
        service.subscribe(simple_key, callback)

        assert callback.called is False

    def test_subscribe_multiple_callbacks(
        self, service: ConfigService, simple_key: str
    ) -> None:
        """Test that multiple callbacks can be registered for the same key."""
        callback1 = FakeCallback()
        callback2 = FakeCallback()
        callback3 = FakeCallback()

        service.subscribe(simple_key, callback1)
        service.subscribe(simple_key, callback2)
        service.subscribe(simple_key, callback3)

        config = SimpleConfig(value=200, name="multi")
        service._config[simple_key] = config
        service._notify_subscribers(simple_key, config)

        assert callback1.called is True
        assert callback1.data == config
        assert callback2.called is True
        assert callback2.data == config
        assert callback3.called is True
        assert callback3.data == config

    def test_subscribe_different_keys_independent(
        self, service: ConfigService, simple_key: str, complex_key: str
    ) -> None:
        """Test that subscribers for different keys are independent."""
        callback1 = FakeCallback()
        callback2 = FakeCallback()

        service.subscribe(simple_key, callback1)
        service.subscribe(complex_key, callback2)

        simple_config = SimpleConfig(value=300, name="key1")
        service._config[simple_key] = simple_config
        service._notify_subscribers(simple_key, simple_config)

        assert callback1.called is True
        assert callback1.data == simple_config
        assert callback2.called is False

    def test_process_incoming_messages_no_bridge(self, service: ConfigService) -> None:
        """Test that process_incoming_messages handles missing bridge gracefully."""
        service.process_incoming_messages()

    def test_process_incoming_messages_empty_queue(
        self, service_with_bridge: tuple[ConfigService, FakeMessageBridge]
    ) -> None:
        """Test processing messages when queue is empty."""
        service, bridge = service_with_bridge

        service.process_incoming_messages()

        assert len(bridge.get_published_messages()) == 0

    def test_process_incoming_messages_valid_data(
        self,
        service_with_bridge: tuple[ConfigService, FakeMessageBridge],
        simple_key: str,
    ) -> None:
        """Test processing incoming messages with valid data."""
        service, bridge = service_with_bridge
        callback = FakeCallback()
        service.subscribe(simple_key, callback)

        bridge.add_incoming_message((simple_key, {"value": 400, "name": "incoming"}))

        service.process_incoming_messages()

        config = service.get_config(simple_key)
        assert config.value == 400
        assert config.name == "incoming"

        assert callback.called is True
        assert callback.data == config

    def test_process_incoming_messages_invalid_data_logged(
        self,
        service_with_bridge: tuple[ConfigService, FakeMessageBridge],
        simple_key: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that invalid incoming data is logged and ignored."""
        service, bridge = service_with_bridge
        callback = FakeCallback()
        service.subscribe(simple_key, callback)

        bridge.add_incoming_message((simple_key, {"invalid_field": "value"}))

        with caplog.at_level('ERROR'):
            service.process_incoming_messages()

        assert len(caplog.records) == 1
        assert "Invalid config data received" in caplog.records[0].message

        assert service.get_config(simple_key) is None
        assert callback.called is False

    def test_process_incoming_messages_unknown_key_ignored(
        self, service_with_bridge: tuple[ConfigService, FakeMessageBridge]
    ) -> None:
        """Test that messages for unknown keys are ignored."""
        service, bridge = service_with_bridge

        bridge.add_incoming_message(("unknown_key", {"some": "data"}))

        service.process_incoming_messages()

        assert service.get_config("unknown_key") is None

    def test_process_incoming_messages_batching(
        self,
        service_with_bridge: tuple[ConfigService, FakeMessageBridge],
        simple_key: str,
    ) -> None:
        """Test that all messages are processed with deduplication."""
        service, bridge = service_with_bridge
        callback = FakeCallback()
        service.subscribe(simple_key, callback)

        for i in range(5):
            bridge.add_incoming_message((simple_key, {"value": i, "name": f"msg{i}"}))

        # Process all messages - deduplication ensures only the last value
        service.process_incoming_messages()

        # All messages should be processed, no remaining messages
        assert len(bridge._incoming_messages) == 0

        # Only the last message should be applied due to deduplication
        config = service.get_config(simple_key)
        assert config.value == 4

    def test_process_incoming_messages_deduplication(
        self,
        service_with_bridge: tuple[ConfigService, FakeMessageBridge],
        simple_key: str,
    ) -> None:
        """Test that only the latest message per key is processed."""
        service, bridge = service_with_bridge
        callback = FakeCallback()
        service.subscribe(simple_key, callback)

        bridge.add_incoming_message((simple_key, {"value": 1, "name": "first"}))
        bridge.add_incoming_message((simple_key, {"value": 2, "name": "second"}))
        bridge.add_incoming_message((simple_key, {"value": 3, "name": "third"}))

        service.process_incoming_messages()

        config = service.get_config(simple_key)
        assert config.value == 3
        assert config.name == "third"

        assert callback.call_count == 1
        assert callback.data == config

    def test_callback_exception_handling(
        self, service: ConfigService, simple_key: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that exceptions in callbacks are handled gracefully."""
        failing_callback = FailingCallback()
        working_callback = FakeCallback()

        service.subscribe(simple_key, failing_callback)
        service.subscribe(simple_key, working_callback)

        config = SimpleConfig(value=500, name="exception_test")

        with caplog.at_level('ERROR'):
            service._notify_subscribers(simple_key, config)

        assert len(caplog.records) == 1
        assert "Error in config subscriber callback" in caplog.records[0].message

        assert failing_callback.call_count == 1
        assert working_callback.called is True
        assert working_callback.data == config

    def test_complex_config_serialization_deserialization(
        self,
        service_with_bridge: tuple[ConfigService, FakeMessageBridge],
        complex_key: str,
    ) -> None:
        """Test complex config with nested data structures."""
        service, bridge = service_with_bridge
        callback = FakeCallback()
        service.subscribe(complex_key, callback)

        config = ComplexConfig(
            items=["item1", "item2", "item3"],
            metadata={"count": 10, "priority": 5},
            threshold=0.75,
        )

        service.update_config(complex_key, config)

        published = bridge.get_published_messages()
        assert len(published) == 1
        assert published[0][1] == {
            "items": ["item1", "item2", "item3"],
            "metadata": {"count": 10, "priority": 5},
            "threshold": 0.75,
        }

        bridge.add_incoming_message((complex_key, published[0][1]))
        callback.reset()

        service.process_incoming_messages()

        # Callback should NOT be called since value hasn't changed
        assert callback.called is False

        # But config should still be stored
        stored_config = service.get_config(complex_key)
        assert stored_config.items == ["item1", "item2", "item3"]
        assert stored_config.metadata == {"count": 10, "priority": 5}
        assert stored_config.threshold == 0.75

    def test_process_incoming_messages_only_notifies_on_value_change(
        self,
        service_with_bridge: tuple[ConfigService, FakeMessageBridge],
        simple_key: str,
    ) -> None:
        """Test that subscribers are only notified when values actually change."""
        service, bridge = service_with_bridge
        callback = FakeCallback()
        service.subscribe(simple_key, callback)

        # Send initial message
        bridge.add_incoming_message((simple_key, {"value": 100}))
        service.process_incoming_messages()

        assert callback.called is True
        assert callback.data.value == 100

        # Send same message again - should NOT trigger callback
        callback.reset()
        bridge.add_incoming_message((simple_key, {"value": 100}))
        service.process_incoming_messages()

        assert callback.called is False

        # Send different message - SHOULD trigger callback
        callback.reset()
        bridge.add_incoming_message((simple_key, {"value": 200}))
        service.process_incoming_messages()

        assert callback.called is True
        assert callback.data.value == 200

    def test_logging_configuration(
        self,
        service_with_bridge: tuple[ConfigService, FakeMessageBridge],
        simple_key: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that debug logging works correctly."""
        service, bridge = service_with_bridge

        config = SimpleConfig(value=1000, name="logging_test")

        with caplog.at_level('DEBUG'):
            service.update_config(simple_key, config)
            bridge.add_incoming_message(
                (simple_key, {"value": 1001, "name": "updated"})
            )
            service.process_incoming_messages()

        assert len(caplog.records) >= 2

    def test_callback_receives_exact_data_object(
        self, service: ConfigService, simple_key: str
    ) -> None:
        """Test that callbacks receive the exact same object instance."""
        callback = FakeCallback()
        service.subscribe(simple_key, callback)

        config = SimpleConfig(value=42, name="identity_test")
        service._config[simple_key] = config
        service._notify_subscribers(simple_key, config)

        assert callback.data is config

    def test_multiple_process_calls_handle_all_messages(
        self,
        service_with_bridge: tuple[ConfigService, FakeMessageBridge],
        simple_key: str,
    ) -> None:
        service, bridge = service_with_bridge
        callback = FakeCallback()
        service.subscribe(simple_key, callback)

        # Add 10 messages
        for i in range(10):
            bridge.add_incoming_message((simple_key, {"value": i, "name": f"msg{i}"}))

        # Process all messages at once
        service.process_incoming_messages()

        # Should have processed all messages (deduplication means only last value)
        config = service.get_config(simple_key)
        assert config.value == 9
        assert len(bridge._incoming_messages) == 0
