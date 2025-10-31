# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import logging
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Generic, TypeVar

import pydantic

from .message_bridge import MessageBridge
from .schema_validator import SchemaValidator

K = TypeVar('K')
V = TypeVar('V')
Serialized = TypeVar('Serialized')


JSONSerialized = dict[str, Any]


class ConfigService(Generic[K, Serialized, V]):
    """
    Service for configuration data with schema validation and message publishing.

    This service initializes from Livedata's Kafka config topic into a local dictionary
    of the latest state. The connection to Kafka is implemented via an implementation of
    :py:class:`MessageBridge`.

    Local producers of new config values, in particular user changes to widget values,
    are set in this service and published to the message bridge, which in turn
    communicates with the Kafka topic.
    """

    def __init__(
        self,
        schema_validator: SchemaValidator[K, Serialized, V],
        message_bridge: MessageBridge[K, Serialized] | None = None,
    ):
        self.schema_validator = schema_validator
        self._message_bridge = message_bridge
        self._subscribers: dict[K, list[Callable[[V], None]]] = defaultdict(list)
        self._logger = logging.getLogger(__name__)
        self._config: dict[K, V] = {}
        self._update_disabled = False

    def subscribe(self, key: K, callback: Callable[[V], None]) -> None:
        """
        Subscribe to configuration updates for a specific key.

        Parameters
        ----------
        key:
            The configuration key to subscribe to.
        callback:
            The callback function to be called when the configuration is updated.
        """
        self._subscribers[key].append(callback)
        if (data := self._config.get(key)) is not None:
            self._invoke(key, callback, data)

    def get_config(self, key: K, default: Any = None) -> V | Any:
        """
        Get the current configuration value for a specific key.

        Parameters
        ----------
        key:
            The configuration key to retrieve.
        default:
            The default value to return if the key is not set.

        Returns
        -------
        The current configuration value or the default if not set.
        """
        return self._config.get(key, default)

    def update_config(self, key: K, value: V) -> None:
        """
        Update the configuration for a specific key.

        This does not trigger callbacks directly. After publishing the update it makes
        it back into the service via :py:meth:`process_incoming_messages`, which will
        then notify all subscribers. This ensures a consistent ordering of updates,
        enforced by Kafka, and avoids potential infinite loops.
        """
        if self._update_disabled:
            return
        if not self.schema_validator.validate(key, value):
            raise ValueError(
                f'No schema registered for key {key} or value does not match schema'
            )
        self._config[key] = value
        if self._message_bridge:
            # Communication with message bridge is using raw JSON.
            self._message_bridge.publish(key, self.schema_validator.serialize(value))

    @contextmanager
    def _disable_updates(self):
        """Context manager to temporarily disable update_config."""
        old_state = self._update_disabled
        self._update_disabled = True
        try:
            yield
        finally:
            self._update_disabled = old_state

    def process_incoming_messages(self) -> None:
        """Process any available incoming messages from the message bridge."""
        if not self._message_bridge:
            return

        # The message bridge should deal with key-based deduplication. To be sure, we
        # convert the messages to a dict, which will overwrite any duplicates.
        messages = dict(self._message_bridge.pop_all())
        # Disable updates while processing incoming messages to avoid callbacks that set
        # widget values, which would otherwise trigger another update to the bridge.
        with self._disable_updates():
            for key, value in messages.items():
                self._handle_config_update(key, value)

    def _handle_config_update(self, key: K, value: Serialized) -> None:
        """Handle a configuration update from the message bridge."""
        try:
            self._logger.debug('Received config update for key %s: %s', key, value)
            validated = self.schema_validator.deserialize(key, value)
            self._logger.debug('Validated config for key %s: %s', key, validated)
            if validated is None:
                return
            # Only notify subscribers if value actually changed
            old_value = self._config.get(key)
            if old_value != validated:
                self._config[key] = validated
                self._notify_subscribers(key, validated)
            else:
                self._config[key] = validated  # Still update to maintain consistency
        except pydantic.ValidationError as e:
            self._logger.error('Invalid config data received for key %s: %s', key, e)

    def _notify_subscribers(self, key: K, data: V) -> None:
        """Notify all subscribers for a given config key."""
        self._logger.debug(
            'Notifying %d subscribers for key %s', len(self._subscribers[key]), key
        )
        for callback in self._subscribers.get(key, []):
            self._invoke(key, callback, data)

    def _invoke(self, key: K, callback: Callable[[V], None], data: V) -> None:
        try:
            callback(data)
        except Exception:
            self._logger.exception(
                'Error in config subscriber callback for key %s.', key
            )
