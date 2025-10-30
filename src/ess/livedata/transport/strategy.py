# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Transport strategy abstractions for creating sources and sinks."""

from typing import Any, Protocol

from ..config import config_names
from ..config.config_loader import load_config
from ..config.streams import stream_kind_to_topic
from ..core.message import MessageSink, MessageSource, StreamKind
from ..http_transport import (
    DA00MessageSerializer,
    GenericJSONMessageSerializer,
    HTTPMessageSource,
    HTTPMultiEndpointSink,
    MultiHTTPSource,
    StatusMessageSerializer,
)
from ..kafka import consumer as kafka_consumer
from ..kafka.message_adapter import MessageAdapter
from ..kafka.sink import KafkaSink
from ..kafka.source import BackgroundMessageSource


class TransportStrategy(Protocol):
    """
    Protocol for transport strategies that create sources and sinks.

    Transport strategies encapsulate the logic for creating message sources
    and sinks for different transport mechanisms (Kafka, HTTP, etc.).
    """

    def create_source(
        self,
        stream_kinds: list[StreamKind],
        instrument: str,
        adapter: MessageAdapter | None = None,
    ) -> MessageSource:
        """
        Create a message source for the given stream kinds.

        Parameters
        ----------
        stream_kinds:
            List of stream kinds to consume messages from.
        instrument:
            Instrument name for topic/endpoint derivation.
        adapter:
            Optional message adapter for transforming raw messages.

        Returns
        -------
        :
            Message source for consuming messages.
        """
        ...

    def create_sink(
        self, stream_kinds: list[StreamKind], instrument: str
    ) -> MessageSink:
        """
        Create a message sink for the given stream kinds.

        Parameters
        ----------
        stream_kinds:
            List of stream kinds to publish messages to.
        instrument:
            Instrument name for topic/endpoint derivation.

        Returns
        -------
        :
            Message sink for publishing messages.
        """
        ...


class KafkaStrategy:
    """
    Strategy for creating Kafka-based sources and sinks.

    Uses existing Kafka infrastructure with BackgroundMessageSource for consuming
    and KafkaSink for publishing.

    Parameters
    ----------
    kafka_config:
        Kafka configuration dictionary (broker settings, auth, etc.).
    """

    def __init__(self, kafka_config: dict[str, Any]):
        self._kafka_config = kafka_config

    def create_source(
        self,
        stream_kinds: list[StreamKind],
        instrument: str,
        adapter: MessageAdapter | None = None,
    ) -> MessageSource:
        """
        Create a Kafka message source.

        Creates a BackgroundMessageSource wrapping a KafkaConsumer for the
        given stream kinds. Topics are derived from stream kinds using
        stream_kind_to_topic(). If an adapter is provided, wraps the source
        with AdaptingMessageSource to convert raw Kafka messages to typed Messages.

        Note: This implementation creates Kafka consumers without explicit cleanup
        handlers. For production use with proper resource management, consider using
        the full service factory pattern with ExitStack.

        Parameters
        ----------
        stream_kinds:
            List of stream kinds to consume.
        instrument:
            Instrument name.
        adapter:
            Optional message adapter for transforming raw Kafka messages to typed
            Messages. If not provided, returns raw Kafka messages.

        Returns
        -------
        :
            Message source yielding typed Messages (if adapter provided) or raw
            Kafka messages (if no adapter). The returned source is a context manager
            and should be used with `with` statement or an ExitStack for proper cleanup.
        """
        # Convert stream kinds to topics
        topics = [
            stream_kind_to_topic(instrument=instrument, kind=kind)
            for kind in stream_kinds
        ]

        # Load consumer config and create consumer
        consumer_config = load_config(namespace=config_names.raw_data_consumer, env='')
        kafka_upstream_config = load_config(namespace=config_names.kafka_upstream)
        combined_config = {
            **consumer_config,
            **kafka_upstream_config,
            **self._kafka_config,
        }

        # Create consumer - note: context manager cleanup is caller's responsibility
        consumer_cm = kafka_consumer.make_consumer_from_config(
            topics=topics,
            config=combined_config,
            group='transport_strategy',
        )

        # Enter context to get consumer
        # In production, the caller should manage this with an ExitStack
        consumer = consumer_cm.__enter__()

        # Wrap in background source for async consumption
        source: MessageSource = BackgroundMessageSource(consumer=consumer)

        # Apply adapter if provided to convert raw Kafka messages to typed Messages
        if adapter is not None:
            from ..kafka.message_adapter import AdaptingMessageSource

            source = AdaptingMessageSource(
                source=source, adapter=adapter, raise_on_error=False
            )

        return source

    def create_sink(
        self, stream_kinds: list[StreamKind], instrument: str
    ) -> MessageSink:
        """
        Create a Kafka message sink.

        Creates a KafkaSink for publishing to topics derived from stream kinds.

        Parameters
        ----------
        stream_kinds:
            List of stream kinds to publish to (used for validation).
        instrument:
            Instrument name.

        Returns
        -------
        :
            KafkaSink for publishing messages.
        """
        kafka_downstream_config = load_config(namespace=config_names.kafka_downstream)
        combined_config = {**kafka_downstream_config, **self._kafka_config}

        return KafkaSink(
            instrument=instrument,
            kafka_config=combined_config,
        )


class HttpStrategy:
    """
    Strategy for creating HTTP-based sources and sinks.

    Uses HTTP polling sources and multi-endpoint sinks for service-to-service
    communication without Kafka.

    Parameters
    ----------
    base_url:
        Base URL for HTTP endpoints (e.g., "http://localhost:5011").
    """

    def __init__(self, base_url: str):
        from urllib.parse import urlparse

        self._base_url = base_url
        parsed = urlparse(base_url)
        self._host = parsed.hostname or '0.0.0.0'  # noqa: S104
        self._port = parsed.port or 8000

    def create_source(
        self,
        stream_kinds: list[StreamKind],
        instrument: str,
        adapter: MessageAdapter | None = None,
    ) -> MessageSource:
        """
        Create an HTTP message source.

        Creates HTTPMessageSource instances for each stream kind, with appropriate
        serializers based on the stream type. Combines multiple sources using
        MultiHTTPSource.

        HTTP sources directly deserialize to typed Messages, so the adapter parameter
        is accepted for protocol compliance but is not used.

        Parameters
        ----------
        stream_kinds:
            List of stream kinds to poll.
        instrument:
            Instrument name.
        adapter:
            Message adapter (unused - HTTP sources already return typed Messages).

        Returns
        -------
        :
            HTTPMessageSource or MultiHTTPSource yielding typed Messages.
        """
        sources = []

        for kind in stream_kinds:
            # Convert stream kind to topic, then to endpoint
            topic = stream_kind_to_topic(instrument=instrument, kind=kind)
            endpoint = f"/{topic.removeprefix(f'{instrument}_')}"

            # Select serializer based on stream kind
            if kind == StreamKind.LIVEDATA_STATUS:
                serializer = StatusMessageSerializer()
            elif kind == StreamKind.LIVEDATA_CONFIG:
                serializer = GenericJSONMessageSerializer()
            else:
                # Default to DA00 for data streams
                serializer = DA00MessageSerializer()

            http_source = HTTPMessageSource(
                base_url=self._base_url,
                endpoint=endpoint,
                serializer=serializer,
            )
            sources.append(http_source)

        # Return single source or combined multi-source
        if len(sources) == 1:
            return sources[0]
        return MultiHTTPSource(sources)

    def create_sink(
        self, stream_kinds: list[StreamKind], instrument: str
    ) -> MessageSink:
        """
        Create an HTTP multi-endpoint sink.

        Creates HTTPMultiEndpointSink with appropriate serializers for each
        stream kind.

        Parameters
        ----------
        stream_kinds:
            List of stream kinds to expose endpoints for.
        instrument:
            Instrument name.

        Returns
        -------
        :
            HTTPMultiEndpointSink for publishing messages via HTTP.
        """
        # Reuse serializer instances of the same type to avoid validation errors
        # when multiple stream kinds map to the same endpoint
        da00_serializer = DA00MessageSerializer()
        status_serializer = StatusMessageSerializer()
        json_serializer = GenericJSONMessageSerializer()

        # Build serializer mapping for stream kinds
        stream_serializers = {}
        for kind in stream_kinds:
            if kind == StreamKind.LIVEDATA_STATUS:
                stream_serializers[kind] = status_serializer
            elif kind == StreamKind.LIVEDATA_CONFIG:
                stream_serializers[kind] = json_serializer
            else:
                # Default to DA00 for data streams
                stream_serializers[kind] = da00_serializer

        # HTTPMultiEndpointSink derives endpoints from stream kinds internally
        return HTTPMultiEndpointSink(
            instrument=instrument,
            stream_serializers=stream_serializers,
            host=self._host,
            port=self._port,
        )
