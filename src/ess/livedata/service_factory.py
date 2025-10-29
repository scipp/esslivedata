# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: S104  # Binding to 0.0.0.0 is intentional for HTTP services
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any, Generic, NoReturn, TypeVar

from .config import config_names
from .config.config_loader import load_config
from .core import MessageSink, Processor
from .core.handler import JobBasedPreprocessorFactoryBase, PreprocessorFactory
from .core.message import Message, MessageSource, StreamKind
from .core.orchestrating_processor import OrchestratingProcessor
from .core.service import Service
from .http_transport.serialization import (
    DA00MessageSerializer,
    StatusMessageSerializer,
)
from .http_transport.service import HTTPMultiEndpointSink
from .kafka import KafkaTopic
from .kafka import consumer as kafka_consumer
from .kafka.message_adapter import AdaptingMessageSource, MessageAdapter
from .kafka.sink import KafkaSink, UnrollingSinkAdapter
from .kafka.source import (
    BackgroundMessageSource,
    KafkaConsumer,
    KafkaMessageSource,
    MultiConsumer,
)
from .sinks import PlotToPngSink

Traw = TypeVar("Traw")
Tin = TypeVar("Tin")
Tout = TypeVar("Tout")


class DataServiceBuilder(Generic[Traw, Tin, Tout]):
    def __init__(
        self,
        *,
        instrument: str,
        name: str,
        log_level: int = logging.INFO,
        adapter: MessageAdapter | None = None,
        preprocessor_factory: PreprocessorFactory[Tin, Tout],
        startup_messages: list[Message[Tout]] | None = None,
        processor_cls: type[Processor] = OrchestratingProcessor,
    ) -> None:
        """
        Parameters
        ----------
        instrument:
            The name of the instrument.
        name:
            The name of the service.
        log_level:
            The log level to use for the service.
        adapter:
            The message adapter to use for incoming messages.
        preprocessor_factory:
            The factory to use for creating preprocessors for messages.
        startup_messages:
            A list of messages to publish before starting the service.
        processor_cls:
            The processor class to use for processing messages. Defaults to
            `OrchestratingProcessor`.
        """
        self._name = f'{instrument}_{name}'
        self._log_level = log_level
        self._topics: list[KafkaTopic] | None = None
        self._instrument = instrument
        self._adapter = adapter
        self._preprocessor_factory = preprocessor_factory
        self._startup_messages = startup_messages or []
        self._processor_cls = processor_cls
        if isinstance(preprocessor_factory, JobBasedPreprocessorFactoryBase):
            # Ensure only jobs from the active namespace can be created by JobFactory.
            preprocessor_factory.instrument.active_namespace = name

    @property
    def instrument(self) -> str:
        """Returns the instrument name."""
        return self._instrument

    def from_consumer_config(
        self,
        kafka_config: dict[str, Any],
        sink: MessageSink[Tout],
        raise_on_adapter_error: bool = False,
        use_background_source: bool = False,
    ) -> Service:
        """Create a service from a consumer config."""
        resources = ExitStack()
        try:
            config_topic, config_consumer = resources.enter_context(
                kafka_consumer.make_control_consumer(instrument=self._instrument)
            )
            topics = self._adapter.topics
            data_topics = [topic for topic in topics if topic != config_topic]
            data_consumer = resources.enter_context(
                kafka_consumer.make_consumer_from_config(
                    topics=data_topics, config=kafka_config, group=self._name
                )
            )
            consumer = MultiConsumer([config_consumer, data_consumer])

            if use_background_source:
                source = resources.enter_context(
                    BackgroundMessageSource(consumer=consumer)
                )
            else:
                source = KafkaMessageSource(consumer=consumer)

            # Ownership of resource stack transferred to the service
            return self.from_source(
                source=source,
                sink=sink,
                resources=resources.pop_all(),
                raise_on_adapter_error=raise_on_adapter_error,
            )
        except Exception:
            resources.close()
            raise

    def from_consumer(
        self,
        consumer: KafkaConsumer,
        sink: MessageSink[Tout],
        resources: ExitStack | None = None,
        raise_on_adapter_error: bool = False,
        use_background_source: bool = False,
    ) -> Service:
        if resources is None:
            resources = ExitStack()

        if use_background_source:
            source = resources.enter_context(BackgroundMessageSource(consumer=consumer))
        else:
            source = KafkaMessageSource(consumer=consumer)

        return self.from_source(
            source=source,
            sink=sink,
            resources=resources,
            raise_on_adapter_error=raise_on_adapter_error,
        )

    def from_source(
        self,
        source: MessageSource,
        sink: MessageSink[Tout],
        resources: ExitStack | None = None,
        raise_on_adapter_error: bool = False,
    ) -> Service:
        processor = self._processor_cls(
            source=source
            if self._adapter is None
            else AdaptingMessageSource(
                source=source,
                adapter=self._adapter,
                raise_on_error=raise_on_adapter_error,
            ),
            sink=sink,
            preprocessor_factory=self._preprocessor_factory,
        )
        sink.publish_messages(self._startup_messages)
        return Service(
            processor=processor,
            name=self._name,
            log_level=self._log_level,
            resources=resources,
        )


class DataServiceRunner:
    def __init__(
        self,
        *,
        pretty_name: str,
        make_builder: Callable[..., DataServiceBuilder],
    ) -> None:
        self._make_builder = make_builder
        self._parser = Service.setup_arg_parser(description=f'{pretty_name} Service')
        self._parser.add_argument(
            '--source-type',
            choices=['kafka', 'http'],
            default='kafka',
            help='Select source type: kafka or http',
        )
        self._parser.add_argument(
            '--http-data-source',
            help='HTTP URL for data source (e.g., http://localhost:8000)',
        )
        self._parser.add_argument(
            '--http-config-source',
            help='HTTP URL for config source (e.g., http://localhost:9000)',
        )
        self._parser.add_argument(
            '--sink-type',
            choices=['kafka', 'png', 'http'],
            default='kafka',
            help='Select sink type: kafka, png, or http',
        )
        self._parser.add_argument(
            '--http-host',
            default='0.0.0.0',
            help='HTTP server host (when using http sink)',
        )
        self._parser.add_argument(
            '--http-port',
            type=int,
            default=8000,
            help='HTTP server port (when using http sink)',
        )

    @property
    def parser(self) -> argparse.ArgumentParser:
        """
        Returns the argument parser.

        Use this to add extra arguments the `make_builder` function needs.
        """
        return self._parser

    def run(
        self,
    ) -> NoReturn:
        args = vars(self._parser.parse_args())

        source_type = args.pop('source_type')
        http_data_source = args.pop('http_data_source')
        http_config_source = args.pop('http_config_source')
        sink_type = args.pop('sink_type')
        http_host = args.pop('http_host')
        http_port = args.pop('http_port')

        builder = self._make_builder(**args)

        # Create sink
        http_sink_instance = None  # Track HTTP sink for starting/stopping
        if sink_type == 'kafka':
            kafka_downstream_config = load_config(
                namespace=config_names.kafka_downstream
            )
            sink = KafkaSink(
                instrument=builder.instrument, kafka_config=kafka_downstream_config
            )
        elif sink_type == 'http':
            # Use multi-endpoint sink for proper topic separation
            http_sink_instance = HTTPMultiEndpointSink(
                instrument=builder.instrument,
                stream_serializers={
                    StreamKind.LIVEDATA_DATA: DA00MessageSerializer(),
                    StreamKind.LIVEDATA_STATUS: StatusMessageSerializer(),
                },
                host=http_host,
                port=http_port,
            )
            sink = http_sink_instance
        else:
            sink = PlotToPngSink()
        sink = UnrollingSinkAdapter(sink)

        # Create source based on source type
        if source_type == 'kafka':
            consumer_config = load_config(
                namespace=config_names.raw_data_consumer, env=''
            )
            kafka_upstream_config = load_config(namespace=config_names.kafka_upstream)

            # Start HTTP sink if we have one
            if http_sink_instance is not None:
                http_sink_instance.start()

            try:
                with builder.from_consumer_config(
                    kafka_config={**consumer_config, **kafka_upstream_config},
                    sink=sink,
                    use_background_source=True,
                ) as service:
                    service.start()
            finally:
                # Stop HTTP sink on exit
                if http_sink_instance is not None:
                    http_sink_instance.stop()
        elif source_type == 'http':
            # HTTP source mode
            from .http_transport import (
                GenericJSONMessageSerializer,
                HTTPMessageSource,
                MultiHTTPSource,
            )

            if not http_data_source:
                raise ValueError(
                    "--http-data-source is required when using --source-type http"
                )

            # Create HTTP sources
            sources = []

            # Data source (DA00 format for monitor/detector data)
            data_source = HTTPMessageSource(
                base_url=http_data_source,
                serializer=DA00MessageSerializer(),
            )
            sources.append(data_source)

            # Config source (JSON format for config messages)
            if http_config_source:
                from .kafka.message_adapter import RawConfigItem

                # Create adapter to convert JSON dict to RawConfigItem
                class HTTPConfigAdapter:
                    """Adapts HTTP JSON config messages to RawConfigItem format."""

                    def adapt(self, message: Message) -> Message:
                        """Convert dict config to RawConfigItem."""
                        if not isinstance(message.value, dict):
                            return message
                        if 'key' in message.value and 'value' in message.value:
                            import json

                            # Convert JSON dict format to RawConfigItem
                            raw_item = RawConfigItem(
                                key=message.value['key'].encode('utf-8'),
                                value=json.dumps(message.value['value']).encode(
                                    'utf-8'
                                ),
                            )
                            return Message(
                                timestamp=message.timestamp,
                                stream=message.stream,
                                value=raw_item,
                            )
                        return message

                config_http_source = HTTPMessageSource(
                    base_url=http_config_source,
                    endpoint='/config',
                    serializer=GenericJSONMessageSerializer(),
                )

                # Wrap with adapter to convert to RawConfigItem
                config_source = AdaptingMessageSource(
                    source=config_http_source,
                    adapter=HTTPConfigAdapter(),
                    raise_on_error=False,
                )
                sources.append(config_source)

            # Combine sources
            if len(sources) == 1:
                source = sources[0]
            else:
                source = MultiHTTPSource(sources)

            # HTTP sources return typed Messages, so we don't need the adapter
            # Temporarily clear it for HTTP mode
            original_adapter = builder._adapter
            builder._adapter = None

            # Build service with HTTP source
            service = builder.from_source(
                source=source,
                sink=sink,
                resources=None,
                raise_on_adapter_error=False,
            )

            # Restore adapter (in case builder is reused, though unlikely)
            builder._adapter = original_adapter

            # Start HTTP sink if we have one
            if http_sink_instance is not None:
                http_sink_instance.start()

            try:
                service.start()
            finally:
                # Stop HTTP sink on exit
                if http_sink_instance is not None:
                    http_sink_instance.stop()
        else:
            raise ValueError(f"Unknown source type: {source_type}")
