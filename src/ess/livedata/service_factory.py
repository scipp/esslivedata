# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any, Generic, NoReturn, TypeVar

import structlog

from .config import config_names
from .config.config_loader import load_config
from .core import MessageSink, Processor
from .core.handler import JobBasedPreprocessorFactoryBase, PreprocessorFactory
from .core.message import Message, MessageSource
from .core.orchestrating_processor import OrchestratingProcessor
from .core.service import Service
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
from .logging_config import configure_logging
from .sinks import PlotToPngSink

logger = structlog.get_logger(__name__)

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
        logger.info(
            "service_created",
            service=self._name,
            instrument=self._instrument,
            preprocessor_factory=type(self._preprocessor_factory).__name__,
        )
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
            '--sink-type',
            choices=['kafka', 'png'],
            default='kafka',
            help='Select sink type: kafka or png',
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

        # Configure logging with parsed arguments
        log_level = getattr(logging, args.pop('log_level'))
        log_json_file = args.pop('log_json_file')
        no_stdout_log = args.pop('no_stdout_log')
        configure_logging(
            level=log_level,
            json_file=log_json_file,
            disable_stdout=no_stdout_log,
        )

        logger.info("service_starting", **args)
        consumer_config = load_config(namespace=config_names.raw_data_consumer, env='')
        kafka_downstream_config = load_config(namespace=config_names.kafka_downstream)
        kafka_upstream_config = load_config(namespace=config_names.kafka_upstream)

        sink_type = args.pop('sink_type')
        builder = self._make_builder(**args)

        if sink_type == 'kafka':
            kafka_sink = KafkaSink(
                instrument=builder.instrument, kafka_config=kafka_downstream_config
            )
        else:
            kafka_sink = PlotToPngSink()

        with ExitStack() as resources:
            sink = resources.enter_context(kafka_sink)
            sink = UnrollingSinkAdapter(sink)

            with builder.from_consumer_config(
                kafka_config={**consumer_config, **kafka_upstream_config},
                sink=sink,
                use_background_source=True,
            ) as service:
                service.start()
