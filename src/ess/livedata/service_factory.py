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
from .core.message_batcher import (
    AdaptiveMessageBatcher,
    MessageBatcher,
    SimpleMessageBatcher,
)
from .core.orchestrating_processor import OrchestratingProcessor
from .core.rate_aware_batcher import RateAwareMessageBatcher
from .core.service import Service
from .kafka import KafkaTopic
from .kafka import consumer as kafka_consumer
from .kafka.message_adapter import AdaptingMessageSource, MessageAdapter
from .kafka.sink import KafkaSink, UnrollingSinkAdapter
from .kafka.sink_serializers import make_default_sink_serializer
from .kafka.source import (
    BackgroundMessageSource,
    KafkaConsumer,
    KafkaMessageSource,
)
from .kafka.stream_counter import StreamCounter
from .kafka.stream_mapping import StreamMapping
from .logging_config import configure_logging
from .sinks import PlotToPngSink

logger = structlog.get_logger(__name__)

Traw = TypeVar("Traw")
Tin = TypeVar("Tin")
Tout = TypeVar("Tout")


_INNER_BATCHER_FACTORIES: dict[str, Callable[[float], MessageBatcher]] = {
    'simple': SimpleMessageBatcher,
    'rate-aware': RateAwareMessageBatcher,
}


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
        job_threads: int = 5,
        stream_counter: StreamCounter | None = None,
        message_batcher: MessageBatcher | None = None,
        stream_mapping: StreamMapping | None = None,
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
        job_threads:
            Number of threads for parallel job execution. When > 1, job
            accumulation and finalization are run in a thread pool.
        message_batcher:
            Message batcher for the processor. If ``None``, the processor's
            default is used.  Services that require a specific batcher should
            set this explicitly; otherwise ``DataServiceRunner`` will assign
            one based on its CLI argument.
        """
        self._name = f'{instrument}_{name}'
        self._log_level = log_level
        self._topics: list[KafkaTopic] | None = None
        self._instrument = instrument
        self._adapter = adapter
        self._preprocessor_factory = preprocessor_factory
        self._startup_messages = startup_messages or []
        self._processor_cls = processor_cls
        self._job_threads = job_threads
        self._stream_counter = stream_counter
        self._message_batcher = message_batcher
        self._stream_mapping = stream_mapping
        if isinstance(preprocessor_factory, JobBasedPreprocessorFactoryBase):
            # Ensure only jobs from the active namespace can be created by JobFactory.
            preprocessor_factory.instrument.active_namespace = name

    @property
    def instrument(self) -> str:
        """Returns the instrument name."""
        return self._instrument

    @property
    def job_threads(self) -> int:
        return self._job_threads

    @job_threads.setter
    def job_threads(self, value: int) -> None:
        self._job_threads = value

    @property
    def message_batcher(self) -> MessageBatcher | None:
        return self._message_batcher

    @message_batcher.setter
    def message_batcher(self, value: MessageBatcher) -> None:
        self._message_batcher = value

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
            consumer = resources.enter_context(
                kafka_consumer.make_consumer_from_config(
                    topics=self._adapter.topics,
                    config=kafka_config,
                    group=self._name,
                )
            )

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
                stream_counter=self._stream_counter,
            ),
            sink=sink,
            preprocessor_factory=self._preprocessor_factory,
            stream_mapping=self._stream_mapping,
            job_threads=self._job_threads,
            stream_stats_provider=self._stream_counter,
            message_batcher=self._message_batcher,
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
            '--sync-scheduler',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Use synchronous dask scheduler instead of threaded'
            ' (reduces GIL contention)',
        )
        self._parser.add_argument(
            '--job-threads',
            type=int,
            default=5,
            help='Number of threads for parallel job execution (1=sequential)',
        )
        self._parser.add_argument(
            '--sink-type',
            choices=['kafka', 'png'],
            default='kafka',
            help='Select sink type: kafka or png',
        )
        self._parser.add_argument(
            '--batcher',
            choices=list(_INNER_BATCHER_FACTORIES),
            default='simple',
            help='Inner batcher wrapped by AdaptiveMessageBatcher: "simple" uses'
            ' SimpleMessageBatcher (pre-existing default); "rate-aware" uses'
            ' RateAwareMessageBatcher (per-stream pulse-slot completion).',
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

        sync_scheduler = args.pop('sync_scheduler')
        if sync_scheduler:
            import dask
            from dask.local import SynchronousExecutor

            dask.config.set(pool=SynchronousExecutor())
            logger.info("dask_scheduler", mode="synchronous")

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
        kafka_config = load_config(namespace=config_names.kafka)

        sink_type = args.pop('sink_type')
        job_threads = args.pop('job_threads')
        batcher_name = args.pop('batcher')
        builder = self._make_builder(**args)
        builder.job_threads = job_threads
        if job_threads > 1:
            logger.info("job_threads", threads=job_threads)

        if builder.message_batcher is None:
            inner_factory = _INNER_BATCHER_FACTORIES[batcher_name]
            builder.message_batcher = AdaptiveMessageBatcher(
                inner_factory=inner_factory
            )
            logger.info("message_batcher", inner=batcher_name)
        else:
            logger.info(
                "message_batcher",
                inner=type(builder.message_batcher).__name__,
                overridden_by_builder=True,
            )

        if sink_type == 'kafka':
            kafka_sink = KafkaSink(
                kafka_config=kafka_config,
                serializer=make_default_sink_serializer(
                    instrument=builder.instrument,
                ),
            )
        else:
            kafka_sink = PlotToPngSink()

        with ExitStack() as resources:
            sink = resources.enter_context(kafka_sink)
            sink = UnrollingSinkAdapter(sink)

            with builder.from_consumer_config(
                kafka_config={**consumer_config, **kafka_config},
                sink=sink,
                use_background_source=True,
            ) as service:
                service.start()
