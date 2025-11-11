# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Dashboard service composition and setup."""

import logging
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

import scipp as sc

from ess.livedata.config import config_names, instrument_registry
from ess.livedata.config.config_loader import load_config
from ess.livedata.config.instruments import get_config
from ess.livedata.config.streams import get_stream_mapping, stream_kind_to_topic
from ess.livedata.config.workflow_spec import ResultKey
from ess.livedata.core.message import StreamKind
from ess.livedata.handlers.config_handler import ConfigUpdate
from ess.livedata.kafka import consumer as kafka_consumer
from ess.livedata.kafka.message_adapter import AdaptingMessageSource
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.kafka.sink import KafkaSink, serialize_dataarray_to_da00
from ess.livedata.kafka.source import BackgroundMessageSource

from .command_service import CommandService
from .config_store import InMemoryConfigStore
from .correlation_histogram import CorrelationHistogramController
from .data_service import DataService
from .job_controller import JobController
from .job_service import JobService
from .orchestrator import Orchestrator
from .plotting_controller import PlottingController
from .roi_publisher import ROIPublisher
from .stream_manager import StreamManager
from .workflow_config_service import WorkflowConfigService
from .workflow_controller import WorkflowController


class DashboardServices:
    """
    Manages dashboard service setup and dependencies.

    This class encapsulates all the service creation and wiring logic needed
    by both the GUI dashboard and headless testing backend. It uses composition
    to manage the various services without imposing lifecycle management, which
    is left to the caller.

    Parameters
    ----------
    instrument:
        Instrument name (e.g., 'dummy', 'dream', 'bifrost')
    dev:
        Use dev mode with simplified topic structure
    exit_stack:
        ExitStack for managing resource cleanup (caller manages lifecycle)
    logger:
        Logger instance for logging
    pipe_factory:
        Factory function for creating pipes for StreamManager.
        For GUI: use holoviews.streams.Pipe
        For tests: use lambda data: None (no-op)
    """

    def __init__(
        self,
        *,
        instrument: str,
        dev: bool,
        exit_stack: ExitStack,
        logger: logging.Logger,
        pipe_factory: Callable[[Any], Any],
    ):
        self._instrument = instrument
        self._dev = dev
        self._exit_stack = exit_stack
        self._logger = logger
        self._pipe_factory = pipe_factory

        # Config stores for workflow and plotter persistent UI state
        self.workflow_config_store = InMemoryConfigStore(
            max_configs=100, cleanup_fraction=0.2
        )
        self.plotter_config_store = InMemoryConfigStore(
            max_configs=100, cleanup_fraction=0.2
        )

        # Setup all services
        self._setup_data_infrastructure()
        self._setup_workflow_management()

        self._logger.info("DashboardServices initialized for %s", instrument)

    def _setup_data_infrastructure(self) -> None:
        """Set up data services, forwarder, and orchestrator."""
        # Sink for commands
        kafka_downstream_config = load_config(namespace=config_names.kafka_downstream)
        command_sink = self._exit_stack.enter_context(
            KafkaSink[ConfigUpdate](
                kafka_config=kafka_downstream_config,
                instrument=self._instrument,
                logger=self._logger,
            )
        )
        self.command_service = CommandService(sink=command_sink, logger=self._logger)
        self.workflow_config_service = WorkflowConfigService(logger=self._logger)

        # da00 of backend services converted to scipp.DataArray
        ScippDataService = DataService[ResultKey, sc.DataArray]
        self.data_service = ScippDataService()
        self.stream_manager = StreamManager(
            data_service=self.data_service, pipe_factory=self._pipe_factory
        )
        self.job_service = JobService(
            data_service=self.data_service, logger=self._logger
        )
        self.job_controller = JobController(
            command_service=self.command_service, job_service=self.job_service
        )

        # Create ROI publisher for publishing ROI updates to Kafka
        kafka_upstream_config = load_config(namespace=config_names.kafka_upstream)
        roi_sink = self._exit_stack.enter_context(
            KafkaSink(
                kafka_config=kafka_upstream_config,
                instrument=self._instrument,
                serializer=serialize_dataarray_to_da00,
                logger=self._logger,
            )
        )
        roi_publisher = ROIPublisher(sink=roi_sink, logger=self._logger)

        self.plotting_controller = PlottingController(
            job_service=self.job_service,
            config_store=self.plotter_config_store,
            stream_manager=self.stream_manager,
            logger=self._logger,
            roi_publisher=roi_publisher,
        )
        self.orchestrator = Orchestrator(
            self._setup_kafka_consumer(),
            data_service=self.data_service,
            job_service=self.job_service,
            workflow_config_service=self.workflow_config_service,
        )
        self._logger.info("Data infrastructure setup complete")

    def _setup_kafka_consumer(self) -> AdaptingMessageSource:
        """Set up unified Kafka consumer for all dashboard message streams."""
        consumer_config = load_config(
            namespace=config_names.reduced_data_consumer, env=''
        )
        kafka_downstream_config = load_config(namespace=config_names.kafka_downstream)
        topics = [
            stream_kind_to_topic(instrument=self._instrument, kind=kind)
            for kind in [
                StreamKind.LIVEDATA_DATA,
                StreamKind.LIVEDATA_STATUS,
                StreamKind.LIVEDATA_RESPONSES,
            ]
        ]
        consumer = self._exit_stack.enter_context(
            kafka_consumer.make_consumer_from_config(
                topics=topics,
                config={**consumer_config, **kafka_downstream_config},
                group='dashboard',
            )
        )
        # Store BackgroundMessageSource for lifecycle management (start/stop)
        self.background_source = self._exit_stack.enter_context(
            BackgroundMessageSource(consumer=consumer)
        )
        stream_mapping = get_stream_mapping(instrument=self._instrument, dev=self._dev)
        adapter = (
            RoutingAdapterBuilder(stream_mapping=stream_mapping)
            .with_livedata_data_route()
            .with_livedata_status_route()
            .with_livedata_responses_route()
            .build()
        )
        return AdaptingMessageSource(source=self.background_source, adapter=adapter)

    def _setup_workflow_management(self) -> None:
        """Initialize workflow controller and related components."""
        # Load the module to register the instrument's workflows
        self.instrument_module = get_config(self._instrument)
        self.processor_factory = instrument_registry[self._instrument].workflow_factory

        self.correlation_controller = CorrelationHistogramController(self.data_service)
        self.workflow_controller = WorkflowController(
            command_service=self.command_service,
            workflow_config_service=self.workflow_config_service,
            source_names=sorted(self.processor_factory.source_names),
            workflow_registry=self.processor_factory,
            config_store=self.workflow_config_store,
            data_service=self.data_service,
            correlation_histogram_controller=self.correlation_controller,
        )
