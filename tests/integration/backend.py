# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Reusable dashboard backend for integration testing without GUI components."""

import logging
from contextlib import ExitStack
from types import TracebackType

import scipp as sc

from ess.livedata.config import config_names, instrument_registry
from ess.livedata.config.config_loader import load_config
from ess.livedata.config.instruments import get_config
from ess.livedata.config.streams import get_stream_mapping, stream_kind_to_topic
from ess.livedata.config.workflow_spec import ResultKey
from ess.livedata.core.message import StreamKind
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.config_store import InMemoryConfigStore
from ess.livedata.dashboard.correlation_histogram import CorrelationHistogramController
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_controller import JobController
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.orchestrator import Orchestrator
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.roi_publisher import ROIPublisher
from ess.livedata.dashboard.stream_manager import StreamManager
from ess.livedata.dashboard.workflow_config_service import WorkflowConfigService
from ess.livedata.dashboard.workflow_controller import WorkflowController
from ess.livedata.handlers.config_handler import ConfigUpdate
from ess.livedata.kafka import consumer as kafka_consumer
from ess.livedata.kafka.message_adapter import AdaptingMessageSource
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.kafka.sink import KafkaSink, serialize_dataarray_to_da00
from ess.livedata.kafka.source import BackgroundMessageSource


class DashboardBackend:
    """
    Reusable dashboard backend for integration tests (no GUI).

    This class extracts the core backend functionality from DashboardBase,
    providing access to all the services and controllers without the Panel/GUI
    components. It's designed to be used in integration tests as a context manager.

    Parameters
    ----------
    instrument:
        Instrument name (e.g., 'dummy', 'dream', 'bifrost')
    dev:
        Use dev mode with simplified topic structure
    log_level:
        Logging level
    """

    def __init__(
        self,
        *,
        instrument: str = 'dummy',
        dev: bool = True,
        log_level: int = logging.INFO,
    ):
        self._instrument = instrument
        self._dev = dev
        self._logger = logging.getLogger(f'{instrument}_dashboard_backend')
        self._logger.setLevel(log_level)

        self._exit_stack = ExitStack()
        self._exit_stack.__enter__()

        # Config stores for workflow and plotter persistent state
        self._workflow_config_store = InMemoryConfigStore(
            max_configs=100, cleanup_fraction=0.2
        )
        self._plotter_config_store = InMemoryConfigStore(
            max_configs=100, cleanup_fraction=0.2
        )

        # Setup data infrastructure (services, Kafka, orchestrator)
        self._setup_data_infrastructure(instrument=instrument, dev=dev)

        # Load instrument module and setup workflow management
        self._instrument_module = get_config(instrument)
        self._processor_factory = instrument_registry[self._instrument].workflow_factory
        self._setup_workflow_management()

        self._logger.info("DashboardBackend initialized for %s", instrument)

    def _setup_data_infrastructure(self, instrument: str, dev: bool) -> None:
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

        # Data services
        ScippDataService = DataService[ResultKey, sc.DataArray]
        self.data_service = ScippDataService()
        # For integration tests, we use a no-op pipe factory since we don't have GUI
        self.stream_manager = StreamManager(
            data_service=self.data_service, pipe_factory=lambda data: None
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
                instrument=instrument,
                serializer=serialize_dataarray_to_da00,
                logger=self._logger,
            )
        )
        roi_publisher = ROIPublisher(sink=roi_sink, logger=self._logger)

        self.plotting_controller = PlottingController(
            job_service=self.job_service,
            config_store=self._plotter_config_store,
            stream_manager=self.stream_manager,
            logger=self._logger,
            roi_publisher=roi_publisher,
        )

        # Setup orchestrator with Kafka consumer
        self.orchestrator = Orchestrator(
            self._setup_kafka_consumer(instrument=instrument, dev=dev),
            data_service=self.data_service,
            job_service=self.job_service,
            workflow_config_service=self.workflow_config_service,
        )
        self._logger.info("Data infrastructure setup complete")

    def _setup_kafka_consumer(
        self, instrument: str, dev: bool
    ) -> AdaptingMessageSource:
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
        # Store BackgroundMessageSource for lifecycle management
        self._background_source = self._exit_stack.enter_context(
            BackgroundMessageSource(consumer=consumer)
        )
        stream_mapping = get_stream_mapping(instrument=instrument, dev=dev)
        adapter = (
            RoutingAdapterBuilder(stream_mapping=stream_mapping)
            .with_livedata_data_route()
            .with_livedata_status_route()
            .with_livedata_responses_route()
            .build()
        )
        return AdaptingMessageSource(source=self._background_source, adapter=adapter)

    def _setup_workflow_management(self) -> None:
        """Initialize workflow controller."""
        self.correlation_controller = CorrelationHistogramController(self.data_service)
        self.workflow_controller = WorkflowController(
            command_service=self.command_service,
            workflow_config_service=self.workflow_config_service,
            source_names=sorted(self._processor_factory.source_names),
            workflow_registry=self._processor_factory,
            config_store=self._workflow_config_store,
            data_service=self.data_service,
            correlation_histogram_controller=self.correlation_controller,
        )

    def start(self) -> None:
        """Start the background message source."""
        self._background_source.start()
        self._logger.info("DashboardBackend started")

    def update(self) -> None:
        """Process one batch of messages from Kafka."""
        self.orchestrator.update()

    def stop(self) -> None:
        """Stop the background message source and clean up resources."""
        if hasattr(self, '_background_source'):
            self._background_source.stop()
        self._exit_stack.__exit__(None, None, None)
        self._logger.info("DashboardBackend stopped")

    def __enter__(self) -> 'DashboardBackend':
        """Enter context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.stop()
