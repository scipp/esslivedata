# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Common functionality for implementing dashboards."""

import logging
from abc import ABC, abstractmethod
from contextlib import ExitStack

import panel as pn
import scipp as sc
from holoviews import Dimension, streams

from ess.livedata import ServiceBase
from ess.livedata.config import config_names, instrument_registry
from ess.livedata.config.config_loader import load_config
from ess.livedata.config.instruments import get_config
from ess.livedata.config.streams import get_stream_mapping, stream_kind_to_topic
from ess.livedata.config.workflow_spec import ResultKey
from ess.livedata.core.message import StreamKind
from ess.livedata.kafka import consumer as kafka_consumer
from ess.livedata.kafka.message_adapter import AdaptingMessageSource
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.kafka.sink import KafkaSink, serialize_dataarray_to_da00
from ess.livedata.kafka.source import BackgroundMessageSource

from .config_store import InMemoryConfigStore
from .correlation_histogram import CorrelationHistogramController
from .data_service import DataService
from .job_controller import JobController
from .job_service import JobService
from .kafka_job_command_service import KafkaJobCommandService
from .kafka_workflow_config_service import KafkaWorkflowConfigService
from .orchestrator import Orchestrator
from .plotting_controller import PlottingController
from .roi_publisher import ROIPublisher
from .stream_manager import StreamManager
from .widgets.plot_creation_widget import PlotCreationWidget
from .widgets.reduction_widget import ReductionWidget
from .workflow_controller import WorkflowController

# Global throttling for sliders, etc.
pn.config.throttled = True


class DashboardBase(ServiceBase, ABC):
    """Base class for dashboard applications providing common functionality."""

    def __init__(
        self,
        *,
        instrument: str = 'dummy',
        dev: bool = False,
        log_level: int = logging.INFO,
        dashboard_name: str,
        port: int = 5007,
    ):
        name = f'{instrument}_{dashboard_name}'
        super().__init__(name=name, log_level=log_level)
        self._instrument = instrument
        self._port = port
        self._dev = dev

        self._exit_stack = ExitStack()
        self._exit_stack.__enter__()

        self._callback = None
        # Separate config stores for workflow and plotter persistent UI state.
        # Note that the cleanup approach was carried over from when this was persisted
        # in Kafka. With a short-lived in-memory store this is not so important, but we
        # may move to a file-based approach (or back to Kafka) in the future.
        self._workflow_config_store = InMemoryConfigStore(
            max_configs=100, cleanup_fraction=0.2
        )
        self._plotter_config_store = InMemoryConfigStore(
            max_configs=100, cleanup_fraction=0.2
        )
        self._setup_config_service()
        self._setup_data_infrastructure(instrument=instrument, dev=dev)
        self._logger.info("%s initialized", self.__class__.__name__)

        # Global unit format.
        Dimension.unit_format = ' [{unit}]'

        # Load the module to register the instrument's workflows.
        self._instrument_module = get_config(instrument)
        self._processor_factory = instrument_registry[self._instrument].workflow_factory
        self._setup_workflow_management()

    @abstractmethod
    def create_sidebar_content(self) -> pn.viewable.Viewable:
        """Override this method to create the sidebar content."""
        pass

    @abstractmethod
    def create_main_content(self) -> pn.viewable.Viewable:
        """Override this method to create the main dashboard content."""
        # Currently unused, should this allow for defining a custom layout where plots
        # should be placed?

    def _setup_config_service(self) -> None:
        """Set up configuration services using backend MessageSource/MessageSink."""
        kafka_downstream_config = load_config(namespace=config_names.kafka_downstream)

        # Dashboard subscribes to RESPONSES to receive backend status updates.
        # Dashboard publishes to COMMANDS to send configuration to backend services.
        responses_topic = stream_kind_to_topic(
            instrument=self._instrument, kind=StreamKind.LIVEDATA_RESPONSES
        )

        consumer = self._exit_stack.enter_context(
            kafka_consumer.make_consumer_from_config(
                topics=[responses_topic],
                config={**kafka_downstream_config, 'auto.offset.reset': 'latest'},
                group='dashboard',
            )
        )

        # Create BackgroundMessageSource for consuming responses
        self._message_source = self._exit_stack.enter_context(
            BackgroundMessageSource(consumer=consumer)
        )

        # Create KafkaSink for publishing commands (using backend abstraction)
        # Note: serializer is not used for LIVEDATA_COMMANDS messages
        from ess.livedata.core.message import Message
        from ess.livedata.handlers.config_handler import ConfigUpdate

        def _config_serializer(value: Message[ConfigUpdate]) -> bytes:
            raise NotImplementedError("ConfigUpdate serialization handled by KafkaSink")

        self._message_sink = KafkaSink[ConfigUpdate](
            kafka_config=kafka_downstream_config,
            instrument=self._instrument,
            logger=self._logger,
            serializer=_config_serializer,
        )

        # Create focused services using MessageSource/MessageSink
        self._workflow_config_service = KafkaWorkflowConfigService(
            source=self._message_source,
            sink=self._message_sink,
            logger=self._logger,
        )

        self._job_command_service = KafkaJobCommandService(
            sink=self._message_sink,
            logger=self._logger,
        )

        self._logger.info("Config services setup complete")

    def _setup_data_infrastructure(self, instrument: str, dev: bool) -> None:
        """Set up data services, forwarder, and orchestrator."""
        # da00 of backend services converted to scipp.DataArray
        ScippDataService = DataService[ResultKey, sc.DataArray]
        self._data_service = ScippDataService()
        self._stream_manager = StreamManager(
            data_service=self._data_service, pipe_factory=streams.Pipe
        )
        self._job_service = JobService(
            data_service=self._data_service, logger=self._logger
        )
        self._job_controller = JobController(
            command_service=self._job_command_service, job_service=self._job_service
        )

        # Create ROI publisher for publishing ROI updates to Kafka
        kafka_upstream_config = load_config(namespace=config_names.kafka_upstream)
        roi_sink = KafkaSink(
            kafka_config=kafka_upstream_config,
            instrument=instrument,
            serializer=serialize_dataarray_to_da00,
            logger=self._logger,
        )
        roi_publisher = ROIPublisher(sink=roi_sink, logger=self._logger)

        self._plotting_controller = PlottingController(
            job_service=self._job_service,
            config_store=self._plotter_config_store,
            stream_manager=self._stream_manager,
            logger=self._logger,
            roi_publisher=roi_publisher,
        )
        self._orchestrator = Orchestrator(
            self._setup_kafka_consumer(instrument=instrument, dev=dev),
            data_service=self._data_service,
            job_service=self._job_service,
        )
        self._logger.info("Data infrastructure setup complete")

    def _setup_kafka_consumer(
        self, instrument: str, dev: bool
    ) -> AdaptingMessageSource:
        """Set up Kafka consumer for data streams."""
        consumer_config = load_config(
            namespace=config_names.reduced_data_consumer, env=''
        )
        kafka_downstream_config = load_config(namespace=config_names.kafka_downstream)
        data_topic = stream_kind_to_topic(
            instrument=self._instrument, kind=StreamKind.LIVEDATA_DATA
        )
        status_topic = stream_kind_to_topic(
            instrument=self._instrument, kind=StreamKind.LIVEDATA_STATUS
        )
        consumer = self._exit_stack.enter_context(
            kafka_consumer.make_consumer_from_config(
                topics=[data_topic, status_topic],
                config={**consumer_config, **kafka_downstream_config},
                group='dashboard',
            )
        )
        source = self._exit_stack.enter_context(
            BackgroundMessageSource(consumer=consumer)
        )
        stream_mapping = get_stream_mapping(instrument=instrument, dev=dev)
        adapter = (
            RoutingAdapterBuilder(stream_mapping=stream_mapping)
            .with_livedata_data_route()
            .with_livedata_status_route()
            .build()
        )
        return AdaptingMessageSource(source=source, adapter=adapter)

    def _setup_workflow_management(self) -> None:
        """Initialize workflow controller and reduction widget."""
        self._correlation_controller = CorrelationHistogramController(
            self._data_service
        )
        self._workflow_controller = WorkflowController(
            service=self._workflow_config_service,
            source_names=sorted(self._processor_factory.source_names),
            workflow_registry=self._processor_factory,
            config_store=self._workflow_config_store,
            data_service=self._data_service,
            correlation_histogram_controller=self._correlation_controller,
        )

        self._reduction_widget = ReductionWidget(controller=self._workflow_controller)

    def _step(self):
        """Step function for periodic updates."""
        # We use hold() to ensure that the UI does not update repeatedly when multiple
        # messages are processed in a single step. This is important to avoid, e.g.,
        # multiple lines in the same plot, or different plots updating in short
        # succession, which is visually distracting.
        # Furthermore, this improves performance by reducing the number of re-renders.
        with pn.io.hold():
            self._orchestrator.update()

    def get_dashboard_title(self) -> str:
        """Get the dashboard title. Override for custom titles."""
        return f"{self._instrument.upper()} — Live Data"

    def get_header_background(self) -> str:
        """Get the header background color. Override for custom colors."""
        return '#2596be'

    def start_periodic_updates(self, period: int = 500) -> None:
        """
        Start periodic updates for the dashboard.

        Parameters
        ----------
        period:
            The period in milliseconds for the periodic update step.
            Default is 500 ms. Even if the backend produces updates, e.g., once per
            second, this default should reduce UI lag somewhat. If there are no new
            messages, the step function should not do anything.
        """
        if self._callback is not None:
            # Callback from previous session, e.g., before reloading the page. As far as
            # I can tell the garbage collector does clean this up eventually, but
            # let's be explicit.
            self._callback.stop()

        def _safe_step():
            try:
                self._step()
                self._workflow_config_service.process_incoming_messages()
            except Exception:
                self._logger.exception("Error in periodic update step.")

        self._callback = pn.state.add_periodic_callback(_safe_step, period=period)
        self._logger.info("Periodic updates started")

    def create_layout(self) -> pn.template.MaterialTemplate:
        """Create the basic dashboard layout."""
        sidebar_content = self.create_sidebar_content()
        main_content = PlotCreationWidget(
            job_service=self._job_service,
            job_controller=self._job_controller,
            plotting_controller=self._plotting_controller,
            workflow_controller=self._workflow_controller,
        ).widget

        template = pn.template.MaterialTemplate(
            title=self.get_dashboard_title(),
            sidebar=sidebar_content,
            main=main_content,
            header_background=self.get_header_background(),
        )
        self.start_periodic_updates()
        return template

    @property
    def server(self):
        """Get the Panel server for WSGI deployment."""
        return pn.serve(
            self.create_layout,
            port=self._port,
            show=False,
            autoreload=False,
            dev=self._dev,
        )

    def _start_impl(self) -> None:
        """Start the dashboard service."""
        self._message_source.start()

    def run_forever(self) -> None:
        """Run the dashboard server."""
        import atexit

        atexit.register(self.stop)
        try:
            pn.serve(
                self.create_layout,
                port=self._port,
                show=False,
                autoreload=True,
                dev=self._dev,
            )
        except KeyboardInterrupt:
            self._logger.info("Keyboard interrupt received, shutting down...")
            self.stop()

    def _stop_impl(self) -> None:
        """Clean shutdown of all components."""
        self._message_source.stop()
        self._exit_stack.__exit__(None, None, None)
