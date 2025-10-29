# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Common functionality for implementing dashboards."""

import logging
import threading
from abc import ABC, abstractmethod
from contextlib import ExitStack

import panel as pn
import scipp as sc
from holoviews import Dimension, streams

from ess.livedata import ServiceBase
from ess.livedata.config import instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.schema_registry import get_schema_registry
from ess.livedata.config.workflow_spec import ResultKey
from ess.livedata.kafka import consumer as kafka_consumer

from .config_service import ConfigService
from .controller_factory import ControllerFactory
from .correlation_histogram import CorrelationHistogramController
from .data_service import DataService
from .job_controller import JobController
from .job_service import JobService
from .message_bridge import BackgroundMessageBridge
from .orchestrator import Orchestrator
from .plotting_controller import PlottingController
from .roi_publisher import ROIPublisher
from .schema_validator import PydanticSchemaValidator
from .stream_manager import StreamManager
from .transport_factory import (
    TransportType,
    create_config_transport,
    create_data_source,
    create_roi_sink,
)
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
        transport: TransportType = 'kafka',
        http_backend_url: str | None = None,
        http_port: int = 8300,
    ):
        name = f'{instrument}_{dashboard_name}'
        super().__init__(name=name, log_level=log_level)
        self._instrument = instrument
        self._port = port
        self._dev = dev
        self._transport = transport
        self._http_backend_url = http_backend_url
        self._http_port = http_port

        self._exit_stack = ExitStack()
        self._exit_stack.__enter__()

        self._callback = None
        self._http_sink = (
            None  # Track HTTP multi-endpoint sink for lifecycle management
        )

        # Create HTTP sink once if using HTTP transport
        if self._transport == 'http':
            from .transport_factory import create_dashboard_sink

            self._http_sink = create_dashboard_sink(
                instrument=instrument, port=http_port, logger=self._logger
            )

        self._setup_config_service()
        self._setup_data_infrastructure(instrument=instrument, dev=dev)
        self._logger.info(
            "%s initialized with %s transport", self.__class__.__name__, transport
        )

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
        """Set up configuration service with transport bridge."""
        # Create consumer if using Kafka transport
        consumer = None
        if self._transport == 'kafka':
            _, consumer = self._exit_stack.enter_context(
                kafka_consumer.make_control_consumer(
                    instrument=self._instrument,
                    extra_config={'auto.offset.reset': 'earliest'},
                )
            )

        # Create transport using factory
        transport = create_config_transport(
            transport_type=self._transport,
            instrument=self._instrument,
            logger=self._logger,
            http_backend_url=self._http_backend_url,
            http_sink=self._http_sink,
            consumer=consumer,
        )

        self._message_bridge = BackgroundMessageBridge(
            transport=transport, logger=self._logger
        )
        self._config_service = ConfigService(
            message_bridge=self._message_bridge,
            schema_validator=PydanticSchemaValidator(
                schema_registry=get_schema_registry()
            ),
        )
        self._controller_factory = ControllerFactory(
            config_service=self._config_service,
            schema_registry=get_schema_registry(),
        )

        self._message_bridge_thread = threading.Thread(
            target=self._message_bridge.start, daemon=True
        )
        self._logger.info(
            "Config service setup complete with %s transport", self._transport
        )

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
            config_service=self._config_service, job_service=self._job_service
        )

        # Create ROI publisher using transport factory
        roi_sink = create_roi_sink(
            transport_type=self._transport,
            instrument=instrument,
            logger=self._logger,
            http_sink=self._http_sink,
        )

        roi_publisher = ROIPublisher(sink=roi_sink, logger=self._logger)

        self._plotting_controller = PlottingController(
            job_service=self._job_service,
            config_service=self._config_service,
            stream_manager=self._stream_manager,
            logger=self._logger,
            roi_publisher=roi_publisher,
        )

        # Create data source using transport factory
        data_source = create_data_source(
            transport_type=self._transport,
            instrument=instrument,
            dev=dev,
            http_backend_url=self._http_backend_url,
            exit_stack=self._exit_stack,
        )

        self._orchestrator = Orchestrator(
            data_source,
            data_service=self._data_service,
            job_service=self._job_service,
        )
        self._logger.info(
            "Data infrastructure setup complete with %s transport", self._transport
        )

    def _setup_workflow_management(self) -> None:
        """Initialize workflow controller and reduction widget."""
        self._correlation_controller = CorrelationHistogramController(
            self._data_service
        )
        self._workflow_controller = WorkflowController.from_config_service(
            config_service=self._config_service,
            source_names=sorted(self._processor_factory.source_names),
            workflow_registry=self._processor_factory,
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
                self._config_service.process_incoming_messages()
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
        # Start HTTP multi-endpoint sink if using HTTP transport
        if self._http_sink is not None:
            self._http_sink.start()
            self._logger.info(
                "HTTP multi-endpoint sink started on port %d", self._http_port
            )

        self._message_bridge_thread.start()

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
        # Stop HTTP multi-endpoint sink if using HTTP transport
        if self._http_sink is not None:
            self._http_sink.stop()
            self._logger.info("HTTP multi-endpoint sink stopped")

        self._message_bridge.stop()
        self._message_bridge_thread.join()
        self._exit_stack.__exit__(None, None, None)
