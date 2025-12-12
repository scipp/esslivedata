# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Dashboard service composition and setup."""

import logging
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

import scipp as sc

from ess.livedata.config import instrument_registry
from ess.livedata.config.grid_template import load_raw_grid_templates
from ess.livedata.config.instruments import get_config
from ess.livedata.config.workflow_spec import ResultKey

from .command_service import CommandService
from .config_store import ConfigStoreManager
from .correlation_histogram import (
    CorrelationHistogram1dTemplate,
    CorrelationHistogram2dTemplate,
    CorrelationHistogramController,
)
from .data_service import DataService
from .job_controller import JobController
from .job_orchestrator import JobOrchestrator
from .job_service import JobService
from .orchestrator import Orchestrator
from .plot_orchestrator import PlotOrchestrator
from .plotting_controller import PlottingController
from .roi_publisher import ROIPublisher
from .stream_manager import StreamManager
from .transport import Transport
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
    transport:
        Transport instance for message sources and sinks.
        For Kafka: DashboardKafkaTransport(instrument, dev, logger)
        For testing: NullTransport()
    config_manager:
        ConfigStoreManager instance for creating config stores.
        Controls whether stores are file-backed or in-memory.
        For GUI: ConfigStoreManager(instrument, store_type='file')
        For tests: ConfigStoreManager(instrument, store_type='memory')
    """

    def __init__(
        self,
        *,
        instrument: str,
        dev: bool,
        exit_stack: ExitStack,
        logger: logging.Logger,
        pipe_factory: Callable[[Any], Any],
        transport: Transport,
        config_manager: ConfigStoreManager,
    ):
        self._instrument = instrument
        self._dev = dev
        self._exit_stack = exit_stack
        self._logger = logger
        self._pipe_factory = pipe_factory
        self._transport = transport
        self._config_manager = config_manager

        # Config stores for workflow and plotter persistent UI state
        self.workflow_config_store = config_manager.get_store('workflow_configs')
        self.plotter_config_store = config_manager.get_store('plotter_configs')

        # Setup all services
        self._setup_data_infrastructure()
        self._setup_workflow_management()
        self._setup_plot_orchestrator()

        self._logger.info("DashboardServices initialized for %s", instrument)

    def start(self) -> None:
        """Start background tasks (e.g., message polling)."""
        self._transport.start()

    def stop(self) -> None:
        """Stop background tasks (e.g., message polling)."""
        self._transport.stop()

    def _setup_data_infrastructure(self) -> None:
        """Set up data services, forwarder, and orchestrator."""
        # Set up transport and get resources
        transport_resources = self._exit_stack.enter_context(self._transport)

        self.command_service = CommandService(
            sink=transport_resources.command_sink, logger=self._logger
        )
        self.workflow_config_service = WorkflowConfigService(logger=self._logger)

        # da00 of backend services converted to scipp.DataArray
        ScippDataService = DataService[ResultKey, sc.DataArray]
        self.data_service = ScippDataService()
        self.stream_manager = StreamManager(
            data_service=self.data_service, pipe_factory=self._pipe_factory
        )
        self.job_service = JobService(logger=self._logger)
        self.job_controller = JobController(
            command_service=self.command_service, job_service=self.job_service
        )

        # Create ROI publisher for publishing ROI updates to Kafka
        roi_publisher = ROIPublisher(
            sink=transport_resources.roi_sink, logger=self._logger
        )

        self.plotting_controller = PlottingController(
            job_service=self.job_service,
            stream_manager=self.stream_manager,
            logger=self._logger,
            roi_publisher=roi_publisher,
        )
        self.orchestrator = Orchestrator(
            transport_resources.message_source,
            data_service=self.data_service,
            job_service=self.job_service,
            workflow_config_service=self.workflow_config_service,
        )
        self._logger.info("Data infrastructure setup complete")

    def _setup_plot_orchestrator(self) -> None:
        """Set up PlotOrchestrator for managing plot grids."""
        # Must be called after _setup_workflow_management (needs job_orchestrator)
        plot_config_store = self._config_manager.get_store('plot_configs')
        raw_templates = load_raw_grid_templates(self._instrument)
        self.plot_orchestrator = PlotOrchestrator(
            plotting_controller=self.plotting_controller,
            job_orchestrator=self.job_orchestrator,
            data_service=self.data_service,
            config_store=plot_config_store,
            raw_templates=raw_templates,
        )
        self._logger.info("PlotOrchestrator setup complete")

    def _setup_workflow_management(self) -> None:
        """Initialize workflow controller and related components."""
        # Load the module to register the instrument's workflows
        self.instrument_module = get_config(self._instrument)
        self.processor_factory = instrument_registry[self._instrument].workflow_factory

        self.correlation_controller = CorrelationHistogramController(self.data_service)

        # Create workflow templates for dynamic workflow creation
        # Templates use the correlation controller for data access and execution
        templates = [
            CorrelationHistogram1dTemplate(controller=self.correlation_controller),
            CorrelationHistogram2dTemplate(controller=self.correlation_controller),
        ]

        self.job_orchestrator = JobOrchestrator(
            command_service=self.command_service,
            workflow_config_service=self.workflow_config_service,
            workflow_registry=self.processor_factory,
            config_store=self.workflow_config_store,
            templates=templates,
        )
        self.workflow_controller = WorkflowController(
            job_orchestrator=self.job_orchestrator,
            data_service=self.data_service,
        )

        # Register hard-coded correlation histogram instances for testing
        self._register_test_correlation_histograms()

    def _register_test_correlation_histograms(self) -> None:
        """Register hard-coded correlation histogram templates for testing.

        This registers correlation histogram workflow instances for Bifrost
        timeseries to allow testing the template system before a UI for
        template instantiation exists.
        """
        if self._instrument != 'bifrost':
            return

        # Register 1D correlation histograms for each Bifrost timeseries
        bifrost_timeseries = [
            'sample_temperature',
            'detector_rotation',
            'sample_rotation',
        ]

        for axis_name in bifrost_timeseries:
            workflow_id = self.job_orchestrator.register_from_template(
                'correlation_histogram_1d', {'x_param': axis_name}
            )
            if workflow_id is not None:
                self._logger.info(
                    'Registered test correlation histogram: %s', workflow_id
                )
