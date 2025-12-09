# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Common functionality for implementing dashboards."""

import logging
import os
import time
from abc import ABC, abstractmethod
from contextlib import ExitStack

import panel as pn
from holoviews import Dimension, streams

from ess.livedata import ServiceBase

from .config_store import ConfigStoreManager
from .dashboard_services import DashboardServices
from .kafka_transport import DashboardKafkaTransport
from .transport import NullTransport, Transport

# Global throttling for sliders, etc.
pn.config.throttled = True

# Profiling support: set LIVEDATA_PROFILE=1 to enable pyinstrument profiling
_PROFILING_ENABLED = os.environ.get('LIVEDATA_PROFILE', '').lower() in (
    '1',
    'true',
    'yes',
)
_profiler = None
if _PROFILING_ENABLED:
    try:
        from pyinstrument import Profiler

        _profiler = Profiler(async_mode='disabled')
        print("Profiling enabled")
    except ImportError:
        logging.getLogger(__name__).warning(
            "LIVEDATA_PROFILE is set but pyinstrument is not installed. "
            "Install with: pip install pyinstrument"
        )


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
        transport: str = 'kafka',
    ):
        name = f'{instrument}_{dashboard_name}'
        super().__init__(name=name, log_level=log_level)
        self._instrument = instrument
        self._port = port
        self._dev = dev

        self._exit_stack = ExitStack()
        self._exit_stack.__enter__()

        self._callback = None

        # Config store manager for file-backed persistent UI state (GUI dashboards)
        config_manager = ConfigStoreManager(instrument=instrument, store_type='file')

        # Setup all dashboard services
        self._services = DashboardServices(
            instrument=instrument,
            dev=dev,
            exit_stack=self._exit_stack,
            logger=self._logger,
            pipe_factory=streams.Pipe,
            transport=self._create_transport(transport),
            config_manager=config_manager,
        )

        self._logger.info("%s initialized", self.__class__.__name__)

        # Global unit format
        Dimension.unit_format = ' [{unit}]'

    def _create_transport(self, transport: str) -> Transport:
        """
        Create transport instance based on transport type.

        Parameters
        ----------
        transport:
            Transport type ('kafka' or 'none')

        Returns
        -------
        :
            Transport instance
        """
        if transport == 'kafka':
            return DashboardKafkaTransport(
                instrument=self._instrument, dev=self._dev, logger=self._logger
            )
        elif transport == 'none':
            return NullTransport()
        else:
            raise ValueError(f"Unknown transport type: {transport}")

    @abstractmethod
    def create_sidebar_content(self) -> pn.viewable.Viewable:
        """Override this method to create the sidebar content."""
        pass

    @abstractmethod
    def create_main_content(self) -> pn.viewable.Viewable:
        """Override this method to create the main dashboard content."""

    def _step(self):
        """Step function for periodic updates."""
        # We use hold() to ensure that the UI does not update repeatedly when multiple
        # messages are processed in a single step. This is important to avoid, e.g.,
        # multiple lines in the same plot, or different plots updating in short
        # succession, which is visually distracting.
        # Furthermore, this improves performance by reducing the number of re-renders.
        if _profiler is not None:
            _profiler.start()

        t0 = time.perf_counter()
        with pn.io.hold():
            t1 = time.perf_counter()
            self._services.orchestrator.update()
            t2 = time.perf_counter()
        t3 = time.perf_counter()

        if _profiler is not None:
            _profiler.stop()

        # Only log if there was actual work (avoid noise from idle cycles)
        process_ms = (t2 - t1) * 1000
        hold_exit_ms = (t3 - t2) * 1000
        total_ms = (t3 - t0) * 1000
        if total_ms > 1.0:
            self._logger.info(
                "Step timing: process=%.1fms, hold_exit=%.1fms, total=%.1fms",
                process_ms,
                hold_exit_ms,
                total_ms,
            )

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
            except Exception:
                self._logger.exception("Error in periodic update step.")

        self._callback = pn.state.add_periodic_callback(_safe_step, period=period)
        self._logger.info("Periodic updates started")

    def create_layout(self) -> pn.template.MaterialTemplate:
        """Create the basic dashboard layout."""
        sidebar_content = self.create_sidebar_content()
        main_content = self.create_main_content()

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
        self._services.start()

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
        self._write_profile_report()
        self._services.stop()
        self._exit_stack.__exit__(None, None, None)

    def _write_profile_report(self) -> None:
        """Write pyinstrument profile report to file on shutdown."""
        if _profiler is None:
            return
        html_path = "profile_report.html"
        with open(html_path, 'w') as f:
            f.write(_profiler.output_html())
        self._logger.info("Profile report written to %s", html_path)
