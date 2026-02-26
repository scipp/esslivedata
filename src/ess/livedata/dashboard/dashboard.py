# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Common functionality for implementing dashboards."""

import logging
from abc import ABC, abstractmethod
from contextlib import ExitStack
from pathlib import Path

import panel as pn
from holoviews import Dimension

from ess.livedata import ServiceBase

from .config_store import ConfigStoreManager
from .dashboard_services import DashboardServices
from .kafka_transport import DashboardKafkaTransport
from .session_registry import SessionId
from .session_updater import SessionUpdater
from .transport import NullTransport, Transport

# Global throttling for sliders, etc.
pn.config.throttled = True

_TEMPLATES_DIR = Path(__file__).parent / 'templates'
_LOGIN_TEMPLATE = str(_TEMPLATES_DIR / 'login.html')
_LOGOUT_TEMPLATE = str(_TEMPLATES_DIR / 'logout.html')
_STATIC_DIRS = {'assets': str(_TEMPLATES_DIR)}


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
        basic_auth_password: str | None = None,
        basic_auth_cookie_secret: str | None = None,
    ):
        name = f'{instrument}_{dashboard_name}'
        super().__init__(name=name, log_level=log_level)
        self._instrument = instrument
        self._port = port
        self._dev = dev
        self._basic_auth_password = basic_auth_password
        self._basic_auth_cookie_secret = basic_auth_cookie_secret

        self._exit_stack = ExitStack()
        self._exit_stack.__enter__()

        # Config store manager for file-backed persistent UI state (GUI dashboards)
        config_manager = ConfigStoreManager(instrument=instrument, store_type='file')

        # Setup all dashboard services (includes session registry)
        self._services = DashboardServices(
            instrument=instrument,
            dev=dev,
            exit_stack=self._exit_stack,
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
            return DashboardKafkaTransport(instrument=self._instrument, dev=self._dev)
        elif transport == 'none':
            return NullTransport()
        else:
            raise ValueError(f"Unknown transport type: {transport}")

    @abstractmethod
    def create_sidebar_content(self) -> pn.viewable.Viewable:
        """Override this method to create the sidebar content."""
        pass

    @abstractmethod
    def create_main_content(
        self, session_updater: SessionUpdater
    ) -> pn.viewable.Viewable:
        """
        Override this method to create the main dashboard content.

        Parameters
        ----------
        session_updater:
            The session updater for this browser session. Widgets that need
            to register handlers for periodic updates should receive this
            in their constructor.
        """

    def get_dashboard_title(self) -> str:
        """Get the dashboard title. Override for custom titles."""
        return f"{self._instrument.upper()} — Live Data"

    def get_header_background(self) -> str:
        """Get the header background color. Override for custom colors."""
        return '#2596be'

    def _get_session_id(self) -> SessionId:
        """Get the current session ID from Panel state."""
        session_context = pn.state.curdoc.session_context
        if session_context is not None:
            return SessionId(session_context.id)
        # Fallback for non-session contexts (e.g., testing)
        return SessionId('unknown')

    def _create_session_updater(self) -> SessionUpdater:
        """
        Create a SessionUpdater for the current browser session.

        The updater auto-registers with the session registry.
        """
        return SessionUpdater(
            session_id=self._get_session_id(),
            session_registry=self._services.session_registry,
            notification_queue=self._services.notification_queue,
        )

    def _start_periodic_callback(
        self, session_updater: SessionUpdater, period: int = 500
    ) -> None:
        """
        Start the periodic callback for a session.

        Parameters
        ----------
        session_updater:
            The session updater to drive with the periodic callback.
        period:
            The period in milliseconds for the periodic update step.
        """
        session_id = session_updater.session_id
        session_registry = self._services.session_registry

        def _safe_step():
            try:
                session_updater.periodic_update()
            except Exception:
                self._logger.exception("Error in periodic update step.")

        callback = pn.state.add_periodic_callback(_safe_step, period=period)
        session_updater.set_periodic_callback(callback)

        def _cleanup_session(session_context):
            self._logger.info("Session destroyed: %s", session_id)
            session_registry.unregister(session_id)

        pn.state.on_session_destroyed(_cleanup_session)
        self._logger.info("Periodic updates started for session %s", session_id)

    def _create_logout_header(self) -> list[pn.viewable.Viewable]:
        """Create a logout button for the header when auth is enabled."""
        # ruff: disable[E501]
        logout_link = pn.pane.HTML(
            """<div style="text-align: right; padding-right: 8px;">
            <a href="/logout" style="
                color: white;
                text-decoration: none;
                font-size: 13px;
                font-weight: 500;
                letter-spacing: 0.5px;
                padding: 6px 18px;
                border: 1.5px solid rgba(255, 255, 255, 0.7);
                border-radius: 20px;
                display: inline-block;
                transition: background 0.2s, border-color 0.2s;
                " onmouseover="this.style.background='rgba(255,255,255,0.2)';this.style.borderColor='white'"
                onmouseout="this.style.background='none';this.style.borderColor='rgba(255,255,255,0.7)'"
            >Log out</a></div>""",
            sizing_mode='stretch_width',
        )
        # ruff: enable[E501]
        return [logout_link]

    def create_layout(self) -> pn.template.MaterialTemplate:
        """Create the basic dashboard layout."""
        # Create session updater first so widgets can register handlers
        session_updater = self._create_session_updater()

        sidebar_content = self.create_sidebar_content()
        main_content = self.create_main_content(session_updater)

        # Include heartbeat widget in layout (invisible but required for
        # browser heartbeat JavaScript to run)
        main_with_heartbeat = pn.Column(
            session_updater.heartbeat_widget,
            main_content,
            sizing_mode='stretch_both',
        )

        header = self._create_logout_header() if self._basic_auth_password else []

        template = pn.template.MaterialTemplate(
            title=self.get_dashboard_title(),
            sidebar=sidebar_content,
            main=main_with_heartbeat,
            header_background=self.get_header_background(),
            header=header,
        )
        # Inject CSS for offline mode (replaces Material Icons font with Unicode)
        template.config.raw_css.extend(self.get_raw_css())
        self._start_periodic_callback(session_updater)
        return template

    def get_raw_css(self) -> list[str]:
        """
        Get additional raw CSS to inject into the template.

        Override this method to add custom CSS. The default provides a fix for
        the hamburger menu icon when running in offline mode (without Google Fonts).
        """
        # SVG hamburger icon (3 horizontal lines), white color, as data URI
        hamburger_svg = (
            "data:image/svg+xml,"
            "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
            "fill='white'%3E%3Cpath d='M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z'"
            "/%3E%3C/svg%3E"
        )
        return [
            f"""
            /* Offline mode: Replace Material Icons menu with SVG hamburger */
            button.mdc-top-app-bar__navigation-icon.material-icons {{
                font-size: 0 !important;
                background-image: url("{hamburger_svg}");
                background-repeat: no-repeat;
                background-position: center;
                background-size: 30px 36px;
            }}
            """
        ]

    @property
    def server(self):
        """Get the Panel server for WSGI deployment."""
        return pn.serve(
            self.create_layout,
            port=self._port,
            show=False,
            autoreload=False,
            dev=self._dev,
            basic_auth=self._basic_auth_password,
            cookie_secret=self._basic_auth_cookie_secret,
            login_template=_LOGIN_TEMPLATE,
            logout_template=_LOGOUT_TEMPLATE,
            static_dirs=_STATIC_DIRS,
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
                basic_auth=self._basic_auth_password,
                cookie_secret=self._basic_auth_cookie_secret,
                login_template=_LOGIN_TEMPLATE,
                logout_template=_LOGOUT_TEMPLATE,
                static_dirs=_STATIC_DIRS,
            )
        except KeyboardInterrupt:
            self._logger.info("Keyboard interrupt received, shutting down...")
            self.stop()

    def _stop_impl(self) -> None:
        """Clean shutdown of all components."""
        self._services.stop()
        self._exit_stack.__exit__(None, None, None)
