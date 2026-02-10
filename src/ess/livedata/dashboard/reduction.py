# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import argparse
import urllib.request
from urllib.error import URLError
from urllib.request import urlopen

import holoviews as hv
import panel as pn
from holoviews.plotting.bokeh.plot import LayoutPlot
from panel.io.resources import CDN_DIST
from panel.theme.material import Material

from ess.livedata import Service
from ess.livedata.logging_config import configure_logging

from .dashboard import DashboardBase
from .session_updater import SessionUpdater
from .widgets.job_status_widget import JobStatusListWidget
from .widgets.log_producer_widget import LogProducerWidget
from .widgets.plot_grid_tabs import PlotGridTabs
from .widgets.reduction_widget import ReductionWidget
from .widgets.system_status_widget import SystemStatusWidget
from .widgets.workflow_status_widget import WorkflowStatusListWidget

# Remove external Google Fonts dependencies from MaterialTemplate.
# This allows the dashboard to work in firewalled environments without internet access.
# Text will fall back to system fonts (sans-serif).
# The material-components-web CSS/JS are bundled with Panel and served locally
# when BOKEH_RESOURCES=server is set.
Material._resources = {
    'css': {
        'material': (
            f"{CDN_DIST}bundled"
            "/material-components-web@7.0.0/dist/material-components-web.min.css"
        )
    },
    'font': {},  # Removed: Google Fonts (Roboto, Material Icons)
    'js': {
        'material': (
            f"{CDN_DIST}bundled"
            "/material-components-web@7.0.0/dist/material-components-web.min.js"
        )
    },
}

ANNOUNCEMENTS_URL = (
    'https://public.esss.dk/groups/scipp/esslivedata/_static/announcements.md'
)

pn.extension('holoviews', 'modal', notifications=True, template='material')
hv.extension('bokeh')

# Remove Bokeh logo from Layout toolbars by patching LayoutPlot.initialize_plot

_original_layout_initialize = LayoutPlot.initialize_plot


def _patched_layout_initialize(self, *args, **kwargs):
    result = _original_layout_initialize(self, *args, **kwargs)
    if hasattr(self, 'state') and hasattr(self.state, 'toolbar'):
        self.state.toolbar.logo = None
    return result


LayoutPlot.initialize_plot = _patched_layout_initialize


class ReductionApp(DashboardBase):
    """Reduction dashboard application."""

    def __init__(
        self,
        *,
        instrument: str = 'dummy',
        dev: bool = False,
        log_level: int,
        transport: str = 'kafka',
        fetch_announcements: bool = True,
    ):
        super().__init__(
            instrument=instrument,
            dev=dev,
            log_level=log_level,
            dashboard_name='reduction_dashboard',
            port=5009,  # Default port for reduction dashboard
            transport=transport,
        )
        self._fetch_announcements = fetch_announcements
        self._logger.info("Reduction dashboard initialized")

    def _create_announcements_pane(self) -> pn.pane.Markdown:
        """Create a Markdown pane that periodically reloads from URL."""
        if not self._fetch_announcements:
            return pn.pane.Markdown(
                "*Announcements disabled.*", sizing_mode='stretch_width'
            )

        def read_announcements() -> str:
            try:
                req = urllib.request.Request(ANNOUNCEMENTS_URL)  # noqa: S310
                with urlopen(req, timeout=10) as response:  # noqa: S310
                    return response.read().decode('utf-8')
            except (URLError, TimeoutError) as e:
                self._logger.warning("Failed to fetch announcements: %s", e)
                return "*Unable to load announcements.*"

        pane = pn.pane.Markdown(read_announcements(), sizing_mode='stretch_width')

        def refresh():
            pane.object = read_announcements()

        pn.state.add_periodic_callback(refresh, period=300_000)  # 5 minutes
        return pane

    def create_sidebar_content(self) -> pn.viewable.Viewable:
        """Create the sidebar content with workflow controls."""
        # Create reduction widget (per-session)
        reduction_widget = ReductionWidget(
            controller=self._services.workflow_controller
        )

        # Create log producer widget only in dev mode (per-session)
        dev_content = []
        if self._dev:
            dev_widget = LogProducerWidget(
                instrument=self._instrument,
                exit_stack=self._exit_stack,
            )
            dev_content = [dev_widget.panel, pn.layout.Divider()]

        return pn.Column(
            *dev_content,
            self._create_announcements_pane(),
            pn.layout.Divider(),
            pn.pane.Markdown("## Data Reduction"),
            pn.pane.Markdown(
                "**Starting workflows here is legacy, prefer using the *Workflows* "
                "tab.**"
            ),
            reduction_widget.widget,
        )

    def create_main_content(
        self, session_updater: SessionUpdater
    ) -> pn.viewable.Viewable:
        """Create the main content area with plot grid tabs."""
        job_status_widget = JobStatusListWidget(
            job_service=self._services.job_service,
            job_controller=self._services.job_controller,
        )
        job_status_widget.register_periodic_refresh(session_updater)

        workflow_status_widget = WorkflowStatusListWidget(
            orchestrator=self._services.job_orchestrator,
            job_service=self._services.job_service,
        )
        workflow_status_widget.register_periodic_refresh(session_updater)

        system_status_widget = SystemStatusWidget(
            session_registry=self._services.session_registry,
            service_registry=self._services.service_registry,
            current_session_id=session_updater.session_id,
        )
        system_status_widget.register_periodic_refresh(session_updater)

        plot_grid_tabs = PlotGridTabs(
            plot_orchestrator=self._services.plot_orchestrator,
            # Temporary hack, will likely get this from JobOrchestrator, or make
            # registry more accessible.
            workflow_registry=self._services.workflow_controller._workflow_registry,
            plotting_controller=self._services.plotting_controller,
            job_status_widget=job_status_widget,
            workflow_status_widget=workflow_status_widget,
            system_status_widget=system_status_widget,
            plot_data_service=self._services.plot_data_service,
            session_updater=session_updater,
        )

        return plot_grid_tabs.panel


def get_arg_parser() -> argparse.ArgumentParser:
    parser = Service.setup_arg_parser(description='ESSlivedata Dashboard')
    parser.add_argument(
        '--transport',
        choices=['kafka', 'none'],
        default='kafka',
        help='Transport backend for message handling',
    )
    parser.add_argument(
        '--no-fetch-announcements',
        action='store_false',
        dest='fetch_announcements',
        help='Disable fetching announcements from external URL',
    )
    return parser


def main() -> None:
    import logging

    parser = get_arg_parser()
    args = vars(parser.parse_args())

    # Configure logging with parsed arguments
    log_level = getattr(logging, args.pop('log_level'))
    log_json_file = args.pop('log_json_file')
    no_stdout_log = args.pop('no_stdout_log')
    configure_logging(
        level=log_level,
        json_file=log_json_file,
        disable_stdout=no_stdout_log,
    )

    app = ReductionApp(log_level=log_level, **args)
    app.start(blocking=True)


if __name__ == "__main__":
    main()
