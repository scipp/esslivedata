# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import argparse

import holoviews as hv
import panel as pn
from holoviews.plotting.bokeh.plot import LayoutPlot

from ess.livedata import Service

from .dashboard import DashboardBase
from .widgets.job_status_widget import JobStatusListWidget
from .widgets.log_producer_widget import LogProducerWidget
from .widgets.plot_grid_tabs import PlotGridTabs
from .widgets.reduction_widget import ReductionWidget

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
    ):
        super().__init__(
            instrument=instrument,
            dev=dev,
            log_level=log_level,
            dashboard_name='reduction_dashboard',
            port=5009,  # Default port for reduction dashboard
            transport=transport,
        )
        self._logger.info("Reduction dashboard initialized")

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
                logger=self._logger,
                exit_stack=self._exit_stack,
            )
            dev_content = [dev_widget.panel, pn.layout.Divider()]

        return pn.Column(
            *dev_content,
            pn.pane.Markdown("## Data Reduction"),
            reduction_widget.widget,
        )

    def create_main_content(self) -> pn.viewable.Viewable:
        """Create the main content area with plot grid tabs."""
        # Create job status widget
        job_status_widget = JobStatusListWidget(
            job_service=self._services.job_service,
            job_controller=self._services.job_controller,
        )

        # Create UI widget connected to shared orchestrator
        plot_grid_tabs = PlotGridTabs(
            plot_orchestrator=self._services.plot_orchestrator,
            # Temporary hack, will likely get this from JobOrchestrator, or make
            # registry more accessible.
            workflow_registry=self._services.workflow_controller._workflow_registry,
            plotting_controller=self._services.plotting_controller,
            job_status_widget=job_status_widget,
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
    return parser


def main() -> None:
    parser = get_arg_parser()
    app = ReductionApp(**vars(parser.parse_args()))
    app.start(blocking=True)


if __name__ == "__main__":
    main()
