# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import argparse
from collections.abc import Callable
from uuid import uuid4

import holoviews as hv
import panel as pn

from ess.livedata import Service
from ess.livedata.config.workflow_spec import JobNumber, WorkflowId

from .dashboard import DashboardBase
from .plot_orchestrator import PlotOrchestrator, SubscriptionId
from .widgets.log_producer_widget import LogProducerWidget
from .widgets.plot_grid_tabs import PlotGridTabs

pn.extension('holoviews', 'modal', notifications=True, template='material')
hv.extension('bokeh')


class StubJobOrchestrator:
    """
    Temporary stub for JobOrchestrator to enable testing PlotGridTabs.

    This provides the minimal interface needed by PlotOrchestrator but doesn't
    actually trigger any callbacks. A real implementation would notify when
    workflows are committed/restarted.
    """

    def __init__(self):
        self._subscriptions: dict[SubscriptionId, tuple[WorkflowId, Callable]] = {}

    def subscribe_to_workflow(
        self, workflow_id: WorkflowId, callback: Callable[[JobNumber], None]
    ) -> SubscriptionId:
        """Subscribe to workflow availability notifications."""
        subscription_id = SubscriptionId(uuid4())
        self._subscriptions[subscription_id] = (workflow_id, callback)
        return subscription_id

    def unsubscribe(self, subscription_id: SubscriptionId) -> None:
        """Unsubscribe from workflow availability notifications."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]


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

        # Create log producer widget only in dev mode
        self._dev_widget = None
        if dev:
            self._dev_widget = LogProducerWidget(
                instrument=instrument, logger=self._logger, exit_stack=self._exit_stack
            )

        # Create PlotOrchestrator and PlotGridTabs for testing
        stub_job_orchestrator = StubJobOrchestrator()
        self._plot_orchestrator = PlotOrchestrator(
            plotting_controller=self._services.plotting_controller,
            job_orchestrator=stub_job_orchestrator,
            config_store=self._services.plotter_config_store,
        )
        self._plot_grid_tabs = PlotGridTabs(plot_orchestrator=self._plot_orchestrator)

        self._logger.info("Reduction dashboard initialized")

    def create_sidebar_content(self) -> pn.viewable.Viewable:
        """Create the sidebar content with workflow controls."""
        if self._dev_widget is not None:
            dev_content = [self._dev_widget.panel, pn.layout.Divider()]
        else:
            dev_content = []
        return pn.Column(
            *dev_content,
            pn.pane.Markdown("## Data Reduction"),
            self._reduction_widget.widget,
        )

    def create_main_content(self) -> pn.viewable.Viewable:
        """Create the main content area."""
        return self._plot_grid_tabs.panel

    def create_layout(self) -> pn.template.MaterialTemplate:
        """Create the dashboard layout with PlotGridTabs in a sub-tab."""
        from .widgets.plot_creation_widget import PlotCreationWidget

        sidebar_content = self.create_sidebar_content()

        # Create the original plot creation widget
        plot_creation_widget = PlotCreationWidget(
            job_service=self._services.job_service,
            job_controller=self._services.job_controller,
            plotting_controller=self._services.plotting_controller,
            workflow_controller=self._services.workflow_controller,
        ).widget

        # Create tabs with both old and new interfaces
        main_tabs = pn.Tabs(
            ('Legacy interface', plot_creation_widget),
            (
                'Future interface (incomplete and not functional)',
                self._plot_grid_tabs.panel,
            ),
            sizing_mode='stretch_both',
        )

        template = pn.template.MaterialTemplate(
            title=self.get_dashboard_title(),
            sidebar=sidebar_content,
            main=main_tabs,
            header_background=self.get_header_background(),
        )
        self.start_periodic_updates()
        return template


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
