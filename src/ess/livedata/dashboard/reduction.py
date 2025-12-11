# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import argparse

import holoviews as hv
import panel as pn
from holoviews.plotting.bokeh.plot import LayoutPlot

from ess.livedata import Service

from .dashboard import DashboardBase
from .widgets.configuration_widget import ConfigurationModal
from .widgets.job_status_widget import JobStatusListWidget
from .widgets.log_producer_widget import LogProducerWidget
from .widgets.plot_grid_tabs import PlotGridTabs
from .widgets.reduction_widget import ReductionWidget
from .widgets.workflow_status_widget import WorkflowStatusListWidget

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
        # Modal container for workflow configuration
        self._workflow_config_modal_container = pn.Column()
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

        # Create on_configure callback for workflow status widget
        # Note: workflow_status_widget is referenced in closure but defined below
        def on_configure_workflow(workflow_id, source_names):
            """Handle gear button click from workflow status widget."""
            try:
                # Get workflow adapter with pre-configured sources
                adapter = self._services.workflow_controller.create_workflow_adapter(
                    workflow_id
                )
                # Pre-select the specified sources
                if hasattr(adapter, 'set_selected_sources'):
                    adapter.set_selected_sources(source_names)

                # Create and show modal
                def on_success():
                    self._cleanup_workflow_config_modal()
                    # Widget rebuilds automatically via orchestrator subscription

                modal = ConfigurationModal(
                    config=adapter,
                    start_button_text="Apply",
                    success_callback=on_success,
                )
                self._workflow_config_modal_container.clear()
                self._workflow_config_modal_container.append(modal.modal)
                modal.show()
            except Exception as e:
                self._logger.exception("Failed to create workflow configuration modal")
                pn.state.notifications.error(f"Configuration error: {e}")

        # Create workflow status widget with callback
        workflow_status_widget = WorkflowStatusListWidget(
            orchestrator=self._services.job_orchestrator,
            job_service=self._services.job_service,
            on_configure=on_configure_workflow,
        )

        # Create UI widget connected to shared orchestrator
        plot_grid_tabs = PlotGridTabs(
            plot_orchestrator=self._services.plot_orchestrator,
            # Temporary hack, will likely get this from JobOrchestrator, or make
            # registry more accessible.
            workflow_registry=self._services.workflow_controller._workflow_registry,
            plotting_controller=self._services.plotting_controller,
            job_status_widget=job_status_widget,
            workflow_status_widget=workflow_status_widget,
        )

        # Return panel with modal container at top level
        return pn.Column(
            plot_grid_tabs.panel,
            self._workflow_config_modal_container,
            sizing_mode='stretch_both',
        )

    def _cleanup_workflow_config_modal(self) -> None:
        """Clean up workflow configuration modal."""
        self._workflow_config_modal_container.clear()


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
