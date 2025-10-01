# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import argparse

import holoviews as hv
import panel as pn

from ess.livedata import Service

from .correlation_histogram import CorrelationHistogramController
from .dashboard import DashboardBase
from .widgets.correlation_histogram_widget import CorrelationHistogramWidget
from .widgets.log_producer_widget import LogProducerWidget

pn.extension('holoviews', 'modal', template='material')
hv.extension('bokeh')


class ReductionApp(DashboardBase):
    """Reduction dashboard application."""

    def __init__(self, *, instrument: str = 'dummy', dev: bool = False, log_level: int):
        super().__init__(
            instrument=instrument,
            dev=dev,
            log_level=log_level,
            dashboard_name='reduction_dashboard',
            port=5009,  # Default port for reduction dashboard
        )
        self._correlation_controller = CorrelationHistogramController(
            self._data_service
        )
        self._correlation_widget = CorrelationHistogramWidget(
            correlation_histogram_controller=self._correlation_controller
        )

        # Create log producer widget only in dev mode
        self._log_producer_widget = None
        if dev:
            self._log_producer_widget = LogProducerWidget(
                instrument=instrument, logger=self._logger
            )

        self._logger.info("Reduction dashboard initialized")

    def create_sidebar_content(self) -> pn.viewable.Viewable:
        """Create the sidebar content with workflow controls."""
        content = []

        # Add log producer widget at the top if in dev mode
        if self._log_producer_widget is not None:
            content.extend(
                [
                    self._log_producer_widget.panel,
                    pn.layout.Divider(),
                ]
            )

        content.extend(
            [
                pn.pane.Markdown("## Data Reduction"),
                self._reduction_widget.widget,
                pn.pane.Markdown("## Correlation Histograms"),
                self._correlation_widget.panel,
            ]
        )

        return pn.Column(*content)

    def create_main_content(self) -> pn.viewable.Viewable:
        """Create the main content area."""
        return pn.Row()


def get_arg_parser() -> argparse.ArgumentParser:
    return Service.setup_arg_parser(description='ESSlivedata Dashboard')


def main() -> None:
    parser = get_arg_parser()
    app = ReductionApp(**vars(parser.parse_args()))
    app.start(blocking=True)


if __name__ == "__main__":
    main()
