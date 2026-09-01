# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import argparse
import os
import threading
import urllib.request
from urllib.error import URLError
from urllib.request import urlopen

import holoviews as hv
import panel as pn
from holoviews.core.options import Compositor
from holoviews.plotting.bokeh.plot import LayoutPlot
from holoviews.plotting.util import apply_nodata, process_cmap
from panel.io.resources import CDN_DIST
from panel.theme.material import Material

from ess.livedata import Service
from ess.livedata.config.device_contract import DeviceContract
from ess.livedata.logging_config import configure_logging

from .dashboard import (
    DEFAULT_UNUSED_SESSION_LIFETIME,
    DashboardBase,
)
from .dashboard_services import DEFAULT_SESSION_STALE_TIMEOUT
from .session_updater import SessionUpdater
from .theme import DEFAULT_THEME, THEMES
from .widgets.log_producer_widget import LogProducerWidget
from .widgets.plot_grid_tabs import PlotGridTabs
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

# HoloViews defaults its Bokeh renderer to `webgl=True`, which sets
# `output_backend='webgl'` on every figure. That gives each plot a second, WebGL
# canvas: Bokeh then resizes, scissors and clears it before every paint
# (`prepare_webgl`) and blits it over the 2D canvas afterwards (`blit_webgl` ->
# a full-canvas `drawImage`), per plot, per renderer, per frame. Because that
# canvas is shared by the whole page, resizing it to each plot's frame
# reallocates its drawing buffer whenever neighbouring cells differ in size,
# which ours do. The cost is therefore fixed per figure, whatever it draws,
# while the saving scales with content -- and our content stays below where the
# GL path starts to pay: curves are a few hundred points against a break-even
# near 30k, and detector images are mostly 320^2 or smaller. Bokeh does have a
# WebGL path for images, it simply does not earn the round trip at these sizes.
# On a 12-cell grid of mixed cell sizes, a data update costs the browser's main
# thread 82 ms on the 2D canvas against 134 ms in WebGL; with uniform cell sizes
# the reallocation term does not arise and the gap is ~9 ms. Any re-measurement
# has to run on real graphics hardware: without a GPU, Chromium rasterizes WebGL
# glyphs on the CPU, which inflates the gap by roughly an order of magnitude and
# hides the reallocation term behind glyph work. Individual figures can still opt
# in via `backend_opts={'plot.output_backend': 'webgl'}`; see #1218 for where
# that might be worth doing.
hv.renderer('bokeh').webgl = False

# HoloViews registers `apply_nodata` as a data-mode compositor for Image, Raster,
# QuadMesh and ImageStack, implementing the `nodata` plot option: an integer
# sentinel value is rewritten to NaN, which draws transparent. It cannot fire here
# -- no plotter offers the option, and the 2D path converts to float64, which the
# operation passes through -- but the machinery around it runs on every frame of
# every 2D layer regardless, wrapping the element in overlays, matching patterns
# and cloning the result back in, ~1.4 ms per layer per update.
#
# `Compositor.definitions` is global and matched per element type, so there is no
# re-enabling this for one plot: masking values for display is ours to do in
# `Plotter.compute` (see `Plotter._prepare_2d_image_data`), where it costs one
# pass per frame rather than one per frame *per session*.
Compositor.definitions = [
    definition
    for definition in Compositor.definitions
    if definition.operation is not apply_nodata
]

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
        port: int = 5009,
        transport: str = 'kafka',
        config_dir: str | None = None,
        auto_start: bool = False,
        collapsed_sidebar: bool = True,
        fetch_announcements: bool = True,
        basic_auth_password: str | None = None,
        basic_auth_cookie_secret: str | None = None,
        theme: str = DEFAULT_THEME.name,
        session_stale_timeout_seconds: float = DEFAULT_SESSION_STALE_TIMEOUT,
        unused_session_lifetime_seconds: float = DEFAULT_UNUSED_SESSION_LIFETIME,
    ):
        super().__init__(
            instrument=instrument,
            dev=dev,
            log_level=log_level,
            dashboard_name='reduction_dashboard',
            port=port,
            transport=transport,
            config_dir=config_dir,
            auto_start=auto_start,
            collapsed_sidebar=collapsed_sidebar,
            basic_auth_password=basic_auth_password,
            basic_auth_cookie_secret=basic_auth_cookie_secret,
            theme=theme,
            session_stale_timeout_seconds=session_stale_timeout_seconds,
            unused_session_lifetime_seconds=unused_session_lifetime_seconds,
        )
        self._fetch_announcements = fetch_announcements
        # Load (and validate) the NICOS derived-device contract once. Fails loud
        # on an invalid contract, before any session is served.
        self._device_contract = DeviceContract.from_instrument(
            self._services.instrument_config
        )
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

        pane = pn.pane.Markdown("", sizing_mode='stretch_width')

        def fetch_and_update():
            pane.object = read_announcements()

        def refresh():
            threading.Thread(target=fetch_and_update, daemon=True).start()

        refresh()
        pn.state.add_periodic_callback(refresh, period=300_000)  # 5 minutes
        return pane

    def create_sidebar_content(
        self, session_updater: SessionUpdater
    ) -> pn.viewable.Viewable:
        """Create the sidebar content."""
        # Create log producer widget only in dev mode (per-session)
        dev_content = []
        if self._dev:
            dev_widget = LogProducerWidget(instrument=self._instrument)
            # Release the per-session Kafka producer on session teardown. Routed
            # through the session updater (not on_session_destroyed directly) so
            # the heartbeat-based stale reaper also triggers cleanup when Panel's
            # on_session_destroyed fails to fire.
            session_updater.register_cleanup_handler(dev_widget.close)
            dev_content = [dev_widget.panel, pn.layout.Divider()]

        return pn.Column(
            *dev_content,
            self._create_announcements_pane(),
        )

    def create_main_content(
        self, session_updater: SessionUpdater
    ) -> pn.viewable.Viewable:
        """Create the main content area with plot grid tabs."""
        workflow_status_widget = WorkflowStatusListWidget(
            orchestrator=self._services.job_orchestrator,
            job_service=self._services.job_service,
            device_contract=self._device_contract,
        )

        system_status_widget = SystemStatusWidget(
            session_registry=self._services.session_registry,
            service_registry=self._services.service_registry,
            current_session_id=session_updater.session_id,
            notification_queue=self._services.notification_queue,
        )

        plot_grid_tabs = PlotGridTabs(
            plot_orchestrator=self._services.plot_orchestrator,
            # Temporary hack, will likely get this from JobOrchestrator, or make
            # registry more accessible.
            workflow_registry=self._services.workflow_controller._workflow_registry,
            plotting_controller=self._services.plotting_controller,
            workflow_status_widget=workflow_status_widget,
            system_status_widget=system_status_widget,
            plot_data_service=self._services.plot_data_service,
            session_updater=session_updater,
            theme=self._theme,
        )

        # PlotGridTabs registers its own two-tier teardown on the session
        # updater (see PlotGridTabs.__init__), so it runs on both the clean
        # disconnect and the stale-session reaper paths, releasing this
        # session's interest tokens when the browser session ends.

        # Register refresh with visibility gate: skip updates when the
        # Workflows tab (index 0) is not the active tab.
        workflow_status_widget.register_periodic_refresh(
            session_updater,
            is_visible=lambda: plot_grid_tabs.active_tab_index == 0,
        )
        system_status_widget.register_periodic_refresh(
            session_updater,
            is_visible=lambda: plot_grid_tabs.active_tab_index == 1,
        )

        return plot_grid_tabs.panel


def get_arg_parser() -> argparse.ArgumentParser:
    parser = Service.setup_arg_parser(description='ESSlivedata Dashboard')
    parser.add_argument(
        '--port',
        type=int,
        default=5009,
        help='Port for the Bokeh server. Override to run several dashboards at '
        'once or to sidestep a port left in use by a prior instance.',
    )
    parser.add_argument(
        '--transport',
        choices=['kafka', 'none', 'fake'],
        default='kafka',
        help='Transport backend for message handling. "fake" runs an in-process '
        'backend that synthesizes data for started workflows (no Kafka).',
    )
    parser.add_argument(
        '--config-dir',
        default=None,
        help='Base directory for persistent UI config (workflow/plot YAML). '
        'Overrides LIVEDATA_CONFIG_DIR. Point at a seeded fixture (copied to a '
        'scratch dir, since the dashboard writes to it) to launch pre-configured.',
    )
    parser.add_argument(
        '--auto-start',
        action='store_true',
        default=False,
        help='Commit every workflow that has staged config on startup. Requires '
        '--transport fake; combine with --config-dir to launch a fully live '
        'dashboard with no UI interaction (e.g. for screenshots).',
    )
    parser.add_argument(
        '--collapsed-sidebar',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Start with the sidebar drawer collapsed, giving plots the full '
        'window width. It holds announcements and the version label, neither '
        'of which needs to be on screen while watching plots.',
    )
    parser.add_argument(
        '--theme',
        choices=sorted(THEMES),
        default=DEFAULT_THEME.name,
        help='Shell look and feel. "nicos" (the default) adopts the NICOS '
        'client\'s teal chrome and puts the main tab strip in a left rail, for '
        'running the two side by side; "classic" is the previous look, with '
        'the tabs along the top.',
    )
    parser.add_argument(
        '--no-fetch-announcements',
        action='store_false',
        dest='fetch_announcements',
        help='Disable fetching announcements from external URL',
    )
    parser.add_argument(
        '--basic-auth-password',
        default=os.environ.get('LIVEDATA_BASIC_AUTH_PASSWORD'),
        help='Password for basic authentication. '
        'Any username will be accepted. '
        'Can also be set via LIVEDATA_BASIC_AUTH_PASSWORD env var.',
    )
    parser.add_argument(
        '--basic-auth-cookie-secret',
        default=os.environ.get('LIVEDATA_BASIC_AUTH_COOKIE_SECRET'),
        help='Cookie secret for basic authentication sessions. '
        'Can also be set via LIVEDATA_BASIC_AUTH_COOKIE_SECRET env var.',
    )
    parser.add_argument(
        '--session-stale-timeout-seconds',
        type=float,
        default=DEFAULT_SESSION_STALE_TIMEOUT,
        help='Seconds without a heartbeat before a session is dropped and its '
        'per-layer state released. Lower it where browsers vanish often, raise it '
        'where links are slow.',
    )
    parser.add_argument(
        '--unused-session-lifetime-seconds',
        type=float,
        default=DEFAULT_UNUSED_SESSION_LIFETIME,
        help='Seconds Bokeh keeps a session with no connections left before '
        'dropping its document. Also the poll interval, so a closed session is '
        'released somewhere between one and two times this.',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        default=False,
        help='Construct the dashboard for the selected instrument and exit '
        'without starting the Bokeh server. Combine with --transport none to '
        'verify all required dependencies are importable without contacting '
        'Kafka.',
    )
    return parser


def _warm_up_colormaps() -> None:
    """Register colorcet's colormaps with matplotlib before serving.

    Resolving a colormap imports colorcet, which registers hundreds of colormaps with
    matplotlib. Left lazy, that lands on a session's IOLoop during its first plot
    render and blocks every request behind it. Pay it before the server starts, where
    blocking is free. Costs ~70 ms; with matplotlib < 3.11.2 it is ~2.8 s, since each
    registration pays difflib "did you mean" generation (matplotlib#32172). The warmup
    is worth keeping independent of that fix -- remove it only if first-render
    profiling says otherwise.

    Deliberately not called at import: it is the single largest import cost in the test
    suite, where nothing renders.
    """
    process_cmap('viridis')


def main() -> None:
    import logging

    _warm_up_colormaps()

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

    check = args.pop('check')
    app = ReductionApp(log_level=log_level, **args)
    if check:
        return
    app.start(blocking=True)


if __name__ == "__main__":
    main()
