# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
CellWidget - Per-session view of a single plot cell.

A cell is one position in a plot grid. Its widget bundles a titlebar (cell
title, rename, per-cell add-layer, toolbar toggle, freshness/lag pill), a
stack of per-layer toolbars, and a content area showing either the composed
plot or a status placeholder.

All per-session, per-cell state lives here (autoscale controller, freshness
pane, per-layer time panes, toolbar-visibility toggle), so ``PlotGridTabs``
holds a single ``dict[CellId, CellWidget]`` and delegates updates and
disposal to the widget.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import holoviews as hv
import panel as pn
import structlog

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec
from ess.livedata.core.timestamp import Timestamp

from ..cell_autoscale import CellAutoscaleController, build_controller_from_layers
from ..data_roles import PRIMARY
from ..format_utils import extract_error_summary
from ..frame_aspect import make_frame_aspect_hook_from_config
from ..plot_data_service import LayerState, LayerStateMachine, PlotDataService
from ..plot_orchestrator import (
    CellId,
    DataSourceConfig,
    LayerId,
    PlotCell,
    PlotConfig,
    PlotOrchestrator,
)
from ..plot_params import PlotAspectType, StretchMode
from ..plots import TimeBounds, format_time_info, merge_time_bounds
from ..plotting_controller import PlottingController
from ..save_filename import build_save_filename_from_cell, make_save_filename_hook
from ..session_layer import SessionLayer
from .plot_grid import GridCellStyles
from .plot_widgets import (
    create_cell_titlebar,
    create_layer_toolbar,
    derive_cell_title,
    get_plot_cell_display_info,
    get_workflow_display_info,
)
from .styles import Colors, FreshnessPill, StatusColors

logger = structlog.get_logger(__name__)


def _get_sizing_mode(config: PlotConfig) -> str:
    """Extract Panel sizing_mode from plot configuration.

    Parameters
    ----------
    config:
        Plot configuration containing plotter params.

    Returns
    -------
    :
        Panel sizing_mode string ('stretch_both', 'stretch_width', or 'stretch_height').
    """
    params = config.params
    if hasattr(params, 'plot_aspect'):
        aspect = params.plot_aspect
        if aspect.aspect_type == PlotAspectType.free:
            return 'stretch_both'
        if aspect.stretch_mode == StretchMode.width:
            return 'stretch_width'
        return 'stretch_height'
    return 'stretch_both'


def _freshness_band(lag_seconds: float) -> tuple[str, str, str]:
    """Return the ``(background, text, dot)`` pill colors for a lag in seconds."""
    if lag_seconds < FreshnessPill.FRESH_MAX_SECONDS:
        return FreshnessPill.FRESH
    if lag_seconds < FreshnessPill.STALE_MAX_SECONDS:
        return FreshnessPill.STALE
    return FreshnessPill.OLD


def _format_lag_short(lag_seconds: float) -> str:
    """Compact lag label, e.g. "2.3s", "41s", "3m"."""
    if lag_seconds < 10:
        return f'{lag_seconds:.1f}s'
    if lag_seconds < 60:
        return f'{lag_seconds:.0f}s'
    return f'{lag_seconds / 60:.0f}m'


def format_freshness_html(lag_seconds: float | None, tooltip: str = '') -> str:
    """Render the titlebar freshness/lag pill, color-banded by staleness.

    Parameters
    ----------
    lag_seconds:
        Data lag in seconds, or None to render nothing (no timing data).
    tooltip:
        Full time-range text shown on hover (the pill itself shows only the
        compact lag).
    """
    if lag_seconds is None:
        return ''
    background, text_color, dot_color = _freshness_band(lag_seconds)
    pill_style = (
        f'display:inline-flex;align-items:center;gap:5px;height:20px;'
        f'padding:0 8px;border-radius:10px;font-size:11px;'
        f'font-variant-numeric:tabular-nums;white-space:nowrap;'
        f'background:{background};color:{text_color};'
    )
    dot_style = (
        f'width:7px;height:7px;border-radius:50%;flex:none;background:{dot_color};'
    )
    if tooltip:
        from html import escape

        title_attr = f' title="{escape(tooltip)}"'
    else:
        title_attr = ''
    return (
        f'<span style="{pill_style}"{title_attr}>'
        f'<span style="{dot_style}"></span>{_format_lag_short(lag_seconds)}</span>'
    )


def create_freshness_pane() -> pn.pane.HTML:
    """Create the titlebar freshness/lag pane, updated in place via ``.object``.

    Content sizes to the pill; ``align='center'`` (``align-self:center``)
    vertically centers it against the title and buttons in the row.
    """
    return pn.pane.HTML(
        '',
        align='center',
        margin=(0, 6),
        styles={'flex': '0 0 auto'},
    )


def format_layer_time_html(text: str) -> str:
    """Render a per-layer time-range/lag label (muted, tabular nums)."""
    if not text:
        return ''
    style = (
        f'font-size:11px;color:{Colors.TEXT_MUTED};white-space:nowrap;'
        'font-variant-numeric:tabular-nums;'
    )
    return f'<span style="{style}">{text}</span>'


def create_layer_time_pane() -> pn.pane.HTML:
    """Create a per-layer time-range pane, updated in place via ``.object``.

    Placed on its own line below the layer title/buttons row; indented and
    pulled up close to the title so it visually belongs to it.
    """
    return pn.pane.HTML(
        '',
        align='start',
        margin=(-4, 6, 0, 14),
        styles={'flex': '0 0 auto'},
    )


@dataclass(frozen=True)
class CellDeps:
    """Shared, session-stable dependencies for building cell widgets.

    Bundled once by ``PlotGridTabs`` so each ``CellWidget`` stays a thin view.
    ``session_layers`` is the session's shared layer render-state registry
    (owned by the poll loop, read here when composing plots); the callbacks
    route modal interactions back to the owning ``PlotGridTabs`` (which holds
    the shared modal container).
    """

    orchestrator: PlotOrchestrator
    workflow_registry: Mapping[WorkflowId, WorkflowSpec]
    plotting_controller: PlottingController
    plot_data_service: PlotDataService
    session_layers: dict[LayerId, SessionLayer]
    on_edit_title: Callable[[CellId, str, bool], None]
    on_add_layer: Callable[[CellId], None]
    on_reconfigure_layer: Callable[[LayerId], None]


class CellWidget:
    """
    Per-session widget for a single plot cell.

    Builds a Panel ``Column`` with a titlebar, a (toggleable) column of
    per-layer toolbars, and a content area showing the composed plot or a
    status placeholder. Owns the cell's autoscale controller and the panes
    updated in place by the poll loop (titlebar freshness pill, per-layer
    time-range labels).

    Parameters
    ----------
    cell_id
        ID of the cell.
    cell
        Plot cell configuration with all layers.
    deps
        Shared dependencies (services and modal callbacks).
    toolbars_visible
        Initial visibility of the per-layer toolbars. Preserved across cell
        rebuilds by the owner so the user's toggle survives version polling.
    """

    def __init__(
        self,
        cell_id: CellId,
        cell: PlotCell,
        deps: CellDeps,
        *,
        toolbars_visible: bool,
    ) -> None:
        self._cell_id = cell_id
        self._cell = cell
        self._deps = deps
        self._toolbars_shown = toolbars_visible
        self._autoscale_controller: CellAutoscaleController | None = None
        self._freshness_pane = create_freshness_pane()
        self._layer_time_panes: dict[LayerId, pn.pane.HTML] = {}
        # Composing builds the autoscale controller as a side effect.
        self._plot = self._compose_plot()
        self._view = self._build()

    @property
    def view(self) -> pn.Column:
        """The Panel widget for this cell."""
        return self._view

    @property
    def has_plot(self) -> bool:
        """Whether the cell currently shows a composed plot (vs. placeholder)."""
        return self._plot is not None

    @property
    def toolbars_shown(self) -> bool:
        """Whether the per-layer toolbars are currently revealed."""
        return self._toolbars_shown

    @property
    def freshness_pane(self) -> pn.pane.HTML:
        """The titlebar freshness/lag pill pane."""
        return self._freshness_pane

    @property
    def layer_time_panes(self) -> dict[LayerId, pn.pane.HTML]:
        """The per-layer time-range panes, keyed by layer ID."""
        return self._layer_time_panes

    @property
    def autoscale_controller(self) -> CellAutoscaleController | None:
        """The cell's autoscale controller, if any layer drives autoscale."""
        return self._autoscale_controller

    def update_freshness(self, bounds_list: list[TimeBounds | None]) -> None:
        """Update the titlebar freshness pane from the layers' time bounds."""
        merged = merge_time_bounds(bounds_list)
        if merged is None:
            html = ''
        else:
            now = Timestamp.now()
            html = format_freshness_html(
                merged.lag_seconds(now), tooltip=format_time_info(merged, now=now)
            )
        if self._freshness_pane.object != html:
            self._freshness_pane.object = html

    def update_layer_time(self, layer_id: LayerId, bounds: TimeBounds | None) -> None:
        """Update a layer toolbar's time-range pane from its time bounds."""
        pane = self._layer_time_panes.get(layer_id)
        if pane is None:
            return
        text = format_time_info(bounds) if bounds is not None else ''
        html = format_layer_time_html(text)
        if pane.object != html:
            pane.object = html

    def dispose(self) -> None:
        """Dispose the autoscale controller, breaking its reference cycle.

        Calling ``dispose()`` breaks the controller → Bokeh-tool →
        on_change-callback → controller reference cycle so long sessions
        don't accumulate detached controllers after cell rebuilds/removals.
        """
        if self._autoscale_controller is not None:
            self._autoscale_controller.dispose()
            self._autoscale_controller = None

    def _layer_states(self) -> dict[LayerId, LayerStateMachine]:
        """Get layer states from PlotDataService for all layers in the cell."""
        result: dict[LayerId, LayerStateMachine] = {}
        for layer in self._cell.layers:
            state = self._deps.plot_data_service.get(layer.layer_id)
            if state is None:
                raise RuntimeError(
                    f"Layer {layer.layer_id} has no state in PlotDataService. "
                    "This indicates a bug: layers should be registered before "
                    "widgets are notified."
                )
            result[layer.layer_id] = state
        return result

    def _build(self) -> pn.Column:
        """Assemble the cell widget: titlebar, layer toolbars, and content."""
        layer_states = self._layer_states()

        # Per-layer toolbars, wrapped in a column the titlebar toggle can hide.
        layer_toolbars = self._build_layer_toolbars(layer_states)
        layers_column = pn.Column(
            *layer_toolbars,
            sizing_mode='stretch_width',
            margin=0,
            visible=self._toolbars_shown,
        )
        titlebar = self._build_titlebar(layers_column)

        # Create content area (placeholder or plot)
        if self._plot is not None:
            content = self._build_plot_content(self._plot)
            border = None
            bg_color = None
        else:
            content = self._build_placeholder(layer_states)
            # Check if any layer has an error
            has_error = any(
                state.error_message is not None for state in layer_states.values()
            )
            if has_error:
                bg_color = '#ffe6e6'
                border = f'2px solid {StatusColors.ERROR}'
            else:
                bg_color = Colors.BG_LIGHT
                border = f'2px dashed {Colors.BORDER}'

        styles = {}
        if bg_color:
            styles['background-color'] = bg_color
        if border:
            styles['border'] = border

        return pn.Column(
            titlebar,
            layers_column,
            content,
            sizing_mode='stretch_both',
            styles=styles,
            margin=GridCellStyles.CELL_MARGIN,
        )

    def _build_titlebar(self, layers_column: pn.Column) -> pn.Row:
        """
        Create the cell-level titlebar (title, add layer, rename, toolbar toggle).

        Parameters
        ----------
        layers_column
            The column wrapping the per-layer toolbars, whose visibility the
            toggle button controls.
        """
        cell = self._cell
        has_user_title = cell.user_title is not None
        title = (
            cell.user_title
            if has_user_title
            else derive_cell_title(
                cell,
                self._deps.workflow_registry,
                get_source_title=self._deps.orchestrator.get_source_title,
            )
        )

        # Union of overlay suggestions across all layers, excluding plotters
        # already present in the cell. Each suggestion remembers the source
        # layer's config so the overlay inherits the right workflow/sources.
        existing_plotter_names = {layer.config.plot_name for layer in cell.layers}
        overlay_specs: list[tuple[str, str, str]] = []
        overlay_sources: dict[tuple[str, str], PlotConfig] = {}
        for layer in cell.layers:
            for spec in self._available_overlays(layer.config):
                output_name, plotter_name, _ = spec
                key = (output_name, plotter_name)
                if plotter_name in existing_plotter_names or key in overlay_sources:
                    continue
                overlay_sources[key] = layer.config
                overlay_specs.append(spec)

        def on_edit_title() -> None:
            self._deps.on_edit_title(self._cell_id, title, has_user_title)

        def on_toggle_toolbars(visible: bool) -> None:
            self._toolbars_shown = visible
            layers_column.visible = visible

        def on_add() -> None:
            self._deps.on_add_layer(self._cell_id)

        def on_overlay(view_name: str, plotter_name: str) -> None:
            source_config = overlay_sources[(view_name, plotter_name)]
            self._create_overlay_layer(source_config, view_name, plotter_name)

        return create_cell_titlebar(
            title=title,
            has_user_title=has_user_title,
            on_edit_title_callback=on_edit_title,
            toolbars_visible=self._toolbars_shown,
            on_toggle_toolbars_callback=on_toggle_toolbars,
            freshness_pane=self._freshness_pane,
            on_add_callback=on_add,
            on_overlay_selected=on_overlay,
            available_overlays=overlay_specs,
        )

    def _available_overlays(self, config: PlotConfig) -> list[tuple[str, str, str]]:
        """
        Get available overlay suggestions for a layer based on its configuration.

        Returns
        -------
        :
            List of (output_name, plotter_name, plotter_title) tuples.
            Returns empty list if overlays are not applicable.
        """
        # Skip for static overlays (no workflow)
        if config.is_static():
            return []

        # Get workflow spec
        workflow_spec = self._deps.workflow_registry.get(config.workflow_id)
        if workflow_spec is None:
            return []

        return self._deps.plotting_controller.get_available_overlays(
            workflow_spec, config.plot_name
        )

    def _create_overlay_layer(
        self,
        base_config: PlotConfig,
        view_name: str,
        plotter_name: str,
    ) -> None:
        """
        Create an overlay layer inheriting configuration from a base layer.

        Parameters
        ----------
        base_config
            Configuration of the base layer (e.g., image layer).
        view_name
            Name of the output view for the overlay (e.g., 'roi_rectangle').
        plotter_name
            Name of the plotter to use for the overlay.
        """
        # Get default params for the plotter
        spec = self._deps.plotting_controller.get_spec(plotter_name)
        params = spec.params() if spec.params else None

        # Create PlotConfig inheriting workflow/sources from base layer
        overlay_config = PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=base_config.workflow_id,
                    source_names=list(base_config.source_names),
                    view_name=view_name,
                )
            },
            plot_name=plotter_name,
            params=params,
        )

        self._deps.orchestrator.add_layer(self._cell_id, overlay_config)

    def _build_layer_toolbars(
        self,
        layer_states: dict[LayerId, LayerStateMachine],
    ) -> list[pn.Row | pn.Column]:
        """
        Create per-layer toolbars (title + gear/close) for all layers in the cell.

        Parameters
        ----------
        layer_states
            Per-layer runtime state from PlotDataService.

        Returns
        -------
        :
            List of toolbar widgets, one per layer.
        """
        toolbars = []
        for layer in self._cell.layers:
            layer_id = layer.layer_id
            config = layer.config
            state = layer_states[layer_id]

            # Get display info for this layer
            title, description = get_plot_cell_display_info(
                config,
                self._deps.workflow_registry,
                get_source_title=self._deps.orchestrator.get_source_title,
            )

            # Add state info to description using explicit state enum
            stopped = False
            match state.state:
                case LayerState.ERROR:
                    description = f"{description}\n\nError: {state.error_message}"
                case LayerState.STOPPED:
                    stopped = True
                    description = f"{description}\n\nStatus: Workflow ended"
                case LayerState.WAITING_FOR_DATA:
                    description = f"{description}\n\nStatus: Waiting for data..."
                case LayerState.WAITING_FOR_JOB:
                    description = f"{description}\n\nStatus: Waiting for workflow"
                case LayerState.READY:
                    pass  # No extra description for ready state

            def make_close_callback(lid: LayerId) -> Callable[[], None]:
                def on_close() -> None:
                    self._deps.orchestrator.remove_layer(lid)

                return on_close

            def make_gear_callback(lid: LayerId) -> Callable[[], None]:
                def on_gear() -> None:
                    self._deps.on_reconfigure_layer(lid)

                return on_gear

            time_pane = create_layer_time_pane()
            self._layer_time_panes[layer_id] = time_pane

            toolbar = create_layer_toolbar(
                on_gear_callback=make_gear_callback(layer_id),
                on_close_callback=make_close_callback(layer_id),
                title=title,
                description=description,
                stopped=stopped,
                time_pane=time_pane,
            )
            toolbars.append(toolbar)

        return toolbars

    def _build_placeholder(
        self,
        layer_states: dict[LayerId, LayerStateMachine],
    ) -> pn.pane.Markdown:
        """
        Create placeholder content showing layer status.

        Parameters
        ----------
        layer_states
            Per-layer runtime state from PlotDataService.

        Returns
        -------
        :
            Markdown pane showing status for all layers.
        """
        # Build status info for each layer
        status_lines = []
        for layer in self._cell.layers:
            config = layer.config
            state = layer_states[layer.layer_id]

            workflow_title, output_title = get_workflow_display_info(
                self._deps.workflow_registry, config.workflow_id, config.view_name
            )

            # Determine status from explicit state enum
            match state.state:
                case LayerState.ERROR:
                    error_text = state.error_message or "Unknown error"
                    status = f"Error: {extract_error_summary(error_text)}"
                    text_color = StatusColors.ERROR
                case LayerState.STOPPED:
                    status = "Workflow ended"
                    text_color = Colors.TEXT
                case LayerState.WAITING_FOR_DATA:
                    status = "Waiting for data..."
                    text_color = Colors.TEXT_MUTED
                case LayerState.WAITING_FOR_JOB:
                    status = "Waiting for workflow..."
                    text_color = Colors.TEXT_MUTED
                case LayerState.READY:
                    # Defensive: READY should have displayable plot and not
                    # reach placeholder. Log if this happens.
                    logger.warning(
                        "Layer %s in READY state but showing placeholder",
                        layer.layer_id,
                    )
                    status = "Ready (loading...)"
                    text_color = Colors.TEXT_MUTED

            status_lines.append(
                f"**{workflow_title} → {output_title}**: "
                f"<span style='color: {text_color}'>{status}</span>"
            )

        content = "\n\n".join(status_lines)

        return pn.pane.Markdown(
            content,
            styles={'text-align': 'left', 'padding': '20px'},
        )

    def _build_plot_content(
        self,
        plot: hv.DynamicMap | hv.Element | hv.Overlay,
    ) -> pn.pane.HoloViews:
        """
        Create plot content widget.

        Parameters
        ----------
        plot
            The composed plot.

        Returns
        -------
        :
            HoloViews pane containing the plot.
        """
        # Use sizing mode from first layer (they should be consistent for overlay)
        if self._cell.layers:
            sizing_mode = _get_sizing_mode(self._cell.layers[0].config)
        else:
            sizing_mode = 'stretch_both'

        # Use .layout to preserve widgets for DynamicMaps with kdims.
        # When pn.pane.HoloViews wraps a DynamicMap with kdims, it generates
        # widgets. However, these widgets don't render when the pane is placed
        # in a Panel layout (Tabs, Column, etc.). The .layout property contains
        # both the plot and widgets, which renders correctly in layouts.
        # See: https://github.com/holoviz/panel/issues/5628
        #
        # CRITICAL: Use linked_axes=False to prevent unintended axis linking (#607)
        #
        # Problem: By default, Panel's HoloViews pane links axes across different
        # plots based on their axis labels (e.g., all plots with 'x' and 'y' axes
        # get linked). For detector panels in different grid cells, this is unwanted:
        # - Different detector panels have independent spatial coordinates
        # - Zooming one panel shouldn't affect others
        # - Each panel needs independent autoscaling
        #
        # Previous workarounds and why they failed:
        # - shared_axes=False (HoloViews): Breaks dynamic features that rely on
        #   shared axis infrastructure
        # - Wrapping in hv.Layout: Prevents multi-layer composition with hv.Overlay,
        #   which was needed for the layer system (#606)
        #
        # Solution: linked_axes=False on the Panel pane
        # - Disables Panel's cross-plot axis linking while preserving HoloViews
        #   features (autoscaling, dynamic updates)
        # - Allows proper multi-layer composition via hv.Overlay
        # - Each grid cell's plot remains independent
        plot_pane_wrapper = pn.pane.HoloViews(
            plot, sizing_mode=sizing_mode, linked_axes=False
        )
        return plot_pane_wrapper.layout

    def _compose_plot(self) -> hv.DynamicMap | hv.Element | None:
        """
        Compose the cell's plot from session-local DynamicMaps or static elements.

        Ensures session components exist when data is available. Sets a
        descriptive SaveTool filename on the result so that browser "Save"
        downloads get a meaningful name, and builds the cell's autoscale
        controller as a side effect.

        Returns
        -------
        :
            Composed plot from session DMaps/elements, or None if none available.
        """
        plots = []
        has_layout = False
        cell_plotters: list = []
        for layer in self._cell.layers:
            layer_id = layer.layer_id
            session_layer = self._deps.session_layers.get(layer_id)
            if session_layer is None:
                continue

            # Ensure components exist if data is now available
            state = self._deps.plot_data_service.get(layer_id)
            if state is not None:
                session_layer.ensure_components(state)
                # Static overlays (vlines/hlines/rectangles) are not Plotters
                # and carry no autoscale axes; exclude them from the controller.
                if state.plotter is not None and not layer.config.is_static():
                    cell_plotters.append(state.plotter)

            dmap = session_layer.dmap
            if dmap is not None:
                # Check the DynamicMap's resolved type (set after Bokeh
                # renders it) and the plotter's cached state.  Either
                # being a Layout means hooks must be skipped.
                if isinstance(dmap, hv.DynamicMap) and dmap.type is hv.Layout:
                    has_layout = True
                elif (
                    state is not None
                    and state.plotter is not None
                    and isinstance(state.plotter.get_cached_state(), hv.Layout)
                ):
                    has_layout = True
                plots.append(dmap)

        if not plots:
            return None

        result: hv.DynamicMap | hv.Element
        if len(plots) == 1:
            result = plots[0]
        else:
            # Collate so hooks survive for any number of DynamicMap layers.
            # Without collation, HoloViews drops overlay-level opts (including
            # hooks) when an Overlay contains 3+ DynamicMaps.  Collating first
            # produces a single DynamicMap whose outputs are plain Overlays;
            # opts applied afterwards land on the OverlayPlot and persist.
            result = hv.Overlay(plots).collate()

        # Skip hooks for Layouts — each sub-figure has its own SaveTool,
        # so a single cell-level filename is not meaningful.
        if has_layout:
            return result

        hooks: list = []
        filename = build_save_filename_from_cell(
            self._cell,
            self._deps.workflow_registry,
            self._deps.orchestrator.get_source_title,
        )
        if filename is not None:
            hooks.append(make_save_filename_hook(filename))
        if self._cell.layers:
            params = self._cell.layers[0].config.params
            if hasattr(params, 'plot_aspect'):
                aspect_hook = make_frame_aspect_hook_from_config(params.plot_aspect)
                if aspect_hook is not None:
                    hooks.append(aspect_hook)
        # Fresh controller on every cell rebuild: keeps tools and toggle state
        # local to this session's Bokeh document. The previous cell widget's
        # controller is disposed by the owner after the swap, breaking its
        # on_change reference cycle (would otherwise leak across the session).
        controller = build_controller_from_layers(cell_plotters)
        if controller is not None:
            self._autoscale_controller = controller
            hooks.append(controller.make_hook())
        if hooks:
            result = result.opts(hooks=hooks)

        return result
