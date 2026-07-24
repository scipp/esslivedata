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

from ..cell_autoscale import CellAutoscaleController, build_controller_from_layers
from ..format_utils import extract_error_summary
from ..hover_suspend import make_hover_suspend_hook
from ..plot_data_service import LayerState, LayerStateMachine, PlotDataService
from ..plot_orchestrator import (
    CellGeometry,
    CellId,
    LayerId,
    PlotCell,
    PlotConfig,
    PlotOrchestrator,
)
from ..plot_params import PlotAspectType, StretchMode
from ..plots import TimeBounds, format_time_info, merge_time_bounds
from ..save_filename import build_save_filename_from_cell, make_save_filename_hook
from ..session_layer import SessionLayer
from .plot_grid import GridCellStyles
from .plot_widgets import (
    create_cell_titlebar,
    create_layer_info_row,
    derive_cell_title,
    get_plot_cell_display_info,
    get_workflow_display_info,
)
from .styles import Colors, StatusColors, StatusPill

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


# Data-age thresholds in seconds (now minus the oldest data's end time) that
# classify freshness for the titlebar pill.
_FRESH_MAX_SECONDS = 5.0
_STALE_MAX_SECONDS = 30.0


def _freshness_band(age_seconds: float) -> tuple[str, str, str]:
    """Return the ``(background, text, dot)`` pill colors for a data age."""
    if age_seconds < _FRESH_MAX_SECONDS:
        return StatusPill.FRESH
    if age_seconds < _STALE_MAX_SECONDS:
        return StatusPill.STALE
    return StatusPill.OLD


def _format_age_short(age_seconds: float) -> str:
    """Compact age label, e.g. "2.3s", "41s", "3m"."""
    if age_seconds < 10:
        return f'{age_seconds:.1f}s'
    if age_seconds < 60:
        return f'{age_seconds:.0f}s'
    return f'{age_seconds / 60:.0f}m'


def _pill_html(
    band: tuple[str, str, str], label: str, *, square_dot: bool = False
) -> str:
    """Render a status pill: colored dot + short label on a tinted background.

    Shared by the titlebar pill and the per-layer status badges so job state
    speaks one visual language at both levels. ``square_dot`` renders the dot
    as a square — the conventional "stop" glyph.
    """
    background, text_color, dot_color = band
    pill_style = (
        f'display:inline-flex;align-items:center;gap:5px;height:20px;'
        f'padding:0 8px;border-radius:10px;font-size:11px;'
        f'font-variant-numeric:tabular-nums;white-space:nowrap;'
        f'background:{background};color:{text_color};'
    )
    dot_radius = '1px' if square_dot else '50%'
    dot_style = (
        f'width:7px;height:7px;border-radius:{dot_radius};flex:none;'
        f'background:{dot_color};'
    )
    return f'<span style="{pill_style}"><span style="{dot_style}"></span>{label}</span>'


def format_freshness_html(age_seconds: float | None) -> str:
    """Render the titlebar pill for live data: wall-clock age of what is shown.

    Shows ``now - data_end`` for the oldest live layer, color-banded by
    staleness. Because every cell uses a shared ``now``, ages are directly
    comparable across plots and grow visibly while a stream is stalled.
    Stopped and errored cells freeze the pill via :func:`format_stopped_html`
    / :func:`format_error_html` instead — a growing age on a deliberately
    stopped job would read as a pipeline problem.

    No hover tooltip: the pill re-renders as the age ticks, which would tear
    down a native ``title`` tooltip on every update. The absolute time range and
    the frozen pipeline lag live in the per-layer toolbar row instead.

    Parameters
    ----------
    age_seconds:
        Data age in seconds, or None to render nothing (no timing data).
    """
    if age_seconds is None:
        return ''
    return _pill_html(_freshness_band(age_seconds), _format_age_short(age_seconds))


def format_stopped_html() -> str:
    """Neutral gray "stopped" pill: the job ended, the snapshot is deliberate."""
    return _pill_html(StatusPill.STOPPED, 'stopped', square_dot=True)


def format_error_html() -> str:
    """Red "error" pill for a failed layer; details live in the layer tooltip."""
    return _pill_html(StatusPill.ERROR, 'error')


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
    plot_data_service: PlotDataService
    session_layers: dict[LayerId, SessionLayer]
    on_edit_title: Callable[[CellId, str, bool], None]
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
        self._stopped_layers: frozenset[LayerId] = frozenset()
        self._pill_frozen = False
        self._layer_time_panes: dict[LayerId, pn.pane.HTML] = {}
        # Composing builds the autoscale controller as a side effect.
        self._plot = self._compose_plot()
        self._view = self._build()

    @property
    def view(self) -> pn.Column:
        """The Panel widget for this cell."""
        return self._view

    @property
    def geometry(self) -> CellGeometry:
        """Grid geometry (position and span) of this cell."""
        return self._cell.geometry

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

    def update_freshness(
        self, bounds_by_layer: Mapping[LayerId, TimeBounds | None]
    ) -> None:
        """Update the titlebar pill with the worst-case data age of live layers.

        No-op while the pill is frozen to a status ("stopped"/"error", set at
        build time). Stopped layers are excluded from the merge so their
        ever-growing age cannot drag the band to red while other layers are
        live.
        """
        if self._pill_frozen:
            return
        merged = merge_time_bounds(
            b for lid, b in bounds_by_layer.items() if lid not in self._stopped_layers
        )
        age = merged.age_seconds() if merged is not None else None
        html = format_freshness_html(age)
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

    def _freeze_pill_for_status(
        self, layer_states: dict[LayerId, LayerStateMachine]
    ) -> None:
        """Freeze the titlebar pill to a status pill when the cell is not live.

        Any errored layer wins (needs attention); otherwise all dynamic layers
        stopped means the cell shows a deliberate frozen snapshot. Static
        overlay layers never run a job, so they are ignored for the stopped
        aggregate. State changes bump layer versions and rebuild the cell, so
        deciding at build time is sufficient.
        """
        self._stopped_layers = frozenset(
            layer_id
            for layer_id, state in layer_states.items()
            if state.state is LayerState.STOPPED
        )
        dynamic_ids = [
            layer.layer_id
            for layer in self._cell.layers
            if not layer.config.is_static()
        ]
        if any(s.state is LayerState.ERROR for s in layer_states.values()):
            frozen = format_error_html()
        elif dynamic_ids and all(lid in self._stopped_layers for lid in dynamic_ids):
            frozen = format_stopped_html()
        else:
            frozen = None
        self._pill_frozen = frozen is not None
        if frozen is not None:
            self._freshness_pane.object = frozen

    def _build(self) -> pn.Column:
        """Assemble the cell widget: titlebar, layer toolbars, and content."""
        layer_states = self._layer_states()
        self._freeze_pill_for_status(layer_states)

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

        configure_layers = [
            (
                layer.layer_id,
                get_plot_cell_display_info(
                    layer.config,
                    self._deps.workflow_registry,
                    get_source_title=self._deps.orchestrator.get_source_title,
                )[0],
            )
            for layer in cell.layers
        ]

        def on_edit_title() -> None:
            self._deps.on_edit_title(self._cell_id, title, has_user_title)

        def on_toggle_toolbars(visible: bool) -> None:
            self._toolbars_shown = visible
            layers_column.visible = visible

        geometry = cell.geometry
        return create_cell_titlebar(
            title=title,
            has_user_title=has_user_title,
            on_edit_title_callback=on_edit_title,
            configure_layers=configure_layers,
            on_configure_layer=self._deps.on_reconfigure_layer,
            toolbars_visible=self._toolbars_shown,
            on_toggle_toolbars_callback=on_toggle_toolbars,
            freshness_pane=self._freshness_pane,
            # Per-cell automation hook: a rebuilt cell's DOM position is not
            # stable, so the grid position addresses it (unique per grid, and
            # only the active tab's grid is rendered).
            css_classes=[f'lt-cell-r{geometry.row}c{geometry.col}'],
        )

    def _build_layer_toolbars(
        self,
        layer_states: dict[LayerId, LayerStateMachine],
    ) -> list[pn.Row | pn.Column]:
        """
        Create per-layer info rows (title + time range) for the cell's layers.

        Parameters
        ----------
        layer_states
            Per-layer runtime state from PlotDataService.

        Returns
        -------
        :
            List of info-row widgets, one per layer.
        """
        rows = []
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
            badge = None
            match state.state:
                case LayerState.ERROR:
                    badge = format_error_html()
                    description = f"{description}\n\nError: {state.error_message}"
                case LayerState.STOPPED:
                    badge = format_stopped_html()
                    description = f"{description}\n\nStatus: Workflow not running"
                case LayerState.WAITING_FOR_DATA:
                    description = f"{description}\n\nStatus: Waiting for data..."
                case LayerState.READY:
                    pass  # No extra description for ready state

            time_pane = create_layer_time_pane()
            self._layer_time_panes[layer_id] = time_pane

            rows.append(
                create_layer_info_row(
                    title=title,
                    description=description,
                    badge=badge,
                    time_pane=time_pane,
                )
            )

        return rows

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
                    status = "Workflow not running"
                    text_color = Colors.TEXT
                case LayerState.WAITING_FOR_DATA:
                    status = "Waiting for data..."
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
        downloads get a meaningful name, suspends hover while an ROI edit tool
        is active, and builds the cell's autoscale controller as a side effect.

        Returns
        -------
        :
            Composed plot from session DMaps/elements, or None if none available.
        """
        plots = []
        non_overlayable = False
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
                # Tables and layout-mode plotters produce a DataTable/Layout with
                # no single figure for the SaveTool/aspect hooks to act on. Such
                # layers are forbidden from sharing a cell, so this flags a
                # solitary non-overlayable layer; hooks are skipped below.
                if state.plotter is not None and not getattr(
                    state.plotter, 'is_overlayable', True
                ):
                    non_overlayable = True

            dmap = session_layer.dmap
            if dmap is not None:
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

        # Skip the cell-level hooks for a non-overlayable layer: a Layout's
        # sub-figures each carry their own SaveTool, and a Table's DataTable
        # widget has no figure for the SaveTool/autoscale hooks to act on. The
        # frame-aspect hook is not among these — it is declared per element type
        # by the plotter (see Plotter._sizing_opts), so it reaches every
        # sub-figure of a Layout regardless.
        if non_overlayable:
            return result

        hooks: list = [make_hover_suspend_hook()]
        filename = build_save_filename_from_cell(
            self._cell,
            self._deps.workflow_registry,
            self._deps.orchestrator.get_source_title,
        )
        if filename is not None:
            hooks.append(make_save_filename_hook(filename))
        # Fresh controller on every cell rebuild: keeps tools and toggle state
        # local to this session's Bokeh document. The previous cell widget's
        # controller is disposed by the owner after the swap, breaking its
        # on_change reference cycle (would otherwise leak across the session).
        controller = build_controller_from_layers(cell_plotters)
        if controller is not None:
            self._autoscale_controller = controller
            hooks.append(controller.make_hook())
        return result.opts(hooks=hooks)
