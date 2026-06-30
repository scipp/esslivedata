# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Widget creation utilities for plot cells."""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import panel as pn

from ...config.workflow_spec import WorkflowId, WorkflowSpec
from ..plot_params import TimeWindowMode
from .buttons import ButtonStyles, create_tool_button, create_tool_button_stylesheet
from .icons import get_icon
from .styles import Colors, HoverColors, StatusColors

if TYPE_CHECKING:
    from ..plot_data_service import LayerId
    from ..plot_orchestrator import PlotCell, PlotConfig
    from ..plotting_controller import PlottingController


def _available_overlays_for_config(
    config: PlotConfig,
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
    plotting_controller: PlottingController,
) -> list[tuple[str, str, str]]:
    """Overlay suggestions for a single layer: ``(view, plotter, title)``."""
    if config.is_static():
        return []
    workflow_spec = workflow_registry.get(config.workflow_id)
    if workflow_spec is None:
        return []
    return plotting_controller.get_available_overlays(workflow_spec, config.plot_name)


def overlay_suggestions_for_layer(
    config: PlotConfig,
    existing_plotters: frozenset[str],
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
    plotting_controller: PlottingController,
) -> list[tuple[str, str, str]]:
    """
    Overlay suggestions derived from one layer, minus plotters already present.

    Overlays inherit the layer's workflow/sources, so a suggestion is offered
    per base layer. Plotters already present anywhere in the cell are excluded
    to avoid suggesting a duplicate.

    Returns
    -------
    :
        List of ``(view_name, plotter_name, plotter_title)`` tuples.
    """
    return [
        (view_name, plotter_name, title)
        for view_name, plotter_name, title in _available_overlays_for_config(
            config, workflow_registry, plotting_controller
        )
        if plotter_name not in existing_plotters
    ]


def create_overlay_add_button(
    *,
    overlays: list[tuple[str, str, str]],
    on_overlay_selected: Callable[[str, str], None],
) -> pn.widgets.MenuButton:
    """
    Create a ``+`` dropdown adding an overlay derived from a layer.

    The dropdown lists the overlay titles (always a menu, even for a single
    suggestion, so the choice is always labeled). Callers build this only when
    ``overlays`` is non-empty.

    Parameters
    ----------
    overlays:
        ``(view_name, plotter_name, plotter_title)`` tuples for the layer.
    on_overlay_selected:
        Callback when an overlay is chosen: ``(view_name, plotter_name)``.

    Returns
    -------
    :
        MenuButton listing the overlay suggestions.
    """
    button_color = StatusColors.SUCCESS
    hover_color = HoverColors.SUCCESS

    items: list[str] = []
    overlay_map: dict[str, tuple[str, str]] = {}
    for view_name, plotter_name, title in overlays:
        items.append(title)
        overlay_map[title] = (view_name, plotter_name)

    stylesheets = create_tool_button_stylesheet(button_color, hover_color)
    stylesheets.append(
        """
        .bk-menu {
            min-width: 200px !important;
            right: 0 !important;
            left: auto !important;
        }
        """
    )
    menu_button = pn.widgets.MenuButton(
        label='',
        items=items,
        icon=get_icon('plus'),
        icon_size='1.5em',
        width=ButtonStyles.TOOL_BUTTON_SIZE,
        height=ButtonStyles.TOOL_BUTTON_SIZE,
        color='light',
        sizing_mode='fixed',
        margin=0,
        stylesheets=stylesheets,
    )

    def on_menu_click(event: pn.param.parameterized.Event) -> None:
        if event.new in overlay_map:
            view_name, plotter_name = overlay_map[event.new]
            on_overlay_selected(view_name, plotter_name)

    menu_button.on_click(on_menu_click)
    return menu_button


def create_layer_info_row(
    *,
    title: str | None = None,
    description: str | None = None,
    stopped: bool = False,
    time_pane: pn.pane.HTML | None = None,
) -> pn.Row | pn.Column:
    """
    Create a per-layer info row showing the layer title and its time range.

    Purely informational: it shows the layer title (with a tooltip for the
    description) and, when a ``time_pane`` is given, the layer's data time
    range on its own line below. Layer configuration lives on the cell titlebar
    gear; add/remove and rename live on the cell properties modal — no controls
    are placed here.

    Parameters
    ----------
    title:
        Optional title text to display on the left side of the row.
    description:
        Optional description shown as tooltip when hovering over the title.
    stopped:
        If True, adds a visual indicator (border) showing workflow has ended.
    time_pane:
        Optional pane showing this layer's data time range/lag, placed on its
        own line below the title and updated in place by the caller.

    Returns
    -------
    :
        Panel Row (or Column when a time pane is given) for one layer.
    """
    # Build left content: title and optional tooltip icon
    left_items: list = []
    if title:
        title_html = pn.pane.HTML(
            f'<span style="font-size: 12px; color: {Colors.TEXT};">{title}</span>',
            sizing_mode='fixed',
            height=ButtonStyles.TOOL_BUTTON_SIZE,
            styles={
                'display': 'flex',
                'align-items': 'center',
            },
        )
        left_items.append(title_html)

        if description:
            from bokeh.models import Tooltip
            from bokeh.models.dom import HTML

            # Convert newlines to <br> for HTML rendering
            html_description = description.replace('\n', '<br>')
            tooltip = Tooltip(
                content=HTML(html_description),
                position='right',
                stylesheets=[':host { max-width: 350px; }'],
            )
            tooltip_icon = pn.widgets.TooltipIcon(
                value=tooltip,
                margin=0,
            )
            left_items.append(tooltip_icon)

    margin = ButtonStyles.CELL_MARGIN

    # Add border when workflow is stopped
    styles = {}
    if stopped:
        styles['border'] = f'2px solid {Colors.TEXT}'
        styles['border-radius'] = '4px'

    title_row = pn.Row(
        *left_items,
        pn.Spacer(sizing_mode='stretch_width'),
        sizing_mode='stretch_width',
        height=ButtonStyles.TOOL_BUTTON_SIZE,
        align='end',
    )

    if time_pane is None:
        title_row.margin = (margin, margin, margin, margin)
        title_row.styles = styles
        return title_row

    # Time info goes on its own line below the title row to avoid competing
    # with the title for horizontal space.
    return pn.Column(
        title_row,
        time_pane,
        sizing_mode='stretch_width',
        margin=(margin, margin, margin, margin),
        styles=styles,
    )


def _cell_title_html(title: str, has_user_title: bool) -> str:
    """Render the cell title span; derived titles are muted and italic.

    User-defined titles are free text and HTML-escaped. Derived titles come
    from display-info helpers and may contain HTML entities (e.g. ``&rarr;``),
    so they are rendered as-is.
    """
    if has_user_title:
        from html import escape

        style = f'font-size:12.5px;font-weight:600;color:{Colors.TEXT_DARK};'
        title = escape(title)
    else:
        style = f'font-size:12.5px;color:{Colors.TEXT_MUTED};font-style:italic;'
    return (
        f'<span style="{style}line-height:{ButtonStyles.TOOL_BUTTON_SIZE}px;'
        f'white-space:nowrap;overflow:hidden;'
        f'text-overflow:ellipsis;display:block;">{title}</span>'
    )


def _create_toolbar_visibility_button(
    *,
    visible: bool,
    on_toggle: Callable[[bool], None],
) -> pn.widgets.Button:
    """Adjustments button toggling per-layer info-row visibility for a cell."""

    def _icon(shown: bool) -> str:
        return get_icon('stack-2' if shown else 'stack')

    def _tip(shown: bool) -> str:
        return 'Hide layer details' if shown else 'Show layer details'

    button = pn.widgets.Button(
        label='',
        icon=_icon(visible),
        icon_size='1.5em',
        width=ButtonStyles.TOOL_BUTTON_SIZE,
        height=ButtonStyles.TOOL_BUTTON_SIZE,
        color='light',
        sizing_mode='fixed',
        margin=0,
        description=_tip(visible),
        # Hand-rolled (toggling icon) so it bypasses create_tool_button; tag it
        # with the stable automation hooks by hand. Icon name varies, so use a
        # semantic suffix rather than lt-tool-{icon_name}.
        css_classes=['lt-tool', 'lt-tool-layer-details'],
        stylesheets=create_tool_button_stylesheet(Colors.TEXT_MUTED, HoverColors.MUTED),
    )
    state = {'visible': visible}

    def _toggle(_) -> None:
        state['visible'] = not state['visible']
        button.icon = _icon(state['visible'])
        button.description = _tip(state['visible'])
        on_toggle(state['visible'])

    button.on_click(_toggle)
    return button


def _create_configure_button_or_menu(
    *,
    layers: list[tuple[LayerId, str]],
    on_configure: Callable[[LayerId], None],
) -> pn.widgets.Button | pn.widgets.MenuButton:
    """
    Gear button (single layer) or menu picking which layer to configure.

    For a single-layer cell the gear configures that layer directly. For
    multiple layers it becomes a dropdown listing the layer titles, routing the
    selection to ``on_configure`` with the chosen layer's ID.

    Parameters
    ----------
    layers:
        ``(layer_id, title)`` pairs for the cell's layers, in display order.
    on_configure:
        Callback invoked with the layer ID to configure.
    """
    button_color = Colors.TEXT_MUTED
    hover_color = HoverColors.MUTED

    if len(layers) == 1:
        layer_id = layers[0][0]
        return create_tool_button(
            icon_name='settings',
            button_color=button_color,
            hover_color=hover_color,
            on_click_callback=lambda: on_configure(layer_id),
        )

    # Titles carry HTML entities (e.g. '&rarr;'); unescape for plain menu
    # labels and disambiguate duplicates so the label->layer map stays 1:1.
    items: list[str] = []
    label_map: dict[str, LayerId] = {}
    for layer_id, title in layers:
        label = html.unescape(title)
        if label in label_map:
            suffix = 2
            while f'{label} ({suffix})' in label_map:
                suffix += 1
            label = f'{label} ({suffix})'
        items.append(label)
        label_map[label] = layer_id

    stylesheets = create_tool_button_stylesheet(button_color, hover_color)
    stylesheets.append(
        """
        .bk-menu {
            min-width: 200px !important;
            right: 0 !important;
            left: auto !important;
        }
        """
    )
    menu_button = pn.widgets.MenuButton(
        label='',
        items=items,
        icon=get_icon('settings'),
        icon_size='1.5em',
        width=ButtonStyles.TOOL_BUTTON_SIZE,
        height=ButtonStyles.TOOL_BUTTON_SIZE,
        color='light',
        sizing_mode='fixed',
        margin=0,
        # MenuButton (dropdown), so it bypasses create_tool_button; tag it with
        # the same hooks as the single-layer gear for consistent automation.
        css_classes=['lt-tool', 'lt-tool-settings'],
        stylesheets=stylesheets,
    )

    def on_menu_click(event: pn.param.parameterized.Event) -> None:
        if event.new in label_map:
            on_configure(label_map[event.new])

    menu_button.on_click(on_menu_click)
    return menu_button


def create_cell_titlebar(
    *,
    title: str,
    has_user_title: bool,
    on_edit_title_callback: Callable[[], None],
    configure_layers: list[tuple[LayerId, str]],
    on_configure_layer: Callable[[LayerId], None],
    toolbars_visible: bool,
    on_toggle_toolbars_callback: Callable[[bool], None],
    freshness_pane: pn.pane.HTML | None = None,
) -> pn.Row:
    """
    Create the cell-level titlebar shown above the per-layer info rows.

    The titlebar holds the cell title on the left and cell-level actions on the
    right: a gear that configures a layer (directly for one layer, via a layer
    picker for several), edit cell properties (opens a modal for
    rename/add/remove layer), and a toggle that hides/shows the per-layer info
    rows.

    Parameters
    ----------
    title:
        Title to display. Either the user-defined title or a derived one.
    has_user_title:
        Whether ``title`` is user-defined (styled prominently) or derived
        (styled muted/italic as a placeholder).
    on_edit_title_callback:
        Invoked when the edit (pencil) button is clicked; opens the cell
        properties modal.
    configure_layers:
        ``(layer_id, title)`` pairs for the cell's layers, driving the gear
        button/menu.
    on_configure_layer:
        Invoked with a layer ID when the gear (or a layer menu entry) is chosen.
    toolbars_visible:
        Current visibility of the per-layer info rows; sets the toggle icon.
    on_toggle_toolbars_callback:
        Invoked with the new visibility state when the toggle is clicked.
    freshness_pane:
        Optional pane showing the data freshness/lag indicator, placed between
        the title and the action buttons. Updated in place by the caller.

    Returns
    -------
    :
        Panel Row widget containing the cell titlebar.
    """
    title_pane = pn.pane.HTML(
        _cell_title_html(title, has_user_title),
        sizing_mode='stretch_width',
        height=ButtonStyles.TOOL_BUTTON_SIZE,
        margin=(0, 4),
        styles={'overflow': 'hidden', 'min-width': '0'},
    )

    gear_button = _create_configure_button_or_menu(
        layers=configure_layers,
        on_configure=on_configure_layer,
    )
    edit_button = create_tool_button(
        icon_name='pencil',
        button_color=Colors.TEXT_MUTED,
        hover_color=HoverColors.MUTED,
        on_click_callback=on_edit_title_callback,
    )
    toggle_button = _create_toolbar_visibility_button(
        visible=toolbars_visible,
        on_toggle=on_toggle_toolbars_callback,
    )

    right_buttons: list = [gear_button, edit_button, toggle_button]

    # Freshness pill sits at the far left; the stretch title fills the middle and
    # pushes the action buttons to the right.
    left: list = [freshness_pane] if freshness_pane is not None else []

    margin = ButtonStyles.CELL_MARGIN
    return pn.Row(
        *left,
        title_pane,
        *right_buttons,
        sizing_mode='stretch_width',
        min_height=ButtonStyles.TOOL_BUTTON_SIZE,
        align='center',
        margin=(margin, margin, 0, margin),
        styles={
            'background-color': Colors.BG_LIGHT,
            'border-bottom': f'1px solid {Colors.BORDER}',
        },
    )


def get_workflow_display_info(
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
    workflow_id: WorkflowId,
    view_name: str | None,
) -> tuple[str, str]:
    """
    Look up workflow and view display titles from the registry.

    Parameters
    ----------
    workflow_registry:
        Registry mapping workflow IDs to their specifications.
    workflow_id:
        ID of the workflow to look up.
    view_name:
        Name of the output view, or None.

    Returns
    -------
    :
        Tuple of (workflow_title, view_title). If the workflow is not found,
        workflow_title falls back to str(workflow_id). If view_name is None
        or not found in the spec, view_title falls back to view_name or
        empty string.
    """
    workflow_spec = workflow_registry.get(workflow_id)

    # Get workflow title
    if workflow_spec is not None:
        workflow_title = workflow_spec.title
    else:
        workflow_title = str(workflow_id)

    # Get view title from spec if available
    view = (
        workflow_spec.get_output_view(view_name)
        if workflow_spec and view_name
        else None
    )
    view_title = view.title if view is not None else (view_name or '')

    return workflow_title, view_title


def _format_window_info(params) -> str:
    """
    Format window parameters into a human-readable string.

    Only ``since_start`` produces a label here. For window mode the actual time
    range comes from the data and is shown by the titlebar freshness indicator
    (see ``_compute_time_bounds``); the configured ``window_duration_seconds`` is
    a target, not a truth, so showing it in the static title would lie when
    backend cadence exceeds the requested lookback.
    """
    window = getattr(params, 'time_window', None)
    if window is None:
        return ''
    if window.mode == TimeWindowMode.since_start:
        return 'since run start'
    return ''


def _get_static_overlay_display_info(config: PlotConfig) -> tuple[str, str]:
    """Get display info for a static overlay layer."""
    from ..plotter_registry import plotter_registry

    plotter_name = config.plot_name
    try:
        spec = plotter_registry.get_spec(plotter_name)
        plotter_title = spec.title
        plotter_desc = spec.description
    except KeyError:
        plotter_title = plotter_name.replace('_', ' ').title()
        plotter_desc = ''

    # Use the user's custom name from view_name
    custom_name = config.view_name

    # Format title as "Plotter → Custom Name"
    title = f'{plotter_title} &rarr; {custom_name}'

    # Build description for tooltip
    description = f'Static overlay: {plotter_title}\nName: {custom_name}'
    if plotter_desc:
        description += f'\n\n{plotter_desc}'

    return title, description


def get_plot_cell_display_info(
    config: PlotConfig,
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
    get_source_title: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    """
    Assemble title and description for a plot cell toolbar.

    Parameters
    ----------
    config:
        Plot configuration containing workflow, output, and params.
    workflow_registry:
        Registry mapping workflow IDs to their specifications.
    get_source_title:
        Optional function to get display title for a source name.
        If not provided, source names are displayed as-is.

    Returns
    -------
    :
        Tuple of (title, description). Title is a short string for display,
        description is a longer string for tooltip.
    """
    # Handle static overlays (single data source with empty source_names)
    if config.is_static():
        return _get_static_overlay_display_info(config)

    workflow_title, output_title = get_workflow_display_info(
        workflow_registry, config.workflow_id, config.view_name
    )

    # Build title: "Workflow → Output (source, window)"
    # Using HTML entity for arrow since title is rendered in HTML pane
    window_info = _format_window_info(config.params)

    # Helper to get display title for a source
    def _title(source_name: str) -> str:
        return get_source_title(source_name) if get_source_title else source_name

    # Format source info: title if single, count if multiple
    num_sources = len(config.source_names)
    if num_sources == 1:
        source_info = _title(config.source_names[0])
    else:
        source_info = f'{num_sources} sources'

    # Combine source and window info in parentheses
    detail_parts = [source_info]
    if window_info:
        detail_parts.append(window_info)
    details = ', '.join(detail_parts)

    title = f'{workflow_title} &rarr; {output_title} ({details})'

    # Build description for tooltip using display titles
    sources_str = ', '.join(_title(s) for s in config.source_names)
    description_parts = [
        f'Workflow: {workflow_title}',
        f'Output: {output_title}',
        f'Sources: {sources_str}',
    ]
    if window_info:
        description_parts.append(f'Window: {window_info}')

    # Append view description from the workflow spec if available
    workflow_spec = workflow_registry.get(config.workflow_id)
    view = (
        workflow_spec.get_output_view(config.view_name)
        if workflow_spec is not None and config.view_name
        else None
    )
    if view is not None and view.description:
        description_parts.append(f'\n{view.description}')

    # Append plotter-specific description (e.g., usage instructions)
    from ..plotter_registry import plotter_registry

    try:
        plotter_desc = plotter_registry.get_spec(config.plot_name).description
        if plotter_desc:
            description_parts.append(f'\n{plotter_desc}')
    except KeyError:
        pass

    description = '\n'.join(description_parts)

    return title, description


def derive_cell_title(
    cell: PlotCell,
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
    get_source_title: Callable[[str], str] | None = None,
) -> str:
    """
    Derive a default cell title from its layers.

    Deliberately minimal: the title is the first non-static layer's workflow
    title, plus ``" (Source)"`` when the cell's layers reference a single source
    (a single source is otherwise not shown by plot labels). Output names and
    window info are omitted -- base-workflow output names (``Image``,
    ``Histogram``) are implied by the workflow, window/freshness lives in the
    titlebar indicator, and per-source breakdown is visible in the plot. When
    this default is too coarse (e.g. distinct reduction outputs sharing a
    workflow), the user sets an explicit cell title.

    Static layers (decorative overlays) are excluded. A cell with only static
    overlays uses the first static layer's title; an empty cell yields ``''``.

    Parameters
    ----------
    cell:
        The plot cell.
    workflow_registry:
        Registry mapping workflow IDs to their specifications.
    get_source_title:
        Optional function to get display title for a source name.

    Returns
    -------
    :
        Derived title string.
    """
    layers = [layer for layer in cell.layers if not layer.config.is_static()]
    if not layers:
        if not cell.layers:
            return ''
        title, _ = get_plot_cell_display_info(
            cell.layers[0].config, workflow_registry, get_source_title
        )
        return title

    workflow_title, _ = get_workflow_display_info(
        workflow_registry, layers[0].config.workflow_id, None
    )

    sources = {s for layer in layers for s in layer.config.source_names}
    if len(sources) == 1:
        source = next(iter(sources))
        source_title = get_source_title(source) if get_source_title else source
        return f'{workflow_title} ({source_title})'
    return workflow_title
