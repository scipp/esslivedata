# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Widget creation utilities for plot cells."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import panel as pn

from ...config.workflow_spec import WorkflowId, WorkflowSpec
from ..plot_params import WindowMode
from .buttons import ButtonStyles, create_tool_button, create_tool_button_stylesheet
from .icons import get_icon
from .styles import Colors, FreshnessPill, HoverColors, StatusColors

if TYPE_CHECKING:
    from ..plot_orchestrator import PlotCell, PlotConfig


_ADD_LAYER_ITEM = 'Add layer...'


def _create_add_button_or_menu(
    *,
    on_add_callback: Callable[[], None],
    on_overlay_selected: Callable[[str, str], None] | None = None,
    available_overlays: list[tuple[str, str, str]] | None = None,
) -> pn.widgets.Button | pn.widgets.MenuButton:
    """
    Create either a simple add button or a dropdown menu with overlay options.

    Parameters
    ----------
    on_add_callback:
        Callback when "Add layer..." is selected (opens modal).
    on_overlay_selected:
        Callback when an overlay is selected: (output_name, plotter_name).
    available_overlays:
        List of (output_name, plotter_name, plotter_title) tuples.

    Returns
    -------
    :
        Button widget or MenuButton with overlay options.
    """
    button_color = StatusColors.SUCCESS
    hover_color = HoverColors.SUCCESS

    if not available_overlays or on_overlay_selected is None:
        # No overlays available - use simple button
        return create_tool_button(
            icon_name='plus',
            button_color=button_color,
            hover_color=hover_color,
            on_click_callback=on_add_callback,
        )

    # Build menu items: "Add layer..." followed by overlay suggestions
    items = [_ADD_LAYER_ITEM]
    overlay_map: dict[str, tuple[str, str]] = {}

    for output_name, plotter_name, plotter_title in available_overlays:
        items.append(plotter_title)
        overlay_map[plotter_title] = (output_name, plotter_name)

    # Use shared button stylesheet + menu-specific styling
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
        selected = event.new
        if selected == _ADD_LAYER_ITEM:
            on_add_callback()
        elif selected in overlay_map:
            output_name, plotter_name = overlay_map[selected]
            on_overlay_selected(output_name, plotter_name)

    menu_button.on_click(on_menu_click)
    return menu_button


def create_layer_toolbar(
    *,
    on_gear_callback: Callable[[], None],
    on_close_callback: Callable[[], None],
    title: str | None = None,
    description: str | None = None,
    stopped: bool = False,
    time_pane: pn.pane.HTML | None = None,
) -> pn.Row | pn.Column:
    """
    Create a per-layer toolbar with title and gear/close buttons.

    The toolbar displays an optional title on the left (with tooltip for
    description) and gear/close buttons on the right, using flexbox layout
    to avoid overlap with Bokeh's plot toolbar. When a ``time_pane`` is given
    it is placed on its own line below, so the time info does not compete with
    the title for horizontal space. Cell-level actions (add layer, rename,
    toolbar visibility) live on the cell titlebar, not here.

    Parameters
    ----------
    on_gear_callback:
        Callback function to invoke when the gear button is clicked.
    on_close_callback:
        Callback function to invoke when the close button is clicked.
    title:
        Optional title text to display on the left side of the toolbar.
    description:
        Optional description shown as tooltip when hovering over the title.
    stopped:
        If True, adds a visual indicator (border) showing workflow has ended.
    time_pane:
        Optional pane showing this layer's data time range/lag, placed right of
        the title and updated in place by the caller.

    Returns
    -------
    :
        Panel Row widget containing the toolbar.
    """
    gear_button = create_tool_button(
        icon_name='settings',
        button_color=ButtonStyles.PRIMARY_BLUE,
        hover_color=ButtonStyles.PRIMARY_HOVER,
        on_click_callback=on_gear_callback,
    )
    close_button = create_tool_button(
        icon_name='x',
        button_color=ButtonStyles.DANGER_RED,
        hover_color=ButtonStyles.DANGER_HOVER,
        on_click_callback=on_close_callback,
    )

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

    button_row = pn.Row(
        *left_items,
        pn.Spacer(sizing_mode='stretch_width'),
        gear_button,
        close_button,
        sizing_mode='stretch_width',
        height=ButtonStyles.TOOL_BUTTON_SIZE,
        align='end',
    )

    if time_pane is None:
        button_row.margin = (margin, margin, margin, margin)
        button_row.styles = styles
        return button_row

    # Time info goes on its own line below the title/buttons row to avoid
    # competing with the title for horizontal space.
    return pn.Column(
        button_row,
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


def _create_toolbar_visibility_button(
    *,
    visible: bool,
    on_toggle: Callable[[bool], None],
) -> pn.widgets.Button:
    """Adjustments button toggling per-layer toolbar visibility for a cell."""

    def _icon(shown: bool) -> str:
        return get_icon('adjustments' if shown else 'adjustments-off')

    def _tip(shown: bool) -> str:
        return 'Hide layer toolbars' if shown else 'Show layer toolbars'

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


def create_cell_titlebar(
    *,
    title: str,
    has_user_title: bool,
    on_edit_title_callback: Callable[[], None],
    toolbars_visible: bool,
    on_toggle_toolbars_callback: Callable[[bool], None],
    freshness_pane: pn.pane.HTML | None = None,
    on_add_callback: Callable[[], None] | None = None,
    on_overlay_selected: Callable[[str, str], None] | None = None,
    available_overlays: list[tuple[str, str, str]] | None = None,
) -> pn.Row:
    """
    Create the cell-level titlebar shown above the per-layer toolbars.

    The titlebar holds the cell title on the left and cell-level actions on the
    right: add layer (with overlay suggestions), edit title (opens a modal), and
    a toggle that hides/shows the per-layer toolbars.

    Parameters
    ----------
    title:
        Title to display. Either the user-defined title or a derived one.
    has_user_title:
        Whether ``title`` is user-defined (styled prominently) or derived
        (styled muted/italic as a placeholder).
    on_edit_title_callback:
        Invoked when the edit (pencil) button is clicked; opens the rename modal.
    toolbars_visible:
        Current visibility of the per-layer toolbars; sets the toggle icon.
    on_toggle_toolbars_callback:
        Invoked with the new visibility state when the toggle is clicked.
    freshness_pane:
        Optional pane showing the data freshness/lag indicator, placed between
        the title and the action buttons. Updated in place by the caller.
    on_add_callback:
        Optional callback to add a layer (opens modal). If None, no add button.
    on_overlay_selected:
        Optional callback when an overlay suggestion is selected, called with
        (output_name, plotter_name).
    available_overlays:
        Optional list of (output_name, plotter_name, plotter_title) tuples; if
        provided the add button becomes a dropdown menu.

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

    right_buttons: list = []
    if on_add_callback is not None:
        right_buttons.append(
            _create_add_button_or_menu(
                on_add_callback=on_add_callback,
                on_overlay_selected=on_overlay_selected,
                available_overlays=available_overlays,
            )
        )
    right_buttons.extend([edit_button, toggle_button])

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
    window = getattr(params, 'window', None)
    if window is None:
        return ''
    if window.mode == WindowMode.since_start:
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

    Uses the source title when a single source is shared across all non-static
    layers (the common monitoring case). Otherwise falls back to the first
    layer's display title, suffixed with ``" (+N more layers)"`` for
    multi-layer cells.

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
    layers = cell.layers
    if not layers:
        return ''

    source_sets = [
        set(layer.config.source_names)
        for layer in layers
        if not layer.config.is_static()
    ]
    if source_sets:
        common = set.intersection(*source_sets)
        if len(common) == 1:
            source = next(iter(common))
            return get_source_title(source) if get_source_title else source

    first_title, _ = get_plot_cell_display_info(
        layers[0].config, workflow_registry, get_source_title
    )
    if len(layers) == 1:
        return first_title

    extra = len(layers) - 1
    suffix = 'layer' if extra == 1 else 'layers'
    return f'{first_title} (+{extra} more {suffix})'
