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

if TYPE_CHECKING:
    from ..plot_orchestrator import PlotConfig


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
    button_color = '#28a745'
    hover_color = 'rgba(40, 167, 69, 0.1)'

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
        name='',
        items=items,
        icon=get_icon('plus'),
        icon_size='1.5em',
        width=ButtonStyles.TOOL_BUTTON_SIZE,
        height=ButtonStyles.TOOL_BUTTON_SIZE,
        button_type='light',
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


def create_cell_toolbar(
    *,
    on_gear_callback: Callable[[], None],
    on_close_callback: Callable[[], None],
    on_add_callback: Callable[[], None] | None = None,
    on_overlay_selected: Callable[[str, str], None] | None = None,
    available_overlays: list[tuple[str, str, str]] | None = None,
    title: str | None = None,
    description: str | None = None,
    stopped: bool = False,
) -> pn.Row:
    """
    Create a toolbar row containing title and buttons for plot cells.

    The toolbar displays an optional title on the left (with tooltip for
    description) and gear/add/close buttons on the right, using flexbox layout
    to avoid overlap with Bokeh's plot toolbar.

    When overlay suggestions are available, the add button becomes a dropdown
    menu with "Add layer..." (opens modal) and direct overlay options.

    Parameters
    ----------
    on_gear_callback:
        Callback function to invoke when the gear button is clicked.
    on_close_callback:
        Callback function to invoke when the close button is clicked.
    on_add_callback:
        Optional callback to invoke when the add button is clicked.
        If None, the add button is not shown. This allows the toolbar to be
        used in contexts where adding layers doesn't make sense (e.g., read-only
        views, cells with layer limits, or special cell types).
    on_overlay_selected:
        Optional callback when an overlay is selected from dropdown.
        Called with (output_name, plotter_name). Required if available_overlays
        is provided.
    available_overlays:
        Optional list of (output_name, plotter_name, plotter_title) tuples
        representing available overlay options for this layer. If provided,
        the add button becomes a dropdown menu.
    title:
        Optional title text to display on the left side of the toolbar.
    description:
        Optional description shown as tooltip when hovering over the title.
    stopped:
        If True, adds a visual indicator (border) showing workflow has ended.

    Returns
    -------
    :
        Panel Row widget containing the toolbar.
    """
    gear_button = create_tool_button(
        icon_name='settings',
        button_color=ButtonStyles.PRIMARY_BLUE,
        hover_color='rgba(0, 123, 255, 0.1)',
        on_click_callback=on_gear_callback,
    )

    add_button: pn.widgets.Button | pn.widgets.MenuButton | None = None
    if on_add_callback is not None:
        add_button = _create_add_button_or_menu(
            on_add_callback=on_add_callback,
            on_overlay_selected=on_overlay_selected,
            available_overlays=available_overlays,
        )

    close_button = create_tool_button(
        icon_name='x',
        button_color=ButtonStyles.DANGER_RED,
        hover_color='rgba(220, 53, 69, 0.1)',
        on_click_callback=on_close_callback,
    )

    # Build left content: title and optional tooltip icon
    left_items: list = []
    if title:
        title_html = pn.pane.HTML(
            f'<span style="font-size: 12px; color: #495057;">{title}</span>',
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
            tooltip = Tooltip(content=HTML(html_description), position='right')
            tooltip_icon = pn.widgets.TooltipIcon(
                value=tooltip,
                margin=0,
            )
            left_items.append(tooltip_icon)

    margin = ButtonStyles.CELL_MARGIN
    right_buttons = [gear_button]
    if add_button is not None:
        right_buttons.append(add_button)
    right_buttons.append(close_button)

    # Add border when workflow is stopped
    styles = {}
    if stopped:
        styles['border'] = '2px solid #495057'  # Dark grey border
        styles['border-radius'] = '4px'

    return pn.Row(
        *left_items,
        pn.Spacer(sizing_mode='stretch_width'),
        *right_buttons,
        sizing_mode='stretch_width',
        height=ButtonStyles.TOOL_BUTTON_SIZE,
        margin=(margin, margin, margin, margin),
        align='end',
        styles=styles,
    )


def get_workflow_display_info(
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
    workflow_id: WorkflowId,
    output_name: str | None,
) -> tuple[str, str]:
    """
    Look up workflow and output display titles from the registry.

    Parameters
    ----------
    workflow_registry:
        Registry mapping workflow IDs to their specifications.
    workflow_id:
        ID of the workflow to look up.
    output_name:
        Name of the output field, or None.

    Returns
    -------
    :
        Tuple of (workflow_title, output_title). If the workflow is not found,
        workflow_title falls back to str(workflow_id). If output_name is None
        or not found in the spec, output_title falls back to output_name or
        empty string.
    """
    workflow_spec = workflow_registry.get(workflow_id)

    # Get workflow title
    if workflow_spec is not None:
        workflow_title = workflow_spec.title
    else:
        workflow_title = str(workflow_id)

    # Get output title from spec if available
    output_title = output_name or ''
    if workflow_spec and workflow_spec.outputs and output_name:
        output_fields = workflow_spec.outputs.model_fields
        if output_name in output_fields:
            field_info = output_fields[output_name]
            output_title = field_info.title or output_name

    return workflow_title, output_title


def _format_window_info(params) -> str:
    """
    Format window parameters into a human-readable string.

    Parameters
    ----------
    params:
        Plotter params that may contain window settings.

    Returns
    -------
    :
        Formatted string like "current" or "10s average".
    """
    window = getattr(params, 'window', None)
    if window is None:
        return ''

    if window.mode == WindowMode.latest:
        return 'latest'

    # Window mode with duration
    duration = window.window_duration_seconds
    if duration == int(duration):
        duration_str = f'{int(duration)}s'
    else:
        duration_str = f'{duration:.1f}s'

    # Include aggregation if not 'auto'
    if window.aggregation != 'auto':
        agg_display = window.aggregation.replace('nan', '')
        return f'{duration_str} {agg_display}'

    return f'{duration_str} window'


def _get_static_overlay_display_info(config: PlotConfig) -> tuple[str, str]:
    """Get display info for a static overlay layer."""
    from ..plotter_registry import plotter_registry

    plotter_name = config.plot_name
    try:
        spec = plotter_registry.get_spec(plotter_name)
        plotter_title = spec.title
    except KeyError:
        plotter_title = plotter_name.replace('_', ' ').title()

    # Use the user's custom name from output_name
    custom_name = config.output_name

    # Format title as "Plotter → Custom Name"
    title = f'{plotter_title} &rarr; {custom_name}'

    # Build description for tooltip
    description = f'Static overlay: {plotter_title}\nName: {custom_name}'

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
        workflow_registry, config.workflow_id, config.output_name
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

    description = '\n'.join(description_parts)

    return title, description
