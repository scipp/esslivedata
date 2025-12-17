# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Widget creation utilities for plot cells."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import panel as pn

from ...config.workflow_spec import WorkflowId, WorkflowSpec
from ..buttons import ButtonStyles, create_tool_button
from ..plot_params import WindowMode

if TYPE_CHECKING:
    from ..plot_orchestrator import PlotConfig


def create_cell_toolbar(
    *,
    on_gear_callback: Callable[[], None],
    on_close_callback: Callable[[], None],
    on_add_callback: Callable[[], None] | None = None,
    title: str | None = None,
    description: str | None = None,
) -> pn.Row:
    """
    Create a toolbar row containing title and buttons for plot cells.

    The toolbar displays an optional title on the left (with tooltip for
    description) and gear/add/close buttons on the right, using flexbox layout
    to avoid overlap with Bokeh's plot toolbar.

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
    title:
        Optional title text to display on the left side of the toolbar.
    description:
        Optional description shown as tooltip when hovering over the title.

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
    add_button = (
        create_tool_button(
            icon_name='plus',
            button_color='#28a745',  # Green
            hover_color='rgba(40, 167, 69, 0.1)',
            on_click_callback=on_add_callback,
        )
        if on_add_callback
        else None
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

    return pn.Row(
        *left_items,
        pn.Spacer(sizing_mode='stretch_width'),
        *right_buttons,
        sizing_mode='stretch_width',
        height=ButtonStyles.TOOL_BUTTON_SIZE,
        margin=(margin, margin, 0, margin),
        align='end',
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


def get_plot_cell_display_info(
    config: PlotConfig,
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
) -> tuple[str, str]:
    """
    Assemble title and description for a plot cell toolbar.

    Parameters
    ----------
    config:
        Plot configuration containing workflow, output, and params.
    workflow_registry:
        Registry mapping workflow IDs to their specifications.

    Returns
    -------
    :
        Tuple of (title, description). Title is a short string for display,
        description is a longer string for tooltip.
    """
    workflow_title, output_title = get_workflow_display_info(
        workflow_registry, config.workflow_id, config.output_name
    )

    # Build title: "Workflow → Output (source, window)"
    # Using HTML entity for arrow since title is rendered in HTML pane
    window_info = _format_window_info(config.params)

    # Format source info: name if single, count if multiple
    num_sources = len(config.source_names)
    if num_sources == 1:
        source_info = config.source_names[0]
    else:
        source_info = f'{num_sources} sources'

    # Combine source and window info in parentheses
    detail_parts = [source_info]
    if window_info:
        detail_parts.append(window_info)
    details = ', '.join(detail_parts)

    title = f'{workflow_title} &rarr; {output_title} ({details})'

    # Build description for tooltip
    sources_str = ', '.join(config.source_names)
    description_parts = [
        f'Workflow: {workflow_title}',
        f'Output: {output_title}',
        f'Sources: {sources_str}',
    ]
    if window_info:
        description_parts.append(f'Window: {window_info}')

    description = '\n'.join(description_parts)

    return title, description
