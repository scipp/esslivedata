# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Button and widget creation utilities for plot cells."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import panel as pn

from ...config.workflow_spec import WorkflowId, WorkflowSpec


@dataclass(frozen=True)
class ButtonStyles:
    """Styling constants for plot cell buttons."""

    # Colors
    PRIMARY_BLUE = '#007bff'
    DANGER_RED = '#dc3545'

    # Dimensions
    CELL_MARGIN = 2
    TOOL_BUTTON_SIZE = 28
    TOOL_BUTTON_TOP_OFFSET = '5px'
    CLOSE_BUTTON_RIGHT_OFFSET = '5px'
    TOOL_BUTTON_Z_INDEX = '1000'

    # Typography
    TOOL_BUTTON_FONT_SIZE = '20px'


def create_tool_button_stylesheet(button_color: str, hover_color: str) -> list[str]:
    """
    Create a stylesheet for tool buttons (close, gear, etc.).

    Parameters
    ----------
    button_color:
        Color for the button icon.
    hover_color:
        RGBA color for the hover background.

    Returns
    -------
    :
        List containing the stylesheet string.
    """
    return [
        f"""
        button {{
            background-color: transparent !important;
            border: none !important;
            color: {button_color} !important;
            font-weight: bold !important;
            font-size: {ButtonStyles.TOOL_BUTTON_FONT_SIZE} !important;
            padding: 0 !important;
            margin: 0 !important;
            line-height: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            height: 100% !important;
            width: 100% !important;
        }}
        button:hover {{
            background-color: {hover_color} !important;
        }}
        """
    ]


def create_tool_button(
    symbol: str,
    right_offset: str,
    button_color: str,
    hover_color: str,
    on_click_callback: Callable[[], None],
) -> pn.widgets.Button:
    """
    Create a styled tool button for plot cells.

    Parameters
    ----------
    symbol:
        Unicode symbol to display on the button.
    right_offset:
        CSS right offset for button positioning.
    button_color:
        Color for the button icon.
    hover_color:
        RGBA color for the hover background.
    on_click_callback:
        Callback function to invoke when the button is clicked.

    Returns
    -------
    :
        Panel Button widget styled as a tool button.
    """
    button = pn.widgets.Button(
        name=symbol,
        width=ButtonStyles.TOOL_BUTTON_SIZE,
        height=ButtonStyles.TOOL_BUTTON_SIZE,
        button_type='light',
        sizing_mode='fixed',
        margin=(ButtonStyles.CELL_MARGIN, ButtonStyles.CELL_MARGIN),
        styles={
            'position': 'absolute',
            'top': ButtonStyles.TOOL_BUTTON_TOP_OFFSET,
            'right': right_offset,
            'z-index': ButtonStyles.TOOL_BUTTON_Z_INDEX,
        },
        stylesheets=create_tool_button_stylesheet(button_color, hover_color),
    )
    button.on_click(lambda _: on_click_callback())
    return button


def create_close_button(on_close_callback: Callable[[], None]) -> pn.widgets.Button:
    """
    Create a styled close button for plot cells.

    Parameters
    ----------
    on_close_callback:
        Callback function to invoke when the button is clicked.

    Returns
    -------
    :
        Panel Button widget styled as a close button.
    """
    return create_tool_button(
        symbol='\u00d7',  # "X" multiplication sign
        right_offset=ButtonStyles.CLOSE_BUTTON_RIGHT_OFFSET,
        button_color=ButtonStyles.DANGER_RED,
        hover_color='rgba(220, 53, 69, 0.1)',
        on_click_callback=on_close_callback,
    )


def create_gear_button(on_gear_callback: Callable[[], None]) -> pn.widgets.Button:
    """
    Create a styled gear button for plot cells (configuration/settings).

    Parameters
    ----------
    on_gear_callback:
        Callback function to invoke when the button is clicked.

    Returns
    -------
    :
        Panel Button widget styled as a gear button.
    """
    return create_tool_button(
        symbol='\u2699',  # Gear symbol
        # Position to the left of the close button
        right_offset=f'{ButtonStyles.TOOL_BUTTON_SIZE + 10}px',
        button_color=ButtonStyles.PRIMARY_BLUE,
        hover_color='rgba(0, 123, 255, 0.1)',
        on_click_callback=on_gear_callback,
    )


def create_add_layer_button(
    on_add_layer_callback: Callable[[], None],
) -> pn.widgets.Button:
    """
    Create a styled button for adding a layer to a plot cell.

    Parameters
    ----------
    on_add_layer_callback:
        Callback function to invoke when the button is clicked.

    Returns
    -------
    :
        Panel Button widget styled as an add layer button.
    """
    return create_tool_button(
        symbol='\u002b',  # Plus sign
        # Position to the left of the gear button
        right_offset=f'{2 * ButtonStyles.TOOL_BUTTON_SIZE + 15}px',
        button_color='#28a745',  # Green color for add action
        hover_color='rgba(40, 167, 69, 0.1)',
        on_click_callback=on_add_layer_callback,
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
