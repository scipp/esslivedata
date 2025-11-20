# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Button and widget creation utilities for plot cells."""

from __future__ import annotations

from collections.abc import Callable

import panel as pn

from .plot_grid import _CellStyles


def _create_tool_button_stylesheet(button_color: str, hover_color: str) -> list[str]:
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
            font-size: {_CellStyles.TOOL_BUTTON_FONT_SIZE} !important;
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


def _create_tool_button(
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
        width=_CellStyles.TOOL_BUTTON_SIZE,
        height=_CellStyles.TOOL_BUTTON_SIZE,
        button_type='light',
        sizing_mode='fixed',
        margin=(_CellStyles.CELL_MARGIN, _CellStyles.CELL_MARGIN),
        styles={
            'position': 'absolute',
            'top': _CellStyles.TOOL_BUTTON_TOP_OFFSET,
            'right': right_offset,
            'z-index': _CellStyles.TOOL_BUTTON_Z_INDEX,
        },
        stylesheets=_create_tool_button_stylesheet(button_color, hover_color),
    )
    button.on_click(lambda _: on_click_callback())
    return button


def _create_close_button(on_close_callback: Callable[[], None]) -> pn.widgets.Button:
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
    return _create_tool_button(
        symbol='\u00d7',  # "X" multiplication sign
        right_offset=_CellStyles.CLOSE_BUTTON_RIGHT_OFFSET,
        button_color=_CellStyles.DANGER_RED,
        hover_color='rgba(220, 53, 69, 0.1)',
        on_click_callback=on_close_callback,
    )


def _create_gear_button(on_gear_callback: Callable[[], None]) -> pn.widgets.Button:
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
    return _create_tool_button(
        symbol='\u2699',  # Gear symbol
        # Position to the left of the close button
        right_offset=f'{_CellStyles.TOOL_BUTTON_SIZE + 10}px',
        button_color=_CellStyles.PRIMARY_BLUE,
        hover_color='rgba(0, 123, 255, 0.1)',
        on_click_callback=on_gear_callback,
    )
