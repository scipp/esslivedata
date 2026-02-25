# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Button creation utilities for the dashboard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import panel as pn

from .icons import get_icon


@dataclass(frozen=True)
class ButtonStyles:
    """Styling constants for tool buttons."""

    # Colors
    PRIMARY_BLUE = '#007bff'
    DANGER_RED = '#dc3545'

    # Dimensions
    CELL_MARGIN = 2
    TOOL_BUTTON_SIZE = 28


def create_tool_button_stylesheet(
    button_color: str,
    hover_color: str,
    *,
    selector: str = 'button',
    hover_selector: str | None = None,
    include_anchor_styles: bool = False,
) -> list[str]:
    """
    Create a stylesheet for tool buttons (close, gear, etc.).

    Parameters
    ----------
    button_color:
        Color for the button icon.
    hover_color:
        RGBA color for the hover background.
    selector:
        CSS selector for the button element. Default is 'button' for regular
        Panel buttons. Use ':host(.solid) button.bk-btn.bk-btn-primary' for
        FileDownload widgets.
    hover_selector:
        CSS selector for the hover state. If None, defaults to '{selector}:hover'.
    include_anchor_styles:
        Whether to include anchor element styling. Set to True for FileDownload
        widgets which have an anchor inside the button.

    Returns
    -------
    :
        List containing the stylesheet string.
    """
    if hover_selector is None:
        hover_selector = f'{selector}:hover'

    anchor_styles = (
        f"""
        /* FileDownload anchor needs to fill button and center icon */
        {selector} a {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            height: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
            text-decoration: none !important;
            font-size: 0 !important;  /* Hide label text and separator */
            line-height: 0 !important;
        }}
        /* Icon wrapper inside anchor */
        {selector} a > * {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            margin: 0 !important;
            line-height: 0 !important;
        }}
        """
        if include_anchor_styles
        else ''
    )

    return [
        f"""
        {selector} {{
            background-color: transparent !important;
            border: none !important;
            color: {button_color} !important;
            padding: 0 !important;
            margin: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            height: 100% !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }}
        {selector} > * {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }}
        {selector} svg {{
            display: block !important;
            flex-shrink: 0 !important;
        }}{anchor_styles}
        {hover_selector} {{
            background-color: {hover_color} !important;
        }}
        """
    ]


def create_tool_button(
    icon_name: str,
    button_color: str,
    hover_color: str,
    on_click_callback: Callable[[], None],
) -> pn.widgets.Button:
    """
    Create a styled tool button.

    Parameters
    ----------
    icon_name:
        Name of the icon to display (from icons module).
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
        name='',
        icon=get_icon(icon_name),
        icon_size='1.5em',
        width=ButtonStyles.TOOL_BUTTON_SIZE,
        height=ButtonStyles.TOOL_BUTTON_SIZE,
        button_type='light',
        sizing_mode='fixed',
        margin=0,
        stylesheets=create_tool_button_stylesheet(button_color, hover_color),
    )
    button.on_click(lambda _: on_click_callback())
    return button


def create_download_button(
    filename: str,
    callback: Callable[[], str],
) -> pn.widgets.FileDownload:
    """
    Create a styled download button for exporting data.

    Parameters
    ----------
    filename:
        Default filename for the downloaded file.
    callback:
        Callback that returns the file content as a string.

    Returns
    -------
    :
        Panel FileDownload widget styled as a tool button.
    """
    # FileDownload uses different CSS selectors than regular Button
    # and has an anchor element inside that needs centering
    stylesheet = create_tool_button_stylesheet(
        button_color=ButtonStyles.PRIMARY_BLUE,
        hover_color='rgba(0, 123, 255, 0.1)',
        selector=':host(.solid) button.bk-btn.bk-btn-primary',
        include_anchor_styles=True,
    )

    return pn.widgets.FileDownload(
        callback=callback,
        filename=filename,
        # Use a non-breaking space as label - empty string causes Panel to
        # fall back to showing the filename as the button text
        label='\u00a0',
        icon=get_icon('download'),
        width=ButtonStyles.TOOL_BUTTON_SIZE,
        height=ButtonStyles.TOOL_BUTTON_SIZE,
        button_type='primary',
        sizing_mode='fixed',
        margin=0,
        embed=False,  # Generate content on click, not upfront
        stylesheets=stylesheet,
    )
