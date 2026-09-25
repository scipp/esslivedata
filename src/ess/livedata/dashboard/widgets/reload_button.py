# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Button reloading the page, for screens without browser controls."""

from typing import ClassVar

from panel.reactive import ReactiveHTML

from .icons import get_icon_data_uri

_ICON_MASK = (
    f"url('{get_icon_data_uri('refresh', color=None)}') center / contain no-repeat"
)


class ReloadButton(ReactiveHTML):
    """Button that reloads the page in the browser.

    A web page added to an iPhone home screen opens without the browser's
    toolbar, so there is no reload button and no pull-to-refresh. After the
    connection to the server drops, the page is dead until it is reloaded; this
    button is the only way to do that short of closing the app.

    The reload runs in the browser alone: the button is needed exactly when the
    server can no longer be reached, so it must not route through Python.

    The button floats in a fixed corner of the window rather than taking space
    in the layout. Where it goes is up to the caller, through ``styles``, and so
    are its colors, through ``stylesheets`` rules on ``button``: the icon paints
    in the button's text color.
    """

    # ``${{script(...)}}`` is ReactiveHTML's hook into ``_scripts``, doubled for
    # the f-string.
    _template = f"""
        <button id="reload" type="button" title="Reload page"
            aria-label="Reload page" onclick="${{script('reload')}}"
            style="width: 100%; height: 100%; margin: 0; padding: 0;
                border: none; border-radius: 3px; cursor: pointer;
                display: flex; align-items: center; justify-content: center;">
            <span style="width: 24px; height: 24px;
                background-color: currentColor;
                mask: {_ICON_MASK}; -webkit-mask: {_ICON_MASK};">
            </span>
        </button>
    """

    _scripts: ClassVar = {'reload': 'window.location.reload();'}
