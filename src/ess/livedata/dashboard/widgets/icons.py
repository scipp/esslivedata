# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Embedded SVG icons for the dashboard.

Icons are from Tabler Icons (https://tabler.io/icons), MIT licensed.
They are embedded as strings to work in air-gapped environments where
users may not have internet access. Panel's default icon loading fetches
from CDN at runtime in the browser, which fails offline.

To add a new icon: download from https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline/{name}.svg,
copy the SVG path elements, and add an entry to the _ICONS dict using the _svg() helper.

Usage:
    from ess.livedata.dashboard.widgets.icons import get_icon

    button = pn.widgets.Button(name='', icon=get_icon('settings'))
"""

from __future__ import annotations

# Common SVG attributes used by all Tabler icons
_SVG_ATTRS = (
    'xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
    'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
)


def _svg(paths: str) -> str:
    """Wrap path elements in SVG tag with standard Tabler attributes."""
    return f'<svg {_SVG_ATTRS}>{paths}</svg>'


# Icon registry: name -> SVG content
# All icons use stroke="currentColor" so color can be set via CSS
_ICONS: dict[str, str] = {
    # Close/dismiss (X)
    'x': _svg('<path d="M18 6l-12 12"/><path d="M6 6l12 12"/>'),
    # Settings/gear
    'settings': _svg(
        '<path d="M10.325 4.317c.426 -1.756 2.924 -1.756 3.35 0a1.724 1.724 0 0 0 '
        '2.573 1.066c1.543 -.94 3.31 .826 2.37 2.37a1.724 1.724 0 0 0 1.065 2.572c'
        '1.756 .426 1.756 2.924 0 3.35a1.724 1.724 0 0 0 -1.066 2.573c.94 1.543 '
        '-.826 3.31 -2.37 2.37a1.724 1.724 0 0 0 -2.572 1.065c-.426 1.756 -2.924 '
        '1.756 -3.35 0a1.724 1.724 0 0 0 -2.573 -1.066c-1.543 .94 -3.31 -.826 '
        '-2.37 -2.37a1.724 1.724 0 0 0 -1.065 -2.572c-1.756 -.426 -1.756 -2.924 0 '
        '-3.35a1.724 1.724 0 0 0 1.066 -2.573c-.94 -1.543 .826 -3.31 2.37 -2.37c1 '
        '.608 2.296 .07 2.572 -1.065z"/>'
        '<path d="M9 12a3 3 0 1 0 6 0a3 3 0 0 0 -6 0"/>'
    ),
    # Add/plus
    'plus': _svg('<path d="M12 5l0 14"/><path d="M5 12l14 0"/>'),
    # Download
    'download': _svg(
        '<path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2 -2v-2"/>'
        '<path d="M7 11l5 5l5 -5"/>'
        '<path d="M12 4l0 12"/>'
    ),
    # Play
    'player-play': _svg('<path d="M7 4v16l13 -8z"/>'),
    # Stop (rounded square)
    'player-stop': _svg(
        '<path d="M5 5m0 2a2 2 0 0 1 2 -2h10a2 2 0 0 1 2 2v10a2 2 0 0 1 -2 2h-10a2 '
        '2 0 0 1 -2 -2z"/>'
    ),
    # Pause
    'player-pause': _svg(
        '<path d="M6 5m0 1a1 1 0 0 1 1 -1h2a1 1 0 0 1 1 1v12a1 1 0 0 1 -1 1h-2a1 1 '
        '0 0 1 -1 -1z"/>'
        '<path d="M14 5m0 1a1 1 0 0 1 1 -1h2a1 1 0 0 1 1 1v12a1 1 0 0 1 -1 1h-2a1 '
        '1 0 0 1 -1 -1z"/>'
    ),
    # Backspace (reset/clear)
    'backspace': _svg(
        '<path d="M20 6a1 1 0 0 1 1 1v10a1 1 0 0 1 -1 1h-11l-5 -5a1.5 1.5 0 0 1 0 -2l5 '
        '-5l11 0"/>'
        '<path d="M12 10l4 4m0 -4l-4 4"/>'
    ),
    # Refresh/reset
    'refresh': _svg(
        '<path d="M20 11a8.1 8.1 0 0 0 -15.5 -2m-.5 -4v4h4"/>'
        '<path d="M4 13a8.1 8.1 0 0 0 15.5 2m.5 4v-4h-4"/>'
    ),
    # Chevron down (expand)
    'chevron-down': _svg('<path d="M6 9l6 6l6 -6"/>'),
    # Chevron right (collapse)
    'chevron-right': _svg('<path d="M9 6l6 6l-6 6"/>'),
    # Trash/delete
    'trash': _svg(
        '<path d="M4 7l16 0"/>'
        '<path d="M10 11l0 6"/>'
        '<path d="M14 11l0 6"/>'
        '<path d="M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2 -2l1 -12"/>'
        '<path d="M9 7v-3a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v3"/>'
    ),
}


def get_icon(name: str) -> str:
    """
    Get embedded SVG for an icon by name.

    Parameters
    ----------
    name:
        Icon name. Available icons: backspace, chevron-down, chevron-right,
        download, player-pause, player-play, player-stop, plus, refresh,
        settings, trash, x.

    Returns
    -------
    :
        SVG string that can be used with Panel's icon parameter.

    Raises
    ------
    KeyError
        If the icon name is not found.
    """
    if name not in _ICONS:
        available = ', '.join(sorted(_ICONS.keys()))
        raise KeyError(f"Unknown icon '{name}'. Available icons: {available}")
    return _ICONS[name]


def list_icons() -> list[str]:
    """Return a list of all available icon names."""
    return sorted(_ICONS.keys())
