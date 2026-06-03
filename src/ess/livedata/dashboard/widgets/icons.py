# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Embedded SVG icons for the dashboard.

Most icons are taken verbatim from Tabler Icons (https://tabler.io/icons),
MIT licensed. They are embedded as strings to work in air-gapped environments
where users may not have internet access. Panel's default icon loading fetches
from CDN at runtime in the browser, which fails offline.

The ``autoscale-{axis}`` toggle icons are Tabler's ``arrow-autofit-content``
(x, and y rotated 90°) and ``contrast-2`` (color axis), each with an outline
variant (autoscale off) and a solid ``-on`` variant (autoscale on).

To add a new icon: download from https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline/{name}.svg,
copy the SVG path elements, and add an entry to the _ICONS dict using the _svg() helper.

Usage:
    from ess.livedata.dashboard.widgets.icons import get_icon

    button = pn.widgets.Button(label='', icon=get_icon('settings'))
"""

from __future__ import annotations

import base64

# Common SVG attributes used by all Tabler icons
_SVG_ATTRS = (
    'xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
    'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
)


def _svg(paths: str) -> str:
    """Wrap path elements in SVG tag with standard Tabler attributes."""
    return f'<svg {_SVG_ATTRS}>{paths}</svg>'


# Filled Tabler icons paint with ``fill`` and carry no stroke.
_SVG_ATTRS_FILLED = (
    'xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
    'viewBox="0 0 24 24" fill="currentColor"'
)


def _svg_filled(paths: str) -> str:
    """Wrap path elements in SVG tag for solid (filled) Tabler icons."""
    return f'<svg {_SVG_ATTRS_FILLED}>{paths}</svg>'


def _rotate90(paths: str) -> str:
    """Rotate path elements 90° about the 24x24 viewBox centre."""
    return f'<g transform="rotate(90 12 12)">{paths}</g>'


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
    # Chevron up
    'chevron-up': _svg('<path d="M6 15l6 -6l6 6"/>'),
    # Chevron down (expand)
    'chevron-down': _svg('<path d="M6 9l6 6l6 -6"/>'),
    # Chevron right (collapse)
    'chevron-right': _svg('<path d="M9 6l6 6l-6 6"/>'),
    # Pencil (edit/rename)
    'pencil': _svg(
        '<path d="M4 20h4l10.5 -10.5a2.828 2.828 0 1 0 -4 -4l-10.5 10.5v4"/>'
        '<path d="M13.5 6.5l4 4"/>'
    ),
    # Eye (visible/enabled)
    'eye': _svg(
        '<path d="M10 12a2 2 0 1 0 4 0a2 2 0 0 0 -4 0"/>'
        '<path d="M21 12c-2.4 4 -5.4 6 -9 6c-3.6 0 -6.6 -2 -9 -6c2.4 -4 5.4 -6 9 '
        '-6c3.6 0 6.6 2 9 6"/>'
    ),
    # Eye off (hidden/disabled)
    'eye-off': _svg(
        '<path d="M10.585 10.587a2 2 0 0 0 2.829 2.828"/>'
        '<path d="M16.681 16.673a8.717 8.717 0 0 1 -4.681 1.327c-3.6 0 -6.6 -2 -9 '
        '-6c1.272 -2.12 2.712 -3.678 4.32 -4.674m2.86 -1.146a9.354 9.354 0 0 1 1.82 '
        '-.18c3.6 0 6.6 2 9 6c-.666 1.11 -1.379 2.067 -2.138 2.87"/>'
        '<path d="M3 3l18 18"/>'
    ),
    # Check (confirm)
    'check': _svg('<path d="M5 12l5 5l10 -10"/>'),
    # Adjustments/sliders (layer controls visible)
    'adjustments': _svg(
        '<path d="M4 10a2 2 0 1 0 4 0a2 2 0 0 0 -4 0"/>'
        '<path d="M6 4v4"/>'
        '<path d="M6 12v8"/>'
        '<path d="M10 16a2 2 0 1 0 4 0a2 2 0 0 0 -4 0"/>'
        '<path d="M12 4v10"/>'
        '<path d="M12 18v2"/>'
        '<path d="M16 7a2 2 0 1 0 4 0a2 2 0 0 0 -4 0"/>'
        '<path d="M18 4v1"/>'
        '<path d="M18 9v11"/>'
    ),
    # Adjustments off (layer controls hidden): sliders with a diagonal strike
    'adjustments-off': _svg(
        '<path d="M4 10a2 2 0 1 0 4 0a2 2 0 0 0 -4 0"/>'
        '<path d="M6 4v4"/>'
        '<path d="M6 12v8"/>'
        '<path d="M10 16a2 2 0 1 0 4 0a2 2 0 0 0 -4 0"/>'
        '<path d="M12 4v10"/>'
        '<path d="M12 18v2"/>'
        '<path d="M16 7a2 2 0 1 0 4 0a2 2 0 0 0 -4 0"/>'
        '<path d="M18 4v1"/>'
        '<path d="M18 9v11"/>'
        '<path d="M3 3l18 18"/>'
    ),
    # Trash/delete
    'trash': _svg(
        '<path d="M4 7l16 0"/>'
        '<path d="M10 11l0 6"/>'
        '<path d="M14 11l0 6"/>'
        '<path d="M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2 -2l1 -12"/>'
        '<path d="M9 7v-3a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v3"/>'
    ),
    # Arrows-minimize (fit data to viewport)
    'arrows-minimize': _svg(
        '<path d="M5 9l4 0l0 -4"/>'
        '<path d="M3 3l6 6"/>'
        '<path d="M5 15l4 0l0 4"/>'
        '<path d="M3 21l6 -6"/>'
        '<path d="M19 9l-4 0l0 -4"/>'
        '<path d="M15 9l6 -6"/>'
        '<path d="M19 15l-4 0l0 4"/>'
        '<path d="M15 15l6 6"/>'
    ),
}


# Per-axis autoscale-toggle icons. Each toggle has two states, distinguished by
# outline (``autoscale-{axis}``, autoscale off / range frozen) vs solid fill
# (``autoscale-{axis}-on``, autoscale on). x and y use Tabler's
# ``arrow-autofit-content``; y is the same glyph rotated 90°. The color axis (c)
# uses ``contrast-2``, whose split square reads as a heatmap cell.
_AUTOFIT_CONTENT = (  # Tabler ``arrow-autofit-content``
    '<path d="M6 4l-3 3l3 3"/>'
    '<path d="M18 4l3 3l-3 3"/>'
    '<path d="M4 16a2 2 0 0 1 2 -2h12a2 2 0 0 1 2 2v2a2 2 0 0 1 -2 2h-12a2 2 0 0 '
    '1 -2 -2l0 -2"/>'
    '<path d="M10 7h-7"/>'
    '<path d="M21 7h-7"/>'
)
_AUTOFIT_CONTENT_FILLED = (  # Tabler ``arrow-autofit-content`` (filled)
    '<path d="M6.707 3.293a1 1 0 0 1 .083 1.32l-.083 .094l-1.292 1.293h4.585a1 1 0 '
    '0 1 .117 1.993l-.117 .007h-4.585l1.292 1.293a1 1 0 0 1 .083 1.32l-.083 .094a1 '
    '1 0 0 1 -1.32 .083l-.094 -.083l-3 -3a1.008 1.008 0 0 1 -.097 -.112l-.071 -.11l'
    '-.054 -.114l-.035 -.105l-.025 -.118l-.007 -.058l-.004 -.09l.003 -.075l.017 '
    '-.126l.03 -.111l.044 -.111l.052 -.098l.064 -.092l.083 -.094l3 -3a1 1 0 0 1 '
    '1.414 0z"/>'
    '<path d="M18.613 3.21l.094 .083l3 3a.927 .927 0 0 1 .097 .112l.071 .11l.054 '
    '.114l.035 .105l.03 .148l.006 .118l-.003 .075l-.017 .126l-.03 .111l-.044 .111l'
    '-.052 .098l-.074 .104l-.073 .082l-3 3a1 1 0 0 1 -1.497 -1.32l.083 -.094l1.292 '
    '-1.293h-4.585a1 1 0 0 1 -.117 -1.993l.117 -.007h4.585l-1.292 -1.293a1 1 0 0 1 '
    '-.083 -1.32l.083 -.094a1 1 0 0 1 1.32 -.083z"/>'
    '<path d="M18 13h-12a3 3 0 0 0 -3 3v2a3 3 0 0 0 3 3h12a3 3 0 0 0 3 -3v-2a3 3 0 '
    '0 0 -3 -3z"/>'
)
_CONTRAST_2 = (  # Tabler ``contrast-2``
    '<path d="M3 5a2 2 0 0 1 2 -2h14a2 2 0 0 1 2 2v14a2 2 0 0 1 -2 2h-14a2 2 0 0 1 '
    '-2 -2l0 -14"/>'
    '<path d="M3 19h2.25c3.728 0 6.75 -3.134 6.75 -7s3.022 -7 6.75 -7h2.25"/>'
)
_CONTRAST_2_FILLED = (  # Tabler ``contrast-2`` (filled)
    '<path d="M19 2a3 3 0 0 1 3 3v14a3 3 0 0 1 -3 3h-14a3 3 0 0 1 -3 -3v-14a3 3 0 0 '
    '1 3 -3zm0 2h-14a1 1 0 0 0 -1 1v14a1 1 0 0 0 .769 .973c3.499 -.347 7.082 -4.127 '
    '7.226 -7.747l.005 -.226c0 -3.687 3.66 -7.619 7.232 -7.974a1 1 0 0 0 -.232 '
    '-.026"/>'
)
_ICONS['autoscale-x'] = _svg(_AUTOFIT_CONTENT)
_ICONS['autoscale-x-on'] = _svg_filled(_AUTOFIT_CONTENT_FILLED)
_ICONS['autoscale-y'] = _svg(_rotate90(_AUTOFIT_CONTENT))
_ICONS['autoscale-y-on'] = _svg_filled(_rotate90(_AUTOFIT_CONTENT_FILLED))
_ICONS['autoscale-c'] = _svg(_CONTRAST_2)
_ICONS['autoscale-c-on'] = _svg_filled(_CONTRAST_2_FILLED)


def get_icon(name: str) -> str:
    """
    Get embedded SVG for an icon by name.

    Parameters
    ----------
    name:
        Icon name. Available icons: adjustments, adjustments-off,
        arrows-minimize, autoscale-c, autoscale-c-on, autoscale-x,
        autoscale-x-on, autoscale-y, autoscale-y-on, backspace, check,
        chevron-down, chevron-right, chevron-up, download, eye, eye-off, pencil,
        player-pause, player-play, player-stop, plus, refresh, settings, trash,
        x.

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


BOKEH_TOOLBAR_ICON_COLOR = '#a1a6a9'
"""Stroke color of Bokeh's built-in toolbar tool icons.

This is Bokeh's ``--bokeh-icon-color`` default (the ``--icon-color`` CSS
variable fallback in BokehJS' ``:host`` block).

Data-URI icons render as ``<img>`` rather than inline SVG, so
``stroke="currentColor"`` does not inherit the surrounding text color and
falls back to black. Substituting this value matches Bokeh's own toolbar
icons visually.
"""


def get_icon_data_uri(name: str, *, color: str = BOKEH_TOOLBAR_ICON_COLOR) -> str:
    """Return an icon as a ``data:image/svg+xml;base64,...`` URI.

    Useful for Bokeh ``CustomAction(icon=...)``, which accepts a data URI with
    an ``image/*`` MIME type but not a raw SVG string. ``color`` replaces the
    SVG's ``currentColor`` -- on the ``stroke`` for outline icons and on the
    ``fill`` for solid icons -- so the icon renders in a fixed color rather than
    falling back to black.
    """
    svg = (
        get_icon(name)
        .replace('stroke="currentColor"', f'stroke="{color}"')
        .replace('fill="currentColor"', f'fill="{color}"')
    )
    encoded = base64.b64encode(svg.encode('utf-8')).decode('ascii')
    return f'data:image/svg+xml;base64,{encoded}'
