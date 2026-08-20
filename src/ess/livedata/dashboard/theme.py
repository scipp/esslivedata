# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Selectable look and feel for the dashboard shell.

The dashboard runs on the same screens as the NICOS client, so an operator's
eye moves between the two all day. The ``nicos`` theme is an experiment in
making that move cheaper: it adopts NICOS's dark teal chrome and moves the main
tab strip from the top edge to a left rail, matching where NICOS puts its own
panel selector.

A theme covers the shell only -- header color, tab placement, tab strip
palette. Widget-level colors (:mod:`~.widgets.styles`), plot colormaps and the
Panel/Material design are deliberately outside its reach: those are shared with
the plots and carry semantics (status greens and reds) that must not shift with
the surrounding chrome.
"""

from dataclasses import dataclass
from typing import Literal

from .widgets.styles import Colors

_DEFAULT_TAB_CSS = f"""
    .bk-tab {{
        border-bottom: 1px solid {Colors.TAB_BORDER} !important;
    }}
    .bk-tab.bk-active {{
        background-color: {Colors.TAB_ACTIVE_BG} !important;
        border: 1px solid {Colors.TAB_BORDER} !important;
        border-bottom: none !important;
    }}
"""

# Sampled from a NICOS client screenshot: teal chrome, light grey tabs, white
# selected tab with a bright blue marker.
_NICOS_TEAL = '#1f5366'
_NICOS_BLUE = '#0094ca'
_NICOS_TAB_BG = '#d9d9d9'
_NICOS_TAB_HOVER_BG = '#e9eaea'

# Panel's Material design styles tabs through ``:host(.bk-<side>) .bk-header
# .bk-tab``, painting them transparent with a colored bottom/side border. Match
# that selector so these rules are not simply outranked, and mark the properties
# it also sets, since the design stylesheet may be applied after this one.
_NICOS_TAB_CSS = f"""
    :host(.bk-left) .bk-header {{
        background-color: {_NICOS_TEAL};
        padding: 8px 0 8px 8px;
        border-right: none;
    }}
    :host(.bk-left) .bk-header .bk-tab {{
        background: {_NICOS_TAB_BG} !important;
        color: {_NICOS_TEAL} !important;
        text-align: left;
        padding: 10px 14px !important;
        min-width: 150px;
        border-style: solid !important;
        border-width: 0 0 0 4px !important;
        border-color: transparent !important;
        border-radius: 3px 0 0 3px;
        margin: 0 0 3px 0 !important;
    }}
    :host(.bk-left) .bk-header .bk-tab:hover {{
        background: {_NICOS_TAB_HOVER_BG} !important;
    }}
    :host(.bk-left) .bk-header .bk-tab.bk-active {{
        background: white !important;
        border-left-color: {_NICOS_BLUE} !important;
        font-weight: bold;
    }}
"""


@dataclass(frozen=True)
class Theme:
    """Shell appearance: chrome color and main tab placement."""

    name: str
    header_background: str
    tabs_location: Literal['above', 'left']
    """Edge the tab strip sits on. Extend only together with matching CSS."""
    tab_strip_css: str
    """Rules layered over Bokeh's tab CSS, valid for ``tabs_location``."""


DEFAULT_THEME = Theme(
    name='default',
    header_background='#2596be',
    tabs_location='above',
    tab_strip_css=_DEFAULT_TAB_CSS,
)

NICOS_THEME = Theme(
    name='nicos',
    header_background=_NICOS_TEAL,
    tabs_location='left',
    tab_strip_css=_NICOS_TAB_CSS,
)

THEMES = {theme.name: theme for theme in (DEFAULT_THEME, NICOS_THEME)}
