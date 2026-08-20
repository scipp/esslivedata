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

# The rail has to bleed into the window corners for the teal to read as chrome
# rather than as a floating panel, so the theme takes the main area's padding
# away from Panel and re-applies it inside the tab strip, around the content
# only. Same spacing as Material's own (``.main-content`` in
# ``panel/template/material/material.css``); it just moves.
_MAIN_CONTENT_PADDING = '10px 20px 20px 10px'

_NICOS_TEMPLATE_CSS = """
    .main-content {
        padding: 0;
    }
"""

# Selecting the panel a tab shows is done by exclusion, because Bokeh 3.9 gives
# it nothing to select: ``TabsView`` appends each child view's own element to
# the shadow root and toggles visibility on it, so there is no ``.bk-panel``
# wrapper to hold the padding (a later Bokeh reintroduces one -- prefer it when
# it lands). The children auto-place into the ``stack`` grid area left by the
# header; ``box-sizing`` is what makes the padding shrink the child rather than
# push it off the right edge.
#
# Panel's Material design styles tabs through ``:host(.bk-<side>) .bk-header
# .bk-tab``, painting them transparent with a colored bottom/side border. Match
# that selector so these rules are not simply outranked, and mark the properties
# it also sets, since the design stylesheet may be applied after this one.
#
# Tabs are laid out in a stretch-aligned column, so each one's *margin* box
# spans the strip: a right margin on the inactive tabs (and none on the active
# one) leaves the strip's teal showing as the hairline that separates them from
# the content area, while the selected tab runs into it. That is NICOS's cue for
# which tab owns the panel, and it is why the margins are not shorthand.
_NICOS_TAB_CSS = f"""
    :host(.bk-left) .bk-header {{
        background-color: {_NICOS_TEAL};
        padding: 0 0 8px 8px;
        border-right: none;
    }}
    :host(.bk-left) > :not(.bk-header):not(style):not(link) {{
        box-sizing: border-box;
        padding: {_MAIN_CONTENT_PADDING};
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
        margin-top: 0 !important;
        margin-bottom: 3px !important;
        margin-left: 0 !important;
        margin-right: 1px !important;
    }}
    :host(.bk-left) .bk-header .bk-tab:hover {{
        background: {_NICOS_TAB_HOVER_BG} !important;
    }}
    :host(.bk-left) .bk-header .bk-tab.bk-active {{
        background: white !important;
        border-left-color: {_NICOS_BLUE} !important;
        font-weight: bold;
        margin-right: 0 !important;
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
    template_css: str = ''
    """Rules for the page around the tabs, injected as the template's raw CSS."""


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
    template_css=_NICOS_TEMPLATE_CSS,
)

THEMES = {theme.name: theme for theme in (DEFAULT_THEME, NICOS_THEME)}
