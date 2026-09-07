# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Selectable look and feel for the dashboard shell.

The dashboard runs on the same screens as the NICOS client, so an operator's
eye moves between the two all day. The default ``nicos`` theme makes that move
cheaper: it adopts NICOS's dark teal chrome and puts the main tab strip in a
left rail, where NICOS puts its own panel selector. ``classic`` is the look
that preceded it -- Panel Material's, with the tabs along the top.

A theme covers the shell only -- header color, tab placement, tab strip
palette. Widget-level colors (:mod:`~.widgets.styles`), plot colormaps and the
Panel/Material design are deliberately outside its reach: those are shared with
the plots and carry semantics (status greens and reds) that must not shift with
the surrounding chrome.
"""

import colorsys
from dataclasses import dataclass
from typing import Literal

from .widgets.styles import Colors


def _lighten(color: str, amount: float) -> str:
    """Raise a ``#rrggbb`` color's lightness, keeping its hue and saturation."""
    r, g, b = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    rgb = colorsys.hls_to_rgb(hue, min(1.0, lightness + amount), saturation)
    return '#' + ''.join(f'{round(c * 255):02x}' for c in rgb)


# How far a floating window's title bar is lifted off the chrome color. Enough
# to read as a separate surface where a window overlaps the header or the rail,
# little enough that it is plainly the same color.
_FLOATING_LIFT = 0.12

_CLASSIC_TAB_CSS = f"""
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
# only. Uniform, unlike Material's ``10px 20px 20px 10px`` (``.main-content`` in
# ``panel/template/material/material.css``): a plot grid fills its tab edge to
# edge, which makes an asymmetric frame around it plain to see.
_NICOS_CONTENT_PADDING = '10px'

# Material sizes the main area ``calc(100vh - 84px)``: the top app bar's 64px
# (``.mdc-top-app-bar__row``, same stylesheet) plus a 20px gap it leaves at the
# bottom of the window. The gap is white, so a rail that stops short of it does
# not read as chrome; the theme takes it back and lets the content's own bottom
# padding do that job instead.
_NICOS_TEMPLATE_CSS = """
    .main-content {
        padding: 0;
        height: calc(100vh - 64px);
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
        padding: {_NICOS_CONTENT_PADDING};
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

    @property
    def floating_header_background(self) -> str:
        """Title-bar color for windows floating above the page.

        A floating window is chrome too, so it wears the chrome color -- but
        not the same shade: matched exactly, its title bar disappears into the
        header and the rail wherever it overlaps them. Lifted, it still reads
        as part of the shell while marking itself as something above the page.
        """
        return _lighten(self.header_background, _FLOATING_LIFT)


CLASSIC_THEME = Theme(
    name='classic',
    header_background='#2596be',
    tabs_location='above',
    tab_strip_css=_CLASSIC_TAB_CSS,
)

NICOS_THEME = Theme(
    name='nicos',
    header_background=_NICOS_TEAL,
    tabs_location='left',
    tab_strip_css=_NICOS_TAB_CSS,
    template_css=_NICOS_TEMPLATE_CSS,
)

THEMES = {theme.name: theme for theme in (NICOS_THEME, CLASSIC_THEME)}
DEFAULT_THEME = NICOS_THEME
