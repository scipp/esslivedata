# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Per-instrument browser-tab icon.

Each instrument gets a tile styled like an element in the periodic table: the
first two letters of the instrument name in a framed square. The tile colour
encodes the measurement technique, the way the periodic table colours element
groups. Users with dashboards of several instruments open can then tell the tabs
apart at the 16 px size browsers draw favicons at.
"""

from urllib.parse import quote

# (stroke and text, fill) per technique.
_TECHNIQUE_COLORS = {
    'diffraction': ('#1f5fa8', '#e6f0fa'),
    'sans': ('#7b3fa0', '#f1e8f7'),
    'imaging': ('#2e7d32', '#e7f4e8'),
    'spectroscopy': ('#c25e00', '#fbeee2'),
    'reflectometry': ('#00838f', '#e0f4f5'),
    'test': ('#616161', '#eeeeee'),
}

_INSTRUMENT_TECHNIQUE = {
    'beer': 'diffraction',
    'bifrost': 'spectroscopy',
    'dream': 'diffraction',
    'dummy': 'test',
    'estia': 'reflectometry',
    'freia': 'reflectometry',
    'loki': 'sans',
    'magic': 'diffraction',
    'nmx': 'diffraction',
    'odin': 'imaging',
    'tbl': 'test',
}


def instrument_favicon(instrument: str) -> str:
    """Return the favicon of an instrument as an SVG ``data:`` URI.

    The URI ends in a ``#svg`` fragment, which browsers ignore. It is there
    because Panel's templates accept a favicon URI only if it ends in a known
    image file extension.
    """
    stroke, fill = _TECHNIQUE_COLORS[_INSTRUMENT_TECHNIQUE[instrument]]
    symbol = instrument[:2].capitalize()
    # The thick frame (5 of 64 units) is needed for it to stay visible at 16 px.
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        f'<rect x="3" y="3" width="58" height="58" rx="5" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="5"/>'
        '<text x="32" y="45" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-weight="700" '
        f'font-size="34" fill="{stroke}">{symbol}</text></svg>'
    )
    return f'data:image/svg+xml,{quote(svg)}#svg'
