# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Report whether the browser window is in portrait orientation."""

from typing import ClassVar

import param
from panel.reactive import ReactiveHTML


class OrientationReporter(ReactiveHTML):
    """Invisible widget syncing the window's orientation to Python.

    The server cannot see the screen, but some choices depend on its shape: on
    a phone, where a color bar goes decides whether a plot keeps a usable size.
    ``portrait`` is true while the window is taller than wide, and follows
    rotations and resizes; watch it to react.

    Mirrors the ReactiveHTML ``_scripts['render']`` pattern of
    :class:`~ess.livedata.dashboard.widgets.modal_escape_closer.ModalEscapeCloser`.
    """

    portrait = param.Boolean(default=True, doc="Window taller than wide.")

    _template = """<div id="orientation" style="display:none;"></div>"""

    _scripts: ClassVar = {
        'render': """
            state.handler = () => {
                const portrait = window.innerHeight > window.innerWidth;
                if (data.portrait !== portrait) { data.portrait = portrait; }
            };
            window.addEventListener('resize', state.handler);
            state.handler();
        """,
        'remove': """
            if (state.handler) {
                window.removeEventListener('resize', state.handler);
            }
        """,
    }

    def __init__(self, **params):
        params.setdefault('width', 0)
        params.setdefault('height', 0)
        params.setdefault('sizing_mode', 'fixed')
        params.setdefault('visible', False)
        super().__init__(**params)
