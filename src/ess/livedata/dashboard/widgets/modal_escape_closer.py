# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Close any open Panel modal when Escape is pressed."""

from typing import ClassVar

from panel.reactive import ReactiveHTML


class ModalEscapeCloser(ReactiveHTML):
    """Invisible widget that closes the open Panel modal on Escape.

    Panel's ``Modal`` (backed by a11y-dialog) binds its Escape handler to the
    main ``document.body``, but the dialog renders inside Bokeh's shadow DOM and
    Panel focuses the dialog *container* on open. From that initial focus the
    keydown never triggers a11y-dialog's handler, so a modal only closes on
    Escape once focus has moved into its content -- surprising for users who just
    opened it. One instance per session installs a ``document``-level Escape
    listener that clicks the close button of whichever dialog is currently shown,
    covering every modal in the app regardless of where it is mounted.

    Mirrors the ReactiveHTML ``_scripts['render']`` pattern of
    :class:`~ess.livedata.dashboard.widgets.heartbeat_widget.HeartbeatWidget`.
    """

    _template = """<div id="modal_esc" style="display:none;"></div>"""

    _scripts: ClassVar = {
        'render': """
            if (window.__esslivedataModalEsc) { return; }
            window.__esslivedataModalEsc = true;
            const findShownClose = (node) => {
                if (node.id === 'pnx_dialog' && node.style &&
                        node.style.display !== 'none') {
                    const btn = node.querySelector('.pnx-dialog-close');
                    if (btn) { return btn; }
                }
                if (node.shadowRoot) {
                    const r = findShownClose(node.shadowRoot);
                    if (r) { return r; }
                }
                for (const child of node.children || []) {
                    const r = findShownClose(child);
                    if (r) { return r; }
                }
                return null;
            };
            state.handler = (e) => {
                if (e.key !== 'Escape') { return; }
                const btn = findShownClose(document.documentElement);
                if (btn) { btn.click(); }
            };
            document.body.addEventListener('keydown', state.handler);
        """,
        'remove': """
            if (state.handler) {
                document.body.removeEventListener('keydown', state.handler);
                window.__esslivedataModalEsc = false;
            }
        """,
    }

    def __init__(self, **params):
        params.setdefault('width', 0)
        params.setdefault('height', 0)
        params.setdefault('sizing_mode', 'fixed')
        params.setdefault('visible', False)
        super().__init__(**params)
