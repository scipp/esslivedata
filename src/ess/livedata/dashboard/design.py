# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Panel :class:`~panel.theme.Design` used by the dashboard template.

Exists to keep markup panes visible when they are built after page load, and
to keep every modal dialog inside the viewport.

Panel keeps a markup pane's content container behind ``visibility: hidden`` from
the moment it renders and reveals it only once every ``<link>`` stylesheet in the
pane's shadow root has fired a ``load`` event
(``PanelMarkupView.watch_stylesheets`` -> ``style_redraw``). That reveal is armed
exactly once, during ``render()``.

A pane built after the page has loaded -- every cell the plot poll loop rebuilds,
every widget a callback adds to a live layout -- first renders with the design's
stylesheet URLs still pointing at cdn.holoviz.org, because its model is not
attached to a document yet and Panel falls back to ``CDN_DIST``. Panel then
patches those URLs to the locally served copies, swapping the pane's ``<link>``
elements for new ones. The load events the reveal is waiting on belong to the
discarded elements and never arrive, so the pane stays invisible for the rest of
the session while its Bokeh model, text and layout are all correct and
live-updating (#1154, holoviz/panel#8696).

Overriding the reveal is safe here: our markup carries inline styling, so there
is no unstyled-content flash to suppress, and the content has to show the moment
it is inserted. Drop this design once a Panel release re-arms the reveal.

Modals are sized in pixels by their owners, which Panel writes as inline styles
on the dialog box. On a screen smaller than that -- a phone, or a short laptop
screen for the height -- the box runs past the viewport, taking its close
button and the action buttons of its footer with it. Capping the box to the
viewport and letting it scroll keeps all of it reachable; footers that must
stay in view while the rest scrolls make themselves sticky
(``ModalSizing.STICKY_FOOTER_STYLES``).
"""

from typing import Any, ClassVar

from panel.layout import Modal
from panel.pane.markup import HTMLBasePane
from panel.theme import Material
from panel.theme.base import Inherit
from panel.viewable import Viewable

_ALWAYS_VISIBLE_MARKUP = ':host > div { visibility: visible !important; }'

# ``!important`` outranks the inline pixel sizes. Panel writes ``width`` as
# ``min-width`` too, which would win over ``max-width``, so it is released.
# ``dvh`` rather than ``vh``: on a phone ``100vh`` includes the height hidden
# behind the browser's toolbars.
_MODAL_FITS_VIEWPORT = """
    .dialog-content {
        box-sizing: border-box;
        min-width: 0 !important;
        max-width: calc(100vw - 16px) !important;
        max-height: calc(100dvh - 16px) !important;
        overflow-y: auto;
    }
"""


class LivedataDesign(Material):
    """Material design that keeps markup panes visible when built post-load."""

    modifiers: ClassVar[dict[type[Viewable], dict[str, Any]]] = {
        HTMLBasePane: {'stylesheets': [Inherit, _ALWAYS_VISIBLE_MARKUP]},
        Modal: {'stylesheets': [Inherit, _MODAL_FITS_VIEWPORT]},
    }
