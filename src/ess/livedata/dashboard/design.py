# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Panel :class:`~panel.theme.Design` used by the dashboard template.

Exists to keep markup panes visible when they are built after page load.

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
live-updating (#1154).

Overriding the reveal is safe here: our markup carries inline styling, so there
is no unstyled-content flash to suppress, and the content has to show the moment
it is inserted.
"""

from typing import Any, ClassVar

from panel.pane.markup import HTMLBasePane
from panel.theme import Material
from panel.theme.base import Inherit
from panel.viewable import Viewable

_ALWAYS_VISIBLE_MARKUP = ':host > div { visibility: visible !important; }'


class LivedataDesign(Material):
    """Material design that keeps markup panes visible when built post-load."""

    modifiers: ClassVar[dict[type[Viewable], dict[str, Any]]] = {
        HTMLBasePane: {'stylesheets': [Inherit, _ALWAYS_VISIBLE_MARKUP]}
    }
