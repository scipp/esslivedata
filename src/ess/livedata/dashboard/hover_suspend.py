# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Suspend plot hover tooltips while a Bokeh edit tool is active.

The hover readout (x, y, value) obscures the plot area exactly where the user
is drawing or dragging an ROI. This module wires the figure so that activating
any edit tool (BoxEdit for rectangles, PolyDraw for polygons) deactivates the
hover tools, and deactivating it restores their previous state. Merely having
the edit tools on the toolbar does not affect hover.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Marks a figure as already wired, so the repeated renders of a DynamicMap do
# not stack duplicate callbacks. Figure-scoped rather than plot-scoped because
# HoloViews swaps the figure on kdim/Layout transitions.
_INSTALLED_TAG = 'lt-hover-suspend'

# Bokeh's ``Tool.active`` is a client-side-only property: activating a tool from
# the toolbar never reaches the server, so the whole toggle has to live in JS.
# Runs on every layout pass, hence the idempotency guard on the figure object.
_SUSPEND_JS = """
const fig = cb_obj;
if (fig._hover_suspend_wired) { return; }
fig._hover_suspend_wired = true;

// Non-null while hover is suspended, holding the states to restore. Keeping
// them means a user who switched hover off by hand does not get it switched
// back on when they leave the edit tool.
let saved = null;

const update = () => {
    const editing = edits.some((tool) => tool.active);
    if (editing && saved === null) {
        saved = hovers.map((hover) => hover.active);
        for (const hover of hovers) { hover.active = false; }
    } else if (!editing && saved !== null) {
        hovers.forEach((hover, i) => { hover.active = saved[i]; });
        saved = null;
    }
};

for (const tool of edits) { tool.properties.active.change.connect(update); }
update();
"""


def make_hover_suspend_hook() -> Callable[[Any, Any], None]:
    """
    Create a HoloViews hook suspending hover while an edit tool is active.

    The hook is a no-op on figures that have no edit tool or no hover tool, so
    it can be applied to every plot cell unconditionally.

    Returns
    -------
    :
        A hook function compatible with ``hv.Element.opts(hooks=[...])``.
    """

    def hook(plot: Any, element: Any) -> None:
        del element
        from bokeh.models import CustomJS
        from bokeh.models.tools import EditTool, HoverTool

        figure = getattr(plot, 'state', None)
        toolbar = getattr(figure, 'toolbar', None)
        if toolbar is None or _INSTALLED_TAG in figure.tags:
            return
        edits = [tool for tool in toolbar.tools if isinstance(tool, EditTool)]
        hovers = [tool for tool in toolbar.tools if isinstance(tool, HoverTool)]
        if not edits or not hovers:
            return
        # inner_width is written by BokehJS on every layout pass, giving a
        # reliable "figure has rendered" trigger for wiring up the JS signals.
        figure.js_on_change(
            'inner_width',
            CustomJS(args={'edits': edits, 'hovers': hovers}, code=_SUSPEND_JS),
        )
        figure.tags = [*figure.tags, _INSTALLED_TAG]

    return hook
