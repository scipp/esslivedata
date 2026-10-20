# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Reading what a rendered Panel view of a plot shows."""

from typing import Any

from bokeh.document import Document
from bokeh.plotting import figure as Figure


def shown_values(view: Any) -> list[list[float]]:
    """The values each curve of a Panel view shows, in drawing order.

    Renders the view if needed.
    """
    (figure,) = view.get_root(Document()).select({'type': Figure})
    return [
        list(renderer.data_source.data['values'])
        for renderer in figure.renderers
        if 'values' in renderer.data_source.data
    ]
