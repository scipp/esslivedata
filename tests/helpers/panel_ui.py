# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Driving Panel widget trees from unit tests via the ``lt-*`` automation hooks.

The same stable CSS classes browser automation targets (see
``.claude/rules/dashboard-widgets.md``) address a button in-process, so a test
clicks what a user clicks instead of indexing into a layout that a refactor
will renumber.
"""

from __future__ import annotations

from typing import Any


def find_by_css_class(widget: Any, css_class: str) -> Any | None:
    """Depth-first search of a Panel widget tree for a component's hook class."""
    if css_class in (getattr(widget, 'css_classes', None) or []):
        return widget
    for child in getattr(widget, 'objects', []):
        found = find_by_css_class(child, css_class)
        if found is not None:
            return found
    return None


def click_tool(widget: Any, css_class: str) -> None:
    """Click the tool button carrying ``css_class`` somewhere under ``widget``."""
    button = find_by_css_class(widget, css_class)
    if button is None:
        raise AssertionError(f"No widget with css class {css_class!r}")
    button.clicks += 1
