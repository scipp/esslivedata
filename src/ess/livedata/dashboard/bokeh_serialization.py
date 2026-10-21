# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Faster serialization of Bokeh models, backporting the Bokeh 4.0 ordering.

Bokeh sends a model to the browser with only the properties that differ from
their defaults, collected by ``HasProps.query_properties_with_values`` with
``include_defaults=False``. Up to Bokeh 3.10 that method reads every property
before checking whether it is wanted, and reading an unset property copies and
validates its default, only to discard it. Revealing a plot grid tab re-sends
every model of the grid, so this dominates the reveal: on an 11-cell LOKI grid it
took ~180 ms of a ~550 ms reveal, ~45 ms with the replacement below (#1401).
Bokeh 4.0 checks first (bokeh#15355).

The replacement decides from the raw default instead, and keeps the stock result
and side effects: a callable default is still produced, since Bokeh stores it on
the model, and a property without a default still raises ``UnsetValueError``.

It relies on Bokeh internals, so :func:`install` accepts Bokeh 3.9 only. Delete
this module when moving to a newer Bokeh.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import bokeh
from bokeh.core.has_props import HasProps
from bokeh.core.property.descriptors import PropertyDescriptor, UnsetValueError
from bokeh.core.property.singletons import Undefined
from bokeh.core.property.wrappers import PropertyValueContainer


def query_properties_with_values(
    self: HasProps,
    query: Callable[[PropertyDescriptor[Any]], bool],
    *,
    include_defaults: bool = True,
    include_undefined: bool = False,
) -> dict[str, Any]:
    """``HasProps.query_properties_with_values`` of Bokeh 3.9.2, reordered.

    Identical to the original except for the early ``continue`` on keys that are
    neither set, themed, nor a materialized unstable default.
    """
    themed_keys: set[str] = set()
    result: dict[str, Any] = {}

    keys = self.properties(_with_props=True)
    if include_defaults:
        selected_keys = set(keys)
    else:
        selected_keys = set(self._property_values.keys()) | set(
            self._unstable_default_values.keys()
        )
        themed_values = self.themed_values()
        if themed_values is not None:
            themed_keys = set(themed_values.keys())
            selected_keys |= themed_keys

    overridden_defaults = self._overridden_defaults()

    for key in keys:
        descriptor = self.lookup(key)
        if not query(descriptor):
            continue

        if key not in selected_keys:
            default = overridden_defaults.get(
                key, getattr(descriptor.property, '_default', Undefined)
            )
            if not (callable(default) or default is Undefined):
                continue

        try:
            value = descriptor.get_value(self)
        except UnsetValueError:
            if include_undefined:
                value = Undefined
            else:
                raise
        else:
            if key not in selected_keys:
                continue

            if not include_defaults and key not in themed_keys:
                if (
                    isinstance(value, PropertyValueContainer)
                    and key in self._unstable_default_values
                ):
                    continue

        result[key] = value

    return result


def install() -> None:
    """Replace Bokeh's method for the whole process.

    Raises
    ------
    RuntimeError:
        If Bokeh is not 3.9, whose internals the replacement was written against.
    """
    if bokeh.__version__.split('.')[:2] != ['3', '9']:
        raise RuntimeError(
            f'{__name__} patches Bokeh 3.9 internals, found Bokeh '
            f'{bokeh.__version__}. Bokeh 4.0 contains the fix (bokeh#15355); '
            'for any other version, delete the patch or port it.'
        )
    HasProps.query_properties_with_values = query_properties_with_values
