# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import re

import bokeh
import holoviews as hv
import numpy as np
import panel as pn
import pytest
from bokeh.core.has_props import HasProps
from bokeh.core.json_encoder import serialize_json
from bokeh.core.property.descriptors import PropertyDescriptor
from bokeh.document import Document
from bokeh.models import Range1d

from ess.livedata.dashboard import bokeh_serialization

hv.extension('bokeh')


@pytest.fixture
def stock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore Bokeh's own method after the test, whatever the test installs."""
    monkeypatch.setattr(
        HasProps, 'query_properties_with_values', HasProps.query_properties_with_values
    )


def _document() -> Document:
    image = hv.Image(np.arange(64.0).reshape(8, 8))
    elements = [
        image.opts(colorbar=True, tools=['hover'], cmap='viridis'),
        hv.Curve(np.arange(5.0)) * hv.Scatter(np.arange(5.0)),
        hv.Bars([('a', 1), ('b', 2)]),
        hv.Table({'x': [1, 2], 'y': ['u', 'v']}),
    ]
    layout = pn.Column(
        *[pn.pane.HoloViews(element) for element in elements],
        pn.pane.HTML('<b>x</b>'),
        pn.widgets.Select(options=['a', 'b']),
        pn.Tabs(('a', pn.pane.Markdown('m')), ('b', 'n'), dynamic=True),
    )
    doc = Document()
    doc.add_root(layout.get_root(doc))
    return doc


def _serialize(doc: Document) -> str:
    """The document as sent to the browser, with model IDs numbered by appearance.

    IDs (Bokeh's ``p1234`` and Panel's ``Column01234`` names) come from
    process-wide counters, so two otherwise identical documents differ only there.
    """
    text = serialize_json(doc.to_json())
    ids: dict[str, str] = {}
    return re.sub(
        r'"(p\d+|[A-Za-z]+\d{5})"',
        lambda m: ids.setdefault(m.group(1), f'"id{len(ids)}"'),
        text,
    )


@pytest.mark.usefixtures('stock')
def test_serialization_matches_stock_bokeh() -> None:
    expected_doc = _document()
    expected = [_serialize(expected_doc), _serialize(expected_doc)]

    bokeh_serialization.install()
    doc = _document()
    # The second pass catches defaults that only the first one would materialize.
    assert [_serialize(doc), _serialize(doc)] == expected


@pytest.mark.usefixtures('stock')
def test_skipped_defaults_are_not_produced(monkeypatch: pytest.MonkeyPatch) -> None:
    bokeh_serialization.install()
    model = Range1d(start=1, end=2)

    def fail(self: PropertyDescriptor, obj: HasProps) -> None:
        raise AssertionError(f'default of {self.name} was produced')

    monkeypatch.setattr(PropertyDescriptor, 'instance_default', fail)
    assert model.properties_with_values(include_defaults=False) == {
        'start': 1,
        'end': 2,
    }


@pytest.mark.usefixtures('stock')
def test_install_refuses_other_bokeh_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    stock_method = HasProps.query_properties_with_values
    monkeypatch.setattr(bokeh, '__version__', '3.10.0')
    with pytest.raises(RuntimeError, match=r'Bokeh 3\.10\.0'):
        bokeh_serialization.install()
    assert HasProps.query_properties_with_values is stock_method
