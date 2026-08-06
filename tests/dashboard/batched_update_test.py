# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable

import pytest
from bokeh.document import Document
from bokeh.models import Column, Div
from panel.io.state import set_curdoc

from ess.livedata.dashboard.batched_update import batched_update


class RecomputeCounter:
    """Counts the document's model-graph recomputes without suppressing them."""

    def __init__(self, doc: Document) -> None:
        self._models = doc.models
        self._inner = doc.models.recompute
        self.count = 0
        doc.models.recompute = self._recompute  # type: ignore[method-assign]

    def _recompute(self) -> None:
        self.count += 1
        self._inner()

    def restore(self) -> None:
        del self._models.recompute


@pytest.fixture
def doc() -> Document:
    document = Document()
    document.add_root(Column(children=[Div(text="root")]))
    return document


@pytest.fixture
def recomputes(doc: Document) -> Callable[[], int]:
    counter = RecomputeCounter(doc)
    yield lambda: counter.count
    counter.restore()


def test_pass_that_mutates_nothing_does_not_recompute(
    doc: Document, recomputes: Callable[[], int]
) -> None:
    with set_curdoc(doc), batched_update():
        pass
    assert recomputes() == 0


def test_pass_that_only_reads_the_graph_does_not_recompute(
    doc: Document, recomputes: Callable[[], int]
) -> None:
    with set_curdoc(doc), batched_update():
        assert len(doc.models) == 2
        assert doc.roots[0].children[0].text == "root"
    assert recomputes() == 0


def test_added_model_is_attached_to_the_document(doc: Document) -> None:
    added = Div(text="added")
    with set_curdoc(doc), batched_update():
        doc.roots[0].children = [*doc.roots[0].children, added]
    assert doc.models.get_by_id(added.id) is added
    assert added.document is doc


def test_removed_model_is_detached_from_the_document(doc: Document) -> None:
    removed = doc.roots[0].children[0]
    with set_curdoc(doc), batched_update():
        doc.roots[0].children = []
    assert doc.models.get_by_id(removed.id) is None
    assert removed.document is None


def test_many_mutations_recompute_once(
    doc: Document, recomputes: Callable[[], int]
) -> None:
    with set_curdoc(doc), batched_update():
        for i in range(5):
            doc.roots[0].children = [*doc.roots[0].children, Div(text=str(i))]
    assert recomputes() == 1
    assert len(doc.models) == 7


def test_root_added_during_pass_is_attached(doc: Document) -> None:
    root = Column(children=[Div(text="second root")])
    with set_curdoc(doc), batched_update():
        doc.add_root(root)
    assert doc.models.get_by_id(root.children[0].id) is not None


def test_nested_pass_recomputes_once_for_both(
    doc: Document, recomputes: Callable[[], int]
) -> None:
    inner_added = Div(text="inner")
    outer_added = Div(text="outer")
    with set_curdoc(doc), batched_update():
        doc.roots[0].children = [*doc.roots[0].children, outer_added]
        with batched_update():
            doc.roots[0].children = [*doc.roots[0].children, inner_added]
    assert recomputes() == 1
    assert doc.models.get_by_id(inner_added.id) is inner_added
    assert doc.models.get_by_id(outer_added.id) is outer_added


def test_raising_pass_leaves_the_document_usable(doc: Document) -> None:
    with pytest.raises(RuntimeError):  # noqa: PT012
        with set_curdoc(doc), batched_update():
            doc.roots[0].children = [*doc.roots[0].children, Div(text="added")]
            raise RuntimeError("handler blew up")

    # A document left frozen would swallow this mutation until the next freeze.
    added = Div(text="after")
    doc.roots[0].children = [*doc.roots[0].children, added]
    assert doc.models.get_by_id(added.id) is added


def test_works_without_a_current_document() -> None:
    with batched_update():
        pass
