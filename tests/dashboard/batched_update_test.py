# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable, Iterator

import pytest
from bokeh.document import Document
from bokeh.models import Column, Div
from bokeh.plotting import figure
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
def recomputes(doc: Document) -> Iterator[Callable[[], int]]:
    counter = RecomputeCounter(doc)
    yield lambda: counter.count
    counter.restore()


def assert_graph_is_consistent(doc: Document) -> None:
    """Assert every model reachable from the roots is attached, and no other."""
    reachable = {model for root in doc.roots for model in root.references()}
    assert set(doc.models) == reachable
    assert all(model.document is doc for model in reachable)


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


def test_data_only_pass_does_not_recompute(
    doc: Document, recomputes: Callable[[], int]
) -> None:
    """Plot updates reach the browser as data patches, which must stay free.

    Bokeh routes ``ColumnDataSource`` patches and streams through hinted events
    that deliberately skip invalidation. Recomputing on them would put an
    O(models) walk on every tick of every plot; skipping is only correct because
    data columns cannot hold model references.
    """
    plot = figure()
    plot.line([1, 2, 3], [1, 4, 9])
    doc.roots[0].children = [*doc.roots[0].children, plot]
    source = plot.renderers[0].data_source
    baseline = recomputes()

    with set_curdoc(doc), batched_update():
        source.stream({'x': [4], 'y': [16]})
        source.patch({'y': [(0, 2)]})

    assert recomputes() == baseline
    assert_graph_is_consistent(doc)


def test_inner_bokeh_freeze_does_not_recompute_mid_pass(
    doc: Document, recomputes: Callable[[], int]
) -> None:
    """Panel's ``freeze_doc`` and HoloViews' ``hold_render`` freeze per update.

    Their ``_pop_freeze`` must find the counter still raised, or the batch
    collapses into one recompute per widget update.
    """
    with set_curdoc(doc), batched_update():
        for i in range(3):
            with doc.models.freeze():
                doc.roots[0].children = [*doc.roots[0].children, Div(text=str(i))]
            assert recomputes() == 0

    assert recomputes() == 1
    assert_graph_is_consistent(doc)


def test_root_added_by_nested_pass_is_attached(doc: Document) -> None:
    """The outer batch snapshots the roots, so it must see the inner's change."""
    root = Column(children=[Div(text="from the inner pass")])
    with set_curdoc(doc), batched_update():
        with batched_update():
            doc.add_root(root)
    assert_graph_is_consistent(doc)


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
