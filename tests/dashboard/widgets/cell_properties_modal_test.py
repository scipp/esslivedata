# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for CellPropertiesModal title persistence.

Regression coverage for issue 1072: the title field is pre-filled via
``TextInput.value``, but Panel only mirrors typed keystrokes into
``value_input`` -- it does not seed ``value_input`` from ``value`` on
construction. Reading ``value_input`` in ``_persist_title`` therefore saw an
empty string whenever the modal was saved without the user touching the
field, erasing any existing custom title.
"""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

import pydantic
import pytest

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec
from ess.livedata.dashboard.plot_orchestrator import CellGeometry, CellId, PlotCell
from ess.livedata.dashboard.widgets.cell_properties_modal import CellPropertiesModal

EXISTING_TITLE = 'Existing Title'


class FakePlottingController:
    """Stand-in for PlottingController.

    ``CellPropertiesModal`` only queries the plotting controller while
    rendering layer rows, which none of these title-persistence tests
    exercise (the fixture cell has no layers); raising on use turns any
    unexpected call into a loud test failure instead of a silent fake.
    """

    def is_overlayable(self, plot_name: str, params: dict | pydantic.BaseModel) -> bool:
        raise AssertionError('unexpected call for a cell with no layers')


class FakeOrchestrator:
    """Orchestrator fake exposing only what CellPropertiesModal calls.

    Mirrors ``PlotOrchestrator.set_cell_title``'s normalization (falsy title
    becomes ``None``) so tests can assert on the resulting ``PlotCell``
    state directly, the same way callers observe persistence in production.
    """

    def __init__(self, cell: PlotCell) -> None:
        self._cell = cell

    def get_cell(self, cell_id: CellId) -> PlotCell:
        return self._cell

    def set_cell_title(self, cell_id: CellId, title: str | None) -> None:
        self._cell.user_title = title or None


@pytest.fixture
def cell_id() -> CellId:
    return CellId(uuid4())


@pytest.fixture
def cell() -> PlotCell:
    return PlotCell(
        geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
        layers=[],
        user_title=EXISTING_TITLE,
    )


@pytest.fixture
def orchestrator(cell: PlotCell) -> FakeOrchestrator:
    return FakeOrchestrator(cell)


@pytest.fixture
def workflow_registry() -> Mapping[WorkflowId, WorkflowSpec]:
    return {}


@pytest.fixture
def modal(
    orchestrator: FakeOrchestrator,
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
    cell_id: CellId,
) -> CellPropertiesModal:
    return CellPropertiesModal(
        orchestrator=orchestrator,
        workflow_registry=workflow_registry,
        plotting_controller=FakePlottingController(),
        cell_id=cell_id,
        current_title=EXISTING_TITLE,
        has_user_title=True,
        on_add_layer=lambda cell_id: None,
        on_close=lambda: None,
    )


class TestPersistTitle:
    def test_save_without_typing_preserves_existing_title(
        self,
        modal: CellPropertiesModal,
        orchestrator: FakeOrchestrator,
        cell_id: CellId,
    ) -> None:
        """Opening the modal and saving without touching the field is a no-op.

        The field is pre-filled via ``value`` only; ``value_input`` stays at
        Panel's default (``''``) until a keystroke lands in the browser.
        """
        modal._persist_title()
        assert orchestrator.get_cell(cell_id).user_title == EXISTING_TITLE

    def test_typed_text_persists(
        self,
        modal: CellPropertiesModal,
        orchestrator: FakeOrchestrator,
        cell_id: CellId,
    ) -> None:
        """A real edit is saved.

        A keystroke updates ``value_input``; focus then leaving the field
        (e.g. clicking Save) updates ``value``. Set both here, in that
        order, to match what production sees by the time the Save button's
        callback runs.
        """
        modal._text.value_input = 'New Title'
        modal._text.value = 'New Title'
        modal._persist_title()
        assert orchestrator.get_cell(cell_id).user_title == 'New Title'

    def test_clearing_field_persists_none(
        self,
        modal: CellPropertiesModal,
        orchestrator: FakeOrchestrator,
        cell_id: CellId,
    ) -> None:
        """Deleting all text and saving clears the custom title."""
        modal._text.value_input = ''
        modal._text.value = ''
        modal._persist_title()
        assert orchestrator.get_cell(cell_id).user_title is None
