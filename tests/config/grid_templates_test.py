# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for grid template loading and GridSpec data model."""

import pydantic
import pytest

from ess.livedata.config.grid_templates import GridSpec, load_raw_grid_templates
from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.plot_orchestrator import (
    CellGeometry,
    PlotCell,
    PlotConfig,
)


class EmptyParams(pydantic.BaseModel):
    """Empty params model for testing."""

    pass


def _make_cell(row: int, col: int, row_span: int = 1, col_span: int = 1) -> PlotCell:
    """Helper to create a PlotCell for testing."""
    return PlotCell(
        geometry=CellGeometry(
            row=row,
            col=col,
            row_span=row_span,
            col_span=col_span,
        ),
        config=PlotConfig(
            workflow_id=WorkflowId.from_string('test/ns/wf/1'),
            output_name=None,
            source_names=['source1'],
            plot_name='lines',
            params=EmptyParams(),
        ),
    )


def _make_raw_cell(row: int, col: int, row_span: int = 1, col_span: int = 1) -> dict:
    """Helper to create a raw cell dict for testing."""
    return {
        'geometry': {
            'row': row,
            'col': col,
            'row_span': row_span,
            'col_span': col_span,
        },
        'config': {
            'workflow_id': 'test/ns/wf/1',
            'output_name': None,
            'source_names': ['source1'],
            'plot_name': 'lines',
            'params': {},
        },
    }


class TestGridSpec:
    """Tests for GridSpec dataclass."""

    def test_properties_from_config(self):
        """Test that properties are set correctly."""
        cell = _make_cell(0, 0)
        spec = GridSpec(
            name='Test',
            title='Test Grid',
            description='',
            nrows=3,
            ncols=4,
            cells=(cell,),
        )

        assert spec.name == 'Test'
        assert spec.title == 'Test Grid'
        assert spec.nrows == 3
        assert spec.ncols == 4
        assert len(spec.cells) == 1
        assert isinstance(spec.cells[0], PlotCell)

    def test_min_rows_from_cells(self):
        """Test that min_rows is computed from cell geometries."""
        cells = (
            _make_cell(row=0, col=0, row_span=1, col_span=1),
            _make_cell(row=1, col=0, row_span=2, col_span=1),
        )
        spec = GridSpec(
            name='Test',
            title='Test',
            description='',
            nrows=2,
            ncols=2,
            cells=cells,
        )

        # Cell at row 1 with row_span 2 requires at least 3 rows
        assert spec.min_rows == 3

    def test_min_cols_from_cells(self):
        """Test that min_cols is computed from cell geometries."""
        cell = _make_cell(row=0, col=1, row_span=1, col_span=3)
        spec = GridSpec(
            name='Test',
            title='Test',
            description='',
            nrows=2,
            ncols=2,
            cells=(cell,),
        )

        # Cell at col 1 with col_span 3 requires at least 4 columns
        assert spec.min_cols == 4

    def test_min_rows_cols_with_no_cells(self):
        """Test that min_rows/cols falls back to config values when no cells."""
        spec = GridSpec(
            name='Empty',
            title='Empty Grid',
            description='',
            nrows=5,
            ncols=6,
            cells=(),
        )

        assert spec.min_rows == 5
        assert spec.min_cols == 6

    def test_has_required_fields(self):
        """Test that GridSpec has all required fields."""
        cell = _make_cell(0, 0)
        spec = GridSpec(
            name='Test',
            title='Test Title',
            description='Test Description',
            nrows=3,
            ncols=4,
            cells=(cell,),
        )

        assert spec.name == 'Test'
        assert spec.title == 'Test Title'
        assert spec.description == 'Test Description'
        assert spec.nrows == 3
        assert spec.ncols == 4
        assert len(spec.cells) == 1

    def test_is_immutable(self):
        """Test that GridSpec is frozen."""
        spec = GridSpec(
            name='Test',
            title='Test',
            description='',
            nrows=3,
            ncols=3,
            cells=(),
        )

        with pytest.raises(AttributeError):
            spec.name = 'Changed'


class TestCellGeometry:
    """Tests for CellGeometry dataclass."""

    def test_creates_geometry(self):
        """Test basic geometry creation."""
        geom = CellGeometry(row=1, col=2, row_span=3, col_span=4)

        assert geom.row == 1
        assert geom.col == 2
        assert geom.row_span == 3
        assert geom.col_span == 4

    def test_is_immutable(self):
        """Test that CellGeometry is frozen."""
        geom = CellGeometry(row=0, col=0, row_span=1, col_span=1)

        with pytest.raises(AttributeError):
            geom.row = 5


class TestLoadRawGridTemplates:
    """Tests for load_raw_grid_templates function."""

    def test_loads_templates_for_dummy_instrument(self):
        """Test loading raw templates from dummy instrument."""
        raw_templates = load_raw_grid_templates('dummy')

        assert len(raw_templates) >= 1
        titles = [t.get('title') for t in raw_templates]
        assert 'Detectors' in titles

    def test_returns_empty_for_unknown_instrument(self):
        """Test that unknown instruments return empty list."""
        raw_templates = load_raw_grid_templates('nonexistent_instrument')

        assert raw_templates == []

    def test_returns_empty_for_instrument_without_templates(self):
        """Test instruments without grid_templates directory return empty list."""
        # LOKI exists but doesn't have grid_templates
        raw_templates = load_raw_grid_templates('loki')

        assert raw_templates == []

    def test_raw_template_has_correct_structure(self):
        """Test that loaded raw templates have the expected structure."""
        raw_templates = load_raw_grid_templates('dummy')

        raw = next((t for t in raw_templates if t.get('title') == 'Detectors'), None)
        assert raw is not None

        # Check structure
        assert raw['nrows'] == 2
        assert raw['ncols'] == 2
        assert len(raw['cells']) == 2

        # Check cell structure is raw dict
        cell = raw['cells'][0]
        assert isinstance(cell, dict)
        assert 'geometry' in cell
        assert 'config' in cell
        assert cell['config']['workflow_id'] is not None
