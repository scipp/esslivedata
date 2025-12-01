# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for grid template loading."""

import pydantic
import pytest

from ess.livedata.config.grid_templates import (
    GridSpec,
    load_raw_grid_templates,
    parse_grid_specs,
)
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
        # Check the detector_overview template we created
        titles = [t.get('title') for t in raw_templates]
        assert 'Detector Overview' in titles

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

        # Find the detector overview template
        raw = next(
            (t for t in raw_templates if t.get('title') == 'Detector Overview'), None
        )
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


class FakeOrchestrator:
    """Fake orchestrator for testing parse_grid_specs."""

    def parse_raw_cell(self, cell_data: dict) -> PlotCell | None:
        """Parse a raw cell dict, returning None for unknown plotters."""
        config_data = cell_data['config']
        plot_name = config_data['plot_name']

        # Simulate unknown plotter check
        if plot_name == 'unknown_plotter':
            return None

        geometry = CellGeometry(
            row=cell_data['geometry']['row'],
            col=cell_data['geometry']['col'],
            row_span=cell_data['geometry']['row_span'],
            col_span=cell_data['geometry']['col_span'],
        )
        config = PlotConfig(
            workflow_id=WorkflowId.from_string(config_data['workflow_id']),
            output_name=config_data.get('output_name'),
            source_names=config_data['source_names'],
            plot_name=plot_name,
            params=EmptyParams(),
        )
        return PlotCell(geometry=geometry, config=config)


class TestParseGridSpecs:
    """Tests for parse_grid_specs function."""

    def test_parses_raw_specs(self):
        """Test that raw specs are parsed into GridSpec objects."""
        raw_specs = [
            {
                'title': 'Test Spec',
                'description': 'A test spec',
                'nrows': 2,
                'ncols': 3,
                'cells': [_make_raw_cell(0, 0)],
            }
        ]
        orchestrator = FakeOrchestrator()

        specs = parse_grid_specs(raw_specs, orchestrator)

        assert len(specs) == 1
        spec = specs[0]
        assert spec.name == 'Test Spec'
        assert spec.title == 'Test Spec'
        assert spec.description == 'A test spec'
        assert spec.nrows == 2
        assert spec.ncols == 3
        assert len(spec.cells) == 1
        assert isinstance(spec.cells[0], PlotCell)

    def test_skips_cells_with_unknown_plotters(self):
        """Test that cells with unknown plotters are skipped."""
        raw_specs = [
            {
                'title': 'Mixed Spec',
                'nrows': 2,
                'ncols': 2,
                'cells': [
                    _make_raw_cell(0, 0),  # Valid
                    {
                        'geometry': {'row': 1, 'col': 0, 'row_span': 1, 'col_span': 1},
                        'config': {
                            'workflow_id': 'test/ns/wf/1',
                            'source_names': ['src'],
                            'plot_name': 'unknown_plotter',
                            'params': {},
                        },
                    },
                ],
            }
        ]
        orchestrator = FakeOrchestrator()

        specs = parse_grid_specs(raw_specs, orchestrator)

        assert len(specs) == 1
        # Only one cell should be parsed (the valid one)
        assert len(specs[0].cells) == 1

    def test_parses_spec_with_no_cells(self):
        """Test that specs with no cells are parsed correctly."""
        raw_specs = [
            {
                'title': 'Empty Spec',
                'nrows': 3,
                'ncols': 3,
                'cells': [],
            }
        ]
        orchestrator = FakeOrchestrator()

        specs = parse_grid_specs(raw_specs, orchestrator)

        assert len(specs) == 1
        assert specs[0].cells == ()

    def test_returns_empty_list_for_empty_input(self):
        """Test that empty input returns empty output."""
        orchestrator = FakeOrchestrator()

        specs = parse_grid_specs([], orchestrator)

        assert specs == []

    def test_uses_defaults_for_missing_fields(self):
        """Test that missing fields get default values."""
        raw_specs = [
            {
                'title': 'Minimal',
                # No description, nrows, ncols, cells
            }
        ]
        orchestrator = FakeOrchestrator()

        specs = parse_grid_specs(raw_specs, orchestrator)

        assert len(specs) == 1
        spec = specs[0]
        assert spec.description == ''
        assert spec.nrows == 3
        assert spec.ncols == 3
        assert spec.cells == ()
