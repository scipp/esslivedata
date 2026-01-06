# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from uuid import uuid4

import pydantic
import pytest

from ess.livedata.config.grid_template import GridSpec, load_raw_grid_templates
from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.data_roles import PRIMARY
from ess.livedata.dashboard.plot_orchestrator import (
    CellGeometry,
    DataSourceConfig,
    Layer,
    LayerId,
    PlotCell,
    PlotConfig,
)


class EmptyParams(pydantic.BaseModel):
    """Empty params model for testing."""

    pass


def _make_plot_cell(
    row: int, col: int, row_span: int = 1, col_span: int = 1
) -> PlotCell:
    """Create a PlotCell with the given geometry."""
    config = PlotConfig(
        data_sources={
            PRIMARY: DataSourceConfig(
                workflow_id=WorkflowId.from_string('test/ns/wf/1'),
                output_name='output',
                source_names=['source1'],
            )
        },
        plot_name='lines',
        params=EmptyParams(),
    )
    layer = Layer(layer_id=LayerId(uuid4()), config=config)
    return PlotCell(
        geometry=CellGeometry(row=row, col=col, row_span=row_span, col_span=col_span),
        layers=[layer],
    )


class TestGridSpec:
    """Tests for GridSpec dataclass."""

    def test_min_rows_empty_cells_returns_nrows(self):
        """When cells is empty, min_rows returns nrows."""
        spec = GridSpec(
            name='test',
            title='Test',
            description='',
            nrows=5,
            ncols=3,
            cells=(),
        )
        assert spec.min_rows == 5

    def test_min_cols_empty_cells_returns_ncols(self):
        """When cells is empty, min_cols returns ncols."""
        spec = GridSpec(
            name='test',
            title='Test',
            description='',
            nrows=5,
            ncols=3,
            cells=(),
        )
        assert spec.min_cols == 3

    def test_min_rows_single_cell(self):
        """min_rows computes from a single cell."""
        cell = _make_plot_cell(row=1, col=0, row_span=2)
        spec = GridSpec(
            name='test',
            title='Test',
            description='',
            nrows=2,
            ncols=2,
            cells=(cell,),
        )
        # Cell at row 1 with row_span 2 requires 3 rows (0, 1, 2)
        assert spec.min_rows == 3

    def test_min_cols_single_cell(self):
        """min_cols computes from a single cell."""
        cell = _make_plot_cell(row=0, col=2, col_span=3)
        spec = GridSpec(
            name='test',
            title='Test',
            description='',
            nrows=2,
            ncols=2,
            cells=(cell,),
        )
        # Cell at col 2 with col_span 3 requires 5 cols (0, 1, 2, 3, 4)
        assert spec.min_cols == 5

    def test_min_rows_multiple_cells_returns_max(self):
        """min_rows returns the maximum required by any cell."""
        cells = (
            _make_plot_cell(row=0, col=0, row_span=1),  # needs 1 row
            _make_plot_cell(row=1, col=1, row_span=3),  # needs 4 rows
            _make_plot_cell(row=2, col=2, row_span=1),  # needs 3 rows
        )
        spec = GridSpec(
            name='test',
            title='Test',
            description='',
            nrows=2,
            ncols=3,
            cells=cells,
        )
        assert spec.min_rows == 4

    def test_min_cols_multiple_cells_returns_max(self):
        """min_cols returns the maximum required by any cell."""
        cells = (
            _make_plot_cell(row=0, col=0, col_span=2),  # needs 2 cols
            _make_plot_cell(row=0, col=1, col_span=1),  # needs 2 cols
            _make_plot_cell(row=1, col=2, col_span=2),  # needs 4 cols
        )
        spec = GridSpec(
            name='test',
            title='Test',
            description='',
            nrows=3,
            ncols=2,
            cells=cells,
        )
        assert spec.min_cols == 4

    def test_gridspec_is_frozen(self):
        """GridSpec is immutable (frozen dataclass)."""
        spec = GridSpec(
            name='test',
            title='Test',
            description='',
            nrows=3,
            ncols=3,
            cells=(),
        )
        with pytest.raises(AttributeError):
            spec.nrows = 5


class TestLoadRawGridTemplates:
    """Tests for load_raw_grid_templates function."""

    def test_load_from_dummy_instrument(self):
        """Loading templates from dummy instrument returns non-empty list."""
        templates = load_raw_grid_templates('dummy')
        assert len(templates) >= 1

    def test_loaded_template_has_expected_keys(self):
        """Loaded templates have the expected structure."""
        templates = load_raw_grid_templates('dummy')
        assert len(templates) >= 1
        template = templates[0]
        assert 'title' in template
        assert 'nrows' in template
        assert 'ncols' in template
        assert 'cells' in template

    def test_load_from_nonexistent_instrument_returns_empty(self):
        """Loading from non-existent instrument returns empty list."""
        templates = load_raw_grid_templates('nonexistent_instrument_xyz')
        assert templates == []

    def test_loaded_cells_have_geometry_and_config(self):
        """Loaded template cells have geometry and config dicts."""
        templates = load_raw_grid_templates('dummy')
        assert len(templates) >= 1
        template = templates[0]
        assert len(template['cells']) >= 1
        cell = template['cells'][0]
        assert 'geometry' in cell
        assert 'config' in cell
        geometry = cell['geometry']
        assert 'row' in geometry
        assert 'col' in geometry
        assert 'row_span' in geometry
        assert 'col_span' in geometry
