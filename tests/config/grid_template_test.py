# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from uuid import uuid4

import pydantic
import pytest

from ess.livedata.config.grid_template import GridSpec, load_raw_grid_templates
from ess.livedata.config.instrument import instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config
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
from ess.livedata.dashboard.plotter_registry import plotter_registry


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
                workflow_id=WorkflowId.from_string('test/wf/1'),
                view_name='output',
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

    def test_loaded_cells_have_geometry_and_layers(self):
        """Loaded template cells have geometry dict and layers list."""
        templates = load_raw_grid_templates('dummy')
        assert len(templates) >= 1
        template = templates[0]
        assert len(template['cells']) >= 1
        cell = template['cells'][0]
        assert 'geometry' in cell
        assert 'layers' in cell
        assert isinstance(cell['layers'], list)
        assert len(cell['layers']) >= 1
        geometry = cell['geometry']
        assert 'row' in geometry
        assert 'col' in geometry
        assert 'row_span' in geometry
        assert 'col_span' in geometry

    @pytest.mark.parametrize('instrument', ['bifrost', 'dummy'])
    def test_shipped_template_workflow_ids_parse(self, instrument):
        """Every workflow_id in a shipped template parses as a WorkflowId.

        Guards against the templates drifting out of sync with the
        WorkflowId string format and silently disappearing from the
        UI dropdown when ``_parse_single_spec`` catches the resulting
        ``ValueError``.
        """
        templates = load_raw_grid_templates(instrument)
        assert templates, f'no templates shipped for instrument {instrument!r}'
        for template in templates:
            for cell in template['cells']:
                for layer in cell['layers']:
                    for ds in layer['data_sources'].values():
                        WorkflowId.from_string(ds['workflow_id'])


def _collect_template_data_sources():
    """Yield one param per (non-static) data source in every shipped template.

    Static-plotter layers (overlays like ``rectangles``) reference sentinel
    ``static/...`` workflow ids with no registered workflow, so they are skipped
    here; their params are validated elsewhere.
    """
    static_plotters = set(plotter_registry.get_static_plotters())
    params = []
    for instrument in available_instruments():
        get_config(instrument)  # register specs (factories not needed)
        for template in load_raw_grid_templates(instrument):
            for ci, cell in enumerate(template['cells']):
                for li, layer in enumerate(cell['layers']):
                    plot_name = layer['plot_name']
                    if plot_name in static_plotters:
                        continue
                    for role, ds in layer['data_sources'].items():
                        params.append(
                            pytest.param(
                                instrument,
                                ds['workflow_id'],
                                ds['view_name'],
                                tuple(ds['source_names']),
                                plot_name,
                                id=f"{instrument}/{template['title']}"
                                f"/cell{ci}/layer{li}/{role}",
                            )
                        )
    return params


@pytest.mark.parametrize(
    ('instrument', 'workflow_id_str', 'view_name', 'source_names', 'plot_name'),
    _collect_template_data_sources(),
)
def test_shipped_template_data_sources_resolve(
    instrument: str,
    workflow_id_str: str,
    view_name: str,
    source_names: tuple[str, ...],
    plot_name: str,
):
    """Every shipped-template data source resolves to a renderable plot.

    Joins the shipped templates to the real workflow specs and plotter
    registry: the ``workflow_id`` must resolve, ``source_names`` must be a
    subset of the workflow's sources, ``view_name`` must be a declared output
    view, and that view's template data must satisfy the chosen plotter's
    data requirements.

    The load path (``PlotOrchestrator._parse_single_spec``) swallows these
    mismatches and silently degrades to empty or wrong plots, so without this
    guard a renamed view or a changed output dimensionality ships unnoticed.
    """
    registry = instrument_registry[instrument].workflow_factory
    workflow_id = WorkflowId.from_string(workflow_id_str)
    assert workflow_id in registry, f'{workflow_id} is not a registered workflow'
    spec = registry[workflow_id]

    assert set(source_names) <= set(spec.source_names), (
        f'{workflow_id} sources {source_names} are not a subset of {spec.source_names}'
    )

    view = spec.get_output_view(view_name)
    available = [v.name for v in spec.get_output_views()]
    assert view is not None, (
        f'{workflow_id} has no output view {view_name!r} (available: {available})'
    )

    template = spec.get_output_template(view_name)
    assert template is not None, (
        f'view {view_name!r} of {workflow_id} has no template; add a '
        'default_factory to its backing field'
    )

    requirements = plotter_registry.get_spec(plot_name).data_requirements
    assert requirements.validate_data({view_name: template}), (
        f'plotter {plot_name!r} cannot render view {view_name!r} of '
        f'{workflow_id}: ndim={template.ndim}, dims={template.dims}, '
        f'coords={list(template.coords)}'
    )
