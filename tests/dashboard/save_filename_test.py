# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.plot_orchestrator import (
    CellGeometry,
    DataSourceConfig,
    Layer,
    LayerId,
    PlotCell,
    PlotConfig,
)
from ess.livedata.dashboard.plot_params import PlotParams1d
from ess.livedata.dashboard.widgets.plot_grid_tabs import (
    _build_save_filename_from_cell,
)


def _make_layer(
    instrument: str = 'dream',
    source_names: list[str] | None = None,
    output_name: str = 'counts',
    *,
    static: bool = False,
) -> Layer:
    workflow_id = WorkflowId(
        instrument=instrument, namespace='ns', name='wf', version=1
    )
    sources = [] if static else (source_names or ['monitor'])
    config = PlotConfig(
        data_sources={
            'primary': DataSourceConfig(
                workflow_id=workflow_id,
                source_names=sources,
                output_name=output_name,
            )
        },
        plot_name='test_plot',
        params=PlotParams1d(),
    )
    return Layer(layer_id=LayerId('test'), config=config)


def _make_cell(*layers: Layer) -> PlotCell:
    return PlotCell(
        geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
        layers=list(layers),
    )


class TestBuildSaveFilenameFromCell:
    def test_single_layer(self):
        cell = _make_cell(_make_layer())
        assert _build_save_filename_from_cell(cell) == 'DREAM_monitor_counts'

    def test_multiple_layers_combines_sources(self):
        layer_a = _make_layer(source_names=['alpha'], output_name='counts')
        layer_b = _make_layer(source_names=['beta'], output_name='spectrum')
        cell = _make_cell(layer_a, layer_b)
        result = _build_save_filename_from_cell(cell)
        assert result is not None
        assert 'alpha' in result
        assert 'beta' in result
        assert 'counts' in result
        assert 'spectrum' in result

    def test_static_layers_are_excluded(self):
        data_layer = _make_layer(source_names=['monitor'], output_name='counts')
        static_layer = _make_layer(static=True, output_name='reference_line')
        cell = _make_cell(data_layer, static_layer)
        result = _build_save_filename_from_cell(cell)
        assert result == 'DREAM_monitor_counts'

    def test_all_static_returns_none(self):
        cell = _make_cell(_make_layer(static=True))
        assert _build_save_filename_from_cell(cell) is None

    def test_instrument_from_first_non_static_layer(self):
        layer = _make_layer(instrument='loki', source_names=['det'])
        cell = _make_cell(layer)
        result = _build_save_filename_from_cell(cell)
        assert result is not None
        assert result.startswith('LOKI_')
