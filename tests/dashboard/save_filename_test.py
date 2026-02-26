# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pydantic
import scipp as sc

from ess.livedata.config.workflow_spec import (
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
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

_WF_ID = WorkflowId(instrument='dream', namespace='ns', name='wf', version=1)


class _TestOutputs(WorkflowOutputsBase):
    counts: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.zeros(dims=['x'], shape=[0])),
        title='Counts',
    )
    spectrum: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.zeros(dims=['x'], shape=[0])),
        title='Spectrum',
    )


_WORKFLOW_REGISTRY = {
    _WF_ID: WorkflowSpec(
        instrument='dream',
        name='wf',
        version=1,
        title='Test Workflow',
        description='test',
        params=None,
        outputs=_TestOutputs,
    )
}

_SOURCE_TITLES = {
    'monitor': 'Cave Monitor',
    'det_bank1': 'Mantle',
    'det_bank2': 'SANS',
}


def _get_source_title(name: str) -> str:
    return _SOURCE_TITLES.get(name, name)


def _make_layer(
    source_names: list[str] | None = None,
    output_name: str = 'counts',
    *,
    instrument: str = 'dream',
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


def _build(cell: PlotCell) -> str | None:
    return _build_save_filename_from_cell(cell, _WORKFLOW_REGISTRY, _get_source_title)


class TestBuildSaveFilenameFromCell:
    def test_single_layer_uses_output_and_source_titles(self):
        cell = _make_cell(_make_layer())
        assert _build(cell) == 'DREAM_Counts_Cave-Monitor'

    def test_multiple_layers_combines_titles(self):
        layer_a = _make_layer(source_names=['det_bank1'], output_name='counts')
        layer_b = _make_layer(source_names=['det_bank2'], output_name='spectrum')
        cell = _make_cell(layer_a, layer_b)
        result = _build(cell)
        assert result is not None
        assert 'Mantle' in result
        assert 'SANS' in result
        assert 'Counts' in result
        assert 'Spectrum' in result

    def test_static_layers_are_excluded(self):
        data_layer = _make_layer(source_names=['monitor'], output_name='counts')
        static_layer = _make_layer(static=True, output_name='reference_line')
        cell = _make_cell(data_layer, static_layer)
        assert _build(cell) == 'DREAM_Counts_Cave-Monitor'

    def test_all_static_returns_none(self):
        cell = _make_cell(_make_layer(static=True))
        assert _build(cell) is None

    def test_instrument_from_first_non_static_layer(self):
        layer = _make_layer(source_names=['det_bank1'], instrument='loki')
        cell = _make_cell(layer)
        result = _build(cell)
        assert result is not None
        assert result.startswith('LOKI_')

    def test_unknown_source_falls_back_to_name(self):
        layer = _make_layer(source_names=['unknown_det'])
        cell = _make_cell(layer)
        result = _build(cell)
        assert result is not None
        assert 'unknown_det' in result
