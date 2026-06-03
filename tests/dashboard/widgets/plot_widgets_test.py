# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pydantic
import scipp as sc

from ess.livedata.config.workflow_spec import (
    REDUCTION,
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.dashboard.data_roles import PRIMARY
from ess.livedata.dashboard.plot_orchestrator import (
    CellGeometry,
    DataSourceConfig,
    Layer,
    LayerId,
    PlotCell,
    PlotConfig,
)
from ess.livedata.dashboard.plot_params import WindowMode, WindowParams
from ess.livedata.dashboard.widgets.plot_widgets import (
    _format_window_info,
    derive_cell_title,
    get_plot_cell_display_info,
)


class _FakeParams(pydantic.BaseModel):
    window: WindowParams = pydantic.Field(default_factory=WindowParams)


class TestFormatWindowInfo:
    def test_returns_since_run_start_for_since_start_mode(self) -> None:
        params = _FakeParams(window=WindowParams(mode=WindowMode.since_start))
        assert _format_window_info(params) == 'since run start'

    def test_returns_empty_for_window_mode(self) -> None:
        """Window mode shows no static label; data range comes from the plot."""
        params = _FakeParams(
            window=WindowParams(mode=WindowMode.window, window_duration_seconds=10)
        )
        assert _format_window_info(params) == ''

    def test_returns_empty_for_window_mode_zero_duration(self) -> None:
        params = _FakeParams(
            window=WindowParams(mode=WindowMode.window, window_duration_seconds=0)
        )
        assert _format_window_info(params) == ''

    def test_returns_empty_when_no_window_attr(self) -> None:
        class NoWindowParams(pydantic.BaseModel):
            pass

        assert _format_window_info(NoWindowParams()) == ''


class TestGetPlotCellDisplayInfo:
    def test_cumulative_output_omits_window_info(self) -> None:
        """Toolbar title should not show window mode for cumulative outputs."""

        class CumulativeOutputs(WorkflowOutputsBase):
            cumulative: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x'], shape=[0], unit='counts'),
                    coords={'x': sc.arange('x', 0, unit='m')},
                ),
                title='Cumulative',
            )

        wf_id = WorkflowId(instrument='test', name='wf', version=1)
        spec = WorkflowSpec(
            instrument='test',
            name='wf',
            version=1,
            title='Beam monitor data',
            description='D',
            outputs=CumulativeOutputs,
            params=None,
            source_names=['src1'],
            group=REDUCTION,
        )
        registry = {wf_id: spec}

        config = PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=wf_id,
                    source_names=['src1'],
                    view_name='cumulative',
                )
            },
            plot_name='lines',
            params=_FakeParams(),
            supports_windowing=False,
        )

        title, _ = get_plot_cell_display_info(config, registry)
        assert 'latest' not in title
        assert 'window' not in title

    def test_since_start_mode_shows_label_in_title(self) -> None:
        """Toolbar title should show 'since run start' label for since_start mode."""

        class CurrentOutputs(WorkflowOutputsBase):
            current: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['time', 'x'], shape=[0, 0], unit='counts'),
                    coords={
                        'time': sc.arange('time', 0, unit='s'),
                        'x': sc.arange('x', 0, unit='m'),
                    },
                ),
                title='Current',
            )

        wf_id = WorkflowId(instrument='test', name='wf', version=1)
        spec = WorkflowSpec(
            instrument='test',
            name='wf',
            version=1,
            title='Beam monitor data',
            description='D',
            outputs=CurrentOutputs,
            params=None,
            source_names=['src1'],
            group=REDUCTION,
        )
        registry = {wf_id: spec}

        config = PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=wf_id,
                    source_names=['src1'],
                    view_name='current',
                )
            },
            plot_name='lines',
            params=_FakeParams(window=WindowParams(mode=WindowMode.since_start)),
        )

        title, _ = get_plot_cell_display_info(config, registry)
        assert 'since run start' in title


_GEO = CellGeometry(row=0, col=0, row_span=1, col_span=1)


def _make_layer(workflow_id, source_names, view_name='result', plot_name='lines'):
    from uuid import uuid4

    config = PlotConfig(
        data_sources={
            PRIMARY: DataSourceConfig(
                workflow_id=workflow_id,
                source_names=source_names,
                view_name=view_name,
            )
        },
        plot_name=plot_name,
        params=_FakeParams(),
    )
    return Layer(layer_id=LayerId(uuid4()), config=config)


class TestDeriveCellTitle:
    @staticmethod
    def _wf():
        return WorkflowId(instrument='test', name='wf', version=1)

    def test_shared_single_source_uses_source_title(self) -> None:
        wf = self._wf()
        cell = PlotCell(
            geometry=_GEO,
            layers=[_make_layer(wf, ['s1']), _make_layer(wf, ['s1'])],
        )
        title = derive_cell_title(cell, {}, get_source_title=lambda s: f'Title:{s}')
        assert title == 'Title:s1'

    def test_shared_common_among_multi_source_layers(self) -> None:
        wf = self._wf()
        cell = PlotCell(
            geometry=_GEO,
            layers=[_make_layer(wf, ['s1', 's2']), _make_layer(wf, ['s1', 's3'])],
        )
        assert derive_cell_title(cell, {}) == 's1'

    def test_single_layer_single_source_uses_source(self) -> None:
        cell = PlotCell(geometry=_GEO, layers=[_make_layer(self._wf(), ['s1'])])
        assert derive_cell_title(cell, {}) == 's1'

    def test_disjoint_sources_multi_layer_falls_back_to_count(self) -> None:
        wf = self._wf()
        cell = PlotCell(
            geometry=_GEO,
            layers=[_make_layer(wf, ['s1']), _make_layer(wf, ['s2'])],
        )
        assert derive_cell_title(cell, {}) == '2 layers'

    def test_single_layer_multi_source_falls_back_to_display_title(self) -> None:
        cell = PlotCell(
            geometry=_GEO,
            layers=[_make_layer(self._wf(), ['s1', 's2'], view_name='current')],
        )
        assert '2 sources' in derive_cell_title(cell, {})

    def test_empty_cell_returns_empty(self) -> None:
        cell = PlotCell(geometry=_GEO, layers=[])
        assert derive_cell_title(cell, {}) == ''
