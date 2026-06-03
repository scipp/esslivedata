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
    format_freshness_html,
    format_layer_time_html,
    get_plot_cell_display_info,
)
from ess.livedata.dashboard.widgets.styles import FreshnessPill


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


def _make_layer(
    workflow_id, source_names, view_name='image', plot_name='lines', params=None
):
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
        params=params if params is not None else _FakeParams(),
    )
    return Layer(layer_id=LayerId(uuid4()), config=config)


def _make_static_layer(name, plot_name='rectangle'):
    """A static overlay: single primary source with empty source_names."""
    from uuid import uuid4

    config = PlotConfig(
        data_sources={
            PRIMARY: DataSourceConfig(
                workflow_id=WorkflowId(instrument='test', name='static', version=1),
                source_names=[],
                view_name=name,
            )
        },
        plot_name=plot_name,
        params=_FakeParams(),
    )
    return Layer(layer_id=LayerId(uuid4()), config=config)


class _XYOutputs(WorkflowOutputsBase):
    image: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['x'], shape=[0], unit='counts'),
            coords={'x': sc.arange('x', 0, unit='m')},
        ),
        title='Image',
    )
    roi: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['x'], shape=[0], unit='counts'),
            coords={'x': sc.arange('x', 0, unit='m')},
        ),
        title='ROI',
    )


class TestDeriveCellTitle:
    @staticmethod
    def _wf(name='wf'):
        return WorkflowId(instrument='test', name=name, version=1)

    @staticmethod
    def _registry(wf):
        spec = WorkflowSpec(
            instrument='test',
            name=wf.name,
            version=1,
            title='Detector XY Projection',
            description='D',
            outputs=_XYOutputs,
            params=None,
            source_names=['Front Right'],
            group=REDUCTION,
        )
        return {wf: spec}

    def test_single_layer_uses_full_display_title(self) -> None:
        wf = self._wf()
        cell = PlotCell(geometry=_GEO, layers=[_make_layer(wf, ['Front Right'])])
        assert (
            derive_cell_title(cell, self._registry(wf))
            == 'Detector XY Projection &rarr; Image (Front Right)'
        )

    def test_shared_workflow_and_source_varying_output_drops_output(self) -> None:
        wf = self._wf()
        cell = PlotCell(
            geometry=_GEO,
            layers=[
                _make_layer(wf, ['Front Right'], view_name='image'),
                _make_layer(wf, ['Front Right'], view_name='roi'),
            ],
        )
        assert (
            derive_cell_title(cell, self._registry(wf))
            == 'Detector XY Projection (Front Right)'
        )

    def test_shared_output_multi_layer_flags_extra_layers(self) -> None:
        wf = self._wf()
        cell = PlotCell(
            geometry=_GEO,
            layers=[
                _make_layer(wf, ['Front Right'], view_name='image'),
                _make_layer(wf, ['Front Right'], view_name='image'),
            ],
        )
        assert (
            derive_cell_title(cell, self._registry(wf))
            == 'Detector XY Projection &rarr; Image (Front Right) (+1 more)'
        )

    def test_source_title_callback_applied(self) -> None:
        wf = self._wf()
        cell = PlotCell(geometry=_GEO, layers=[_make_layer(wf, ['s1'])])
        title = derive_cell_title(
            cell, self._registry(wf), get_source_title=lambda s: f'Title:{s}'
        )
        assert title == 'Detector XY Projection &rarr; Image (Title:s1)'

    def test_varying_source_is_dropped(self) -> None:
        wf = self._wf()
        cell = PlotCell(
            geometry=_GEO,
            layers=[_make_layer(wf, ['s1']), _make_layer(wf, ['s2'])],
        )
        assert (
            derive_cell_title(cell, self._registry(wf))
            == 'Detector XY Projection &rarr; Image (+1 more)'
        )

    def test_single_layer_multi_source_drops_source(self) -> None:
        wf = self._wf()
        cell = PlotCell(geometry=_GEO, layers=[_make_layer(wf, ['s1', 's2'])])
        title = derive_cell_title(cell, self._registry(wf))
        assert title == 'Detector XY Projection &rarr; Image'
        assert 'sources' not in title

    def test_shared_window_is_shown(self) -> None:
        wf = self._wf()
        params = _FakeParams(window=WindowParams(mode=WindowMode.since_start))
        cell = PlotCell(
            geometry=_GEO,
            layers=[_make_layer(wf, ['Front Right'], params=params)],
        )
        assert (
            derive_cell_title(cell, self._registry(wf))
            == 'Detector XY Projection &rarr; Image (Front Right, since run start)'
        )

    def test_multiple_workflows_fall_back_to_first_layer(self) -> None:
        wf_a = self._wf('a')
        wf_b = self._wf('b')
        registry = {**self._registry(wf_a), **self._registry(wf_b)}
        cell = PlotCell(
            geometry=_GEO,
            layers=[_make_layer(wf_a, ['Front Right']), _make_layer(wf_b, ['s2'])],
        )
        first_title, _ = get_plot_cell_display_info(cell.layers[0].config, registry)
        assert derive_cell_title(cell, registry) == f'{first_title} (+1 more)'

    def test_static_layers_excluded_from_derivation(self) -> None:
        wf = self._wf()
        cell = PlotCell(
            geometry=_GEO,
            layers=[_make_layer(wf, ['Front Right']), _make_static_layer('Box')],
        )
        assert (
            derive_cell_title(cell, self._registry(wf))
            == 'Detector XY Projection &rarr; Image (Front Right)'
        )

    def test_only_static_layers_uses_first_static_title(self) -> None:
        cell = PlotCell(geometry=_GEO, layers=[_make_static_layer('Box')])
        assert derive_cell_title(cell, {}) == 'Rectangle &rarr; Box'

    def test_empty_cell_returns_empty(self) -> None:
        cell = PlotCell(geometry=_GEO, layers=[])
        assert derive_cell_title(cell, {}) == ''


class TestFreshnessPill:
    def test_none_renders_nothing(self) -> None:
        assert format_freshness_html(None) == ''

    def test_fresh_band_colors_and_label(self) -> None:
        html = format_freshness_html(2.3)
        assert FreshnessPill.FRESH[0] in html  # background
        assert '2.3s' in html
        assert 'border-radius' in html

    def test_stale_band(self) -> None:
        html = format_freshness_html(12.0)
        assert FreshnessPill.STALE[0] in html
        assert '12s' in html

    def test_old_band(self) -> None:
        html = format_freshness_html(41.0)
        assert FreshnessPill.OLD[0] in html
        assert '41s' in html

    def test_minutes_for_large_lag(self) -> None:
        assert '3m' in format_freshness_html(200.0)

    def test_tooltip_is_html_escaped(self) -> None:
        html = format_freshness_html(2.0, tooltip='12:00 <range>')
        assert 'title="12:00 &lt;range&gt;"' in html


class TestLayerTimeHtml:
    def test_empty_renders_nothing(self) -> None:
        assert format_layer_time_html('') == ''

    def test_wraps_text_in_muted_span(self) -> None:
        html = format_layer_time_html('14:35:01 - 14:35:07 (Lag: 2.3s)')
        assert '14:35:01 - 14:35:07 (Lag: 2.3s)' in html
        assert '<span' in html
