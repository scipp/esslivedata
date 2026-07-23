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
from ess.livedata.dashboard.plot_params import (
    TimeWindowMixin,
    TimeWindowMode,
    TimeWindowParams,
)
from ess.livedata.dashboard.widgets.plot_widgets import (
    _create_configure_button_or_menu,
    _create_toolbar_visibility_button,
    _format_window_info,
    create_cell_titlebar,
    derive_cell_title,
    get_plot_cell_display_info,
)


class TestPerPanelToolHooks:
    """Per-panel tools must carry the stable lt-* automation hooks.

    Unlike the create_tool_button-based buttons, these are hand-rolled (a
    toggling icon, a dropdown menu), so they tag themselves and can silently
    drift. See .claude/rules/dashboard-widgets.md.
    """

    def test_toolbar_visibility_toggle_has_layer_details_hook(self) -> None:
        button = _create_toolbar_visibility_button(
            visible=True, on_toggle=lambda _: None
        )
        assert 'lt-tool' in button.css_classes
        assert 'lt-tool-layer-details' in button.css_classes

    def test_multi_layer_configure_menu_has_settings_hook(self) -> None:
        layers = [(LayerId('a'), 'A'), (LayerId('b'), 'B')]
        menu = _create_configure_button_or_menu(
            layers=layers, on_configure=lambda _: None
        )
        assert 'lt-tool' in menu.css_classes
        assert 'lt-tool-settings' in menu.css_classes

    def test_titlebar_pencil_carries_per_cell_context_hook(self) -> None:
        """The edit pencil must carry the caller's per-cell context class.

        A rebuilt cell's DOM position is not stable, so automation addresses
        the pencil via the cell-position context (e.g. lt-cell-r0c0) instead
        of DOM order.
        """
        titlebar = create_cell_titlebar(
            title='T',
            has_user_title=False,
            on_edit_title_callback=lambda: None,
            configure_layers=[(LayerId('a'), 'A')],
            on_configure_layer=lambda _: None,
            toolbars_visible=True,
            on_toggle_toolbars_callback=lambda _: None,
            css_classes=['lt-cell-r0c0'],
        )
        pencils = [
            obj
            for obj in titlebar.objects
            if 'lt-tool-pencil' in (getattr(obj, 'css_classes', None) or [])
        ]
        assert len(pencils) == 1
        assert 'lt-cell-r0c0' in pencils[0].css_classes


class _FakeParams(TimeWindowMixin):
    pass


class TestFormatWindowInfo:
    def test_returns_since_run_start_for_since_start_mode(self) -> None:
        params = _FakeParams(
            time_window=TimeWindowParams(mode=TimeWindowMode.since_start)
        )
        assert _format_window_info(params) == 'since run start'

    def test_returns_empty_for_window_mode(self) -> None:
        """Window mode shows no static label; data range comes from the plot."""
        params = _FakeParams(
            time_window=TimeWindowParams(
                mode=TimeWindowMode.window, window_duration_seconds=10
            )
        )
        assert _format_window_info(params) == ''

    def test_returns_empty_for_window_mode_zero_duration(self) -> None:
        params = _FakeParams(
            time_window=TimeWindowParams(
                mode=TimeWindowMode.window, window_duration_seconds=0
            )
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
            params=_FakeParams(
                time_window=TimeWindowParams(mode=TimeWindowMode.since_start)
            ),
        )

        title, _ = get_plot_cell_display_info(config, registry)
        assert 'since run start' in title


class TestPlotterTitleInDisplayInfo:
    """Plotter title disambiguates layers sharing workflow/output/source (#645)."""

    class _ImageOutputs(WorkflowOutputsBase):
        image: sc.DataArray = pydantic.Field(
            default_factory=lambda: sc.DataArray(
                sc.zeros(dims=['x'], shape=[0], unit='counts'),
                coords={'x': sc.arange('x', 0, unit='m')},
            ),
            title='Image',
        )

    def _config(self, plot_name: str) -> PlotConfig:
        wf_id = WorkflowId(instrument='test', name='wf', version=1)
        spec = WorkflowSpec(
            instrument='test',
            name='wf',
            version=1,
            title='Detector counts',
            description='D',
            outputs=self._ImageOutputs,
            params=None,
            source_names=['src1'],
            group=REDUCTION,
        )
        self._registry = {wf_id: spec}
        return PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=wf_id, source_names=['src1'], view_name='image'
                )
            },
            plot_name=plot_name,
            params=_FakeParams(),
        )

    def test_title_includes_plotter_title(self) -> None:
        config = self._config('rectangles_readback')
        title, _ = get_plot_cell_display_info(config, self._registry)
        assert 'ROI Rectangles (Readback)' in title

    def test_readback_and_request_layers_have_distinct_titles(self) -> None:
        readback, _ = get_plot_cell_display_info(
            self._config('rectangles_readback'), self._registry
        )
        request, _ = get_plot_cell_display_info(
            self._config('rectangles_request'), self._registry
        )
        assert readback != request

    def test_description_includes_plotter_title(self) -> None:
        _, description = get_plot_cell_display_info(
            self._config('rectangles_readback'), self._registry
        )
        assert 'Plotter: ROI Rectangles (Readback)' in description

    def test_plotter_title_suppressed_when_equal_to_output(self) -> None:
        # Output 'Image' rendered by the 'image' plotter (title 'Image').
        config = self._config('image')
        title, description = get_plot_cell_display_info(config, self._registry)
        assert title == 'Detector counts &rarr; Image (src1)'
        assert 'Plotter:' not in description

    def test_unregistered_plotter_omits_plotter_title(self) -> None:
        config = self._config('does_not_exist')
        title, description = get_plot_cell_display_info(config, self._registry)
        assert title == 'Detector counts &rarr; Image (src1)'
        assert 'Plotter:' not in description


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

    def test_single_layer_single_source(self) -> None:
        wf = self._wf()
        cell = PlotCell(geometry=_GEO, layers=[_make_layer(wf, ['Front Right'])])
        assert (
            derive_cell_title(cell, self._registry(wf))
            == 'Detector XY Projection (Front Right)'
        )

    def test_output_name_is_not_shown(self) -> None:
        wf = self._wf()
        cell = PlotCell(
            geometry=_GEO,
            layers=[
                _make_layer(wf, ['Front Right'], view_name='image'),
                _make_layer(wf, ['Front Right'], view_name='roi'),
            ],
        )
        title = derive_cell_title(cell, self._registry(wf))
        assert title == 'Detector XY Projection (Front Right)'
        assert 'Image' not in title
        assert 'ROI' not in title

    def test_source_title_callback_applied(self) -> None:
        wf = self._wf()
        cell = PlotCell(geometry=_GEO, layers=[_make_layer(wf, ['s1'])])
        title = derive_cell_title(
            cell, self._registry(wf), get_source_title=lambda s: f'Title:{s}'
        )
        assert title == 'Detector XY Projection (Title:s1)'

    def test_multiple_sources_drop_source(self) -> None:
        wf = self._wf()
        cell = PlotCell(
            geometry=_GEO,
            layers=[_make_layer(wf, ['s1']), _make_layer(wf, ['s2'])],
        )
        assert derive_cell_title(cell, self._registry(wf)) == 'Detector XY Projection'

    def test_single_layer_multi_source_drops_source(self) -> None:
        wf = self._wf()
        cell = PlotCell(geometry=_GEO, layers=[_make_layer(wf, ['s1', 's2'])])
        assert derive_cell_title(cell, self._registry(wf)) == 'Detector XY Projection'

    def test_uses_first_layer_workflow_title(self) -> None:
        wf_a = self._wf('a')
        wf_b = self._wf('b')
        registry = {**self._registry(wf_a), **self._registry(wf_b)}
        cell = PlotCell(
            geometry=_GEO,
            layers=[_make_layer(wf_a, ['Front Right']), _make_layer(wf_b, ['s2'])],
        )
        # Different sources across layers -> source dropped, first workflow used.
        assert derive_cell_title(cell, registry) == 'Detector XY Projection'

    def test_static_layers_excluded_from_derivation(self) -> None:
        wf = self._wf()
        cell = PlotCell(
            geometry=_GEO,
            layers=[_make_layer(wf, ['Front Right']), _make_static_layer('Box')],
        )
        assert (
            derive_cell_title(cell, self._registry(wf))
            == 'Detector XY Projection (Front Right)'
        )

    def test_only_static_layers_uses_first_static_title(self) -> None:
        cell = PlotCell(geometry=_GEO, layers=[_make_static_layer('Box')])
        assert derive_cell_title(cell, {}) == 'Rectangle &rarr; Box'

    def test_empty_cell_returns_empty(self) -> None:
        cell = PlotCell(geometry=_GEO, layers=[])
        assert derive_cell_title(cell, {}) == ''
