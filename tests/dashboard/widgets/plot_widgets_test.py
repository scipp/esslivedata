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
from ess.livedata.dashboard.plot_orchestrator import DataSourceConfig, PlotConfig
from ess.livedata.dashboard.plot_params import WindowMode, WindowParams
from ess.livedata.dashboard.widgets.plot_widgets import (
    _format_window_info,
    get_plot_cell_display_info,
)


class _FakeParams(pydantic.BaseModel):
    window: WindowParams = pydantic.Field(default_factory=WindowParams)


class TestFormatWindowInfo:
    def test_returns_empty_when_supports_windowing_false(self) -> None:
        params = _FakeParams()
        assert _format_window_info(params, supports_windowing=False) == ''

    def test_returns_latest_for_latest_mode(self) -> None:
        params = _FakeParams(window=WindowParams(mode=WindowMode.latest))
        assert _format_window_info(params) == 'latest'

    def test_returns_window_info_for_window_mode(self) -> None:
        params = _FakeParams(
            window=WindowParams(mode=WindowMode.window, window_duration_seconds=10)
        )
        assert _format_window_info(params) == '10s window'

    def test_returns_empty_when_no_window_attr(self) -> None:
        class NoWindowParams(pydantic.BaseModel):
            pass

        assert _format_window_info(NoWindowParams()) == ''

    def test_supports_windowing_false_overrides_window_mode(self) -> None:
        """Even with window mode set, supports_windowing=False returns empty."""
        params = _FakeParams(
            window=WindowParams(mode=WindowMode.window, window_duration_seconds=5)
        )
        assert _format_window_info(params, supports_windowing=False) == ''


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
                    output_name='cumulative',
                )
            },
            plot_name='lines',
            params=_FakeParams(),
            supports_windowing=False,
        )

        title, _ = get_plot_cell_display_info(config, registry)
        assert 'latest' not in title
        assert 'window' not in title

    def test_current_output_shows_window_info(self) -> None:
        """Toolbar title should show window mode for outputs with time coord."""

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
                    output_name='current',
                )
            },
            plot_name='lines',
            params=_FakeParams(),
        )

        title, _ = get_plot_cell_display_info(config, registry)
        assert 'latest' in title
