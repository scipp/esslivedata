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
