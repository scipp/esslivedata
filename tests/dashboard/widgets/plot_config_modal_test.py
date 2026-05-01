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
from ess.livedata.dashboard.data_roles import X_AXIS, Y_AXIS
from ess.livedata.dashboard.plot_orchestrator import DataSourceConfig
from ess.livedata.dashboard.widgets.plot_config_modal import (
    _build_timeseries_options,
    _inject_axis_source_titles,
    _resolve_axis_source_titles,
    _resolve_output_display_hints,
)


class FakeInstrumentConfig:
    """Minimal fake that provides get_source_title."""

    def __init__(self, titles: dict[str, str]):
        self._titles = titles

    def get_source_title(self, source_name: str) -> str:
        return self._titles.get(source_name, source_name)


def _make_workflow_id(name: str = "timeseries") -> WorkflowId:
    return WorkflowId(instrument="test", name=name, version=1)


class Bins(pydantic.BaseModel):
    x_axis_source: str = pydantic.Field(default="", title="X Axis", frozen=True)
    y_axis_source: str = pydantic.Field(default="", title="Y Axis", frozen=True)
    n_bins: int = 50


class Params(pydantic.BaseModel):
    bins: Bins = Bins()


def _make_axis_sources(
    x_source: str = "monitor_cave",
    y_source: str | None = None,
) -> dict[str, DataSourceConfig]:
    wf_id = _make_workflow_id()
    sources: dict[str, DataSourceConfig] = {
        X_AXIS: DataSourceConfig(
            workflow_id=wf_id, source_names=[x_source], output_name="delta"
        ),
    }
    if y_source is not None:
        sources[Y_AXIS] = DataSourceConfig(
            workflow_id=wf_id, source_names=[y_source], output_name="delta"
        )
    return sources


class TestResolveAxisSourceTitles:
    def test_resolves_titles_with_instrument_config(self):
        axis_sources = _make_axis_sources("monitor_cave", "monitor_bunker")
        instrument = FakeInstrumentConfig(
            {"monitor_cave": "Cave Monitor", "monitor_bunker": "Bunker Monitor"}
        )
        result = _resolve_axis_source_titles(axis_sources, instrument)
        assert result == {
            "x_axis_source": "Cave Monitor",
            "y_axis_source": "Bunker Monitor",
        }

    def test_falls_back_to_source_name_when_title_not_found(self):
        axis_sources = _make_axis_sources("unknown_source")
        instrument = FakeInstrumentConfig({})
        result = _resolve_axis_source_titles(axis_sources, instrument)
        assert result == {"x_axis_source": "unknown_source"}

    def test_empty_axis_sources_returns_empty(self):
        instrument = FakeInstrumentConfig({})
        assert _resolve_axis_source_titles({}, instrument) == {}

    def test_single_output_workflow_omits_output_title(self):
        class SingleOutput(WorkflowOutputsBase):
            delta: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
                title='Delta',
            )

        wf_id = _make_workflow_id()
        spec = _make_workflow_spec("Timeseries data", SingleOutput)
        axis_sources = _make_axis_sources("monitor_cave")
        instrument = FakeInstrumentConfig({"monitor_cave": "Cave Monitor"})

        result = _resolve_axis_source_titles(axis_sources, instrument, {wf_id: spec})
        assert result == {"x_axis_source": "Cave Monitor"}

    def test_multi_output_workflow_appends_output_title(self):
        class MultiOutput(WorkflowOutputsBase):
            delta: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
                title='Delta',
            )
            cumulative: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
                title='Cumulative',
            )

        wf_id = _make_workflow_id()
        spec = _make_workflow_spec("Monitor data", MultiOutput)
        axis_sources = _make_axis_sources("monitor_cave")
        instrument = FakeInstrumentConfig({"monitor_cave": "Cave Monitor"})

        result = _resolve_axis_source_titles(axis_sources, instrument, {wf_id: spec})
        assert result == {"x_axis_source": "Cave Monitor (Delta)"}

    def test_without_workflow_registry_omits_output_title(self):
        axis_sources = _make_axis_sources("monitor_cave")
        instrument = FakeInstrumentConfig({"monitor_cave": "Cave Monitor"})

        result = _resolve_axis_source_titles(axis_sources, instrument)
        assert result == {"x_axis_source": "Cave Monitor"}

    def test_multi_output_both_axes(self):
        class MultiOutput(WorkflowOutputsBase):
            delta: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
                title='Delta',
            )
            cumulative: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
                title='Cumulative',
            )

        wf_id = _make_workflow_id()
        spec = _make_workflow_spec("Monitor data", MultiOutput)
        axis_sources = _make_axis_sources("monitor_cave", "monitor_bunker")
        instrument = FakeInstrumentConfig(
            {"monitor_cave": "Cave Monitor", "monitor_bunker": "Bunker Monitor"}
        )

        result = _resolve_axis_source_titles(axis_sources, instrument, {wf_id: spec})
        assert result == {
            "x_axis_source": "Cave Monitor (Delta)",
            "y_axis_source": "Bunker Monitor (Delta)",
        }


class TestInjectAxisSourceTitles:
    def test_injects_titles_with_instrument_config(self):
        params = Params()
        axis_sources = _make_axis_sources("monitor_cave")
        instrument = FakeInstrumentConfig({"monitor_cave": "Cave Monitor"})
        result = _inject_axis_source_titles(params, axis_sources, instrument)
        assert result.bins.x_axis_source == "Cave Monitor"

    def test_no_change_when_no_axis_sources(self):
        params = Params(bins=Bins(x_axis_source="existing"))
        instrument = FakeInstrumentConfig({})
        result = _inject_axis_source_titles(params, {}, instrument)
        assert result.bins.x_axis_source == "existing"

    def test_no_change_when_no_bins(self):
        class NoBinsParams(pydantic.BaseModel):
            color: str = "red"

        params = NoBinsParams()
        axis_sources = _make_axis_sources("monitor_cave")
        instrument = FakeInstrumentConfig({"monitor_cave": "Cave Monitor"})
        result = _inject_axis_source_titles(params, axis_sources, instrument)
        assert result.color == "red"

    def test_injects_titles_into_dict(self):
        params = {"bins": {"x_axis_source": "old", "n_bins": 50}}
        axis_sources = _make_axis_sources("monitor_cave")
        instrument = FakeInstrumentConfig({"monitor_cave": "Cave Monitor"})
        result = _inject_axis_source_titles(params, axis_sources, instrument)
        assert result["bins"]["x_axis_source"] == "Cave Monitor"
        assert result["bins"]["n_bins"] == 50

    def test_dict_without_bins_unchanged(self):
        params = {"color": "red"}
        axis_sources = _make_axis_sources("monitor_cave")
        instrument = FakeInstrumentConfig({"monitor_cave": "Cave Monitor"})
        result = _inject_axis_source_titles(params, axis_sources, instrument)
        assert result == {"color": "red"}

    def test_injects_output_title_for_multi_output_workflow(self):
        class MultiOutput(WorkflowOutputsBase):
            delta: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
                title='Delta',
            )
            cumulative: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
                title='Cumulative',
            )

        wf_id = _make_workflow_id()
        spec = _make_workflow_spec("Monitor data", MultiOutput)
        params = Params()
        axis_sources = _make_axis_sources("monitor_cave")
        instrument = FakeInstrumentConfig({"monitor_cave": "Cave Monitor"})
        result = _inject_axis_source_titles(
            params, axis_sources, instrument, {wf_id: spec}
        )
        assert result.bins.x_axis_source == "Cave Monitor (Delta)"


def _make_workflow_spec(title: str, outputs: type[WorkflowOutputsBase]) -> WorkflowSpec:
    return WorkflowSpec(
        instrument="test",
        name="test_wf",
        version=1,
        title=title,
        description="",
        params=None,
        outputs=outputs,
        group=REDUCTION,
    )


class TestBuildTimeseriesOptions:
    def test_single_output_workflow_omits_output_name(self):
        """When a workflow has only one output, the display name has no suffix."""

        class SingleOutput(WorkflowOutputsBase):
            delta: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
                title='Delta',
            )

        wf_id = _make_workflow_id()
        spec = _make_workflow_spec("Timeseries data", SingleOutput)
        timeseries = [(wf_id, "mon1", "delta")]
        instrument = FakeInstrumentConfig({"mon1": "Monitor 1"})

        options = _build_timeseries_options(timeseries, {wf_id: spec}, instrument)
        display_names = list(options.keys())
        assert display_names == ["Timeseries data: Monitor 1"]

    def test_multi_output_workflow_includes_output_title(self):
        """When a workflow has multiple outputs, the display name includes the title."""

        class MultiOutput(WorkflowOutputsBase):
            delta: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
                title='Delta',
            )
            cumulative: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
                title='Cumulative',
            )

        wf_id = _make_workflow_id()
        spec = _make_workflow_spec("Monitor data", MultiOutput)
        timeseries = [(wf_id, "mon1", "delta")]
        instrument = FakeInstrumentConfig({"mon1": "Monitor 1"})

        options = _build_timeseries_options(timeseries, {wf_id: spec}, instrument)
        display_names = list(options.keys())
        assert display_names == ["Monitor data: Monitor 1 (Delta)"]

    def test_no_instrument_config_uses_source_name(self):
        class SingleOutput(WorkflowOutputsBase):
            delta: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
                title='Delta',
            )

        wf_id = _make_workflow_id()
        spec = _make_workflow_spec("Timeseries data", SingleOutput)
        timeseries = [(wf_id, "raw_source", "delta")]

        options = _build_timeseries_options(timeseries, {wf_id: spec}, None)
        assert list(options.keys()) == ["Timeseries data: raw_source"]


class TestResolveOutputDisplayHints:
    def test_static_overlay_preselects_all_and_no_hidden_fields(self):
        hints = _resolve_output_display_hints(
            is_static=True, workflow_spec=None, output_name="any"
        )
        assert hints.preselect_all_sources is True
        assert hints.hidden_fields == frozenset()

    def test_0d_output_preselects_all(self):
        class Outputs(WorkflowOutputsBase):
            counts: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
            )

        spec = _make_workflow_spec("Scalar output", Outputs)
        hints = _resolve_output_display_hints(
            is_static=False, workflow_spec=spec, output_name="counts"
        )
        assert hints.preselect_all_sources is True

    def test_1d_output_preselects_all(self):
        class Outputs(WorkflowOutputsBase):
            spectrum: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.zeros(sizes={'wavelength': 0})),
            )

        spec = _make_workflow_spec("1D output", Outputs)
        hints = _resolve_output_display_hints(
            is_static=False, workflow_spec=spec, output_name="spectrum"
        )
        assert hints.preselect_all_sources is True

    def test_2d_output_does_not_preselect_all(self):
        class Outputs(WorkflowOutputsBase):
            image: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.zeros(sizes={'x': 0, 'y': 0})),
            )

        spec = _make_workflow_spec("2D output", Outputs)
        hints = _resolve_output_display_hints(
            is_static=False, workflow_spec=spec, output_name="image"
        )
        assert hints.preselect_all_sources is False

    def test_3d_output_does_not_preselect_all(self):
        class Outputs(WorkflowOutputsBase):
            volume: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(sizes={'x': 0, 'y': 0, 'z': 0})
                ),
            )

        spec = _make_workflow_spec("3D output", Outputs)
        hints = _resolve_output_display_hints(
            is_static=False, workflow_spec=spec, output_name="volume"
        )
        assert hints.preselect_all_sources is False

    def test_output_with_time_coord_does_not_hide_window(self):
        class Outputs(WorkflowOutputsBase):
            spectrum: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(sizes={'time': 0, 'wavelength': 0}),
                    coords={'time': sc.zeros(sizes={'time': 0})},
                ),
            )

        spec = _make_workflow_spec("Windowed output", Outputs)
        hints = _resolve_output_display_hints(
            is_static=False, workflow_spec=spec, output_name="spectrum"
        )
        assert 'window' not in hints.hidden_fields

    def test_output_without_time_coord_hides_window(self):
        class Outputs(WorkflowOutputsBase):
            total: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.zeros(sizes={'wavelength': 0})),
            )

        spec = _make_workflow_spec("Cumulative output", Outputs)
        hints = _resolve_output_display_hints(
            is_static=False, workflow_spec=spec, output_name="total"
        )
        assert 'window' in hints.hidden_fields

    def test_unknown_output_preselects_all(self):
        class Outputs(WorkflowOutputsBase):
            data: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
            )

        spec = _make_workflow_spec("Some workflow", Outputs)
        hints = _resolve_output_display_hints(
            is_static=False, workflow_spec=spec, output_name="nonexistent"
        )
        assert hints.preselect_all_sources is True
