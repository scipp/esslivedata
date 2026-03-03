# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pydantic

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.data_roles import X_AXIS, Y_AXIS
from ess.livedata.dashboard.plot_orchestrator import DataSourceConfig
from ess.livedata.dashboard.widgets.plot_config_modal import (
    _inject_axis_source_names,
    _resolve_axis_source_titles,
)


class FakeInstrumentConfig:
    """Minimal fake that provides get_source_title."""

    def __init__(self, titles: dict[str, str]):
        self._titles = titles

    def get_source_title(self, source_name: str) -> str:
        return self._titles.get(source_name, source_name)


def _make_workflow_id(name: str = "timeseries") -> WorkflowId:
    return WorkflowId(instrument="test", namespace="ns", name=name, version=1)


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


class TestInjectAxisSourceNames:
    def test_injects_titles_with_instrument_config(self):
        params = Params()
        axis_sources = _make_axis_sources("monitor_cave")
        instrument = FakeInstrumentConfig({"monitor_cave": "Cave Monitor"})
        result = _inject_axis_source_names(params, axis_sources, instrument)
        assert result.bins.x_axis_source == "Cave Monitor"

    def test_no_change_when_no_axis_sources(self):
        params = Params(bins=Bins(x_axis_source="existing"))
        instrument = FakeInstrumentConfig({})
        result = _inject_axis_source_names(params, {}, instrument)
        assert result.bins.x_axis_source == "existing"

    def test_no_change_when_no_bins(self):
        class NoBinsParams(pydantic.BaseModel):
            color: str = "red"

        params = NoBinsParams()
        axis_sources = _make_axis_sources("monitor_cave")
        instrument = FakeInstrumentConfig({"monitor_cave": "Cave Monitor"})
        result = _inject_axis_source_names(params, axis_sources, instrument)
        assert result.color == "red"
