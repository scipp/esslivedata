# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

import panel as pn
import pydantic
import pytest
import scipp as sc
from panel.util.warnings import PanelUserWarning

from ess.livedata.config.workflow_spec import (
    DETECTORS,
    MONITORS,
    REDUCTION,
    WindowOutput,
    WorkflowGroup,
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.dashboard.data_roles import PRIMARY, X_AXIS, Y_AXIS
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.plot_orchestrator import (
    CellGeometry,
    DataSourceConfig,
    PlotConfig,
)
from ess.livedata.dashboard.plot_params import (
    PlotParams1d,
    PlotParams2d,
    PlotParamsTimeseries,
)
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.stream_manager import StreamManager
from ess.livedata.dashboard.widgets.plot_config_modal import (
    STATIC_OVERLAY_GROUP,
    PlotConfigModal,
    _build_timeseries_options,
    _inject_axis_source_titles,
    _resolve_axis_source_titles,
    _resolve_output_display_hints,
)
from ess.livedata.dashboard.widgets.plot_grid import PlotGrid


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
            workflow_id=wf_id, source_names=[x_source], view_name="delta"
        ),
    }
    if y_source is not None:
        sources[Y_AXIS] = DataSourceConfig(
            workflow_id=wf_id, source_names=[y_source], view_name="delta"
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


def _make_workflow_spec(
    title: str,
    outputs: type[WorkflowOutputsBase],
    *,
    workflow_id: WorkflowId | None = None,
    source_names: list[str] | None = None,
    group: WorkflowGroup = REDUCTION,
) -> WorkflowSpec:
    workflow_id = workflow_id or _make_workflow_id("test_wf")
    return WorkflowSpec(
        instrument=workflow_id.instrument,
        name=workflow_id.name,
        version=workflow_id.version,
        title=title,
        description=f"{title} description",
        source_names=source_names or [],
        params=None,
        outputs=outputs,
        group=group,
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
            is_static=True,
            workflow_spec=None,
            params_class=PlotParams1d,
            view_name="any",
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
            is_static=False,
            workflow_spec=spec,
            params_class=PlotParams1d,
            view_name="counts",
        )
        assert hints.preselect_all_sources is True

    def test_1d_output_preselects_all(self):
        class Outputs(WorkflowOutputsBase):
            spectrum: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.zeros(sizes={'wavelength': 0})),
            )

        spec = _make_workflow_spec("1D output", Outputs)
        hints = _resolve_output_display_hints(
            is_static=False,
            workflow_spec=spec,
            params_class=PlotParams1d,
            view_name="spectrum",
        )
        assert hints.preselect_all_sources is True

    def test_2d_output_does_not_preselect_all(self):
        class Outputs(WorkflowOutputsBase):
            image: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.zeros(sizes={'x': 0, 'y': 0})),
            )

        spec = _make_workflow_spec("2D output", Outputs)
        hints = _resolve_output_display_hints(
            is_static=False,
            workflow_spec=spec,
            params_class=PlotParams1d,
            view_name="image",
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
            is_static=False,
            workflow_spec=spec,
            params_class=PlotParams1d,
            view_name="volume",
        )
        assert hints.preselect_all_sources is False

    def test_window_control_visibility_follows_the_plotter_params(self):
        # Which window-mode control a params class carries decides what the view
        # must back for it: a per-update-only view backs the duration control but
        # not the timeseries cumulative toggle.
        class Outputs(WorkflowOutputsBase):
            spectrum: WindowOutput = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.zeros(sizes={'wavelength': 0})),
            )

        spec = _make_workflow_spec("Windowed output", Outputs)

        def hidden(params_class: type[pydantic.BaseModel]) -> frozenset[str]:
            return _resolve_output_display_hints(
                is_static=False,
                workflow_spec=spec,
                params_class=params_class,
                view_name="spectrum",
            ).hidden_fields

        assert 'time_window' not in hidden(PlotParams1d)
        assert 'accumulation' in hidden(PlotParamsTimeseries)

    def test_unknown_output_preselects_all(self):
        class Outputs(WorkflowOutputsBase):
            data: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
            )

        spec = _make_workflow_spec("Some workflow", Outputs)
        hints = _resolve_output_display_hints(
            is_static=False,
            workflow_spec=spec,
            params_class=PlotParams1d,
            view_name="nonexistent",
        )
        assert hints.preselect_all_sources is True


def _spectrum() -> sc.DataArray:
    """1D output template, as a monitor histogram workflow declares it."""
    return sc.DataArray(
        sc.zeros(sizes={'wavelength': 0}, unit='counts'),
        coords={'wavelength': sc.zeros(sizes={'wavelength': 1}, unit='angstrom')},
    )


def _image() -> sc.DataArray:
    """2D output template, as a detector view workflow declares it."""
    return sc.DataArray(
        sc.zeros(sizes={'y': 0, 'x': 0}, unit='counts'),
        coords={
            'x': sc.zeros(sizes={'x': 1}, unit='m'),
            'y': sc.zeros(sizes={'y': 1}, unit='m'),
        },
    )


class HistogramOutputs(WorkflowOutputsBase):
    histogram: sc.DataArray = pydantic.Field(
        default_factory=_spectrum, title='Histogram'
    )


class CurrentOnlyOutputs(WorkflowOutputsBase):
    """An output with no cumulative stream, so 'since run start' cannot resolve."""

    current: WindowOutput = pydantic.Field(default_factory=_spectrum, title='Current')


class ImageOutputs(WorkflowOutputsBase):
    image: sc.DataArray = pydantic.Field(default_factory=_image, title='Image')


HISTOGRAM_ID = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)
CURRENT_ID = WorkflowId(instrument='dummy', name='monitor_current', version=1)
IMAGE_ID = WorkflowId(instrument='dummy', name='detector_view', version=1)


@contextmanager
def _headless():
    """Silence Panel's "modal needs a server" warning.

    Opening or closing a ``pn.Modal`` outside a Bokeh server warns; the wizard
    is deliberately driven headless here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PanelUserWarning)
        yield


class WizardDriver:
    """Operates a ``PlotConfigModal`` through its rendered widgets.

    Controls are looked up in the modal's live component tree and driven by
    assignment and clicks, so tests exercise the same handlers the browser
    does rather than the wizard steps' internals. Completed configurations
    land in :attr:`configs`, cancellations increment :attr:`cancels`.
    """

    def __init__(
        self,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        plotting_controller: PlottingController,
        initial_config: PlotConfig | None = None,
    ) -> None:
        self.configs: list[PlotConfig] = []
        self.cancels = 0
        self._modal = PlotConfigModal(
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
            success_callback=self.configs.append,
            cancel_callback=self._on_cancel,
            initial_config=initial_config,
        )
        with _headless():
            self._modal.show()

    def _on_cancel(self) -> None:
        self.cancels += 1

    @property
    def is_open(self) -> bool:
        """Whether the modal is still showing."""
        return self._modal.modal.open

    def widget(self, widget_type: type[pn.widgets.Widget], name: str) -> Any:
        """The single rendered widget of the given type carrying ``name``."""
        [widget] = [w for w in self._modal.modal.select(widget_type) if w.name == name]
        return widget

    def radio(self, name: str) -> pn.widgets.RadioButtonGroup:
        return self.widget(pn.widgets.RadioButtonGroup, name)

    def sources(self) -> pn.widgets.MultiChoice:
        return self.widget(pn.widgets.MultiChoice, 'Source Names')

    def labels(self) -> list[str]:
        """Labels of the buttons currently offered."""
        return [b.label for b in self._modal.modal.select(pn.widgets.Button)]

    def button(self, label: str) -> pn.widgets.Button:
        [button] = [
            b for b in self._modal.modal.select(pn.widgets.Button) if b.label == label
        ]
        return button

    def controls(self) -> list[str]:
        """Names of every rendered widget, i.e. the step's visible controls."""
        return [w.name for w in self._modal.modal.select(pn.widgets.Widget)]

    def click(self, label: str) -> None:
        """Click a button, refusing what a user could not click either."""
        button = self.button(label)
        assert not button.disabled, f'{label!r} is disabled'
        with _headless():
            button.param.trigger('clicks')


@pytest.fixture
def workflow_registry() -> dict[WorkflowId, WorkflowSpec]:
    """Registry spanning two groups and the output shapes the wizard branches on.

    Overrides the shared fixture, whose specs declare no output templates:
    plotter selection would then fall back to offering every registered
    plotter rather than the ones the output supports.
    """
    return {
        HISTOGRAM_ID: _make_workflow_spec(
            'Monitor Histogram',
            HistogramOutputs,
            workflow_id=HISTOGRAM_ID,
            source_names=['monitor1', 'monitor2'],
            group=MONITORS,
        ),
        CURRENT_ID: _make_workflow_spec(
            'Monitor Current',
            CurrentOnlyOutputs,
            workflow_id=CURRENT_ID,
            source_names=['monitor1'],
            group=MONITORS,
        ),
        IMAGE_ID: _make_workflow_spec(
            'Detector View',
            ImageOutputs,
            workflow_id=IMAGE_ID,
            source_names=['panel_0', 'panel_1'],
            group=DETECTORS,
        ),
    }


@pytest.fixture
def plotting_controller() -> PlottingController:
    """Controller backed by the real plotter registry."""
    return PlottingController(stream_manager=StreamManager(data_service=DataService()))


@pytest.fixture
def open_wizard(
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
    plotting_controller: PlottingController,
) -> Callable[..., WizardDriver]:
    """Open a plot-configuration wizard, as clicking out a grid region does."""

    def _open(initial_config: PlotConfig | None = None) -> WizardDriver:
        return WizardDriver(workflow_registry, plotting_controller, initial_config)

    return _open


def _advance_to_configuration(
    wizard: WizardDriver, *, workflow_id: WorkflowId, group: str, plot_name: str
) -> None:
    """Drive the wizard from the first step to the configuration step."""
    wizard.radio('Group').value = group
    wizard.radio('Workflow').value = workflow_id
    wizard.click('Next')
    wizard.radio('Plotter Type').value = plot_name
    wizard.click('Next')


class TestWorkflowAndOutputSelection:
    def test_opens_on_the_first_group_workflow_and_output(self, open_wizard) -> None:
        wizard = open_wizard()
        # Groups are ordered by group name, workflows by title.
        assert wizard.radio('Group').value == DETECTORS.name
        assert wizard.radio('Workflow').value == IMAGE_ID
        assert wizard.radio('Output').value == 'image'
        assert not wizard.button('Next').disabled

    def test_switching_group_repopulates_workflows_and_picks_an_output(
        self, open_wizard
    ) -> None:
        wizard = open_wizard()
        wizard.radio('Group').value = MONITORS.name

        assert set(wizard.radio('Workflow').options.values()) == {
            CURRENT_ID,
            HISTOGRAM_ID,
        }
        assert wizard.radio('Workflow').value == CURRENT_ID
        assert wizard.radio('Output').value == 'current'

    def test_switching_workflow_repopulates_outputs(self, open_wizard) -> None:
        wizard = open_wizard()
        wizard.radio('Group').value = MONITORS.name
        wizard.radio('Workflow').value = HISTOGRAM_ID

        assert wizard.radio('Output').options == {'Histogram': 'histogram'}
        assert wizard.radio('Output').value == 'histogram'

    def test_static_overlay_cannot_advance_until_it_is_named(self, open_wizard) -> None:
        wizard = open_wizard()
        wizard.radio('Group').value = STATIC_OVERLAY_GROUP

        # A static overlay has no output to pick, so the user names it instead.
        assert wizard.button('Next').disabled
        wizard.widget(pn.widgets.TextInput, 'Overlay Name').value = 'Beam centre'
        assert not wizard.button('Next').disabled


class TestPlotterSelection:
    @pytest.mark.parametrize(
        ('group', 'workflow_id', 'offered', 'not_offered'),
        [
            (DETECTORS.name, IMAGE_ID, 'image', 'lines'),
            (MONITORS.name, HISTOGRAM_ID, 'lines', 'image'),
        ],
        ids=['2d', '1d'],
    )
    def test_plotters_offered_follow_the_selected_output(
        self, open_wizard, group, workflow_id, offered, not_offered
    ) -> None:
        wizard = open_wizard()
        wizard.radio('Group').value = group
        wizard.radio('Workflow').value = workflow_id
        wizard.click('Next')

        options = set(wizard.radio('Plotter Type').options.values())
        assert offered in options
        assert not_offered not in options

    @pytest.mark.parametrize(
        ('group', 'workflow_id', 'plot_name', 'expected', 'absent'),
        [
            (MONITORS.name, HISTOGRAM_ID, 'lines', 'Line Mode', 'Color Axis Scale'),
            (DETECTORS.name, IMAGE_ID, 'image', 'Color Axis Scale', 'Line Mode'),
        ],
        ids=['lines', 'image'],
    )
    def test_parameter_form_is_generated_for_the_selected_plotter(
        self, open_wizard, group, workflow_id, plot_name, expected, absent
    ) -> None:
        wizard = open_wizard()
        _advance_to_configuration(
            wizard, workflow_id=workflow_id, group=group, plot_name=plot_name
        )

        controls = wizard.controls()
        assert expected in controls
        assert absent not in controls


class TestSourceSelection:
    def test_one_dimensional_output_preselects_every_source(self, open_wizard) -> None:
        wizard = open_wizard()
        _advance_to_configuration(
            wizard,
            workflow_id=HISTOGRAM_ID,
            group=MONITORS.name,
            plot_name='lines',
        )
        assert wizard.sources().value == ['monitor1', 'monitor2']

    def test_two_dimensional_output_preselects_nothing(self, open_wizard) -> None:
        # An image per source cannot be overlaid, so the user picks one.
        wizard = open_wizard()
        _advance_to_configuration(
            wizard, workflow_id=IMAGE_ID, group=DETECTORS.name, plot_name='image'
        )
        assert wizard.sources().value == []


class TestApply:
    def test_applying_yields_the_configured_plot(self, open_wizard) -> None:
        wizard = open_wizard()
        _advance_to_configuration(
            wizard,
            workflow_id=HISTOGRAM_ID,
            group=MONITORS.name,
            plot_name='lines',
        )
        wizard.sources().value = ['monitor2']

        wizard.click('Add Plot')

        [config] = wizard.configs
        assert config.plot_name == 'lines'
        assert config.data_sources[PRIMARY] == DataSourceConfig(
            workflow_id=HISTOGRAM_ID,
            source_names=['monitor2'],
            view_name='histogram',
        )
        assert isinstance(config.params, PlotParams1d)
        assert not wizard.is_open

    def test_empty_source_selection_is_rejected(self, open_wizard) -> None:
        wizard = open_wizard()
        _advance_to_configuration(
            wizard, workflow_id=IMAGE_ID, group=DETECTORS.name, plot_name='image'
        )

        wizard.click('Add Plot')

        assert wizard.configs == []
        assert wizard.is_open

        # The user can fix the selection and apply again without reopening.
        wizard.sources().value = ['panel_1']
        wizard.click('Add Plot')

        [config] = wizard.configs
        assert config.data_sources[PRIMARY].source_names == ['panel_1']
        assert isinstance(config.params, PlotParams2d)

    def test_since_start_is_rejected_for_an_output_without_a_cumulative_stream(
        self, open_wizard
    ) -> None:
        wizard = open_wizard()
        _advance_to_configuration(
            wizard, workflow_id=CURRENT_ID, group=MONITORS.name, plot_name='lines'
        )
        mode = wizard.widget(pn.widgets.Select, 'Mode')
        mode.value = 'since_start'

        wizard.click('Add Plot')

        assert wizard.configs == []
        assert wizard.is_open

        mode.value = 'window'
        wizard.click('Add Plot')
        assert len(wizard.configs) == 1


class TestNavigation:
    def test_back_returns_to_the_previous_step(self, open_wizard) -> None:
        wizard = open_wizard()
        wizard.click('Next')
        wizard.click('Next')
        assert 'Source Names' in wizard.controls()

        wizard.click('Back')
        assert 'Plotter Type' in wizard.controls()

        wizard.click('Back')
        assert wizard.radio('Workflow').value == IMAGE_ID
        assert wizard.radio('Output').value == 'image'

    def test_back_resets_the_plotter_choice_but_not_the_output_choice(
        self, open_wizard
    ) -> None:
        # Entering the plotter step rebuilds its radio group and preselects the
        # first offered plotter, so stepping back and forth drops a non-default
        # choice; the workflow/output step keeps its widgets and its selection.
        wizard = open_wizard()
        wizard.click('Next')
        wizard.radio('Plotter Type').value = 'overlay_1d'
        wizard.click('Next')

        wizard.click('Back')
        assert wizard.radio('Plotter Type').value == 'image'
        wizard.click('Back')
        assert wizard.radio('Workflow').value == IMAGE_ID

    def test_changing_the_output_after_going_back_re_derives_the_plotters(
        self, open_wizard
    ) -> None:
        wizard = open_wizard()
        wizard.click('Next')
        assert 'image' in wizard.radio('Plotter Type').options.values()

        wizard.click('Back')
        wizard.radio('Group').value = MONITORS.name
        wizard.radio('Workflow').value = HISTOGRAM_ID
        wizard.click('Next')

        offered = set(wizard.radio('Plotter Type').options.values())
        assert 'image' not in offered
        assert 'lines' in offered

    def test_cancelling_yields_no_config(self, open_wizard) -> None:
        wizard = open_wizard()
        wizard.click('Next')

        wizard.click('Cancel')

        assert wizard.configs == []
        assert wizard.cancels == 1
        assert not wizard.is_open


class TestEditMode:
    def test_opens_at_the_configuration_step_prefilled(self, open_wizard) -> None:
        wizard = open_wizard()
        _advance_to_configuration(
            wizard,
            workflow_id=HISTOGRAM_ID,
            group=MONITORS.name,
            plot_name='lines',
        )
        wizard.sources().value = ['monitor2']
        wizard.click('Add Plot')
        [config] = wizard.configs

        editor = open_wizard(config)

        assert editor.sources().value == ['monitor2']
        # Edit mode applies rather than adds, and opens at the last step --
        # with the earlier steps still reachable.
        assert 'Update Plot' in editor.labels()
        assert 'Add Plot' not in editor.labels()

        editor.click('Back')
        assert editor.radio('Plotter Type').value == 'lines'
        editor.click('Back')
        assert editor.radio('Workflow').value == HISTOGRAM_ID


def _click_cell(grid: PlotGrid, row: int, col: int) -> None:
    """Click an empty grid cell, addressed by its stable automation hook."""
    hook = f'lt-empty-cell-r{row}c{col}'
    [button] = [
        b for b in grid.panel.select(pn.widgets.Button) if hook in b.css_classes
    ]
    assert not button.disabled, f'cell ({row}, {col}) is disabled'
    button.param.trigger('clicks')


def _is_empty(grid: PlotGrid, row: int, col: int) -> bool:
    hook = f'lt-empty-cell-r{row}c{col}'
    return any(hook in b.css_classes for b in grid.panel.select(pn.widgets.Button))


class TestClickToPlace:
    """The two-click gesture that turns a grid region into a configured plot."""

    @pytest.fixture
    def grid(self) -> tuple[PlotGrid, list[CellGeometry]]:
        requested: list[CellGeometry] = []
        return (
            PlotGrid(nrows=3, ncols=3, plot_request_callback=requested.append),
            requested,
        )

    def test_first_click_only_highlights(self, grid) -> None:
        plot_grid, requested = grid
        _click_cell(plot_grid, 1, 1)
        assert requested == []

    @pytest.mark.parametrize(
        ('first', 'second'),
        [((0, 1), (2, 2)), ((2, 2), (0, 1)), ((0, 2), (2, 1)), ((2, 1), (0, 2))],
    )
    def test_second_click_requests_the_normalized_region(
        self, grid, first, second
    ) -> None:
        # Any pair of opposite corners describes the same region.
        plot_grid, requested = grid
        _click_cell(plot_grid, *first)
        _click_cell(plot_grid, *second)

        assert requested == [CellGeometry(row=0, col=1, row_span=3, col_span=2)]

    def test_completing_the_wizard_configures_the_clicked_region(
        self, open_wizard
    ) -> None:
        placed: list[tuple[CellGeometry, WizardDriver]] = []
        plot_grid = PlotGrid(
            nrows=3,
            ncols=3,
            plot_request_callback=lambda geometry: placed.append(
                (geometry, open_wizard())
            ),
        )

        _click_cell(plot_grid, 1, 0)
        _click_cell(plot_grid, 1, 1)
        [(geometry, wizard)] = placed
        _advance_to_configuration(
            wizard,
            workflow_id=HISTOGRAM_ID,
            group=MONITORS.name,
            plot_name='lines',
        )
        wizard.click('Add Plot')

        assert geometry == CellGeometry(row=1, col=0, row_span=1, col_span=2)
        [config] = wizard.configs
        assert config.plot_name == 'lines'

    def test_cancelling_leaves_the_region_selectable(self, open_wizard) -> None:
        placed: list[tuple[CellGeometry, WizardDriver]] = []
        plot_grid = PlotGrid(
            nrows=3,
            ncols=3,
            plot_request_callback=lambda geometry: placed.append(
                (geometry, open_wizard())
            ),
        )

        _click_cell(plot_grid, 0, 0)
        _click_cell(plot_grid, 0, 1)
        [(_, wizard)] = placed
        wizard.click('Cancel')

        # No cell was placed, and the region can be clicked out again.
        assert wizard.configs == []
        assert _is_empty(plot_grid, 0, 0)
        assert _is_empty(plot_grid, 0, 1)
        _click_cell(plot_grid, 0, 0)
        _click_cell(plot_grid, 0, 1)
        assert len(placed) == 2
