# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Test source selection behavior in ConfigurationWidget."""

import panel as pn
import pydantic
import pytest

from ess.livedata.dashboard.configuration_adapter import ConfigurationAdapter
from ess.livedata.dashboard.plot_configuration_adapter import PlotConfigurationAdapter
from ess.livedata.dashboard.plotting import DataRequirements, PlotterSpec
from ess.livedata.dashboard.widgets.configuration_widget import ConfigurationWidget


class FakeAdapter(ConfigurationAdapter):
    """Minimal adapter for testing source selection."""

    def __init__(
        self,
        source_names: list[str],
        allow_multiple: bool = True,
        initial_source_names: list[str] | None = None,
    ) -> None:
        super().__init__(initial_source_names=initial_source_names)
        self._source_names = source_names
        self._allow_multiple = allow_multiple

    @property
    def title(self) -> str:
        return "Test"

    @property
    def description(self) -> str:
        return "Test description"

    def model_class(self) -> type[pydantic.BaseModel] | None:
        return None

    @property
    def source_names(self) -> list[str]:
        return self._source_names

    @property
    def allow_multiple_sources(self) -> bool:
        return self._allow_multiple

    def start_action(
        self,
        selected_sources: list[str],
        parameter_values: object,
    ) -> None:
        pass


class TestSourceSelectionWidget:
    """Test ConfigurationWidget source selector behavior."""

    def test_multiple_sources_creates_multichoice(self) -> None:
        """When allow_multiple_sources=True, MultiChoice widget is created."""
        adapter = FakeAdapter(
            source_names=['source_a', 'source_b', 'source_c'],
            allow_multiple=True,
        )
        widget = ConfigurationWidget(adapter)

        assert isinstance(widget._source_selector, pn.widgets.MultiChoice)
        assert widget._source_selector.name == "Source Names"

    def test_single_source_creates_select(self) -> None:
        """When allow_multiple_sources=False, Select widget is created."""
        adapter = FakeAdapter(
            source_names=['source_a', 'source_b', 'source_c'],
            allow_multiple=False,
        )
        widget = ConfigurationWidget(adapter)

        assert isinstance(widget._source_selector, pn.widgets.Select)
        assert widget._source_selector.name == "Source Name"

    def test_multichoice_selected_sources_returns_list(self) -> None:
        """MultiChoice selected_sources returns the list directly."""
        adapter = FakeAdapter(
            source_names=['source_a', 'source_b'],
            allow_multiple=True,
            initial_source_names=['source_a'],
        )
        widget = ConfigurationWidget(adapter)

        assert widget.selected_sources == ['source_a']

    def test_select_selected_sources_returns_list(self) -> None:
        """Select selected_sources wraps single value in list."""
        adapter = FakeAdapter(
            source_names=['source_a', 'source_b'],
            allow_multiple=False,
            initial_source_names=['source_a'],
        )
        widget = ConfigurationWidget(adapter)

        # Select returns single value, but selected_sources should be a list
        assert widget.selected_sources == ['source_a']
        assert isinstance(widget.selected_sources, list)

    def test_select_uses_first_initial_source(self) -> None:
        """Select widget uses first initial source name as value."""
        adapter = FakeAdapter(
            source_names=['source_a', 'source_b', 'source_c'],
            allow_multiple=False,
            initial_source_names=['source_b'],
        )
        widget = ConfigurationWidget(adapter)

        assert widget._source_selector.value == 'source_b'

    def test_select_defaults_to_first_initial_source(self) -> None:
        """Select widget uses first of initial_source_names when no explicit initial.

        When initial_source_names is None, ConfigurationAdapter returns all sources
        as initial, so the first one (unsorted) gets selected.
        """
        adapter = FakeAdapter(
            source_names=['source_c', 'source_a', 'source_b'],
            allow_multiple=False,
            initial_source_names=None,
        )
        widget = ConfigurationWidget(adapter)

        # initial_source_names returns all sources when not specified,
        # first one is 'source_c'
        assert widget._source_selector.value == 'source_c'

    def test_no_sources_creates_no_selector(self) -> None:
        """When no sources available, no selector is created."""
        adapter = FakeAdapter(
            source_names=[],
            allow_multiple=True,
        )
        widget = ConfigurationWidget(adapter)

        assert widget._source_selector is None
        assert widget.selected_sources == []


class TestPlotConfigurationAdapterMultipleSources:
    """Test PlotConfigurationAdapter.allow_multiple_sources."""

    @pytest.fixture
    def make_plot_spec(self):
        """Factory for creating PlotterSpec with configurable multiple_datasets."""

        class EmptyParams(pydantic.BaseModel):
            pass

        def _make_spec(multiple_datasets: bool) -> PlotterSpec:
            return PlotterSpec(
                name='test_plotter',
                title='Test Plotter',
                description='Test plotter description',
                params=EmptyParams,
                data_requirements=DataRequirements(
                    min_dims=2,
                    max_dims=2,
                    multiple_datasets=multiple_datasets,
                ),
            )

        return _make_spec

    def test_allow_multiple_sources_true_when_multiple_datasets_true(
        self, make_plot_spec
    ) -> None:
        """allow_multiple_sources is True when multiple_datasets is True."""
        spec = make_plot_spec(multiple_datasets=True)
        adapter = PlotConfigurationAdapter(
            plot_spec=spec,
            source_names=['a', 'b'],
            success_callback=lambda *args: None,
        )

        assert adapter.allow_multiple_sources is True

    def test_allow_multiple_sources_false_when_multiple_datasets_false(
        self, make_plot_spec
    ) -> None:
        """allow_multiple_sources is False when multiple_datasets is False."""
        spec = make_plot_spec(multiple_datasets=False)
        adapter = PlotConfigurationAdapter(
            plot_spec=spec,
            source_names=['a', 'b'],
            success_callback=lambda *args: None,
        )

        assert adapter.allow_multiple_sources is False


class TestConfigurationAdapterDefaultBehavior:
    """Test default allow_multiple_sources behavior."""

    def test_default_allow_multiple_sources_is_true(self) -> None:
        """ConfigurationAdapter.allow_multiple_sources defaults to True."""

        class MinimalAdapter(ConfigurationAdapter):
            @property
            def title(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return ""

            def model_class(self) -> type[pydantic.BaseModel] | None:
                return None

            @property
            def source_names(self) -> list[str]:
                return []

            def start_action(self, selected_sources, parameter_values) -> None:
                pass

        adapter = MinimalAdapter()
        assert adapter.allow_multiple_sources is True
