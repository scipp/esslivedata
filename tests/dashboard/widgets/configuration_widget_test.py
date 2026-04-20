# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for ConfigurationWidget / ConfigurationPanel tab routing and rendering."""

from __future__ import annotations

import panel as pn
import pydantic
import pytest

from ess.livedata.dashboard.configuration_adapter import ConfigurationAdapter
from ess.livedata.dashboard.widgets.configuration_widget import (
    ConfigurationPanel,
    ConfigurationWidget,
    NoParamsWidget,
)
from ess.livedata.dashboard.widgets.model_widget import ModelWidget


class _Params1(pydantic.BaseModel):
    a: int = 1


class _Params2(pydantic.BaseModel):
    b: int = pydantic.Field(default=0, ge=0, le=10)


class _MultiGroupModel(pydantic.BaseModel):
    group_one: _Params1 = pydantic.Field(default_factory=_Params1, title='Group One')
    group_two: _Params2 = pydantic.Field(default_factory=_Params2, title='Group Two')


class _EmptyModel(pydantic.BaseModel):
    pass


class _Adapter(ConfigurationAdapter):
    """Minimal ConfigurationAdapter for driving the widget in tests."""

    def __init__(
        self,
        *,
        title: str = 'Test',
        description: str = 'desc',
        model: type[pydantic.BaseModel] | None = _MultiGroupModel,
        source_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._description = description
        self._model = model
        self._source_names = source_names if source_names is not None else ['s1', 's2']
        self.started_with: tuple[list[str], pydantic.BaseModel | None] | None = None

    @property
    def title(self) -> str:
        return self._title

    @property
    def description(self) -> str:
        return self._description

    def model_class(self) -> type[pydantic.BaseModel] | None:
        return self._model

    @property
    def source_names(self) -> list[str]:
        return self._source_names

    def start_action(
        self, selected_sources: list[str], parameter_values: pydantic.BaseModel
    ) -> None:
        self.started_with = (selected_sources, parameter_values)


def _param_tab_index(widget: ConfigurationWidget, field_name: str) -> int:
    return widget._tab_field_order.index(field_name)


class TestTitlePaneEscaping:
    def test_title_and_description_are_html_escaped(self) -> None:
        adapter = _Adapter(
            title='Evil <script>alert(1)</script>',
            description='Also "quoted" & stuff',
        )
        widget = ConfigurationWidget(adapter)
        html_text = widget.title_pane.object
        assert '<script>' not in html_text
        assert '&lt;script&gt;' in html_text
        assert '&amp;' in html_text


class TestBodyShape:
    def test_tabs_created_when_multiple_groups(self) -> None:
        widget = ConfigurationWidget(_Adapter())
        body = widget.widget.objects[0]
        assert isinstance(body, pn.Tabs)
        assert list(body._names) == ['General', 'Group One', 'Group Two']

    def test_no_tabs_when_nothing_to_configure(self) -> None:
        """No sources + no params → body falls back to NoParamsWidget directly."""
        adapter = _Adapter(model=_EmptyModel, source_names=[])
        widget = ConfigurationWidget(adapter)
        body = widget.widget.objects[0]
        assert not isinstance(body, pn.Tabs)
        assert widget._tabs is None

    def test_activate_first_error_tab_is_noop_when_no_tabs(self) -> None:
        adapter = _Adapter(model=_EmptyModel, source_names=[])
        widget = ConfigurationWidget(adapter)
        widget.activate_first_error_tab()  # must not raise


class TestErrorTabRouting:
    def test_source_error_activates_general_tab(self) -> None:
        widget = ConfigurationWidget(_Adapter())
        # Clear source selection, switch to a non-General tab, then validate.
        widget._source_selector.value = []
        body = widget.widget.objects[0]
        assert isinstance(body, pn.Tabs)
        body.active = _param_tab_index(widget, 'group_two')
        valid, _ = widget.validate_configuration()
        assert not valid
        widget.activate_first_error_tab()
        assert body.active == _param_tab_index(widget, None)  # General

    def test_param_error_activates_owning_tab(self) -> None:
        widget = ConfigurationWidget(_Adapter())
        # Inject out-of-range value into group_two.b (must be 0..10).
        param_widget = widget._model_widget.get_parameter_widget('group_two')
        assert param_widget is not None
        param_widget.set_values({'b': 999})
        body = widget.widget.objects[0]
        assert isinstance(body, pn.Tabs)
        body.active = _param_tab_index(widget, None)  # General
        valid, _ = widget.validate_configuration()
        assert not valid
        widget.activate_first_error_tab()
        assert body.active == _param_tab_index(widget, 'group_two')


class TestFailingFieldTracking:
    def test_failing_names_populated_by_validate(self) -> None:
        mw = ModelWidget(_MultiGroupModel)
        mw.get_parameter_widget('group_two').set_values({'b': 999})
        valid, _ = mw.validate_parameters()
        assert not valid
        assert mw.get_failing_field_names() == ['group_two']

    def test_failing_names_cleared_on_success(self) -> None:
        mw = ModelWidget(_MultiGroupModel)
        mw.get_parameter_widget('group_two').set_values({'b': 999})
        mw.validate_parameters()
        mw.get_parameter_widget('group_two').set_values({'b': 1})
        valid, _ = mw.validate_parameters()
        assert valid
        assert mw.get_failing_field_names() == []

    def test_failing_names_cleared_by_clear_validation_errors(self) -> None:
        mw = ModelWidget(_MultiGroupModel)
        mw.get_parameter_widget('group_two').set_values({'b': 999})
        mw.validate_parameters()
        mw.clear_validation_errors()
        assert mw.get_failing_field_names() == []


class TestConfigurationPanelShape:
    def test_panel_includes_title_then_body(self) -> None:
        panel = ConfigurationPanel(_Adapter())
        combined = panel.panel
        assert combined.objects[0] is panel._config_widget.title_pane
        # The second child is the body column holding the widget + error pane.
        assert combined.objects[1] is panel._body

    def test_panel_property_is_cached(self) -> None:
        panel = ConfigurationPanel(_Adapter())
        assert panel.panel is panel.panel

    def test_split_for_sticky_header_returns_same_slots(self) -> None:
        panel = ConfigurationPanel(_Adapter())
        title, body = panel.split_for_sticky_header()
        assert title is panel._config_widget.title_pane
        assert body is panel._body


class TestNoParamsWidget:
    def test_no_params_widget_exposes_empty_tab_list(self) -> None:
        widget = NoParamsWidget()
        assert widget.param_group_tabs == []
        assert widget.get_failing_field_names() == []


@pytest.fixture(autouse=True)
def _panel_extensions_loaded():
    """Ensure Panel extensions required by pn.Tabs etc. are loaded."""
    pn.extension()
