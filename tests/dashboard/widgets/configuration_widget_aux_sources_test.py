# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for AuxSourcesWidget rendering and interaction."""

import pytest

from ess.livedata.config.workflow_spec import AuxInput, AuxSources
from ess.livedata.dashboard.widgets.configuration_widget import AuxSourcesWidget


class TestAuxSourcesWidgetLabels:
    """Test that AuxSourcesWidget uses titles for widget names and option labels."""

    def test_widget_name_uses_aux_input_title(self) -> None:
        aux = AuxSources(
            {
                'incident_monitor': AuxInput(
                    choices=('mon1',),
                    default='mon1',
                    title='Incident Monitor',
                ),
            }
        )
        widget = AuxSourcesWidget(aux)
        select = widget.select_widgets['incident_monitor']
        assert select.name == 'Incident Monitor'

    def test_widget_name_falls_back_to_field_name(self) -> None:
        aux = AuxSources({'rotation': 'rot_angle'})
        widget = AuxSourcesWidget(aux)
        select = widget.select_widgets['rotation']
        assert select.name == 'rotation'

    def test_option_labels_use_get_source_title(self) -> None:
        titles = {'mon1': 'Incident Monitor', 'mon2': 'Transmission Monitor'}
        aux = AuxSources(
            {
                'monitor': AuxInput(
                    choices=('mon1', 'mon2'),
                    default='mon1',
                    title='Monitor',
                ),
            }
        )
        widget = AuxSourcesWidget(aux, get_source_title=titles.get)
        select = widget.select_widgets['monitor']
        assert select.options == {
            'Incident Monitor': 'mon1',
            'Transmission Monitor': 'mon2',
        }

    def test_option_labels_without_title_callback_show_raw_names(self) -> None:
        aux = AuxSources(
            {
                'monitor': AuxInput(
                    choices=('mon1', 'mon2'),
                    default='mon1',
                ),
            }
        )
        widget = AuxSourcesWidget(aux)
        select = widget.select_widgets['monitor']
        assert select.options == {'mon1': 'mon1', 'mon2': 'mon2'}


class TestAuxSourcesWidgetValues:
    """Test value reading and initial values."""

    def test_get_values_returns_defaults(self) -> None:
        aux = AuxSources(
            {
                'incident': AuxInput(choices=('mon1', 'mon2'), default='mon1'),
                'transmission': AuxInput(choices=('mon3', 'mon4'), default='mon4'),
            }
        )
        widget = AuxSourcesWidget(aux)
        assert widget.get_values() == {'incident': 'mon1', 'transmission': 'mon4'}

    def test_initial_values_override_defaults(self) -> None:
        aux = AuxSources(
            {
                'monitor': AuxInput(choices=('mon1', 'mon2'), default='mon1'),
            }
        )
        widget = AuxSourcesWidget(aux, initial_values={'monitor': 'mon2'})
        assert widget.get_values() == {'monitor': 'mon2'}

    def test_initial_values_missing_key_uses_default(self) -> None:
        aux = AuxSources(
            {
                'a': AuxInput(choices=('x', 'y'), default='x'),
                'b': AuxInput(choices=('p', 'q'), default='p'),
            }
        )
        # Only override 'a', 'b' should use default
        widget = AuxSourcesWidget(aux, initial_values={'a': 'y'})
        assert widget.get_values() == {'a': 'y', 'b': 'p'}


class TestAuxSourcesWidgetSingleChoice:
    """Test single-choice (fixed) aux source behavior."""

    def test_single_choice_string_shorthand(self) -> None:
        aux = AuxSources({'rotation': 'det_rotation'})
        widget = AuxSourcesWidget(aux)
        select = widget.select_widgets['rotation']
        assert select.options == {'det_rotation': 'det_rotation'}
        assert select.value == 'det_rotation'

    def test_single_choice_is_not_disabled(self) -> None:
        """Single-choice widgets should remain enabled for readability."""
        aux = AuxSources({'rotation': 'det_rotation'})
        widget = AuxSourcesWidget(aux)
        select = widget.select_widgets['rotation']
        assert not select.disabled


class TestAuxSourcesWidgetMultipleInputs:
    """Test widget with multiple aux source inputs."""

    def test_panel_contains_all_widgets(self) -> None:
        aux = AuxSources(
            {
                'incident_monitor': AuxInput(
                    choices=('mon1',),
                    default='mon1',
                    title='Incident Monitor',
                ),
                'transmission_monitor': AuxInput(
                    choices=('mon2',),
                    default='mon2',
                    title='Transmission Monitor',
                ),
            }
        )
        widget = AuxSourcesWidget(aux)
        assert len(widget.select_widgets) == 2
        assert len(widget.panel.objects) == 2

    @pytest.fixture
    def loki_style_widget(self) -> AuxSourcesWidget:
        """Widget simulating LOKI's aux source configuration."""
        titles = {
            'beam_monitor_m1': 'Incident Monitor',
            'beam_monitor_m3': 'Transmission Monitor',
        }
        aux = AuxSources(
            {
                'incident_monitor': AuxInput(
                    choices=('beam_monitor_m1',),
                    default='beam_monitor_m1',
                    title='Incident Monitor',
                ),
                'transmission_monitor': AuxInput(
                    choices=('beam_monitor_m3',),
                    default='beam_monitor_m3',
                    title='Transmission Monitor',
                ),
            }
        )
        return AuxSourcesWidget(aux, get_source_title=titles.get)

    def test_loki_style_incident_monitor_label(
        self, loki_style_widget: AuxSourcesWidget
    ) -> None:
        select = loki_style_widget.select_widgets['incident_monitor']
        assert select.name == 'Incident Monitor'
        assert select.options == {'Incident Monitor': 'beam_monitor_m1'}

    def test_loki_style_transmission_monitor_label(
        self, loki_style_widget: AuxSourcesWidget
    ) -> None:
        select = loki_style_widget.select_widgets['transmission_monitor']
        assert select.name == 'Transmission Monitor'
        assert select.options == {'Transmission Monitor': 'beam_monitor_m3'}

    def test_loki_style_get_values(self, loki_style_widget: AuxSourcesWidget) -> None:
        assert loki_style_widget.get_values() == {
            'incident_monitor': 'beam_monitor_m1',
            'transmission_monitor': 'beam_monitor_m3',
        }
