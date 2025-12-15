# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unit tests for ConfigurationAdapter."""

import pydantic

from ess.livedata.dashboard.configuration_adapter import (
    ConfigurationAdapter,
    ConfigurationState,
)


class SimpleParams(pydantic.BaseModel):
    """Simple parameter model for testing."""

    value: int = 10
    name: str = "default"


class ConcreteAdapter(ConfigurationAdapter[SimpleParams]):
    """Concrete implementation for testing."""

    def __init__(
        self,
        available_sources: list[str],
        config_state: ConfigurationState | None = None,
        initial_source_names: list[str] | None = None,
    ) -> None:
        super().__init__(
            config_state=config_state, initial_source_names=initial_source_names
        )
        self._available_sources = available_sources

    @property
    def title(self) -> str:
        return "Test Adapter"

    @property
    def description(self) -> str:
        return "Test description"

    def model_class(self) -> type[SimpleParams]:
        return SimpleParams

    @property
    def source_names(self) -> list[str]:
        return self._available_sources

    def start_action(
        self,
        selected_sources: list[str],
        parameter_values: SimpleParams,
    ) -> None:
        pass


class TestConfigurationStateSchema:
    """Tests for the ConfigurationState schema."""

    def test_empty_state(self) -> None:
        state = ConfigurationState()
        assert state.params == {}
        assert state.aux_source_names == {}

    def test_state_with_params(self) -> None:
        state = ConfigurationState(
            params={'value': 42, 'name': 'test'},
            aux_source_names={'monitor': 'mon1'},
        )
        assert state.params == {'value': 42, 'name': 'test'}
        assert state.aux_source_names == {'monitor': 'mon1'}

    def test_state_serialization_roundtrip(self) -> None:
        state = ConfigurationState(
            params={'value': 42, 'name': 'test'},
            aux_source_names={'monitor': 'mon1'},
        )
        dumped = state.model_dump()
        restored = ConfigurationState.model_validate(dumped)
        assert restored == state


class TestInitialSourceNames:
    """Tests for initial_source_names behavior."""

    def test_no_initial_source_names_returns_all_available(self) -> None:
        """Without initial_source_names, returns all available sources."""
        adapter = ConcreteAdapter(
            available_sources=['source1', 'source2', 'source3'],
        )
        assert adapter.initial_source_names == ['source1', 'source2', 'source3']

    def test_initial_source_names_respected(self) -> None:
        """Initial source names are returned as-is when all are available."""
        adapter = ConcreteAdapter(
            available_sources=['source1', 'source2', 'source3'],
            initial_source_names=['source2', 'source3'],
        )
        assert adapter.initial_source_names == ['source2', 'source3']

    def test_initial_source_names_filtered_to_available(self) -> None:
        """Initial source names are filtered to available sources."""
        adapter = ConcreteAdapter(
            available_sources=['source1', 'source2'],
            initial_source_names=['source1', 'nonexistent'],
        )
        assert adapter.initial_source_names == ['source1']

    def test_initial_source_names_fallback_when_all_unavailable(self) -> None:
        """Falls back to all available when initial sources don't exist."""
        adapter = ConcreteAdapter(
            available_sources=['source1', 'source2'],
            initial_source_names=['nonexistent1', 'nonexistent2'],
        )
        assert adapter.initial_source_names == ['source1', 'source2']


class TestInitialParameterValues:
    """Tests for initial_parameter_values behavior."""

    def test_no_config_state_returns_empty_dict(self) -> None:
        """Without config_state, returns empty dict for defaults."""
        adapter = ConcreteAdapter(available_sources=['source1'])
        assert adapter.initial_parameter_values == {}

    def test_params_from_config_state(self) -> None:
        """Parameters come from config_state."""
        state = ConfigurationState(params={'value': 42, 'name': 'test'})
        adapter = ConcreteAdapter(
            available_sources=['source1'],
            config_state=state,
        )
        assert adapter.initial_parameter_values == {'value': 42, 'name': 'test'}

    def test_incompatible_params_fall_back_to_defaults(self) -> None:
        """Params with no field overlap trigger fallback to defaults."""
        state = ConfigurationState(
            params={'completely_unknown_field': 'value'},
        )
        adapter = ConcreteAdapter(
            available_sources=['source1'],
            config_state=state,
        )
        # Should return empty dict, triggering Pydantic defaults
        assert adapter.initial_parameter_values == {}

    def test_partial_params_returned_for_pydantic_validation(self) -> None:
        """Params with partial overlap are returned for Pydantic to handle."""
        state = ConfigurationState(
            params={'value': 99, 'unknown_field': 'ignored'},
        )
        adapter = ConcreteAdapter(
            available_sources=['source1'],
            config_state=state,
        )
        # Return all params - Pydantic will ignore extra fields and use defaults
        assert adapter.initial_parameter_values == {
            'value': 99,
            'unknown_field': 'ignored',
        }


class TestAuxSourceNames:
    """Tests for aux_source_names behavior."""

    class AdapterWithAuxSources(ConcreteAdapter):
        """Adapter with auxiliary sources defined."""

        class AuxSourcesModel(pydantic.BaseModel):
            monitor: str
            detector: str

        @property
        def aux_sources(self) -> type[pydantic.BaseModel]:
            return self.AuxSourcesModel

    def test_no_aux_sources_returns_empty(self) -> None:
        """Adapter without aux_sources returns empty dict."""
        adapter = ConcreteAdapter(available_sources=['source1'])
        assert adapter.initial_aux_source_names == {}

    def test_aux_source_names_from_config_state(self) -> None:
        """Aux source names come from config_state."""
        state = ConfigurationState(
            params={},
            aux_source_names={'monitor': 'mon1', 'detector': 'det1'},
        )
        adapter = self.AdapterWithAuxSources(
            available_sources=['source1'],
            config_state=state,
        )
        assert adapter.initial_aux_source_names == {
            'monitor': 'mon1',
            'detector': 'det1',
        }

    def test_aux_source_names_filtered_to_valid_fields(self) -> None:
        """Aux source names are filtered to valid model fields."""
        state = ConfigurationState(
            params={},
            aux_source_names={
                'monitor': 'mon1',
                'detector': 'det1',
                'invalid_field': 'ignored',
            },
        )
        adapter = self.AdapterWithAuxSources(
            available_sources=['source1'],
            config_state=state,
        )
        # 'invalid_field' should be filtered out
        assert adapter.initial_aux_source_names == {
            'monitor': 'mon1',
            'detector': 'det1',
        }

    def test_no_config_state_returns_empty(self) -> None:
        """Without config_state, returns empty dict."""
        adapter = self.AdapterWithAuxSources(available_sources=['source1'])
        assert adapter.initial_aux_source_names == {}
