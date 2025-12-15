# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unit tests for ConfigurationAdapter and per-source configuration."""

import pydantic

from ess.livedata.dashboard.configuration_adapter import (
    ConfigurationAdapter,
    ConfigurationState,
    JobConfigState,
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
    ) -> None:
        super().__init__(config_state=config_state)
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
        assert state.jobs == {}
        assert state.source_names == []

    def test_state_with_jobs(self) -> None:
        state = ConfigurationState(
            jobs={
                'source1': JobConfigState(
                    params={'value': 1},
                    aux_source_names={'aux': 'stream1'},
                ),
                'source2': JobConfigState(
                    params={'value': 2},
                    aux_source_names={'aux': 'stream2'},
                ),
            }
        )
        assert state.source_names == ['source1', 'source2']
        assert state.jobs['source1'].params == {'value': 1}
        assert state.jobs['source2'].params == {'value': 2}

    def test_state_serialization_roundtrip(self) -> None:
        state = ConfigurationState(
            jobs={
                'source1': JobConfigState(
                    params={'value': 42, 'name': 'test'},
                    aux_source_names={'monitor': 'mon1'},
                ),
            }
        )
        dumped = state.model_dump()
        restored = ConfigurationState.model_validate(dumped)
        assert restored == state


class TestSetSelectedSources:
    """Tests for the set_selected_sources functionality."""

    def test_initial_source_names_without_scoping(self) -> None:
        """Without scoping, returns all persisted sources."""
        state = ConfigurationState(
            jobs={
                'source1': JobConfigState(params={'value': 1}),
                'source2': JobConfigState(params={'value': 2}),
                'source3': JobConfigState(params={'value': 3}),
            }
        )
        adapter = ConcreteAdapter(
            available_sources=['source1', 'source2', 'source3'],
            config_state=state,
        )
        assert adapter.initial_source_names == ['source1', 'source2', 'source3']

    def test_initial_source_names_with_scoping(self) -> None:
        """When scoped, returns only the scoped sources."""
        state = ConfigurationState(
            jobs={
                'source1': JobConfigState(params={'value': 1}),
                'source2': JobConfigState(params={'value': 2}),
                'source3': JobConfigState(params={'value': 3}),
            }
        )
        adapter = ConcreteAdapter(
            available_sources=['source1', 'source2', 'source3'],
            config_state=state,
        )
        adapter.set_selected_sources(['source2', 'source3'])
        assert adapter.initial_source_names == ['source2', 'source3']

    def test_scoped_sources_filtered_to_available(self) -> None:
        """Scoped sources are filtered to available sources."""
        state = ConfigurationState(
            jobs={
                'source1': JobConfigState(params={'value': 1}),
            }
        )
        adapter = ConcreteAdapter(
            available_sources=['source1', 'source2'],
            config_state=state,
        )
        # Scope to source that exists and one that doesn't
        adapter.set_selected_sources(['source1', 'nonexistent'])
        assert adapter.initial_source_names == ['source1']

    def test_parameter_values_from_scoped_source(self) -> None:
        """Parameters come from the first scoped source's config."""
        state = ConfigurationState(
            jobs={
                'source1': JobConfigState(params={'value': 1, 'name': 'first'}),
                'source2': JobConfigState(params={'value': 2, 'name': 'second'}),
                'source3': JobConfigState(params={'value': 3, 'name': 'third'}),
            }
        )
        adapter = ConcreteAdapter(
            available_sources=['source1', 'source2', 'source3'],
            config_state=state,
        )

        # Without scoping, get params from first source in config
        assert adapter.initial_parameter_values == {'value': 1, 'name': 'first'}

        # Scope to source2 and source3
        adapter.set_selected_sources(['source2', 'source3'])

        # Params should now come from source2 (first scoped source)
        assert adapter.initial_parameter_values == {'value': 2, 'name': 'second'}

    def test_parameter_values_fallback_when_scoped_source_not_in_state(self) -> None:
        """When scoped sources aren't in state, falls back to first available."""
        state = ConfigurationState(
            jobs={
                'source1': JobConfigState(params={'value': 1, 'name': 'first'}),
            }
        )
        adapter = ConcreteAdapter(
            available_sources=['source1', 'source2', 'source3'],
            config_state=state,
        )

        # Scope to source that has no config
        adapter.set_selected_sources(['source2'])

        # Should fall back to first source in config state
        assert adapter.initial_parameter_values == {'value': 1, 'name': 'first'}


class TestAuxSourceNamesWithScoping:
    """Tests for aux_source_names with per-source configuration."""

    class AdapterWithAuxSources(ConcreteAdapter):
        """Adapter with auxiliary sources defined."""

        class AuxSourcesModel(pydantic.BaseModel):
            monitor: str
            detector: str

        @property
        def aux_sources(self) -> type[pydantic.BaseModel]:
            return self.AuxSourcesModel

    def test_aux_source_names_from_scoped_source(self) -> None:
        """Aux source names come from the first scoped source's config."""
        state = ConfigurationState(
            jobs={
                'source1': JobConfigState(
                    params={},
                    aux_source_names={'monitor': 'mon1', 'detector': 'det1'},
                ),
                'source2': JobConfigState(
                    params={},
                    aux_source_names={'monitor': 'mon2', 'detector': 'det2'},
                ),
            }
        )
        adapter = self.AdapterWithAuxSources(
            available_sources=['source1', 'source2'],
            config_state=state,
        )

        # Without scoping
        assert adapter.initial_aux_source_names == {
            'monitor': 'mon1',
            'detector': 'det1',
        }

        # Scope to source2
        adapter.set_selected_sources(['source2'])
        assert adapter.initial_aux_source_names == {
            'monitor': 'mon2',
            'detector': 'det2',
        }


class TestBackwardCompatibility:
    """Tests for handling edge cases in configuration state."""

    def test_empty_config_state_uses_all_available_sources(self) -> None:
        """Empty config state defaults to all available sources."""
        adapter = ConcreteAdapter(
            available_sources=['source1', 'source2'],
            config_state=None,
        )
        assert adapter.initial_source_names == ['source1', 'source2']

    def test_empty_jobs_uses_all_available_sources(self) -> None:
        """Empty jobs dict defaults to all available sources."""
        adapter = ConcreteAdapter(
            available_sources=['source1', 'source2'],
            config_state=ConfigurationState(jobs={}),
        )
        assert adapter.initial_source_names == ['source1', 'source2']

    def test_sources_filtered_to_available(self) -> None:
        """Persisted sources not in available list are filtered out."""
        state = ConfigurationState(
            jobs={
                'source1': JobConfigState(params={}),
                'removed_source': JobConfigState(params={}),
            }
        )
        adapter = ConcreteAdapter(
            available_sources=['source1', 'source2'],
            config_state=state,
        )
        # 'removed_source' should be filtered out
        assert adapter.initial_source_names == ['source1']

    def test_incompatible_params_fall_back_to_defaults(self) -> None:
        """Params with no field overlap trigger fallback to defaults."""
        state = ConfigurationState(
            jobs={
                'source1': JobConfigState(
                    params={'completely_unknown_field': 'value'},
                ),
            }
        )
        adapter = ConcreteAdapter(
            available_sources=['source1'],
            config_state=state,
        )
        # Should return empty dict, triggering Pydantic defaults
        assert adapter.initial_parameter_values == {}
