# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pydantic
import pytest

from ess.livedata.config import instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.config.workflow_spec import WorkflowSpec
from ess.livedata.dashboard.widgets.configuration_widget import (
    ConfigurationAdapter,
    ConfigurationWidget,
)


def collect_workflow_specs():
    """Collect all workflow specs for parameterization."""
    specs = []
    for instrument_name in available_instruments():
        _ = get_config(instrument_name)  # Load the module to register the instrument
        instrument = instrument_registry[instrument_name]
        for workflow_id, spec in instrument.workflow_factory.items():
            specs.append(pytest.param(spec, id=str(workflow_id)))
    return specs


class FakeConfigurationAdapter(ConfigurationAdapter):
    """Fake adapter for testing."""

    def __init__(self, spec: WorkflowSpec) -> None:
        super().__init__(config_state=None)
        self._spec = spec

    @property
    def title(self) -> str:
        return self._spec.title

    @property
    def description(self) -> str:
        return self._spec.description

    def model_class(self) -> type[pydantic.BaseModel] | None:
        return self._spec.params

    @property
    def source_names(self) -> list[str]:
        return self._spec.source_names

    @property
    def initial_source_names(self) -> list[str]:
        return []

    @property
    def initial_parameter_values(self) -> dict[str, object]:
        return {}

    def start_action(
        self,
        selected_sources: list[str],
        parameter_values: object,
    ) -> None:
        pass


class TestWorkflowParams:
    @pytest.mark.parametrize('spec', collect_workflow_specs())
    def test_can_create_widget_for_workflow(self, spec: WorkflowSpec) -> None:
        """
        Test that we can create a configuration widget for all workflows.

        Expected to fail if an incompatible Pydantic model is used as workflow params.
        """
        adapter = FakeConfigurationAdapter(spec)
        # This will raise if there is an issue
        _ = ConfigurationWidget(adapter)
