# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Test all registered workflow specs.

This test file validates workflow specs WITHOUT loading factory implementations.
These tests run fast because they don't import heavy dependencies.

For tests that require factory implementations, see
registered_workflow_factories_test.py.
"""

import pydantic
import pytest

from ess.livedata.config.instrument import instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.config.workflow_spec import WorkflowConfig, WorkflowId
from ess.livedata.dashboard.plotter_registry import plotter_registry
from ess.livedata.dashboard.workflow_configuration_adapter import (
    WorkflowConfigurationAdapter,
)


def _collect_workflow_specs():
    """Collect workflow specs WITHOUT loading factories (fast).

    This only imports instrument spec modules, not factory implementations,
    allowing tests to run much faster.
    """
    workflows = []
    for instrument_name in available_instruments():
        _ = get_config(instrument_name)  # Load specs only
        instrument = instrument_registry[instrument_name]
        # DO NOT call instrument.load_factories()
        workflows.extend(
            [
                pytest.param(instrument_name, workflow_id, id=str(workflow_id))
                for workflow_id in instrument.workflow_factory
            ]
        )
    return workflows


@pytest.mark.parametrize(("instrument_name", "workflow_id"), _collect_workflow_specs())
def test_workflow_spec_params_validation(instrument_name: str, workflow_id: WorkflowId):
    """Test that spec.params is None or a valid Pydantic BaseModel class.

    Since params are now explicitly registered (not inferred from factory),
    this validates they're properly set in the spec.
    """
    instrument = instrument_registry[instrument_name]
    spec = instrument.workflow_factory[workflow_id]

    if spec.params is not None:
        assert issubclass(spec.params, pydantic.BaseModel), (
            f"spec.params for {workflow_id} should be a Pydantic BaseModel subclass, "
            f"got {spec.params}"
        )
        # Verify we can instantiate with defaults
        instance = spec.params()
        assert isinstance(instance, pydantic.BaseModel)


@pytest.mark.parametrize(("instrument_name", "workflow_id"), _collect_workflow_specs())
def test_workflow_spec_aux_sources_validation(
    instrument_name: str, workflow_id: WorkflowId
):
    """Test that spec.aux_sources is None or a valid Pydantic BaseModel class.

    Since aux_sources are now explicitly registered (not inferred from factory),
    this validates they're properly set in the spec.
    """
    instrument = instrument_registry[instrument_name]
    spec = instrument.workflow_factory[workflow_id]

    if spec.aux_sources is not None:
        assert issubclass(spec.aux_sources, pydantic.BaseModel), (
            f"spec.aux_sources for {workflow_id} should be a "
            f"Pydantic BaseModel subclass, got {spec.aux_sources}"
        )
        # Verify we can instantiate with defaults
        instance = spec.aux_sources()
        assert isinstance(instance, pydantic.BaseModel)


@pytest.mark.parametrize(("instrument_name", "workflow_id"), _collect_workflow_specs())
def test_workflow_spec_outputs_validation(
    instrument_name: str, workflow_id: WorkflowId
):
    """Test that spec.outputs is None or a valid Pydantic BaseModel class.

    The outputs field defines workflow outputs with metadata for UI display.
    """
    instrument = instrument_registry[instrument_name]
    spec = instrument.workflow_factory[workflow_id]

    if spec.outputs is not None:
        assert issubclass(spec.outputs, pydantic.BaseModel), (
            f"spec.outputs for {workflow_id} should be a "
            f"Pydantic BaseModel subclass, got {spec.outputs}"
        )
        # Verify we can instantiate (outputs model typically has no defaults needed)
        try:
            instance = spec.outputs()
            assert isinstance(instance, pydantic.BaseModel)
        except pydantic.ValidationError:
            # Outputs may require fields - that's OK, just check the class is valid
            pass


@pytest.mark.parametrize(("instrument_name", "workflow_id"), _collect_workflow_specs())
def test_workflow_params_serialization_roundtrip(
    instrument_name: str, workflow_id: WorkflowId
):
    """Test that params can be serialized and deserialized without loss.

    This simulates the frontend→backend→frontend cycle:
    - Frontend creates params model with defaults
    - WorkflowConfig.from_params serializes to dict (via model.model_dump())
    - Backend deserializes from dict (model.model_validate(dict))
    - Values should remain identical
    """
    instrument = instrument_registry[instrument_name]
    spec = instrument.workflow_factory[workflow_id]

    if spec.params is None:
        pytest.skip("Workflow has no parameters")

    # Create instance with defaults
    original = spec.params()

    # Simulate WorkflowConfig.from_params serialization
    workflow_config = WorkflowConfig.from_params(
        workflow_id=workflow_id,
        params=original.model_dump(),
    )
    serialized = workflow_config.params

    # Simulate backend deserialization (WorkflowFactory.create)
    deserialized = spec.params.model_validate(serialized)

    # Should be identical after roundtrip
    assert deserialized.model_dump() == original.model_dump()


@pytest.mark.parametrize(("instrument_name", "workflow_id"), _collect_workflow_specs())
def test_workflow_aux_sources_serialization_roundtrip(
    instrument_name: str, workflow_id: WorkflowId
):
    """Test that aux_sources can be serialized and deserialized without loss.

    This simulates the frontend→backend cycle for auxiliary sources:
    - Frontend creates aux_sources model with defaults
    - WorkflowConfig.from_params stores as dict
    - Backend deserializes from dict (model.model_validate(dict))
    """
    instrument = instrument_registry[instrument_name]
    spec = instrument.workflow_factory[workflow_id]

    if spec.aux_sources is None:
        pytest.skip("Workflow has no auxiliary sources")

    # Create instance with defaults
    original = spec.aux_sources()

    # Simulate WorkflowConfig.from_params storing aux sources
    workflow_config = WorkflowConfig.from_params(
        workflow_id=workflow_id,
        aux_source_names=original.model_dump(),
    )
    serialized = workflow_config.aux_source_names

    # Simulate backend deserialization (WorkflowFactory.create)
    deserialized = spec.aux_sources.model_validate(serialized)

    # Should be identical after roundtrip
    assert deserialized.model_dump() == original.model_dump()


@pytest.mark.parametrize(("instrument_name", "workflow_id"), _collect_workflow_specs())
def test_workflow_config_widget_adapter_compatibility(
    instrument_name: str, workflow_id: WorkflowId
):
    """Test that WorkflowSpec is compatible with WorkflowConfigurationAdapter.

    This verifies that WorkflowConfigurationAdapter can be created and used
    to extract default values, simulating what ConfigurationWidget does.
    """
    instrument = instrument_registry[instrument_name]
    spec = instrument.workflow_factory[workflow_id]

    # Create adapter (this is what WorkflowController.create_workflow_adapter does)
    adapter = WorkflowConfigurationAdapter(
        spec=spec,
        config_state=None,
        start_callback=lambda *args, **kwargs: True,
    )

    # Verify adapter properties work
    assert adapter.title == spec.title
    assert adapter.description == spec.description
    assert adapter.source_names == spec.source_names
    assert adapter.aux_sources == spec.aux_sources

    # Verify we can get the model class
    # First set aux sources (instantiate if available, otherwise None)
    aux_sources_model = adapter.aux_sources() if adapter.aux_sources else None
    model_class = adapter.set_aux_sources(aux_sources_model)
    if spec.params is not None:
        assert model_class == spec.params
        # Verify we can instantiate with defaults
        instance = model_class()
        assert isinstance(instance, pydantic.BaseModel)
    else:
        assert model_class is None

    # Verify aux_sources can be instantiated
    if spec.aux_sources is not None:
        aux_instance = spec.aux_sources()
        assert isinstance(aux_instance, pydantic.BaseModel)


def _collect_workflow_outputs():
    """Collect (instrument, workflow_id, output_name) for all workflow outputs.

    This enables per-output testing for finer-grained failure reporting.
    """
    outputs = []
    for instrument_name in available_instruments():
        _ = get_config(instrument_name)  # Load specs only
        instrument = instrument_registry[instrument_name]
        # DO NOT call instrument.load_factories()
        for workflow_id in instrument.workflow_factory:
            spec = instrument.workflow_factory[workflow_id]
            outputs.extend(
                pytest.param(
                    instrument_name,
                    workflow_id,
                    output_name,
                    id=f"{workflow_id}/{output_name}",
                )
                for output_name in spec.outputs.model_fields
            )
    return outputs


@pytest.mark.parametrize(
    ("instrument_name", "workflow_id", "output_name"), _collect_workflow_outputs()
)
def test_workflow_output_has_compatible_plotter(
    instrument_name: str, workflow_id: WorkflowId, output_name: str
):
    """Test that each workflow output has at least one compatible plotter.

    This ensures the dashboard can offer valid plotting options for every
    declared workflow output. The plotter matching uses the output's template
    DataArray (from default_factory) to validate against plotter requirements.
    """
    instrument = instrument_registry[instrument_name]
    spec = instrument.workflow_factory[workflow_id]

    template = spec.get_output_template(output_name)
    if template is None:
        pytest.fail(
            f"Output '{output_name}' of {workflow_id} has no template. "
            f"Add default_factory to the field definition to enable plotter matching."
        )

    compatible = plotter_registry.get_compatible_plotters_with_spec(
        {output_name: template}, spec.aux_sources
    )

    assert compatible, (
        f"Output '{output_name}' of {workflow_id} has no compatible plotter. "
        f"Template: ndim={template.ndim}, dims={template.dims}, "
        f"coords={list(template.coords)}"
    )
