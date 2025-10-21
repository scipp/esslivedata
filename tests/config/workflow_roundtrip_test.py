# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Test workflow registration → configuration → instantiation roundtrip.

This test ensures that for each registered workflow:
1. WorkflowSpec is properly created with params and aux_sources type hints
2. Default values can be obtained from Pydantic models (via adapter)
3. WorkflowConfig can be created from defaults (via from_params)
4. Backend can instantiate the workflow from the config
"""

import uuid

import pydantic
import pytest

from ess.livedata.config.instrument import instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.config.workflow_spec import WorkflowConfig, WorkflowId
from ess.livedata.core.job_manager import JobFactory, JobId
from ess.livedata.dashboard.workflow_configuration_adapter import (
    WorkflowConfigurationAdapter,
)


def _collect_workflows():
    """Collect all workflows from all instruments for parameterization."""
    import importlib

    workflows = []
    for instrument_name in available_instruments():
        _ = get_config(instrument_name)  # Load module to register instrument

        # Load factories for instruments using new submodule structure
        try:
            importlib.import_module(
                f'ess.livedata.config.instruments.{instrument_name}.factories'
            )
        except ModuleNotFoundError:
            # Instrument may not have been converted to submodule structure yet
            pass

        instrument = instrument_registry[instrument_name]
        workflows.extend(
            [
                pytest.param(instrument_name, workflow_id, id=str(workflow_id))
                for workflow_id in instrument.workflow_factory
            ]
        )
    return workflows


@pytest.mark.parametrize(("instrument_name", "workflow_id"), _collect_workflows())
def test_workflow_roundtrip(instrument_name: str, workflow_id: WorkflowId):
    """Test complete roundtrip for a registered workflow.

    This test validates the chain:
    1. Workflow registration creates proper WorkflowSpec with type hints
    2. Default parameter values can be extracted via adapter
    3. WorkflowConfig can be created using from_params helper
    4. Backend can instantiate workflow from config (JobFactory)
    """
    # Skip known workflows that require data files not available in CI
    if str(workflow_id) == "dream/data_reduction/powder_reduction_with_vanadium/1":
        pytest.skip(
            "Workflow requires vanadium data file "
            "(268227_00024779_Vana_inc_BC_offset_240_deg_wlgth.hdf) "
            "not available in test environment"
        )

    instrument = instrument_registry[instrument_name]
    workflow_factory = instrument.workflow_factory

    # Step 1: Verify WorkflowSpec was created with proper type hints
    spec = workflow_factory[workflow_id]
    assert spec is not None, f"WorkflowSpec not found for {workflow_id}"

    # Step 2: Use WorkflowConfigurationAdapter to get model classes
    # This simulates what the frontend (ConfigurationWidget) does
    adapter = WorkflowConfigurationAdapter(
        spec=spec,
        persistent_config=None,  # No saved config, use defaults
        start_callback=lambda *args, **kwargs: True,  # Dummy callback for testing
    )

    # Get aux sources model first (like ConfigurationWidget does)
    aux_sources_model = None
    if adapter.aux_sources is not None:
        # Instantiate with defaults
        aux_sources_model = adapter.aux_sources()

    # Set aux sources and get parameter model class (like ConfigurationWidget does)
    params_model = None
    params_class = adapter.set_aux_sources(aux_sources_model)
    if params_class is not None:
        # Instantiate with defaults (ConfigurationWidget uses initial_parameter_values
        # if available, otherwise creates with defaults)
        params_model = params_class()

    # Step 3: Create WorkflowConfig using the helper method
    # This simulates what WorkflowController.start_workflow does
    workflow_config = WorkflowConfig.from_params(
        workflow_id=workflow_id,
        params=params_model,
        aux_source_names=aux_sources_model,
    )

    # Step 4: Instantiate workflow via backend path (JobFactory → WorkflowFactory)
    # Pick the first available source, or use empty string if none specified
    source_name = spec.source_names[0] if spec.source_names else "test_source"

    # Set active namespace to match the workflow namespace
    # (in production this is set when the service starts)
    original_namespace = instrument.active_namespace
    instrument.active_namespace = workflow_id.namespace
    try:
        job_factory = JobFactory(instrument)
        job_id = JobId(source_name=source_name, job_number=uuid.uuid4())

        # This should not raise - it validates params and aux_sources internally
        job = job_factory.create(job_id=job_id, config=workflow_config)
    finally:
        # Restore original namespace
        instrument.active_namespace = original_namespace

    # Verify job was created successfully
    assert job is not None
    assert job.job_id == job_id
    assert job.workflow_id == workflow_id


@pytest.mark.parametrize(("instrument_name", "workflow_id"), _collect_workflows())
def test_workflow_spec_has_type_hints(instrument_name: str, workflow_id: WorkflowId):
    """Test that WorkflowSpec has type hints extracted from factory function.

    This verifies that the decorator in WorkflowFactory.register properly extracted
    type hints and populated spec.params and spec.aux_sources.
    """
    instrument = instrument_registry[instrument_name]
    spec = instrument.workflow_factory[workflow_id]

    # Type hints should be extracted if factory has params/aux_sources arguments
    # We can't easily check the factory signature here without accessing private state,
    # but we can verify that if type hints exist, they're valid Pydantic models

    if spec.params is not None:
        assert issubclass(spec.params, pydantic.BaseModel), (
            f"spec.params for {workflow_id} should be a Pydantic BaseModel subclass, "
            f"got {spec.params}"
        )
        # Verify we can instantiate with defaults
        instance = spec.params()
        assert isinstance(instance, pydantic.BaseModel)

    if spec.aux_sources is not None:
        assert issubclass(spec.aux_sources, pydantic.BaseModel), (
            f"spec.aux_sources for {workflow_id} should be a "
            f"Pydantic BaseModel subclass, got {spec.aux_sources}"
        )
        # Verify we can instantiate with defaults
        instance = spec.aux_sources()
        assert isinstance(instance, pydantic.BaseModel)


@pytest.mark.parametrize(("instrument_name", "workflow_id"), _collect_workflows())
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
        params=original,
    )
    serialized = workflow_config.params

    # Simulate backend deserialization (WorkflowFactory.create)
    deserialized = spec.params.model_validate(serialized)

    # Should be identical after roundtrip
    assert deserialized.model_dump() == original.model_dump()


@pytest.mark.parametrize(("instrument_name", "workflow_id"), _collect_workflows())
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
        aux_source_names=original,  # Pass model instance
    )
    serialized = workflow_config.aux_source_names

    # Simulate backend deserialization (WorkflowFactory.create)
    deserialized = spec.aux_sources.model_validate(serialized)

    # Should be identical after roundtrip
    assert deserialized.model_dump() == original.model_dump()


@pytest.mark.parametrize(("instrument_name", "workflow_id"), _collect_workflows())
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
        persistent_config=None,
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
