# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Test workflow factory registration and instantiation.

This test file validates workflow factories by loading factory implementations.
These tests are slower than workflow_spec_test.py because they import heavy
dependencies (sciline, ess.reduce, etc.).

For fast spec-only tests, see registered_workflow_specs_test.py.
"""

import uuid

import pytest

from ess.livedata.config.instrument import instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.config.workflow_spec import WorkflowConfig, WorkflowId
from ess.livedata.core.job_manager import JobFactory, JobId
from ess.livedata.dashboard.workflow_configuration_adapter import (
    WorkflowConfigurationAdapter,
)


def _is_slow_workflow(workflow_id):
    """Check if a workflow is known to be slow (>2s)."""
    # Bifrost data reduction workflows are slow due to complex spectroscopy setup
    if (
        workflow_id.instrument == 'bifrost'
        and workflow_id.namespace == 'data_reduction'
    ):
        return True
    # LOKI i_of_q workflows are slow due to SANS reduction complexity
    if workflow_id.instrument == 'loki' and workflow_id.namespace == 'data_reduction':
        if workflow_id.name.startswith('i_of_q'):
            return True
    return False


def _collect_workflow_factories():
    """Collect workflows WITH factories loaded (slower).

    This imports both spec modules and factory implementation modules,
    loading heavy dependencies like sciline, ess.reduce, etc.
    """
    workflows = []
    for instrument_name in available_instruments():
        _ = get_config(instrument_name)  # Register instrument
        instrument = instrument_registry[instrument_name]
        instrument.load_factories()
        for workflow_id in instrument.workflow_factory:
            marks = [pytest.mark.slow] if _is_slow_workflow(workflow_id) else []
            workflows.append(
                pytest.param(
                    instrument_name, workflow_id, id=str(workflow_id), marks=marks
                )
            )
    return workflows


@pytest.mark.parametrize(
    ("instrument_name", "workflow_id"), _collect_workflow_factories()
)
def test_workflow_factory_is_attached(instrument_name: str, workflow_id: WorkflowId):
    """Test that each workflow spec has a factory attached.

    This validates the two-phase registration:
    1. Spec was registered via instrument.register_spec()
    2. Factory was attached via handle.attach_factory()

    If this test fails, a spec was registered but the factory was never attached.
    """
    instrument = instrument_registry[instrument_name]
    workflow_factory = instrument.workflow_factory

    # Spec should exist (already validated by parametrization)
    assert workflow_id in workflow_factory

    # Factory should be attached
    assert workflow_id in workflow_factory._factories, (
        f"Workflow spec '{workflow_id}' exists but no factory is attached. "
        f"Did you forget to call handle.attach_factory()?"
    )


@pytest.mark.parametrize(
    ("instrument_name", "workflow_id"), _collect_workflow_factories()
)
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
        config_state=None,  # No saved config, use defaults
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
        params=params_model.model_dump() if params_model is not None else None,
        aux_source_names=aux_sources_model.model_dump()
        if aux_sources_model is not None
        else None,
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
