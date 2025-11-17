# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration tests for workflow configuration persistence via config store."""

from collections.abc import Generator

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.configuration_adapter import ConfigurationState
from ess.livedata.handlers.monitor_workflow_specs import MonitorDataParams
from ess.livedata.parameter_models import Scale, TimeUnit, TOAEdges
from tests.integration.backend import DashboardBackend


@pytest.fixture
def backend_with_null_transport() -> Generator[DashboardBackend, None, None]:
    """Create DashboardBackend with null transport (no Kafka required)."""
    with DashboardBackend(instrument='dummy', dev=True, transport='none') as backend:
        yield backend


def test_workflow_params_stored_and_retrieved_via_config_store(
    backend_with_null_transport: DashboardBackend,
) -> None:
    """
    Test that workflow params are stored in config store and retrieved correctly.

    This test verifies the complete persistence flow:
    1. Start a workflow via WorkflowController with specific params
    2. Verify params are stored in the config store
    3. Create a new adapter via the controller
    4. Verify the adapter retrieves the correct params from the config store

    Note: This test uses null transport (no Kafka required) since it only tests
    the config store persistence mechanism through the controller.
    """
    # Define workflow parameters with non-default values
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )
    source_names = ['monitor1', 'monitor2']

    # Create params with custom values (non-default)
    custom_params = MonitorDataParams(
        toa_edges=TOAEdges(
            start=5.0,
            stop=15.0,
            num_bins=150,
            scale=Scale.LINEAR,
            unit=TimeUnit.MS,
        )
    )

    # Start the workflow with custom parameters
    job_ids = backend_with_null_transport.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=source_names,
        config=custom_params,
    )

    assert len(job_ids) == 2, f"Expected 2 jobs, got {len(job_ids)}"

    # Verify params are stored in config store
    stored_config = backend_with_null_transport.workflow_controller.get_workflow_config(
        workflow_id
    )
    assert stored_config is not None, "Config should be stored in config store"
    assert stored_config.source_names == source_names
    assert stored_config.params == custom_params.model_dump()

    # Create adapter and verify it retrieves correct params from config store
    adapter = backend_with_null_transport.workflow_controller.create_workflow_adapter(
        workflow_id
    )
    assert adapter.initial_source_names == source_names
    assert adapter.initial_parameter_values == custom_params.model_dump()


def test_adapter_filters_removed_sources(
    backend_with_null_transport: DashboardBackend,
) -> None:
    """
    Test that adapter filters out sources that are no longer available.

    This test verifies that if a workflow was started with sources that are
    no longer in the workflow spec, the adapter correctly filters them out
    when restoring the configuration.
    """
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )

    # Start workflow with multiple sources (monitor3 not in spec)
    source_names = ['monitor1', 'monitor2', 'monitor3']
    backend_with_null_transport.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=source_names,
        config=MonitorDataParams(),
    )

    # Verify config is stored with all sources
    stored_config = backend_with_null_transport.workflow_controller.get_workflow_config(
        workflow_id
    )
    assert stored_config is not None
    assert set(stored_config.source_names) == {'monitor1', 'monitor2', 'monitor3'}

    # Create adapter - it should filter to only sources available in spec
    adapter = backend_with_null_transport.workflow_controller.create_workflow_adapter(
        workflow_id
    )

    # Adapter should only return sources that are both persisted AND in the spec
    initial_sources = adapter.initial_source_names
    assert 'monitor1' in initial_sources
    assert 'monitor2' in initial_sources
    # monitor3 should be filtered out as it's not in the workflow spec
    assert 'monitor3' not in initial_sources


def test_config_persists_across_adapter_recreations(
    backend_with_null_transport: DashboardBackend,
) -> None:
    """
    Test that config persists correctly across multiple adapter recreations.

    This verifies that creating multiple adapters from the same stored config
    yields consistent results.
    """
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )
    source_names = ['monitor1']
    custom_params = MonitorDataParams(
        toa_edges=TOAEdges(
            start=1.0,
            stop=10.0,
            num_bins=200,
            scale=Scale.LINEAR,
            unit=TimeUnit.MS,
        )
    )

    backend_with_null_transport.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=source_names,
        config=custom_params,
    )

    # Create two adapters from the same stored config
    adapter1 = backend_with_null_transport.workflow_controller.create_workflow_adapter(
        workflow_id
    )
    adapter2 = backend_with_null_transport.workflow_controller.create_workflow_adapter(
        workflow_id
    )

    # Both adapters should retrieve identical params from config store
    assert adapter1.initial_parameter_values == adapter2.initial_parameter_values
    assert adapter1.initial_parameter_values is not adapter2.initial_parameter_values
    assert adapter1.initial_parameter_values['toa_edges']['num_bins'] == 200


def test_incompatible_config_falls_back_to_defaults(
    backend_with_null_transport: DashboardBackend,
) -> None:
    """
    Test that incompatible config doesn't break adapter creation.

    If stored config has params that are incompatible with the current
    workflow parameter model (e.g., due to schema changes between versions),
    the adapter should validate against the current model and fall back to
    defaults rather than propagating invalid data to the UI.
    """
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )

    # Manually inject completely incompatible config into the store
    # (e.g., from an old version of the workflow with different param structure)
    incompatible_config = ConfigurationState(
        source_names=['monitor1'],
        aux_source_names={},
        params={
            'old_field_that_no_longer_exists': 42,
            'another_invalid_field': 'invalid_value',
            # Completely wrong structure - not matching current MonitorDataParams
        },
    )

    # Directly store incompatible config in the config store
    config_store = backend_with_null_transport.config_manager.get_store(
        'workflow_configs'
    )
    config_store[workflow_id] = incompatible_config.model_dump()

    # Adapter creation should not fail even with incompatible config
    adapter = backend_with_null_transport.workflow_controller.create_workflow_adapter(
        workflow_id
    )

    # Adapter should validate and detect incompatibility, returning empty dict
    # which will cause the UI to use default parameter values
    initial_params = adapter.initial_parameter_values
    assert initial_params == {}, (
        "Expected empty dict for incompatible params to trigger defaults, "
        f"got {initial_params}"
    )

    # Verify source names are still restored (only params validation failed)
    assert adapter.initial_source_names == ['monitor1']
