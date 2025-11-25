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


def test_adapter_filters_removed_sources(tmp_path) -> None:
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

    # Set up legacy config BEFORE creating the backend, so it gets loaded during init.
    # This simulates a scenario where 'motion1' was previously configured for this
    # workflow, but is not in the monitor_histogram workflow spec (only monitor1/2).
    from ess.livedata.dashboard.config_store import ConfigStoreManager

    config_manager = ConfigStoreManager(instrument='dummy', config_dir=tmp_path)
    config_store = config_manager.get_store('workflow_configs')

    source_names = ['monitor1', 'monitor2', 'motion1']
    legacy_config = ConfigurationState(
        source_names=source_names,
        aux_source_names={},
        params=MonitorDataParams().model_dump(),
    )
    config_store[str(workflow_id)] = legacy_config.model_dump()

    # Now create backend with the same config dir - orchestrator loads legacy config
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend:
        # Verify config was loaded into orchestrator's staged_jobs
        staged = backend.workflow_controller._orchestrator.get_staged_config(
            workflow_id
        )
        # Only monitor1 and monitor2 should be staged (motion1 filtered by spec)
        assert set(staged.keys()) == {'monitor1', 'monitor2'}

        # Create adapter - it should only show sources from spec
        adapter = backend.workflow_controller.create_workflow_adapter(workflow_id)

        initial_sources = adapter.initial_source_names
        assert 'monitor1' in initial_sources
        assert 'monitor2' in initial_sources
        # motion1 should be filtered out as it's not in the workflow spec
        assert 'motion1' not in initial_sources


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


def test_incompatible_config_falls_back_to_defaults(tmp_path) -> None:
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

    # Set up incompatible config BEFORE creating the backend
    # This simulates an old version of the workflow with different param structure
    from ess.livedata.dashboard.config_store import ConfigStoreManager

    config_manager = ConfigStoreManager(instrument='dummy', config_dir=tmp_path)
    config_store = config_manager.get_store('workflow_configs')

    incompatible_config = ConfigurationState(
        source_names=['monitor1'],
        aux_source_names={},
        params={
            'old_field_that_no_longer_exists': 42,
            'another_invalid_field': 'invalid_value',
            # Completely wrong structure - not matching current MonitorDataParams
        },
    )
    config_store[str(workflow_id)] = incompatible_config.model_dump()

    # Now create backend with the same config dir
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend:
        # Adapter creation should not fail even with incompatible config
        adapter = backend.workflow_controller.create_workflow_adapter(workflow_id)

        # Adapter should validate and detect incompatibility, returning empty dict
        # which will cause the UI to use default parameter values
        initial_params = adapter.initial_parameter_values
        assert initial_params == {}, (
            "Expected empty dict for incompatible params to trigger defaults, "
            f"got {initial_params}"
        )

        # Verify source names are still restored (only params validation failed)
        assert adapter.initial_source_names == ['monitor1']


def test_plot_orchestrator_persistence_across_backend_restarts(tmp_path) -> None:
    """
    Test that PlotOrchestrator state persists across backend restarts.

    This test verifies the complete persistence flow for PlotOrchestrator:
    1. Create plot grids with cells via PlotOrchestrator
    2. Stop the backend
    3. Create a new backend with the same config dir
    4. Verify all grids and cells are restored correctly
    """
    from ess.livedata.config.workflow_spec import WorkflowId
    from ess.livedata.dashboard.plot_orchestrator import (
        CellGeometry,
        PlotCell,
        PlotConfig,
    )

    # Create first backend instance
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend1:
        orchestrator1 = backend1.plot_orchestrator

        # Add a grid
        grid_id = orchestrator1.add_grid(title='Test Grid', nrows=2, ncols=2)

        # Add a cell with plot configuration
        workflow_id = WorkflowId(
            instrument='dummy',
            namespace='monitor_data',
            name='monitor_histogram',
            version=1,
        )
        plot_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='histogram',
            source_names=['monitor1', 'monitor2'],
            plot_name='test_plotter',
            params={'param1': 'value1', 'param2': 42},
        )
        geometry = CellGeometry(row=0, col=0, row_span=1, col_span=1)
        cell = PlotCell(geometry=geometry, config=plot_config)

        cell_id = orchestrator1.add_plot(grid_id=grid_id, cell=cell)

        # Add another cell in the same grid
        plot_config2 = PlotConfig(
            workflow_id=workflow_id,
            output_name='spectrum',
            source_names=['monitor1'],
            plot_name='another_plotter',
            params={'threshold': 100.0},
        )
        geometry2 = CellGeometry(row=0, col=1, row_span=1, col_span=1)
        cell2 = PlotCell(geometry=geometry2, config=plot_config2)

        cell_id2 = orchestrator1.add_plot(grid_id=grid_id, cell=cell2)

        # Add a second grid with one cell
        grid_id2 = orchestrator1.add_grid(title='Second Grid', nrows=1, ncols=1)
        plot_config3 = PlotConfig(
            workflow_id=workflow_id,
            output_name='output3',
            source_names=['monitor2'],
            plot_name='third_plotter',
            params={},
        )
        geometry3 = CellGeometry(row=0, col=0, row_span=1, col_span=1)
        cell3 = PlotCell(geometry=geometry3, config=plot_config3)

        cell_id3 = orchestrator1.add_plot(grid_id=grid_id2, cell=cell3)

        # Verify grids exist before shutdown
        all_grids1 = orchestrator1.get_all_grids()
        assert len(all_grids1) == 2
        assert grid_id in all_grids1
        assert grid_id2 in all_grids1

    # Backend1 is now stopped and cleaned up

    # Create a new backend with the same config dir
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend2:
        orchestrator2 = backend2.plot_orchestrator

        # Verify grids were restored
        all_grids2 = orchestrator2.get_all_grids()
        assert len(all_grids2) == 2, f"Expected 2 grids, got {len(all_grids2)}"

        # Verify grid IDs match (UUIDs should be preserved)
        restored_grid_ids = set(all_grids2.keys())
        original_grid_ids = {grid_id, grid_id2}
        assert (
            restored_grid_ids == original_grid_ids
        ), f"Grid IDs don't match: {restored_grid_ids} != {original_grid_ids}"

        # Verify first grid configuration
        grid1_restored = all_grids2[grid_id]
        assert grid1_restored.title == 'Test Grid'
        assert grid1_restored.nrows == 2
        assert grid1_restored.ncols == 2
        assert len(grid1_restored.cells) == 2

        # Verify first cell configuration
        assert cell_id in grid1_restored.cells
        cell1_restored = grid1_restored.cells[cell_id]
        assert cell1_restored.geometry.row == 0
        assert cell1_restored.geometry.col == 0
        assert cell1_restored.geometry.row_span == 1
        assert cell1_restored.geometry.col_span == 1
        assert cell1_restored.config.workflow_id == workflow_id
        assert cell1_restored.config.output_name == 'histogram'
        assert cell1_restored.config.source_names == ['monitor1', 'monitor2']
        assert cell1_restored.config.plot_name == 'test_plotter'
        # Params are stored as dict (not validated on load)
        assert cell1_restored.config.params == {'param1': 'value1', 'param2': 42}

        # Verify second cell configuration
        assert cell_id2 in grid1_restored.cells
        cell2_restored = grid1_restored.cells[cell_id2]
        assert cell2_restored.geometry.row == 0
        assert cell2_restored.geometry.col == 1
        assert cell2_restored.config.output_name == 'spectrum'
        assert cell2_restored.config.source_names == ['monitor1']
        assert cell2_restored.config.params == {'threshold': 100.0}

        # Verify second grid configuration
        grid2_restored = all_grids2[grid_id2]
        assert grid2_restored.title == 'Second Grid'
        assert grid2_restored.nrows == 1
        assert grid2_restored.ncols == 1
        assert len(grid2_restored.cells) == 1

        # Verify third cell configuration
        assert cell_id3 in grid2_restored.cells
        cell3_restored = grid2_restored.cells[cell_id3]
        assert cell3_restored.config.output_name == 'output3'
        assert cell3_restored.config.source_names == ['monitor2']
        assert cell3_restored.config.params == {}


def test_plot_orchestrator_persists_pydantic_params_with_enums(tmp_path) -> None:
    """
    Test that PlotOrchestrator correctly serializes Pydantic params containing enums.

    This regression test ensures that enum values (like CombineMode, PlotScale) in
    Pydantic models are properly converted to strings during YAML serialization.
    Previously, enum instances caused serialization failures.
    """
    from ess.livedata.config.workflow_spec import WorkflowId
    from ess.livedata.dashboard.plot_orchestrator import (
        CellGeometry,
        PlotCell,
        PlotConfig,
    )
    from ess.livedata.dashboard.plot_params import (
        CombineMode,
        PlotParams2d,
        PlotScale,
        WindowMode,
    )

    # Create params with enum values (the problematic case)
    params_with_enums = PlotParams2d(
        layout={'combine_mode': CombineMode.layout, 'layout_columns': 2},
        window={'mode': WindowMode.window, 'window_duration_seconds': 10.0},
        plot_scale={
            'x_scale': PlotScale.linear,
            'y_scale': PlotScale.log,
            'color_scale': PlotScale.log,
        },
    )

    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend1:
        orchestrator1 = backend1.plot_orchestrator

        grid_id = orchestrator1.add_grid(title='Enum Test Grid', nrows=1, ncols=1)

        workflow_id = WorkflowId(
            instrument='dummy',
            namespace='monitor_data',
            name='monitor_histogram',
            version=1,
        )
        # Pass Pydantic model directly (this is what the UI does)
        plot_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='image',
            source_names=['monitor1'],
            plot_name='test_plotter',
            params=params_with_enums,
        )
        geometry = CellGeometry(row=0, col=0, row_span=1, col_span=1)
        cell = PlotCell(geometry=geometry, config=plot_config)

        cell_id = orchestrator1.add_plot(grid_id=grid_id, cell=cell)

    # Restart backend - this should not fail during deserialization
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend2:
        orchestrator2 = backend2.plot_orchestrator

        all_grids = orchestrator2.get_all_grids()
        assert len(all_grids) == 1

        restored_grid = all_grids[grid_id]
        assert cell_id in restored_grid.cells

        restored_params = restored_grid.cells[cell_id].config.params
        # Params are stored as dict with string enum values
        assert restored_params['layout']['combine_mode'] == 'layout'
        assert restored_params['window']['mode'] == 'window'
        assert restored_params['window']['window_duration_seconds'] == 10.0
        assert restored_params['plot_scale']['x_scale'] == 'linear'
        assert restored_params['plot_scale']['y_scale'] == 'log'
        assert restored_params['plot_scale']['color_scale'] == 'log'
