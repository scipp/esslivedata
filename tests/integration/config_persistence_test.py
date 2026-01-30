# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration tests for workflow configuration persistence via config store."""

from collections.abc import Generator

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.handlers.monitor_workflow_specs import MonitorDataParams
from ess.livedata.parameter_models import Scale, TimeUnit, TOAEdges
from tests.integration.backend import DashboardBackend


def add_cell_with_layer(orchestrator, grid_id, geometry, config):
    """Helper to add a cell with a single layer."""
    cell_id = orchestrator.add_cell(grid_id, geometry)
    orchestrator.add_layer(cell_id, config)
    return cell_id


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

    # Verify params are stored via orchestrator's staged config
    staged_config = backend_with_null_transport.job_orchestrator.get_staged_config(
        workflow_id
    )
    assert staged_config is not None, "Config should be stored"
    assert set(staged_config.keys()) == set(source_names)
    # Each source has its own config with the same params
    for source in source_names:
        assert staged_config[source].params == custom_params.model_dump()

    # get_workflow_config returns reference config (single source)
    ref_config = backend_with_null_transport.workflow_controller.get_workflow_config(
        workflow_id
    )
    assert ref_config is not None
    assert ref_config.params == custom_params.model_dump()

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
    default_params = MonitorDataParams().model_dump()
    # Raw dict format for persisted config
    legacy_config = {
        'jobs': {
            name: {'params': default_params, 'aux_source_names': {}}
            for name in source_names
        }
    }
    config_store[str(workflow_id)] = legacy_config

    # Now create backend with the same config dir - orchestrator loads legacy config
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend:
        # Verify config was loaded into orchestrator's staged_jobs
        staged = backend.job_orchestrator.get_staged_config(workflow_id)
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

    # Raw dict format for persisted config with incompatible params
    incompatible_config = {
        'jobs': {
            'monitor1': {
                'params': {
                    'old_field_that_no_longer_exists': 42,
                    'another_invalid_field': 'invalid_value',
                    # Completely wrong structure - not matching
                    # current MonitorDataParams
                },
                'aux_source_names': {},
            }
        }
    }
    config_store[str(workflow_id)] = incompatible_config

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

    Note: UUIDs are regenerated on load as they are runtime identity handles.
    We verify content equality, not ID equality.
    """
    from ess.livedata.config.workflow_spec import WorkflowId
    from ess.livedata.dashboard.data_roles import PRIMARY
    from ess.livedata.dashboard.plot_orchestrator import (
        CellGeometry,
        DataSourceConfig,
        PlotConfig,
    )
    from ess.livedata.dashboard.plot_params import PlotParams1d, WindowMode

    # Create first backend instance
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend1:
        orchestrator1 = backend1.plot_orchestrator

        # Add a grid
        orchestrator1.add_grid(title='Test Grid', nrows=2, ncols=2)

        # Add a cell with plot configuration
        workflow_id = WorkflowId(
            instrument='dummy',
            namespace='monitor_data',
            name='monitor_histogram',
            version=1,
        )
        # Use real plotter with custom params
        params1 = PlotParams1d(
            window={'mode': WindowMode.window, 'window_duration_seconds': 5.0}
        )
        plot_config = PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=workflow_id,
                    output_name='histogram',
                    source_names=['monitor1', 'monitor2'],
                )
            },
            plot_name='lines',
            params=params1,
        )
        geometry = CellGeometry(row=0, col=0, row_span=1, col_span=1)

        # Find the grid by title to add cells
        grids1 = orchestrator1.get_all_grids()
        grid_id = next(gid for gid, g in grids1.items() if g.title == 'Test Grid')
        add_cell_with_layer(orchestrator1, grid_id, geometry, plot_config)

        # Add another cell in the same grid with different params
        params2 = PlotParams1d(
            window={'mode': WindowMode.latest, 'window_duration_seconds': 10.0}
        )
        plot_config2 = PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=workflow_id,
                    output_name='spectrum',
                    source_names=['monitor1'],
                )
            },
            plot_name='lines',
            params=params2,
        )
        geometry2 = CellGeometry(row=0, col=1, row_span=1, col_span=1)

        add_cell_with_layer(orchestrator1, grid_id, geometry2, plot_config2)

        # Add a second grid with one cell using default params
        orchestrator1.add_grid(title='Second Grid', nrows=1, ncols=1)
        grids1 = orchestrator1.get_all_grids()
        grid_id2 = next(gid for gid, g in grids1.items() if g.title == 'Second Grid')

        params3 = PlotParams1d()  # Default params
        plot_config3 = PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=workflow_id,
                    output_name='output3',
                    source_names=['monitor2'],
                )
            },
            plot_name='lines',
            params=params3,
        )
        geometry3 = CellGeometry(row=0, col=0, row_span=1, col_span=1)

        add_cell_with_layer(orchestrator1, grid_id2, geometry3, plot_config3)

        # Verify grids exist before shutdown
        all_grids1 = orchestrator1.get_all_grids()
        assert len(all_grids1) == 2

    # Backend1 is now stopped and cleaned up

    # Create a new backend with the same config dir
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend2:
        orchestrator2 = backend2.plot_orchestrator

        # Verify grids were restored
        all_grids2 = orchestrator2.get_all_grids()
        assert len(all_grids2) == 2, f"Expected 2 grids, got {len(all_grids2)}"

        # Find grids by title (UUIDs are regenerated on load)
        grid1_restored = next(g for g in all_grids2.values() if g.title == 'Test Grid')
        grid2_restored = next(
            g for g in all_grids2.values() if g.title == 'Second Grid'
        )

        # Verify first grid configuration
        assert grid1_restored.title == 'Test Grid'
        assert grid1_restored.nrows == 2
        assert grid1_restored.ncols == 2
        assert len(grid1_restored.cells) == 2

        # Find cells by geometry (UUIDs are regenerated)
        cell1_restored = next(
            c for c in grid1_restored.cells.values() if c.geometry.col == 0
        )
        cell2_restored = next(
            c for c in grid1_restored.cells.values() if c.geometry.col == 1
        )

        # Verify first cell configuration
        assert cell1_restored.geometry.row == 0
        assert cell1_restored.geometry.col == 0
        assert cell1_restored.geometry.row_span == 1
        assert cell1_restored.geometry.col_span == 1
        assert cell1_restored.layers[0].config.workflow_id == workflow_id
        assert cell1_restored.layers[0].config.output_name == 'histogram'
        assert cell1_restored.layers[0].config.source_names == ['monitor1', 'monitor2']
        assert cell1_restored.layers[0].config.plot_name == 'lines'
        # Params are validated and restored as model
        assert cell1_restored.layers[0].config.params == params1

        # Verify second cell configuration
        assert cell2_restored.geometry.row == 0
        assert cell2_restored.geometry.col == 1
        assert cell2_restored.layers[0].config.output_name == 'spectrum'
        assert cell2_restored.layers[0].config.source_names == ['monitor1']
        assert cell2_restored.layers[0].config.params == params2

        # Verify second grid configuration
        assert grid2_restored.title == 'Second Grid'
        assert grid2_restored.nrows == 1
        assert grid2_restored.ncols == 1
        assert len(grid2_restored.cells) == 1

        # Verify third cell configuration
        cell3_restored = next(iter(grid2_restored.cells.values()))
        assert cell3_restored.layers[0].config.output_name == 'output3'
        assert cell3_restored.layers[0].config.source_names == ['monitor2']
        assert cell3_restored.layers[0].config.params == params3


def test_plot_orchestrator_persists_pydantic_params_with_enums(tmp_path) -> None:
    """
    Test that PlotOrchestrator correctly serializes Pydantic params containing enums.

    This regression test ensures that enum values (like CombineMode, PlotScale) in
    Pydantic models are properly converted to strings during YAML serialization.
    Previously, enum instances caused serialization failures.
    """
    from ess.livedata.config.workflow_spec import WorkflowId
    from ess.livedata.dashboard.data_roles import PRIMARY
    from ess.livedata.dashboard.plot_orchestrator import (
        CellGeometry,
        DataSourceConfig,
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

        orchestrator1.add_grid(title='Enum Test Grid', nrows=1, ncols=1)

        workflow_id = WorkflowId(
            instrument='dummy',
            namespace='monitor_data',
            name='monitor_histogram',
            version=1,
        )
        # Pass Pydantic model directly (this is what the UI does)
        plot_config = PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=workflow_id,
                    output_name='image',
                    source_names=['monitor1'],
                )
            },
            plot_name='image',  # Use real plotter that accepts PlotParams2d
            params=params_with_enums,
        )
        geometry = CellGeometry(row=0, col=0, row_span=1, col_span=1)

        # Find the grid by title to add the cell
        grids1 = orchestrator1.get_all_grids()
        grid_id = next(gid for gid, g in grids1.items() if g.title == 'Enum Test Grid')
        add_cell_with_layer(orchestrator1, grid_id, geometry, plot_config)

    # Restart backend - this should not fail during deserialization
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend2:
        orchestrator2 = backend2.plot_orchestrator

        all_grids = orchestrator2.get_all_grids()
        assert len(all_grids) == 1

        # Find grid and cell by content (UUIDs are regenerated on load)
        restored_grid = next(iter(all_grids.values()))
        assert restored_grid.title == 'Enum Test Grid'
        assert len(restored_grid.cells) == 1

        restored_cell = next(iter(restored_grid.cells.values()))
        restored_params = restored_cell.layers[0].config.params
        # Params are validated and restored as model, equal to original
        assert restored_params == params_with_enums


def test_plot_orchestrator_persists_multi_layer_cells(tmp_path) -> None:
    """
    Test that PlotOrchestrator correctly persists cells with multiple layers.

    This test verifies that cells with multiple layers are correctly serialized
    and deserialized across backend restarts.
    """
    from ess.livedata.config.workflow_spec import WorkflowId
    from ess.livedata.dashboard.data_roles import PRIMARY
    from ess.livedata.dashboard.plot_orchestrator import (
        CellGeometry,
        DataSourceConfig,
        PlotConfig,
    )
    from ess.livedata.dashboard.plot_params import PlotParams1d, WindowMode

    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )

    # Create two layer configurations
    params1 = PlotParams1d(
        window={'mode': WindowMode.window, 'window_duration_seconds': 5.0}
    )
    config1 = PlotConfig(
        data_sources={
            PRIMARY: DataSourceConfig(
                workflow_id=workflow_id,
                output_name='histogram',
                source_names=['monitor1'],
            )
        },
        plot_name='lines',
        params=params1,
    )

    params2 = PlotParams1d(
        window={'mode': WindowMode.latest, 'window_duration_seconds': 10.0}
    )
    config2 = PlotConfig(
        data_sources={
            PRIMARY: DataSourceConfig(
                workflow_id=workflow_id,
                output_name='spectrum',
                source_names=['monitor2'],
            )
        },
        plot_name='lines',
        params=params2,
    )

    geometry = CellGeometry(row=0, col=0, row_span=1, col_span=1)

    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend1:
        orchestrator1 = backend1.plot_orchestrator

        orchestrator1.add_grid(title='Multi-Layer Test', nrows=1, ncols=1)

        grids1 = orchestrator1.get_all_grids()
        grid_id = next(
            gid for gid, g in grids1.items() if g.title == 'Multi-Layer Test'
        )

        # Create cell with two layers using add_cell + add_layer
        cell_id = orchestrator1.add_cell(grid_id, geometry)
        orchestrator1.add_layer(cell_id, config1)
        orchestrator1.add_layer(cell_id, config2)

        # Verify cell has two layers before shutdown
        grid = orchestrator1.get_grid(grid_id)
        cell = next(iter(grid.cells.values()))
        assert len(cell.layers) == 2

    # Restart backend and verify layers are restored
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend2:
        orchestrator2 = backend2.plot_orchestrator

        all_grids = orchestrator2.get_all_grids()
        assert len(all_grids) == 1

        restored_grid = next(iter(all_grids.values()))
        assert restored_grid.title == 'Multi-Layer Test'
        assert len(restored_grid.cells) == 1

        restored_cell = next(iter(restored_grid.cells.values()))

        # Verify both layers were restored
        assert (
            len(restored_cell.layers) == 2
        ), f"Expected 2 layers, got {len(restored_cell.layers)}"

        # Verify layer configs (order should be preserved)
        restored_layer1 = restored_cell.layers[0]
        restored_layer2 = restored_cell.layers[1]

        assert restored_layer1.config.output_name == 'histogram'
        assert restored_layer1.config.source_names == ['monitor1']
        assert restored_layer1.config.params == params1

        assert restored_layer2.config.output_name == 'spectrum'
        assert restored_layer2.config.source_names == ['monitor2']
        assert restored_layer2.config.params == params2
