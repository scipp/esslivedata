# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for config store implementations."""

import tempfile
from pathlib import Path

import pytest
import yaml

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.config_store import (
    ConfigStoreManager,
    FileBackedConfigStore,
    InMemoryConfigStore,
)


@pytest.fixture
def temp_config_file():
    """Create a temporary config file path for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = Path(f.name)
    # Delete the file so tests start with non-existent file
    temp_path.unlink()
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()
    # Also cleanup any .tmp files
    temp_tmp = temp_path.with_suffix('.tmp')
    if temp_tmp.exists():
        temp_tmp.unlink()


@pytest.fixture
def workflow_id_1():
    """First test WorkflowId."""
    return WorkflowId(
        instrument='dummy', namespace='reduction', name='powder_workflow', version=1
    )


@pytest.fixture
def workflow_id_2():
    """Second test WorkflowId."""
    return WorkflowId(
        instrument='dummy', namespace='reduction', name='imaging_workflow', version=1
    )


@pytest.fixture
def config_value_1():
    """First test config value."""
    return {'source_names': ['detector_1'], 'params': {'threshold': 150.0}}


@pytest.fixture
def config_value_2():
    """Second test config value."""
    return {'source_names': ['detector_2'], 'params': {'threshold': 200.0}}


class TestInMemoryConfigStore:
    """Tests for InMemoryConfigStore."""

    def test_empty_store_creation(self):
        """Test creating an InMemoryConfigStore."""
        store = InMemoryConfigStore()
        assert len(store) == 0

    def test_write_and_read(self, workflow_id_1, config_value_1):
        """Test writing and reading a config."""
        store = InMemoryConfigStore()
        store[workflow_id_1] = config_value_1

        assert workflow_id_1 in store
        assert store[workflow_id_1] == config_value_1

    def test_multiple_configs(
        self, workflow_id_1, workflow_id_2, config_value_1, config_value_2
    ):
        """Test storing multiple configs."""
        store = InMemoryConfigStore()
        store[workflow_id_1] = config_value_1
        store[workflow_id_2] = config_value_2

        assert len(store) == 2
        assert store[workflow_id_1] == config_value_1
        assert store[workflow_id_2] == config_value_2

    def test_update_existing_config(self, workflow_id_1, config_value_1):
        """Test updating an existing config."""
        store = InMemoryConfigStore()
        store[workflow_id_1] = config_value_1

        # Update the config
        new_value = {'source_names': ['detector_3'], 'params': {'threshold': 300.0}}
        store[workflow_id_1] = new_value

        assert len(store) == 1
        assert store[workflow_id_1] == new_value

    def test_delete_config(self, workflow_id_1, config_value_1):
        """Test deleting a config."""
        store = InMemoryConfigStore()
        store[workflow_id_1] = config_value_1
        assert workflow_id_1 in store

        del store[workflow_id_1]
        assert workflow_id_1 not in store
        assert len(store) == 0

    def test_lru_eviction(self):
        """Test LRU eviction when max_configs is exceeded."""
        store = InMemoryConfigStore(max_configs=5, cleanup_fraction=0.4)

        # Add 6 configs (exceeds max_configs=5)
        for i in range(6):
            wf_id = WorkflowId(
                instrument='dummy',
                namespace='reduction',
                name=f'workflow_{i}',
                version=1,
            )
            store[wf_id] = {'params': {'value': i}}

        # Should have evicted 40% (2 configs), leaving 4
        assert len(store) == 4

        # First two should be evicted (oldest)
        wf_id_0 = WorkflowId(
            instrument='dummy', namespace='reduction', name='workflow_0', version=1
        )
        wf_id_1 = WorkflowId(
            instrument='dummy', namespace='reduction', name='workflow_1', version=1
        )
        assert wf_id_0 not in store
        assert wf_id_1 not in store

        # Last four should remain
        for i in range(2, 6):
            wf_id = WorkflowId(
                instrument='dummy',
                namespace='reduction',
                name=f'workflow_{i}',
                version=1,
            )
            assert wf_id in store

    def test_no_max_configs(self):
        """Test store without max_configs limit."""
        store = InMemoryConfigStore(max_configs=None)

        # Add many configs - should not evict
        for i in range(150):
            wf_id = WorkflowId(
                instrument='dummy',
                namespace='reduction',
                name=f'workflow_{i}',
                version=1,
            )
            store[wf_id] = {'params': {'value': i}}

        assert len(store) == 150

    def test_iteration(self, workflow_id_1, workflow_id_2):
        """Test iterating over configs."""
        store = InMemoryConfigStore()
        store[workflow_id_1] = {'params': {'value': 1}}
        store[workflow_id_2] = {'params': {'value': 2}}

        keys = list(store.keys())
        assert len(keys) == 2
        assert workflow_id_1 in keys
        assert workflow_id_2 in keys


class TestFileBackedConfigStore:
    """Tests for FileBackedConfigStore."""

    def test_empty_store_creation(self, temp_config_file):
        """Test creating a FileBackedConfigStore with no existing file."""
        store = FileBackedConfigStore(temp_config_file)
        assert len(store) == 0
        # Parent directory is created, but file itself is not created until first write
        assert temp_config_file.parent.exists()
        assert not temp_config_file.exists()

    def test_write_and_read(self, temp_config_file, workflow_id_1, config_value_1):
        """Test writing and reading a config."""
        store = FileBackedConfigStore(temp_config_file)
        store[workflow_id_1] = config_value_1

        # Verify file was created and contains the config
        assert temp_config_file.exists()
        with open(temp_config_file) as f:
            data = yaml.safe_load(f)
        assert str(workflow_id_1) in data
        assert data[str(workflow_id_1)] == config_value_1

        # Verify reading from store
        assert workflow_id_1 in store
        assert store[workflow_id_1] == config_value_1

    def test_persistence_across_instances(
        self, temp_config_file, workflow_id_1, config_value_1
    ):
        """Test that configs persist when creating a new store instance."""
        # Write with first instance
        store1 = FileBackedConfigStore(temp_config_file)
        store1[workflow_id_1] = config_value_1

        # Read with second instance
        store2 = FileBackedConfigStore(temp_config_file)
        assert workflow_id_1 in store2
        assert store2[workflow_id_1] == config_value_1

    def test_multiple_configs(
        self,
        temp_config_file,
        workflow_id_1,
        workflow_id_2,
        config_value_1,
        config_value_2,
    ):
        """Test storing multiple configs."""
        store = FileBackedConfigStore(temp_config_file)
        store[workflow_id_1] = config_value_1
        store[workflow_id_2] = config_value_2

        assert len(store) == 2
        assert store[workflow_id_1] == config_value_1
        assert store[workflow_id_2] == config_value_2

    def test_update_existing_config(
        self, temp_config_file, workflow_id_1, config_value_1
    ):
        """Test updating an existing config."""
        store = FileBackedConfigStore(temp_config_file)
        store[workflow_id_1] = config_value_1

        # Update the config
        new_value = {'source_names': ['detector_3'], 'params': {'threshold': 300.0}}
        store[workflow_id_1] = new_value

        assert len(store) == 1
        assert store[workflow_id_1] == new_value

        # Verify persistence
        store2 = FileBackedConfigStore(temp_config_file)
        assert store2[workflow_id_1] == new_value

    def test_delete_config(self, temp_config_file, workflow_id_1, config_value_1):
        """Test deleting a config."""
        store = FileBackedConfigStore(temp_config_file)
        store[workflow_id_1] = config_value_1
        assert workflow_id_1 in store

        del store[workflow_id_1]
        assert workflow_id_1 not in store
        assert len(store) == 0

        # Verify deletion persisted
        store2 = FileBackedConfigStore(temp_config_file)
        assert workflow_id_1 not in store2

    def test_lru_eviction(self, temp_config_file):
        """Test LRU eviction when max_configs is exceeded."""
        store = FileBackedConfigStore(
            temp_config_file, max_configs=5, cleanup_fraction=0.4
        )

        # Add 6 configs (exceeds max_configs=5)
        for i in range(6):
            wf_id = WorkflowId(
                instrument='dummy',
                namespace='reduction',
                name=f'workflow_{i}',
                version=1,
            )
            store[wf_id] = {'params': {'value': i}}

        # Should have evicted 40% (2 configs), leaving 4
        assert len(store) == 4

        # First two should be evicted (oldest)
        wf_id_0 = WorkflowId(
            instrument='dummy', namespace='reduction', name='workflow_0', version=1
        )
        wf_id_1 = WorkflowId(
            instrument='dummy', namespace='reduction', name='workflow_1', version=1
        )
        assert wf_id_0 not in store
        assert wf_id_1 not in store

        # Last four should remain
        for i in range(2, 6):
            wf_id = WorkflowId(
                instrument='dummy',
                namespace='reduction',
                name=f'workflow_{i}',
                version=1,
            )
            assert wf_id in store

    def test_corrupted_file_handling(
        self, temp_config_file, workflow_id_1, config_value_1
    ):
        """Test graceful handling of corrupted YAML file."""
        # Write invalid YAML
        with open(temp_config_file, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")

        # Should not crash, should start empty
        store = FileBackedConfigStore(temp_config_file)
        assert len(store) == 0

        # Should be able to write new configs
        store[workflow_id_1] = config_value_1
        assert workflow_id_1 in store

    def test_empty_file_handling(self, temp_config_file):
        """Test graceful handling of empty file."""
        # Create empty file
        temp_config_file.touch()

        store = FileBackedConfigStore(temp_config_file)
        assert len(store) == 0

    def test_invalid_workflow_id_in_file(
        self, temp_config_file, workflow_id_1, config_value_1
    ):
        """Test graceful handling of invalid WorkflowId strings in file."""
        # Write file with invalid WorkflowId format
        with open(temp_config_file, 'w') as f:
            yaml.safe_dump(
                {
                    str(workflow_id_1): config_value_1,  # Valid WorkflowId
                    'invalid_format': {'params': {}},  # Missing slashes
                },
                f,
            )

        store = FileBackedConfigStore(temp_config_file)
        # Should load only the valid entry
        assert len(store) == 1
        assert workflow_id_1 in store

    def test_parent_directory_creation(self):
        """Test that parent directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / 'subdir1' / 'subdir2' / 'config.yaml'

            store = FileBackedConfigStore(nested_path)
            wf_id = WorkflowId(
                instrument='dummy', namespace='reduction', name='test', version=1
            )
            store[wf_id] = {'params': {}}

            assert nested_path.exists()
            assert nested_path.parent.exists()

    def test_iteration(self, temp_config_file, workflow_id_1, workflow_id_2):
        """Test iterating over configs."""
        store = FileBackedConfigStore(temp_config_file)
        store[workflow_id_1] = {'params': {'value': 1}}
        store[workflow_id_2] = {'params': {'value': 2}}

        keys = list(store.keys())
        assert len(keys) == 2
        assert workflow_id_1 in keys
        assert workflow_id_2 in keys

    def test_no_max_configs(self, temp_config_file):
        """Test store without max_configs limit."""
        store = FileBackedConfigStore(temp_config_file, max_configs=None)

        # Add many configs - should not evict
        for i in range(150):
            wf_id = WorkflowId(
                instrument='dummy',
                namespace='reduction',
                name=f'workflow_{i}',
                version=1,
            )
            store[wf_id] = {'params': {'value': i}}

        assert len(store) == 150


class TestConfigStoreManager:
    """Tests for ConfigStoreManager."""

    def test_creates_file_stores_by_default(self):
        """Test that ConfigStoreManager creates file-backed stores by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigStoreManager(instrument='dummy', config_dir=tmpdir)

            workflow_store = manager.get_store('workflow_configs')
            plotter_store = manager.get_store('plotter_configs')

            # Should be FileBackedConfigStore instances
            assert isinstance(workflow_store, FileBackedConfigStore)
            assert isinstance(plotter_store, FileBackedConfigStore)

            # Should create files in the correct location
            assert (Path(tmpdir) / 'workflow_configs.yaml').parent.exists()
            assert (Path(tmpdir) / 'plotter_configs.yaml').parent.exists()

    def test_creates_memory_stores(self):
        """Test that ConfigStoreManager can create in-memory stores."""
        manager = ConfigStoreManager(instrument='dummy', store_type='memory')

        workflow_store = manager.get_store('workflow_configs')
        plotter_store = manager.get_store('plotter_configs')

        # Should be InMemoryConfigStore instances
        assert isinstance(workflow_store, InMemoryConfigStore)
        assert isinstance(plotter_store, InMemoryConfigStore)

    def test_respects_max_configs(self):
        """Test that ConfigStoreManager passes max_configs to stores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigStoreManager(
                instrument='dummy',
                config_dir=tmpdir,
                max_configs=5,
                cleanup_fraction=0.4,
            )

            store = manager.get_store('test_store')

            # Add 6 configs to trigger eviction
            for i in range(6):
                wf_id = WorkflowId(
                    instrument='dummy', namespace='reduction', name=f'wf_{i}', version=1
                )
                store[wf_id] = {'params': {'value': i}}

            # Should have evicted 40% (2 configs), leaving 4
            assert len(store) == 4

    def test_config_dir_property(self):
        """Test that ConfigStoreManager exposes config_dir property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigStoreManager(instrument='dummy', config_dir=tmpdir)
            assert manager.config_dir == Path(tmpdir)

    def test_store_type_property(self):
        """Test that ConfigStoreManager exposes store_type property."""
        manager = ConfigStoreManager(instrument='dummy', store_type='file')
        assert manager.store_type == 'file'

        manager = ConfigStoreManager(instrument='dummy', store_type='memory')
        assert manager.store_type == 'memory'

    def test_resolves_config_dir_from_env(self, monkeypatch):
        """Test that ConfigStoreManager resolves config dir from LIVEDATA_CONFIG_DIR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv('LIVEDATA_CONFIG_DIR', tmpdir)
            manager = ConfigStoreManager(instrument='dummy')

            assert manager.config_dir == Path(tmpdir) / 'dummy'

    def test_resolves_config_dir_from_xdg(self, monkeypatch):
        """Test that ConfigStoreManager uses XDG_CONFIG_HOME as fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.delenv('LIVEDATA_CONFIG_DIR', raising=False)
            monkeypatch.setenv('XDG_CONFIG_HOME', tmpdir)
            manager = ConfigStoreManager(instrument='dummy')

            assert manager.config_dir == Path(tmpdir) / 'esslivedata' / 'dummy'

    def test_file_stores_persist_data(self):
        """Test that file stores created by manager persist data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wf_id = WorkflowId(
                instrument='dummy', namespace='reduction', name='test', version=1
            )
            config_value = {'params': {'value': 42}}

            # Create manager and store data
            manager1 = ConfigStoreManager(instrument='dummy', config_dir=tmpdir)
            store1 = manager1.get_store('workflow_configs')
            store1[wf_id] = config_value

            # Create new manager and verify data persists
            manager2 = ConfigStoreManager(instrument='dummy', config_dir=tmpdir)
            store2 = manager2.get_store('workflow_configs')

            assert wf_id in store2
            assert store2[wf_id] == config_value

    def test_multiple_stores_independent(self):
        """Test that multiple stores created by manager are independent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigStoreManager(instrument='dummy', config_dir=tmpdir)

            workflow_store = manager.get_store('workflow_configs')
            plotter_store = manager.get_store('plotter_configs')

            wf_id1 = WorkflowId(
                instrument='dummy', namespace='reduction', name='wf1', version=1
            )
            wf_id2 = WorkflowId(
                instrument='dummy', namespace='plotting', name='plot1', version=1
            )

            # Add to different stores
            workflow_store[wf_id1] = {'params': {'a': 1}}
            plotter_store[wf_id2] = {'params': {'b': 2}}

            # Should not interfere with each other
            assert wf_id1 in workflow_store
            assert wf_id1 not in plotter_store
            assert wf_id2 in plotter_store
            assert wf_id2 not in workflow_store

    def test_different_instruments_isolated(self):
        """Test that configs for different instruments are stored separately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create managers for different instruments
            dummy_config_dir = Path(tmpdir) / 'dummy'
            dream_config_dir = Path(tmpdir) / 'dream'

            manager_dummy = ConfigStoreManager(
                instrument='dummy', config_dir=dummy_config_dir
            )
            manager_dream = ConfigStoreManager(
                instrument='dream', config_dir=dream_config_dir
            )

            # Get stores from each manager
            dummy_store = manager_dummy.get_store('workflow_configs')
            dream_store = manager_dream.get_store('workflow_configs')

            # Create WorkflowIds for each instrument
            dummy_wf_id = WorkflowId(
                instrument='dummy', namespace='reduction', name='workflow1', version=1
            )
            dream_wf_id = WorkflowId(
                instrument='dream', namespace='reduction', name='workflow1', version=1
            )

            dummy_config = {'params': {'threshold': 100.0}}
            dream_config = {'params': {'threshold': 200.0}}

            # Add configs to respective stores
            dummy_store[dummy_wf_id] = dummy_config
            dream_store[dream_wf_id] = dream_config

            # Verify configs are in separate stores
            assert dummy_wf_id in dummy_store
            assert dummy_wf_id not in dream_store
            assert dream_wf_id in dream_store
            assert dream_wf_id not in dummy_store

            # Verify files are in separate directories
            dummy_config_file = dummy_config_dir / 'workflow_configs.yaml'
            dream_config_file = dream_config_dir / 'workflow_configs.yaml'
            assert dummy_config_file.exists()
            assert dream_config_file.exists()
            assert dummy_config_file != dream_config_file

            # Verify data persists separately when creating new managers
            manager_dummy2 = ConfigStoreManager(
                instrument='dummy', config_dir=dummy_config_dir
            )
            manager_dream2 = ConfigStoreManager(
                instrument='dream', config_dir=dream_config_dir
            )

            dummy_store2 = manager_dummy2.get_store('workflow_configs')
            dream_store2 = manager_dream2.get_store('workflow_configs')

            # Each instrument should only have its own config
            assert dummy_wf_id in dummy_store2
            assert dummy_store2[dummy_wf_id] == dummy_config
            assert dream_wf_id not in dummy_store2

            assert dream_wf_id in dream_store2
            assert dream_store2[dream_wf_id] == dream_config
            assert dummy_wf_id not in dream_store2
