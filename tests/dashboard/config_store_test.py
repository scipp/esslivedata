# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for FileBackedConfigStore."""

import tempfile
from pathlib import Path

import pytest
import yaml

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.config_store import FileBackedConfigStore


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


def test_empty_store_creation(temp_config_file):
    """Test creating a FileBackedConfigStore with no existing file."""
    store = FileBackedConfigStore(temp_config_file)
    assert len(store) == 0
    # Parent directory is created, but file itself is not created until first write
    assert temp_config_file.parent.exists()
    assert not temp_config_file.exists()


def test_write_and_read(temp_config_file, workflow_id_1, config_value_1):
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


def test_persistence_across_instances(temp_config_file, workflow_id_1, config_value_1):
    """Test that configs persist when creating a new store instance."""
    # Write with first instance
    store1 = FileBackedConfigStore(temp_config_file)
    store1[workflow_id_1] = config_value_1

    # Read with second instance
    store2 = FileBackedConfigStore(temp_config_file)
    assert workflow_id_1 in store2
    assert store2[workflow_id_1] == config_value_1


def test_multiple_configs(
    temp_config_file, workflow_id_1, workflow_id_2, config_value_1, config_value_2
):
    """Test storing multiple configs."""
    store = FileBackedConfigStore(temp_config_file)
    store[workflow_id_1] = config_value_1
    store[workflow_id_2] = config_value_2

    assert len(store) == 2
    assert store[workflow_id_1] == config_value_1
    assert store[workflow_id_2] == config_value_2


def test_update_existing_config(temp_config_file, workflow_id_1, config_value_1):
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


def test_delete_config(temp_config_file, workflow_id_1, config_value_1):
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


def test_lru_eviction(temp_config_file):
    """Test LRU eviction when max_configs is exceeded."""
    store = FileBackedConfigStore(temp_config_file, max_configs=5, cleanup_fraction=0.4)

    # Add 6 configs (exceeds max_configs=5)
    for i in range(6):
        wf_id = WorkflowId(
            instrument='dummy', namespace='reduction', name=f'workflow_{i}', version=1
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
            instrument='dummy', namespace='reduction', name=f'workflow_{i}', version=1
        )
        assert wf_id in store


def test_corrupted_file_handling(temp_config_file, workflow_id_1, config_value_1):
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


def test_empty_file_handling(temp_config_file):
    """Test graceful handling of empty file."""
    # Create empty file
    temp_config_file.touch()

    store = FileBackedConfigStore(temp_config_file)
    assert len(store) == 0


def test_invalid_workflow_id_in_file(temp_config_file, workflow_id_1, config_value_1):
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


def test_parent_directory_creation():
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


def test_iteration(temp_config_file, workflow_id_1, workflow_id_2):
    """Test iterating over configs."""
    store = FileBackedConfigStore(temp_config_file)
    store[workflow_id_1] = {'params': {'value': 1}}
    store[workflow_id_2] = {'params': {'value': 2}}

    keys = list(store.keys())
    assert len(keys) == 2
    assert workflow_id_1 in keys
    assert workflow_id_2 in keys


def test_no_max_configs(temp_config_file):
    """Test store without max_configs limit."""
    store = FileBackedConfigStore(temp_config_file, max_configs=None)

    # Add many configs - should not evict
    for i in range(150):
        wf_id = WorkflowId(
            instrument='dummy', namespace='reduction', name=f'workflow_{i}', version=1
        )
        store[wf_id] = {'params': {'value': i}}

    assert len(store) == 150
