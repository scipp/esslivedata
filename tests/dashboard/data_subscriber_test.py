# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import uuid
from typing import Any

import pytest

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.data_subscriber import (
    DataSubscriber,
    FilteredMergingStreamAssembler,
    MergingStreamAssembler,
    Pipe,
    StreamAssembler,
)


class FakeStreamAssembler(StreamAssembler[str]):
    """Fake implementation of StreamAssembler for testing."""

    def __init__(self, keys: set[str], return_value: Any = None) -> None:
        super().__init__(keys)
        self.return_value = return_value
        self.assemble_calls: list[dict[str, Any]] = []

    def assemble(self, data: dict[str, Any]) -> Any:
        self.assemble_calls.append(data.copy())
        return self.return_value


class FakePipe(Pipe):
    """Fake implementation of Pipe for testing."""

    def __init__(self) -> None:
        self.send_calls: list[Any] = []

    def send(self, data: Any) -> None:
        self.send_calls.append(data)


@pytest.fixture
def sample_keys() -> set[str]:
    """Sample data keys for testing."""
    return {'key1', 'key2', 'key3'}


@pytest.fixture
def fake_assembler(sample_keys: set[str]) -> FakeStreamAssembler:
    """Fake assembler with sample keys."""
    return FakeStreamAssembler(sample_keys, 'assembled_data')


@pytest.fixture
def fake_pipe() -> FakePipe:
    """Fake pipe for testing."""
    return FakePipe()


@pytest.fixture
def subscriber(
    fake_assembler: FakeStreamAssembler, fake_pipe: FakePipe
) -> DataSubscriber[str]:
    """DataSubscriber instance for testing."""
    return DataSubscriber(fake_assembler, fake_pipe)


class TestDataSubscriber:
    """Test cases for DataSubscriber class."""

    def test_init_stores_assembler_and_pipe(
        self, fake_assembler: FakeStreamAssembler, fake_pipe: FakePipe
    ) -> None:
        """Test that initialization stores the assembler and pipe correctly."""
        subscriber = DataSubscriber(fake_assembler, fake_pipe)

        assert subscriber._assembler is fake_assembler
        assert subscriber._pipe is fake_pipe

    def test_keys_returns_assembler_keys(
        self, subscriber: DataSubscriber, sample_keys: set[str]
    ) -> None:
        """Test that keys property returns the assembler's keys."""
        assert subscriber.keys == sample_keys

    def test_trigger_with_complete_data(
        self,
        subscriber: DataSubscriber,
        fake_assembler: FakeStreamAssembler,
        fake_pipe: FakePipe,
    ) -> None:
        """Test trigger method when all required keys are present in store."""
        store = {
            'key1': 'value1',
            'key2': 'value2',
            'key3': 'value3',
            'extra_key': 'extra_value',
        }

        subscriber.trigger(store)

        # Verify assembler was called with the correct subset of data
        assert len(fake_assembler.assemble_calls) == 1
        expected_data = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        assert fake_assembler.assemble_calls[0] == expected_data

        # Verify pipe was called with assembled data
        assert len(fake_pipe.send_calls) == 1
        assert fake_pipe.send_calls[0] == 'assembled_data'

    def test_trigger_with_partial_data(
        self,
        subscriber: DataSubscriber,
        fake_assembler: FakeStreamAssembler,
        fake_pipe: FakePipe,
    ) -> None:
        """Test trigger method when only some required keys are present in store."""
        store = {'key1': 'value1', 'key3': 'value3', 'unrelated_key': 'unrelated_value'}

        subscriber.trigger(store)

        # Verify assembler was called with only available keys
        assert len(fake_assembler.assemble_calls) == 1
        expected_data = {'key1': 'value1', 'key3': 'value3'}
        assert fake_assembler.assemble_calls[0] == expected_data

        # Verify pipe was called
        assert len(fake_pipe.send_calls) == 1
        assert fake_pipe.send_calls[0] == 'assembled_data'

    def test_trigger_with_empty_store(
        self,
        subscriber: DataSubscriber,
        fake_assembler: FakeStreamAssembler,
        fake_pipe: FakePipe,
    ) -> None:
        """Test trigger method with an empty store."""
        store: dict[str, Any] = {}

        subscriber.trigger(store)

        # Verify assembler was called with empty data
        assert len(fake_assembler.assemble_calls) == 1
        assert fake_assembler.assemble_calls[0] == {}

        # Verify pipe was called
        assert len(fake_pipe.send_calls) == 1
        assert fake_pipe.send_calls[0] == 'assembled_data'

    def test_trigger_with_no_matching_keys(
        self,
        subscriber: DataSubscriber,
        fake_assembler: FakeStreamAssembler,
        fake_pipe: FakePipe,
    ) -> None:
        """Test trigger method when store contains no matching keys."""
        store = {'other_key1': 'value1', 'other_key2': 'value2'}

        subscriber.trigger(store)

        # Verify assembler was called with empty data
        assert len(fake_assembler.assemble_calls) == 1
        assert fake_assembler.assemble_calls[0] == {}

        # Verify pipe was called
        assert len(fake_pipe.send_calls) == 1
        assert fake_pipe.send_calls[0] == 'assembled_data'

    def test_trigger_multiple_calls(
        self,
        subscriber: DataSubscriber,
        fake_assembler: FakeStreamAssembler,
        fake_pipe: FakePipe,
    ) -> None:
        """Test multiple calls to trigger method."""
        store1 = {'key1': 'value1', 'key2': 'value2'}
        store2 = {'key2': 'updated_value2', 'key3': 'value3'}

        subscriber.trigger(store1)
        subscriber.trigger(store2)

        # Verify both calls were processed
        assert len(fake_assembler.assemble_calls) == 2
        assert fake_assembler.assemble_calls[0] == {'key1': 'value1', 'key2': 'value2'}
        assert fake_assembler.assemble_calls[1] == {
            'key2': 'updated_value2',
            'key3': 'value3',
        }

        assert len(fake_pipe.send_calls) == 2
        assert all(call == 'assembled_data' for call in fake_pipe.send_calls)

    def test_trigger_with_different_assembled_data(self, sample_keys: set[str]) -> None:
        """Test trigger method with assembler that returns different data types."""
        assembled_values = [42, {'result': 'success'}, [1, 2, 3], None]

        for value in assembled_values:
            assembler = FakeStreamAssembler(sample_keys, value)
            pipe = FakePipe()
            subscriber = DataSubscriber(assembler, pipe)

            store = {'key1': 'test_value'}
            subscriber.trigger(store)

            assert len(pipe.send_calls) == 1
            assert pipe.send_calls[0] == value


class TestMergingStreamAssembler:
    """Test cases for MergingStreamAssembler class."""

    @pytest.fixture
    def workflow_id(self) -> WorkflowId:
        """Sample workflow ID."""
        return WorkflowId(
            instrument='test_instrument',
            namespace='test_namespace',
            name='test_workflow',
            version=1,
        )

    @pytest.fixture
    def result_keys(self, workflow_id: WorkflowId) -> set[ResultKey]:
        """Sample result keys for testing."""
        job_id_1 = JobId(source_name='detector_1', job_number=uuid.uuid4())
        job_id_2 = JobId(source_name='detector_2', job_number=uuid.uuid4())
        return {
            ResultKey(workflow_id=workflow_id, job_id=job_id_1, output_name='current'),
            ResultKey(workflow_id=workflow_id, job_id=job_id_2, output_name='current'),
            ResultKey(
                workflow_id=workflow_id, job_id=job_id_1, output_name='cumulative'
            ),
        }

    def test_assemble_returns_only_matching_keys(
        self, result_keys: set[ResultKey]
    ) -> None:
        """Test that assemble only includes keys present in both keys and data."""
        assembler = MergingStreamAssembler(result_keys)

        # Convert to list for indexing
        keys_list = list(result_keys)
        # Create data with some matching and some non-matching keys
        data = {
            keys_list[0]: 'value1',
            keys_list[1]: 'value2',
            # Non-matching key
            ResultKey(
                workflow_id=WorkflowId(
                    instrument='other', namespace='ns', name='wf', version=1
                ),
                job_id=JobId(source_name='other', job_number=uuid.uuid4()),
                output_name='other',
            ): 'value_other',
        }

        result = assembler.assemble(data)

        assert len(result) == 2
        assert keys_list[0] in result
        assert keys_list[1] in result

    def test_assemble_with_empty_data(self, result_keys: set[ResultKey]) -> None:
        """Test assemble with empty data dictionary."""
        assembler = MergingStreamAssembler(result_keys)
        result = assembler.assemble({})
        assert result == {}

    def test_assemble_with_no_matching_keys(self, result_keys: set[ResultKey]) -> None:
        """Test assemble when data contains no matching keys."""
        assembler = MergingStreamAssembler(result_keys)
        other_key = ResultKey(
            workflow_id=WorkflowId(
                instrument='other', namespace='ns', name='wf', version=1
            ),
            job_id=JobId(source_name='other', job_number=uuid.uuid4()),
            output_name='other',
        )
        data = {other_key: 'value'}

        result = assembler.assemble(data)
        assert result == {}

    def test_assemble_sorts_keys_deterministically(
        self, workflow_id: WorkflowId
    ) -> None:
        """Test that assemble returns keys in deterministic sorted order."""
        # Create keys with specific ordering to verify sorting
        job_id_a = JobId(source_name='a_detector', job_number=uuid.uuid4())
        job_id_b = JobId(source_name='b_detector', job_number=uuid.uuid4())

        key_a_current = ResultKey(
            workflow_id=workflow_id, job_id=job_id_a, output_name='current'
        )
        key_a_cumulative = ResultKey(
            workflow_id=workflow_id, job_id=job_id_a, output_name='cumulative'
        )
        key_b_current = ResultKey(
            workflow_id=workflow_id, job_id=job_id_b, output_name='current'
        )

        keys = {key_b_current, key_a_cumulative, key_a_current}
        assembler = MergingStreamAssembler(keys)

        # Create data in random order
        data = {
            key_b_current: 'value_b_current',
            key_a_current: 'value_a_current',
            key_a_cumulative: 'value_a_cumulative',
        }

        result = assembler.assemble(data)

        # Verify keys are sorted by (workflow_id, job_id, output_name)
        result_keys = list(result.keys())
        assert len(result_keys) == 3
        # Should be sorted: a_detector comes before b_detector
        # Within a_detector: cumulative comes before current (alphabetical)
        assert result_keys[0] == key_a_cumulative
        assert result_keys[1] == key_a_current
        assert result_keys[2] == key_b_current

    def test_assemble_sorting_stable_across_calls(
        self, workflow_id: WorkflowId
    ) -> None:
        """Test that sorting is stable across multiple assemble calls."""
        job_ids = [
            JobId(source_name=f'detector_{i}', job_number=uuid.uuid4())
            for i in range(3)
        ]
        keys = {
            ResultKey(workflow_id=workflow_id, job_id=job_id, output_name='current')
            for job_id in job_ids
        }
        assembler = MergingStreamAssembler(keys)

        data = {key: f'value_{i}' for i, key in enumerate(keys)}

        result1 = assembler.assemble(data)
        result2 = assembler.assemble(data)

        # Verify same order in both calls
        assert list(result1.keys()) == list(result2.keys())


class TestFilteredMergingStreamAssembler:
    """Test cases for FilteredMergingStreamAssembler class."""

    @pytest.fixture
    def workflow_id(self) -> WorkflowId:
        """Sample workflow ID."""
        return WorkflowId(
            instrument='test_instrument',
            namespace='test_namespace',
            name='test_workflow',
            version=1,
        )

    @pytest.fixture
    def result_keys(self, workflow_id: WorkflowId) -> set[ResultKey]:
        """Sample result keys with ROI indices."""
        job_id = JobId(source_name='detector', job_number=uuid.uuid4())
        return {
            ResultKey(
                workflow_id=workflow_id, job_id=job_id, output_name='roi_current_0'
            ),
            ResultKey(
                workflow_id=workflow_id, job_id=job_id, output_name='roi_current_1'
            ),
            ResultKey(
                workflow_id=workflow_id, job_id=job_id, output_name='roi_current_2'
            ),
        }

    def test_filter_includes_matching_keys(self, result_keys: set[ResultKey]) -> None:
        """Test that filter function determines which keys are included."""

        # Filter: only include ROI indices 0 and 1
        def filter_fn(key: ResultKey) -> bool:
            if key.output_name is None:
                return False
            parts = key.output_name.rsplit('_', 1)
            if len(parts) != 2:
                return False
            try:
                idx = int(parts[1])
                return idx in {0, 1}
            except ValueError:
                return False

        assembler = FilteredMergingStreamAssembler(result_keys, filter_fn)

        data = {key: f'value_{key.output_name}' for key in result_keys}
        result = assembler.assemble(data)

        # Only roi_current_0 and roi_current_1 should be included
        assert len(result) == 2
        result_output_names = {key.output_name for key in result.keys()}
        assert result_output_names == {'roi_current_0', 'roi_current_1'}

    def test_filter_excludes_all_keys(self, result_keys: set[ResultKey]) -> None:
        """Test filter that excludes all keys."""

        # Filter that rejects everything
        def filter_fn(key: ResultKey) -> bool:
            return False

        assembler = FilteredMergingStreamAssembler(result_keys, filter_fn)
        data = {key: f'value_{key.output_name}' for key in result_keys}
        result = assembler.assemble(data)

        assert result == {}

    def test_filter_includes_all_keys(self, result_keys: set[ResultKey]) -> None:
        """Test filter that includes all keys."""

        # Filter that accepts everything
        def filter_fn(key: ResultKey) -> bool:
            return True

        assembler = FilteredMergingStreamAssembler(result_keys, filter_fn)
        data = {key: f'value_{key.output_name}' for key in result_keys}
        result = assembler.assemble(data)

        assert len(result) == 3

    def test_filter_with_empty_data(self, result_keys: set[ResultKey]) -> None:
        """Test filter with empty data."""

        def filter_fn(key: ResultKey) -> bool:
            return True

        assembler = FilteredMergingStreamAssembler(result_keys, filter_fn)
        result = assembler.assemble({})

        assert result == {}

    def test_filter_respects_parent_sorting(self, workflow_id: WorkflowId) -> None:
        """Test that filtered results maintain parent class sorting."""
        job_id = JobId(source_name='detector', job_number=uuid.uuid4())
        keys = {
            ResultKey(
                workflow_id=workflow_id, job_id=job_id, output_name='roi_current_2'
            ),
            ResultKey(
                workflow_id=workflow_id, job_id=job_id, output_name='roi_current_0'
            ),
            ResultKey(
                workflow_id=workflow_id, job_id=job_id, output_name='roi_current_1'
            ),
        }

        # Filter to include indices 0 and 2
        def filter_fn(key: ResultKey) -> bool:
            if key.output_name is None:
                return False
            return key.output_name in {'roi_current_0', 'roi_current_2'}

        assembler = FilteredMergingStreamAssembler(keys, filter_fn)
        data = {key: f'value_{key.output_name}' for key in keys}
        result = assembler.assemble(data)

        # Should be sorted alphabetically: roi_current_0, roi_current_2
        result_keys = list(result.keys())
        assert len(result_keys) == 2
        assert result_keys[0].output_name == 'roi_current_0'
        assert result_keys[1].output_name == 'roi_current_2'

    def test_filter_with_non_matching_assembler_keys(
        self, result_keys: set[ResultKey]
    ) -> None:
        """Test filter when data contains keys not in assembler keys."""

        def filter_fn(key: ResultKey) -> bool:
            return True

        assembler = FilteredMergingStreamAssembler(result_keys, filter_fn)

        # Add extra key not in assembler keys
        other_key = ResultKey(
            workflow_id=WorkflowId(
                instrument='other', namespace='ns', name='wf', version=1
            ),
            job_id=JobId(source_name='other', job_number=uuid.uuid4()),
            output_name='other',
        )
        data = {key: f'value_{key.output_name}' for key in result_keys}
        data[other_key] = 'value_other'

        result = assembler.assemble(data)

        # Should only include keys from assembler.keys, not the extra key
        assert len(result) == 3
        assert other_key not in result
