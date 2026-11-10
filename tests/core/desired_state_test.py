# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import uuid

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.desired_state import (
    DesiredJob,
    DesiredStateDecodeError,
    DesiredWorkflow,
    InMemoryDesiredStateLog,
    decode_desired_record,
    encode_desired_record,
    fold_updates,
)
from tests.helpers.hostile_wire import malformed_desired_records

WF_A = WorkflowId(instrument='dummy', name='a', version=1)
WF_B = WorkflowId(instrument='dummy', name='b', version=1)


def desired(workflow_id: WorkflowId, *sources: str) -> DesiredWorkflow:
    return DesiredWorkflow(
        workflow_id=workflow_id,
        job_number=uuid.uuid4(),
        jobs={
            source: DesiredJob(
                params={'edges': [1.0, 2.0], 'mode': {'kind': 'tof'}},
                aux_source_names={'monitor': 'monitor1'},
            )
            for source in sources
        },
    )


class TestCodec:
    def test_round_trip(self) -> None:
        record = desired(WF_A, 'det1', 'det2')
        key, value = encode_desired_record(WF_A, record)
        assert key == b'dummy/a/1'
        assert decode_desired_record(key, value) == (WF_A, record)

    def test_tombstone_round_trip(self) -> None:
        key, value = encode_desired_record(WF_A, None)
        assert value is None
        assert decode_desired_record(key, value) == (WF_A, None)

    def test_encode_rejects_record_for_other_workflow(self) -> None:
        with pytest.raises(ValueError, match='dummy/b/1'):
            encode_desired_record(WF_A, desired(WF_B, 'det'))

    @pytest.mark.parametrize(
        ('key', 'value'),
        malformed_desired_records('dummy').values(),
        ids=malformed_desired_records('dummy').keys(),
    )
    def test_decode_rejects_malformed_record(
        self, key: bytes | None, value: bytes | None
    ) -> None:
        with pytest.raises(DesiredStateDecodeError):
            decode_desired_record(key, value)


class TestFoldUpdates:
    def test_records_are_upserted_in_order(self) -> None:
        first, second, other = desired(WF_A, 'x'), desired(WF_A, 'y'), desired(WF_B)
        table = fold_updates({}, [(WF_A, first), (WF_B, other), (WF_A, second)])
        assert table == {WF_A: second, WF_B: other}

    def test_tombstone_deletes(self) -> None:
        table = fold_updates({}, [(WF_A, desired(WF_A)), (WF_A, None)])
        assert table == {}

    def test_tombstone_for_absent_key_is_ignored(self) -> None:
        record = desired(WF_B)
        assert fold_updates({WF_B: record}, [(WF_A, None)]) == {WF_B: record}

    def test_record_after_tombstone_restores_key(self) -> None:
        record = desired(WF_A)
        table = fold_updates({}, [(WF_A, desired(WF_A)), (WF_A, None), (WF_A, record)])
        assert table == {WF_A: record}

    def test_input_table_is_not_modified(self) -> None:
        table = {WF_A: desired(WF_A)}
        fold_updates(table, [(WF_A, None), (WF_B, desired(WF_B))])
        assert list(table) == [WF_A]


class TestInMemoryDesiredStateLog:
    def test_readers_see_all_records_from_offset_zero(self) -> None:
        log = InMemoryDesiredStateLog()
        early = log.reader()
        first = desired(WF_A, 'det')
        log.write(WF_A, first)
        late = log.reader()

        assert early.bootstrap(timeout_s=1.0) == [(WF_A, first)]
        assert late.bootstrap(timeout_s=1.0) == [(WF_A, first)]

    def test_poll_returns_records_appended_since_last_read(self) -> None:
        log = InMemoryDesiredStateLog()
        log.write(WF_A, desired(WF_A))
        reader_1, reader_2 = log.reader(), log.reader()
        reader_1.bootstrap(timeout_s=1.0)
        reader_2.bootstrap(timeout_s=1.0)
        second = desired(WF_B, 'det')
        log.write(WF_B, second)
        log.write(WF_A, None)

        assert reader_1.poll() == [(WF_B, second), (WF_A, None)]
        assert reader_1.poll() == []
        assert reader_2.poll() == [(WF_B, second), (WF_A, None)]

    def test_poll_on_empty_log_returns_nothing(self) -> None:
        reader = InMemoryDesiredStateLog().reader()
        assert reader.bootstrap(timeout_s=1.0) == []
        assert reader.poll() == []

    def test_write_rejects_record_for_other_workflow(self) -> None:
        log = InMemoryDesiredStateLog()
        with pytest.raises(ValueError, match='dummy/b/1'):
            log.write(WF_A, desired(WF_B))
        assert log.reader().poll() == []
