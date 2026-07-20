# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import scipp as sc

from ess.livedata.config.workflow_spec import DataKey, JobId, WorkflowId
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.dashboard.active_job_registry import ActiveJobRegistry, Generation
from ess.livedata.dashboard.data_service import DataService, DataServiceSubscriber
from ess.livedata.dashboard.extractors import LatestValueExtractor
from ess.livedata.dashboard.job_service import JobService

_workflow_id = WorkflowId(instrument="test", name="wf", version=1)
_other_workflow_id = WorkflowId(instrument="test", name="other", version=1)


def _make_data_key(workflow_id=_workflow_id, source_name="det1", output_name="result"):
    return DataKey(
        workflow_id=workflow_id, source_name=source_name, output_name=output_name
    )


def _make_registry():
    ds = DataService()
    js = JobService()
    return ActiveJobRegistry(data_service=ds, job_service=js), ds, js


def _report_active(js: JobService, workflow_id: WorkflowId, job_number) -> None:
    js.status_updated(
        JobStatus(
            job_id=JobId(source_name="det1", job_number=job_number),
            workflow_id=workflow_id,
            state=JobState.active,
        )
    )


class TestIsCurrent:
    def test_only_current_generation_of_that_workflow_is_current(self):
        registry, _ds, _js = _make_registry()
        job_number = uuid.uuid4()

        registry.begin_generation(_workflow_id, job_number, config={})

        assert registry.is_current(_workflow_id, job_number)
        assert not registry.is_current(_workflow_id, uuid.uuid4())
        assert not registry.is_current(_other_workflow_id, job_number)

    def test_flip_replaces_current(self):
        registry, _ds, _js = _make_registry()
        old, new = uuid.uuid4(), uuid.uuid4()

        registry.begin_generation(_workflow_id, old, config={})
        registry.begin_generation(_workflow_id, new, config={})

        assert registry.is_current(_workflow_id, new)
        assert not registry.is_current(_workflow_id, old)


class TestBeginGeneration:
    def test_clears_workflow_buffers(self):
        registry, ds, _js = _make_registry()
        key = _make_data_key()
        ds[key] = sc.scalar(1.0)

        registry.begin_generation(_workflow_id, uuid.uuid4(), config={})

        assert key not in ds

    def test_leaves_other_workflow_buffers_untouched(self):
        registry, ds, _js = _make_registry()
        key = _make_data_key()
        other_key = _make_data_key(workflow_id=_other_workflow_id)
        ds[key] = sc.scalar(1.0)
        ds[other_key] = sc.scalar(2.0)

        registry.begin_generation(_workflow_id, uuid.uuid4(), config={})

        assert key not in ds
        assert other_key in ds

    def test_prunes_job_statuses_of_generation_leaving_the_window(self):
        registry, _ds, js = _make_registry()
        first, second, third = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        for job_number in (first, second):
            registry.begin_generation(_workflow_id, job_number, config={})
            _report_active(js, _workflow_id, job_number)

        assert len(js.job_statuses) == 2
        registry.begin_generation(_workflow_id, third, config={})

        # first dropped off the (current, last) window; second retained.
        statuses = list(js.job_statuses)
        assert len(statuses) == 1
        assert statuses[0].job_number == second

    def test_multi_key_clear_does_not_emit_partial_state(self):
        """A subscriber on multiple keys must not receive a partial-state
        notification while the generation clear is in progress.

        Regression for: stopping a workflow used to make multi-source plots
        re-render with only one source, because per-key eviction fired a
        notification in which the not-yet-evicted keys were still present.
        """
        registry, ds, _ = _make_registry()
        key_a = _make_data_key(source_name="det1")
        key_b = _make_data_key(source_name="det2")
        ds[key_a] = sc.scalar(1.0)
        ds[key_b] = sc.scalar(2.0)

        snapshots: list[dict] = []

        class RecordingSubscriber(DataServiceSubscriber):
            @property
            def extractors(self):
                return {
                    key_a: LatestValueExtractor(),
                    key_b: LatestValueExtractor(),
                }

            def on_updated(self, updated_keys: set[DataKey]) -> None:
                snapshots.append(ds.snapshot(self))

        ds.register_subscriber(RecordingSubscriber())

        registry.begin_generation(_workflow_id, uuid.uuid4(), config={})

        # Every pull must show either both keys or neither — never one.
        assert all(set(s) in ({key_a, key_b}, set()) for s in snapshots), snapshots


class TestDeactivate:
    def test_stops_admitting_data(self):
        registry, _ds, _js = _make_registry()
        job_number = uuid.uuid4()
        registry.begin_generation(_workflow_id, job_number, config={})

        registry.deactivate(_workflow_id)

        assert not registry.is_current(_workflow_id, job_number)

    def test_retains_buffered_data(self):
        registry, ds, _js = _make_registry()
        job_number = uuid.uuid4()
        key = _make_data_key()
        registry.begin_generation(_workflow_id, job_number, config={})
        ds[key] = sc.scalar(1.0)

        registry.deactivate(_workflow_id)

        assert key in ds

    def test_unknown_workflow_is_noop(self):
        registry, _ds, _js = _make_registry()
        registry.deactivate(_workflow_id)  # no exception


class TestIsKnownJob:
    def test_current_and_last_are_known(self):
        registry, _ds, _js = _make_registry()
        old, new = uuid.uuid4(), uuid.uuid4()
        registry.begin_generation(_workflow_id, old, config={})
        registry.begin_generation(_workflow_id, new, config={})

        assert registry.is_known_job(new)
        assert registry.is_known_job(old)
        assert not registry.is_known_job(uuid.uuid4())

    def test_generation_leaving_the_window_is_forgotten(self):
        registry, _ds, _js = _make_registry()
        first = uuid.uuid4()
        registry.begin_generation(_workflow_id, first, config={})
        for _ in range(2):
            registry.begin_generation(_workflow_id, uuid.uuid4(), config={})

        assert not registry.is_known_job(first)


class TestResolveConfig:
    def test_resolves_current_and_last_config(self):
        registry, _ds, _js = _make_registry()
        old, new = uuid.uuid4(), uuid.uuid4()
        registry.begin_generation(_workflow_id, old, config={"threshold": 1})
        registry.begin_generation(_workflow_id, new, config={"threshold": 2})

        assert registry.resolve_config(_workflow_id, new) == {"threshold": 2}
        assert registry.resolve_config(_workflow_id, old) == {"threshold": 1}
        assert registry.resolve_config(_workflow_id, uuid.uuid4()) is None
        assert registry.resolve_config(_other_workflow_id, new) is None


class TestRestore:
    def test_restored_current_admits_data(self):
        registry, _ds, _js = _make_registry()
        job_number = uuid.uuid4()

        registry.restore(
            _workflow_id,
            current=Generation(job_number=job_number, config={"a": 1}),
        )

        assert registry.is_current(_workflow_id, job_number)
        assert registry.resolve_config(_workflow_id, job_number) == {"a": 1}

    def test_restored_last_is_known_but_not_current(self):
        registry, _ds, _js = _make_registry()
        job_number = uuid.uuid4()

        registry.restore(
            _workflow_id,
            current=None,
            last=Generation(job_number=job_number, config={}),
        )

        assert not registry.is_current(_workflow_id, job_number)
        assert registry.is_known_job(job_number)


class TestRecordStale:
    def test_counts_without_side_effects(self):
        registry, ds, _js = _make_registry()
        key = _make_data_key()
        current = uuid.uuid4()
        registry.begin_generation(_workflow_id, current, config={})
        ds[key] = sc.scalar(1.0)

        for _ in range(3):
            registry.record_stale(_workflow_id, uuid.uuid4())

        assert registry.is_current(_workflow_id, current)
        assert key in ds
