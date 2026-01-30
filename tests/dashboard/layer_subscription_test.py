# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for LayerSubscription class."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from ess.livedata.config.workflow_spec import JobNumber, WorkflowId
from ess.livedata.dashboard.data_roles import PRIMARY, X_AXIS
from ess.livedata.dashboard.job_orchestrator import WorkflowCallbacks
from ess.livedata.dashboard.layer_subscription import (
    LayerSubscription,
    SubscriptionReady,
)
from ess.livedata.dashboard.plot_orchestrator import DataSourceConfig

if TYPE_CHECKING:
    pass


# Type aliases for clarity
SubscriptionId = UUID


class FakeJobOrchestrator:
    """Fake JobOrchestrator for testing LayerSubscription."""

    def __init__(self):
        self._subscriptions: dict[SubscriptionId, WorkflowCallbacks] = {}
        self._workflow_subscriptions: dict[WorkflowId, set[SubscriptionId]] = {}
        self._running_jobs: dict[WorkflowId, JobNumber] = {}
        self._next_sub_id = 0

    def add_running_job(self, workflow_id: WorkflowId, job_number: JobNumber) -> None:
        """Mark a workflow as running with the given job number."""
        self._running_jobs[workflow_id] = job_number

    def subscribe_to_workflow(
        self,
        workflow_id: WorkflowId,
        callbacks: WorkflowCallbacks,
    ) -> tuple[SubscriptionId, bool]:
        """Subscribe to workflow lifecycle events."""
        sub_id = SubscriptionId(int=self._next_sub_id)
        self._next_sub_id += 1
        self._subscriptions[sub_id] = callbacks

        # Track which workflow this subscription is for
        if workflow_id not in self._workflow_subscriptions:
            self._workflow_subscriptions[workflow_id] = set()
        self._workflow_subscriptions[workflow_id].add(sub_id)

        # If workflow is already running, invoke on_started immediately
        if workflow_id in self._running_jobs:
            job_number = self._running_jobs[workflow_id]
            if callbacks.on_started:
                callbacks.on_started(job_number)
            return sub_id, True

        return sub_id, False

    def unsubscribe(self, subscription_id: SubscriptionId) -> None:
        """Unsubscribe from workflow notifications."""
        self._subscriptions.pop(subscription_id, None)
        for workflow_subs in self._workflow_subscriptions.values():
            workflow_subs.discard(subscription_id)

    def simulate_workflow_started(
        self, workflow_id: WorkflowId, job_number: JobNumber
    ) -> None:
        """Simulate workflow starting (notifies only subscribers for workflow)."""
        self._running_jobs[workflow_id] = job_number
        for sub_id in self._workflow_subscriptions.get(workflow_id, set()):
            if sub_id in self._subscriptions:
                callbacks = self._subscriptions[sub_id]
                if callbacks.on_started:
                    callbacks.on_started(job_number)

    def simulate_workflow_stopped(
        self, workflow_id: WorkflowId, job_number: JobNumber
    ) -> None:
        """Simulate workflow stopping (notifies only subscribers for workflow)."""
        self._running_jobs.pop(workflow_id, None)
        for sub_id in self._workflow_subscriptions.get(workflow_id, set()):
            if sub_id in self._subscriptions:
                callbacks = self._subscriptions[sub_id]
                if callbacks.on_stopped:
                    callbacks.on_stopped(job_number)


@pytest.fixture
def workflow_id() -> WorkflowId:
    """Sample workflow ID."""
    return WorkflowId(
        instrument='test_instrument',
        namespace='test_namespace',
        name='test_workflow',
        version=1,
    )


@pytest.fixture
def workflow_id_2() -> WorkflowId:
    """Second sample workflow ID."""
    return WorkflowId(
        instrument='test_instrument',
        namespace='test_namespace',
        name='test_workflow_2',
        version=1,
    )


@pytest.fixture
def job_number() -> JobNumber:
    """Sample job number."""
    return uuid.uuid4()


@pytest.fixture
def job_number_2() -> JobNumber:
    """Second sample job number."""
    return uuid.uuid4()


@pytest.fixture
def fake_job_orchestrator() -> FakeJobOrchestrator:
    """Create a fake job orchestrator."""
    return FakeJobOrchestrator()


class TestLayerSubscriptionSingleSource:
    """Tests for LayerSubscription with a single data source."""

    def test_on_ready_fires_when_workflow_already_running(
        self, workflow_id, job_number, fake_job_orchestrator
    ):
        """on_ready should fire immediately if workflow is already running."""
        # Setup: workflow is already running
        fake_job_orchestrator.add_running_job(workflow_id, job_number)

        ready_invocations = []

        def on_ready(ready: SubscriptionReady) -> None:
            ready_invocations.append(ready)

        data_source = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['detector1', 'detector2'],
            output_name='result',
        )

        subscription = LayerSubscription(
            data_sources={PRIMARY: data_source},
            job_orchestrator=fake_job_orchestrator,
            on_ready=on_ready,
        )
        subscription.start()

        # on_ready should fire immediately
        assert len(ready_invocations) == 1
        ready = ready_invocations[0]
        all_keys = [k for keys in ready.keys_by_role.values() for k in keys]
        assert len(all_keys) == 2
        assert all(key.job_id.job_number == job_number for key in all_keys)

    def test_on_ready_fires_when_workflow_starts_later(
        self, workflow_id, job_number, fake_job_orchestrator
    ):
        """on_ready should fire when workflow starts after subscription."""
        ready_invocations = []

        def on_ready(ready: SubscriptionReady) -> None:
            ready_invocations.append(ready)

        data_source = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['detector1'],
            output_name='result',
        )

        subscription = LayerSubscription(
            data_sources={PRIMARY: data_source},
            job_orchestrator=fake_job_orchestrator,
            on_ready=on_ready,
        )
        subscription.start()

        # on_ready should not fire yet
        assert len(ready_invocations) == 0

        # Simulate workflow starting
        fake_job_orchestrator.simulate_workflow_started(workflow_id, job_number)

        # Now on_ready should fire
        assert len(ready_invocations) == 1

    def test_result_keys_built_correctly(
        self, workflow_id, job_number, fake_job_orchestrator
    ):
        """Result keys should be built correctly from data source config."""
        fake_job_orchestrator.add_running_job(workflow_id, job_number)

        ready_invocations = []

        def on_ready(ready: SubscriptionReady) -> None:
            ready_invocations.append(ready)

        data_source = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['detector1', 'detector2'],
            output_name='result',
        )

        subscription = LayerSubscription(
            data_sources={PRIMARY: data_source},
            job_orchestrator=fake_job_orchestrator,
            on_ready=on_ready,
        )
        subscription.start()

        ready = ready_invocations[0]
        all_keys = [k for keys in ready.keys_by_role.values() for k in keys]
        assert len(all_keys) == 2

        # Verify keys are correct
        key_source_names = {key.job_id.source_name for key in all_keys}
        assert key_source_names == {'detector1', 'detector2'}

        for key in all_keys:
            assert key.workflow_id == workflow_id
            assert key.job_id.job_number == job_number
            assert key.output_name == 'result'

    def test_keys_by_role_contains_primary_keys(
        self, workflow_id, job_number, fake_job_orchestrator
    ):
        """keys_by_role should contain keys organized by role."""
        fake_job_orchestrator.add_running_job(workflow_id, job_number)

        ready_invocations = []

        def on_ready(ready: SubscriptionReady) -> None:
            ready_invocations.append(ready)

        data_source = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['detector1', 'detector2'],
            output_name='result',
        )

        subscription = LayerSubscription(
            data_sources={PRIMARY: data_source},
            job_orchestrator=fake_job_orchestrator,
            on_ready=on_ready,
        )
        subscription.start()

        ready = ready_invocations[0]

        # Verify keys_by_role has the PRIMARY role
        assert PRIMARY in ready.keys_by_role
        primary_keys = ready.keys_by_role[PRIMARY]
        assert len(primary_keys) == 2

        # Verify keys are correct
        key_source_names = {key.job_id.source_name for key in primary_keys}
        assert key_source_names == {'detector1', 'detector2'}

        for key in primary_keys:
            assert key.workflow_id == workflow_id
            assert key.job_id.job_number == job_number
            assert key.output_name == 'result'

    def test_on_stopped_propagates(
        self, workflow_id, job_number, fake_job_orchestrator
    ):
        """on_stopped should be called when workflow stops."""
        fake_job_orchestrator.add_running_job(workflow_id, job_number)

        stopped_invocations = []

        def on_stopped(job_num: JobNumber) -> None:
            stopped_invocations.append(job_num)

        data_source = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['detector1'],
            output_name='result',
        )

        subscription = LayerSubscription(
            data_sources={PRIMARY: data_source},
            job_orchestrator=fake_job_orchestrator,
            on_ready=lambda r: None,
            on_stopped=on_stopped,
        )
        subscription.start()

        # Simulate workflow stopping
        fake_job_orchestrator.simulate_workflow_stopped(workflow_id, job_number)

        assert len(stopped_invocations) == 1
        assert stopped_invocations[0] == job_number

    def test_unsubscribe_removes_subscriptions(
        self, workflow_id, job_number, fake_job_orchestrator
    ):
        """unsubscribe should remove all workflow subscriptions."""
        data_source = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['detector1'],
            output_name='result',
        )

        subscription = LayerSubscription(
            data_sources={PRIMARY: data_source},
            job_orchestrator=fake_job_orchestrator,
            on_ready=lambda r: None,
        )
        subscription.start()

        # Should have one subscription
        assert len(fake_job_orchestrator._subscriptions) == 1

        subscription.unsubscribe()

        # Should have no subscriptions
        assert len(fake_job_orchestrator._subscriptions) == 0


class TestLayerSubscriptionMultiSource:
    """Tests for LayerSubscription with multiple data sources."""

    def test_on_ready_waits_for_all_workflows(
        self,
        workflow_id,
        workflow_id_2,
        job_number,
        job_number_2,
        fake_job_orchestrator,
    ):
        """on_ready should wait for all workflows to be ready."""
        # Only first workflow is running
        fake_job_orchestrator.add_running_job(workflow_id, job_number)

        ready_invocations = []

        def on_ready(ready: SubscriptionReady) -> None:
            ready_invocations.append(ready)

        data_source_1 = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['detector1'],
            output_name='data_result',
        )
        data_source_2 = DataSourceConfig(
            workflow_id=workflow_id_2,
            source_names=['temperature'],
            output_name='axis_result',
        )

        subscription = LayerSubscription(
            data_sources={PRIMARY: data_source_1, X_AXIS: data_source_2},
            job_orchestrator=fake_job_orchestrator,
            on_ready=on_ready,
        )
        subscription.start()

        # on_ready should NOT fire yet (waiting for workflow_id_2)
        assert len(ready_invocations) == 0

        # Simulate second workflow starting
        fake_job_orchestrator.simulate_workflow_started(workflow_id_2, job_number_2)

        # Now on_ready should fire
        assert len(ready_invocations) == 1

    def test_result_keys_from_all_data_sources(
        self,
        workflow_id,
        workflow_id_2,
        job_number,
        job_number_2,
        fake_job_orchestrator,
    ):
        """Result keys should include keys from all data sources."""
        fake_job_orchestrator.add_running_job(workflow_id, job_number)
        fake_job_orchestrator.add_running_job(workflow_id_2, job_number_2)

        ready_invocations = []

        def on_ready(ready: SubscriptionReady) -> None:
            ready_invocations.append(ready)

        data_source_1 = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['det1', 'det2'],
            output_name='data',
        )
        data_source_2 = DataSourceConfig(
            workflow_id=workflow_id_2,
            source_names=['temp'],
            output_name='axis',
        )

        subscription = LayerSubscription(
            data_sources={PRIMARY: data_source_1, X_AXIS: data_source_2},
            job_orchestrator=fake_job_orchestrator,
            on_ready=on_ready,
        )
        subscription.start()

        ready = ready_invocations[0]
        all_keys = [k for keys in ready.keys_by_role.values() for k in keys]
        assert len(all_keys) == 3  # 2 from det sources + 1 from temp

        # Verify keys are from both workflows
        workflow_ids = {key.workflow_id for key in all_keys}
        assert workflow_ids == {workflow_id, workflow_id_2}

    def test_keys_by_role_contains_keys_for_each_role(
        self,
        workflow_id,
        workflow_id_2,
        job_number,
        job_number_2,
        fake_job_orchestrator,
    ):
        """keys_by_role should contain keys organized by role."""
        fake_job_orchestrator.add_running_job(workflow_id, job_number)
        fake_job_orchestrator.add_running_job(workflow_id_2, job_number_2)

        ready_invocations = []

        def on_ready(ready: SubscriptionReady) -> None:
            ready_invocations.append(ready)

        data_source_1 = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['det1', 'det2'],
            output_name='data',
        )
        data_source_2 = DataSourceConfig(
            workflow_id=workflow_id_2,
            source_names=['temp'],
            output_name='axis',
        )

        subscription = LayerSubscription(
            data_sources={PRIMARY: data_source_1, X_AXIS: data_source_2},
            job_orchestrator=fake_job_orchestrator,
            on_ready=on_ready,
        )
        subscription.start()

        ready = ready_invocations[0]

        # Verify structure of keys_by_role
        assert PRIMARY in ready.keys_by_role
        assert X_AXIS in ready.keys_by_role

        # Check primary keys
        primary_keys = ready.keys_by_role[PRIMARY]
        assert len(primary_keys) == 2
        primary_source_names = {key.job_id.source_name for key in primary_keys}
        assert primary_source_names == {'det1', 'det2'}

        # Check x_axis keys
        x_axis_keys = ready.keys_by_role[X_AXIS]
        assert len(x_axis_keys) == 1
        assert x_axis_keys[0].job_id.source_name == 'temp'
        assert x_axis_keys[0].workflow_id == workflow_id_2

    def test_on_stopped_fires_for_any_workflow(
        self,
        workflow_id,
        workflow_id_2,
        job_number,
        job_number_2,
        fake_job_orchestrator,
    ):
        """on_stopped should fire when ANY subscribed workflow stops."""
        fake_job_orchestrator.add_running_job(workflow_id, job_number)
        fake_job_orchestrator.add_running_job(workflow_id_2, job_number_2)

        stopped_invocations = []

        def on_stopped(job_num: JobNumber) -> None:
            stopped_invocations.append(job_num)

        data_source_1 = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['det1'],
            output_name='data',
        )
        data_source_2 = DataSourceConfig(
            workflow_id=workflow_id_2,
            source_names=['temp'],
            output_name='axis',
        )

        subscription = LayerSubscription(
            data_sources={PRIMARY: data_source_1, X_AXIS: data_source_2},
            job_orchestrator=fake_job_orchestrator,
            on_ready=lambda r: None,
            on_stopped=on_stopped,
        )
        subscription.start()

        # Stop first workflow
        fake_job_orchestrator.simulate_workflow_stopped(workflow_id, job_number)

        # on_stopped should fire once for this workflow
        assert len(stopped_invocations) == 1
        assert stopped_invocations[0] == job_number

    def test_on_ready_fires_again_when_job_replaced(
        self,
        workflow_id,
        workflow_id_2,
        job_number,
        job_number_2,
        fake_job_orchestrator,
    ):
        """on_ready should fire again when a workflow restarts with a new job_number."""
        ready_invocations = []

        def on_ready(ready: SubscriptionReady) -> None:
            ready_invocations.append(ready)

        data_source_1 = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['det1'],
            output_name='data',
        )
        data_source_2 = DataSourceConfig(
            workflow_id=workflow_id_2,
            source_names=['temp'],
            output_name='axis',
        )

        subscription = LayerSubscription(
            data_sources={PRIMARY: data_source_1, X_AXIS: data_source_2},
            job_orchestrator=fake_job_orchestrator,
            on_ready=on_ready,
        )
        subscription.start()

        # Start both workflows
        fake_job_orchestrator.simulate_workflow_started(workflow_id, job_number)
        fake_job_orchestrator.simulate_workflow_started(workflow_id_2, job_number_2)

        assert len(ready_invocations) == 1

        # Simulate job replacement with NEW job_number
        new_job_number = uuid.uuid4()
        fake_job_orchestrator.simulate_workflow_started(workflow_id, new_job_number)

        # on_ready should fire again with new keys
        assert len(ready_invocations) == 2
        # Verify the new ready has the new job_number
        new_ready = ready_invocations[1]
        all_keys = [k for keys in new_ready.keys_by_role.values() for k in keys]
        det_key = next(k for k in all_keys if k.job_id.source_name == 'det1')
        assert det_key.job_id.job_number == new_job_number

    def test_on_ready_fires_again_after_stop_and_restart(
        self,
        workflow_id,
        workflow_id_2,
        job_number,
        job_number_2,
        fake_job_orchestrator,
    ):
        """on_ready should fire again after a workflow stops and restarts."""
        ready_invocations = []

        def on_ready(ready: SubscriptionReady) -> None:
            ready_invocations.append(ready)

        data_source_1 = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['det1'],
            output_name='data',
        )
        data_source_2 = DataSourceConfig(
            workflow_id=workflow_id_2,
            source_names=['temp'],
            output_name='axis',
        )

        subscription = LayerSubscription(
            data_sources={PRIMARY: data_source_1, X_AXIS: data_source_2},
            job_orchestrator=fake_job_orchestrator,
            on_ready=on_ready,
            on_stopped=lambda _: None,
        )
        subscription.start()

        # Start both workflows
        fake_job_orchestrator.simulate_workflow_started(workflow_id, job_number)
        fake_job_orchestrator.simulate_workflow_started(workflow_id_2, job_number_2)

        assert len(ready_invocations) == 1

        # Stop workflow_id
        fake_job_orchestrator.simulate_workflow_stopped(workflow_id, job_number)

        # Restart with new job_number
        new_job_number = uuid.uuid4()
        fake_job_orchestrator.simulate_workflow_started(workflow_id, new_job_number)

        # on_ready should fire again
        assert len(ready_invocations) == 2


class TestLayerSubscriptionDuplicateWorkflows:
    """Tests for LayerSubscription when multiple data sources share workflow_id."""

    def test_handles_duplicate_workflow_id(
        self, workflow_id, job_number, fake_job_orchestrator
    ):
        """Should handle multiple data sources with same workflow_id."""
        fake_job_orchestrator.add_running_job(workflow_id, job_number)

        ready_invocations = []

        def on_ready(ready: SubscriptionReady) -> None:
            ready_invocations.append(ready)

        # Both data sources use the same workflow_id
        data_source_1 = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['det1'],
            output_name='data',
        )
        data_source_2 = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['det2'],
            output_name='other_data',
        )

        subscription = LayerSubscription(
            data_sources={PRIMARY: data_source_1, X_AXIS: data_source_2},
            job_orchestrator=fake_job_orchestrator,
            on_ready=on_ready,
        )
        subscription.start()

        # on_ready should fire (both use same workflow which is running)
        assert len(ready_invocations) == 1

        ready = ready_invocations[0]
        all_keys = [k for keys in ready.keys_by_role.values() for k in keys]
        assert len(all_keys) == 2

        # Verify keys have different source_names
        source_names = {key.job_id.source_name for key in all_keys}
        assert source_names == {'det1', 'det2'}

        # Both keys should have same job_number
        assert all(key.job_id.job_number == job_number for key in all_keys)
