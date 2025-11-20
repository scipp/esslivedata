# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Shared test fakes for dashboard tests."""

import uuid
from collections.abc import Callable

from ess.livedata.config.workflow_spec import JobNumber, WorkflowId
from ess.livedata.dashboard.plot_orchestrator import SubscriptionId


class FakeJobOrchestrator:
    """
    Fake JobOrchestrator for testing.

    Simulates the JobOrchestrator interface including immediate callback
    invocation when subscribing to already-running workflows.
    """

    def __init__(self):
        self._subscriptions: dict[SubscriptionId, Callable[[JobNumber], None]] = {}
        self._workflow_subscriptions: dict[WorkflowId, set[SubscriptionId]] = {}
        self._current_jobs: dict[WorkflowId, JobNumber] = {}

    def subscribe_to_workflow(
        self, workflow_id: WorkflowId, callback: Callable[[JobNumber], None]
    ) -> SubscriptionId:
        """Subscribe to workflow availability notifications."""
        subscription_id = SubscriptionId(uuid.uuid4())
        self._subscriptions[subscription_id] = callback

        # Track which workflows have subscriptions
        if workflow_id not in self._workflow_subscriptions:
            self._workflow_subscriptions[workflow_id] = set()
        self._workflow_subscriptions[workflow_id].add(subscription_id)

        # If workflow is already running, notify immediately (like real JobOrchestrator)
        if workflow_id in self._current_jobs:
            current_job_number = self._current_jobs[workflow_id]
            callback(current_job_number)

        return subscription_id

    def unsubscribe(self, subscription_id: SubscriptionId) -> None:
        """Unsubscribe from workflow availability notifications."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            # Remove from workflow tracking
            for workflow_subs in self._workflow_subscriptions.values():
                workflow_subs.discard(subscription_id)

    def simulate_workflow_commit(
        self, workflow_id: WorkflowId, job_number: JobNumber
    ) -> None:
        """Simulate a workflow commit by calling all callbacks for that workflow."""
        # Track this as the current job
        self._current_jobs[workflow_id] = job_number

        if workflow_id not in self._workflow_subscriptions:
            return

        for subscription_id in self._workflow_subscriptions[workflow_id]:
            if subscription_id in self._subscriptions:
                self._subscriptions[subscription_id](job_number)

    @property
    def subscription_count(self) -> int:
        """Return the total number of active subscriptions."""
        return len(self._subscriptions)
