# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for clear-at-commit: a recommit starts accumulation afresh."""

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import (
    get_output_data,
    wait_for_backend_condition,
    wait_for_job_data,
)

#: Cumulative updates the first generation must span before the recommit. The
#: post-commit observation can span a batch or two, so the bar needs a few more
#: than that to stay clear of per-batch producer noise.
_PRE_COMMIT_UPDATES = 5


@pytest.mark.integration
@pytest.mark.services('monitor')
def test_recommit_clears_accumulated_data(integration_env: IntegrationEnv) -> None:
    """
    Recommitting a running workflow resets its cumulative output.

    The commit flips the generation and clears the workflow's buffers, so the
    cumulative total observed after the recommit must fall clear of the value
    accumulated before it, instead of continuing to grow.
    """
    backend = integration_env.backend
    workflow_id = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)
    source_name = 'monitor1'

    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=[source_name],
        config=MonitorDataParams(),
    )
    wait_for_job_data(backend, workflow_id, job_ids, timeout=30.0)

    def cumulative_total() -> float | None:
        data = get_output_data(backend, workflow_id, source_name, 'cumulative')
        if data is None:
            return None
        return float(data.sum().value)

    first = cumulative_total()
    assert first is not None

    # Let the first generation accumulate over several backend updates. The
    # comparison below is between two accumulation windows, and the post-commit
    # one is not under the test's control: the dashboard exposes the latest
    # ingested value rather than the new generation's first result, and its own
    # background thread ingests while this test samples. A bar only one update
    # above `first` is therefore decided by producer noise and poll timing
    # instead of by the clear under test. Counting updates rather than sleeping
    # a fixed span keeps this as short as the pipeline allows.
    totals = [first]

    def accumulated_enough() -> bool:
        total = cumulative_total()
        if total is not None and total > totals[-1]:
            totals.append(total)
        return len(totals) > _PRE_COMMIT_UPDATES

    wait_for_backend_condition(
        backend, accumulated_enough, timeout=60.0, poll_interval=0.1
    )
    pre_commit_total = cumulative_total()
    assert pre_commit_total is not None

    # Recommit with identical parameters, as the UI does on Start.
    new_job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=[source_name],
        config=MonitorDataParams(),
    )
    assert new_job_ids[0].job_number != job_ids[0].job_number

    # The commit cleared the workflow's buffers and old-generation data fails
    # the ingest filter, so the next readable cumulative output is the new
    # generation's fresh accumulation. Capture it inside the condition, and
    # poll tightly: the total keeps growing, so every missed poll lets the
    # captured value drift further into the new generation.
    observed: list[float] = []

    def fresh_total_observed() -> bool:
        total = cumulative_total()
        if total is None:
            return False
        observed.append(total)
        return True

    wait_for_backend_condition(
        backend, fresh_total_observed, timeout=30.0, poll_interval=0.05
    )
    post_commit_total = observed[0]
    # A fresh accumulation of at most a batch or two, against a bar of
    # _PRE_COMMIT_UPDATES updates: anything near the bar means the buffers kept
    # accumulating across the commit.
    assert post_commit_total < pre_commit_total / 2, (
        f"Cumulative total {post_commit_total} did not drop clear of the "
        f"pre-commit value {pre_commit_total}"
    )
