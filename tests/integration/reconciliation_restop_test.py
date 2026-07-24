# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration tests for stop reconciliation (ADR 0008).

Covers the #1097 checklist item "a job whose stop was commanded before the
restart is re-stopped by reconciliation", split into what the architecture
provides:

- Within a dashboard's lifetime, a stop the backend has not acted on is
  re-issued by reconciliation while the job keeps a fresh observed status
  (``test_unacted_stop_is_reissued_by_reconciliation``).
- Across a dashboard restart, the desired-stopped intent is not persisted:
  the config store records only presence or absence of ``current_job``, and
  an absent record is indistinguishable from a record miss (crash between
  send and persist, store loss), which ADR 0008 resolves toward adoption
  ("record misses degrade, they do not lie"). The restarted dashboard
  therefore adopts the still-running job instead of re-stopping it; the
  guarantee is that the job is visible and stoppable
  (``test_restart_after_lost_stop_adopts_job_instead_of_restopping``).
"""

import signal

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.job import JobState
from ess.livedata.dashboard.config_store import ConfigStoreManager
from ess.livedata.dashboard.job_orchestrator import STOP_REISSUE_INTERVAL_SECONDS
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.integration.backend import DashboardBackend
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import (
    topic_high_watermark,
    wait_for_backend_condition,
    wait_for_job_data,
    wait_for_watermark_stall,
)

COMMANDS_TOPIC = 'dummy_livedata_commands'
DATA_TOPIC = 'dummy_livedata_data'

WORKFLOW_ID = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)


def _job_active(backend: DashboardBackend, job_id) -> bool:
    status = backend.job_service.job_statuses.get(job_id)
    return status is not None and status.state == JobState.active


@pytest.mark.integration
@pytest.mark.services('monitor')
def test_unacted_stop_is_reissued_by_reconciliation(
    integration_env: IntegrationEnv,
) -> None:
    """
    A stop the backend has not acted on is re-issued by reconciliation.

    The backend service is frozen with SIGSTOP, so a commanded stop is
    neither consumed nor acknowledged while the job's last observed status
    stays fresh (heartbeat staleness is 60 s). Desired state (stopped)
    then contradicts observed state (running), and after
    STOP_REISSUE_INTERVAL_SECONDS the background reconciliation re-issues
    the stop — observed as one extra message on the commands topic that no
    user action produced (ADR 0008). Once the service resumes it consumes
    the stops and the job's result stream goes quiet.
    """
    backend = integration_env.backend
    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=WORKFLOW_ID,
        source_names=['monitor1'],
        config=MonitorDataParams(),
    )
    job_id = job_ids[0]
    wait_for_backend_condition(
        backend, lambda: _job_active(backend, job_id), timeout=30.0
    )
    wait_for_job_data(backend, WORKFLOW_ID, job_ids, timeout=30.0)

    service = integration_env.services['monitor_data']
    service.process.send_signal(signal.SIGSTOP)
    try:
        watermark_before_stop = topic_high_watermark(COMMANDS_TOPIC)
        assert backend.job_orchestrator.stop_workflow(WORKFLOW_ID)
        wait_for_backend_condition(
            backend,
            lambda: topic_high_watermark(COMMANDS_TOPIC) == watermark_before_stop + 1,
            timeout=10.0,
        )

        # The re-issue is rate-bounded per job_number, so it appears one
        # reissue interval after the user's stop. The observed status must
        # still be fresh by then: 30 s reissue bound < 60 s staleness.
        wait_for_backend_condition(
            backend,
            lambda: topic_high_watermark(COMMANDS_TOPIC) >= watermark_before_stop + 2,
            timeout=STOP_REISSUE_INTERVAL_SECONDS + 20.0,
            poll_interval=1.0,
        )
        # The re-issue did not resurrect the workflow: desired state is
        # still stopped.
        assert backend.job_orchestrator.get_active_job_number(WORKFLOW_ID) is None
    finally:
        service.process.send_signal(signal.SIGCONT)

    # The resumed service consumes the stop commands and removes the job; a
    # commanded stop publishes no final status, so the observable effect is
    # the result stream going quiet.
    wait_for_watermark_stall(DATA_TOPIC)


@pytest.mark.integration
def test_restart_after_lost_stop_adopts_job_instead_of_restopping(
    monitor_services, tmp_path
) -> None:
    """
    After a restart, a job whose stop was swallowed is adopted, not re-stopped.

    A stop whose send the producer swallowed (the #1046 scenario ADR 0008
    cites) leaves the persisted state without ``current_job`` while the
    backend keeps running the job. After a dashboard restart that state is
    indistinguishable from a record miss, so the running job is adopted from
    heartbeat observation as running-with-unknown-config — reconciliation
    does not re-issue the stop (no command is written at all). The ADR's
    degraded-mode guarantee is that the job is visible and stoppable, which
    the final stop exercises end-to-end.
    """
    with DashboardBackend(
        instrument='dummy', dev=True, config_dir=tmp_path
    ) as backend_a:
        job_ids = backend_a.workflow_controller.start_workflow(
            workflow_id=WORKFLOW_ID,
            source_names=['monitor1'],
            config=MonitorDataParams(),
        )
        job_id = job_ids[0]
        wait_for_backend_condition(
            backend_a, lambda: _job_active(backend_a, job_id), timeout=30.0
        )
        wait_for_job_data(backend_a, WORKFLOW_ID, job_ids, timeout=30.0)
    # Dashboard gone; the service keeps running the job and heartbeating.

    # Reproduce the persisted state a swallowed stop leaves behind:
    # stop_workflow writes the store entry without ``current_job``, but its
    # stop command never reaches Kafka. Editing the store directly yields
    # exactly that state without the durable command a real send would leave
    # in the commands topic (which the service would consume on its own).
    store = ConfigStoreManager(instrument='dummy', config_dir=tmp_path).get_store(
        'workflow_configs'
    )
    entry = store[str(WORKFLOW_ID)]
    assert 'current_job' in entry, "commit should have persisted the job record"
    del entry['current_job']
    store[str(WORKFLOW_ID)] = entry

    commands_watermark = topic_high_watermark(COMMANDS_TOPIC)
    with DashboardBackend(
        instrument='dummy', dev=True, config_dir=tmp_path
    ) as backend_b:
        # Adoption from heartbeat observation: same job_number, data admitted.
        wait_for_backend_condition(
            backend_b,
            lambda: (
                backend_b.job_orchestrator.get_active_job_number(WORKFLOW_ID)
                == job_id.job_number
            ),
            timeout=30.0,
        )
        # Adopted without a record: params provenance is degraded.
        active = backend_b.job_orchestrator.get_active_config(WORKFLOW_ID)
        assert active[job_id.source_name].params == {}
        wait_for_job_data(backend_b, WORKFLOW_ID, job_ids, timeout=30.0)

        # Reconciliation issued nothing: desired and observed state agree
        # once the job is adopted, so no stop (and no start) was written.
        assert topic_high_watermark(COMMANDS_TOPIC) == commands_watermark

        # The adopted job is stoppable — the degraded-mode guarantee.
        assert backend_b.job_orchestrator.stop_workflow(WORKFLOW_ID)
        wait_for_watermark_stall(DATA_TOPIC)
