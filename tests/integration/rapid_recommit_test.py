# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test: rapid recommits admit only the latest generation's data."""

from collections.abc import Iterable, Mapping

import pytest

from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.config.workflow_spec import DataKey, JobId, JobNumber, WorkflowId
from ess.livedata.dashboard.data_service import DataServiceSubscriber
from ess.livedata.dashboard.extractors import LatestValueExtractor, UpdateExtractor
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import (
    topic_high_watermark,
    wait_for_backend_condition,
    wait_for_condition,
    wait_for_job_data,
)

#: Recommits issued back-to-back, with no backend update in between, so several
#: generations are in flight at once while the producer keeps running.
_BURST_RECOMMITS = 3

#: Backend updates to keep watching after the final generation's data arrived,
#: to catch a late-arriving stale message being admitted.
_CONFIRM_POLLS = 25


class _StampReader(DataServiceSubscriber[DataKey]):
    """Throwaway view reading the generation stamps of a set of keys.

    ``snapshot_with_stamps`` is the only public read of the provenance stamps
    recorded at ingest, and it is subscriber-shaped. Registration is not needed
    for a read: extraction of the latest value neither requires history
    retention nor notifications.
    """

    def __init__(self, keys: Iterable[DataKey]) -> None:
        self._extractors = {key: LatestValueExtractor() for key in keys}
        super().__init__()

    @property
    def extractors(self) -> Mapping[DataKey, UpdateExtractor]:
        return self._extractors

    def on_updated(self, updated_keys: set[DataKey]) -> None:
        pass


@pytest.mark.integration
@pytest.mark.services('monitor')
def test_rapid_recommit_admits_only_latest_generation(
    integration_env: IntegrationEnv,
) -> None:
    """
    Recommitting repeatedly under a live producer admits only the last commit.

    Each commit flips the dashboard's generation before its commands even reach
    the backend, so results computed for the superseded generations keep
    arriving afterwards. They must be dropped and counted, and the data that
    does reach DataService must carry the final generation's job_number only.
    """
    backend = integration_env.backend
    registry = backend.job_orchestrator.active_job_registry
    data_topic = get_stream_mapping(
        instrument=integration_env.instrument, dev=True
    ).topics.livedata_data
    workflow_id = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)
    source_name = 'monitor1'

    def commit() -> JobId:
        job_ids = backend.workflow_controller.start_workflow(
            workflow_id=workflow_id,
            source_names=[source_name],
            config=MonitorDataParams(),
        )
        return job_ids[0]

    def stamps() -> dict[DataKey, JobNumber]:
        keys = [key for key in backend.data_service if key.workflow_id == workflow_id]
        _data, stamps = backend.data_service.snapshot_with_stamps(_StampReader(keys))
        return stamps

    first_job = commit()
    wait_for_job_data(backend, workflow_id, [first_job], timeout=30.0)

    # Burst of recommits, each superseding a generation the backend may not
    # even have started yet. No backend.update() in between: the messages the
    # producer emits meanwhile stay unforwarded, so they meet the generation
    # filter only after the burst.
    job_numbers = {first_job.job_number}
    for _ in range(_BURST_RECOMMITS):
        job_numbers.add(commit().job_number)

    # Wait for the producer to publish at least once more, reading the broker's
    # watermark rather than consuming. This makes the assertion on stale drops
    # deterministic instead of a race: the observed message predates the flip
    # below, so whichever generation stamped it is superseded by the time the
    # dashboard forwards it.
    watermark = topic_high_watermark(data_topic)
    wait_for_condition(
        lambda: topic_high_watermark(data_topic) > watermark,
        timeout=60.0,
        poll_interval=1.0,
    )

    final_job = commit()
    job_numbers.add(final_job.job_number)
    assert len(job_numbers) == _BURST_RECOMMITS + 2, "commits reused a job_number"

    # The final commit cleared the buffers, so data reappears only once the
    # backend runs the final generation. Stale drops are counted against the
    # same generation record and are not reset again: no further commit
    # follows, and a heartbeat of a superseded generation is recognized as such
    # instead of being adopted.
    def settled() -> bool:
        return bool(stamps()) and registry.stale_count(workflow_id) > 0

    wait_for_backend_condition(backend, settled, timeout=90.0, poll_interval=0.1)

    polls: list[dict[DataKey, JobNumber]] = []

    def watched_enough() -> bool:
        polls.append(stamps())
        return len(polls) >= _CONFIRM_POLLS

    wait_for_backend_condition(backend, watched_enough, timeout=60.0, poll_interval=0.2)

    admitted = {stamp for poll in polls for stamp in poll.values()}
    assert admitted == {final_job.job_number}, (
        f"data from superseded generations reached DataService: "
        f"{admitted - {final_job.job_number}}"
    )
    assert registry.stale_count(workflow_id) > 0, (
        "no stale-generation drop was counted, although data published before "
        "the final commit was still unforwarded when it happened"
    )
