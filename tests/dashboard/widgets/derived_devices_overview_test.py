# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the read-only Derived devices overview."""

import pydantic
import pytest
import scipp as sc

from ess.livedata.config.device_contract import (
    DeviceContract,
    DeviceContractEntry,
)
from ess.livedata.config.workflow_spec import (
    REDUCTION,
    OutputView,
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.dashboard.active_job_registry import ActiveJobRegistry
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_orchestrator import JobOrchestrator
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.widgets.derived_devices_overview import (
    DerivedDevicesOverview,
)
from ess.livedata.fakes import FakeMessageSink


class Params(pydantic.BaseModel):
    threshold: float = 1.0


class MonitorOutputs(WorkflowOutputsBase):
    counts_total_cumulative: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0.0, unit='counts')),
        title='Total',
    )

    output_views = (
        OutputView(
            name='total',
            title='Total',
            streams={'since_start': 'counts_total_cumulative'},
        ),
    )


@pytest.fixture
def workflow_id() -> WorkflowId:
    return WorkflowId(instrument='dummy', name='monitor', version=1)


@pytest.fixture
def registry(workflow_id):
    spec = WorkflowSpec(
        instrument=workflow_id.instrument,
        name=workflow_id.name,
        version=workflow_id.version,
        title='Monitor',
        description='',
        source_names=['monitor1', 'monitor2'],
        params=Params,
        outputs=MonitorOutputs,
        group=REDUCTION,
    )
    return {workflow_id: spec}


@pytest.fixture
def contract(workflow_id, registry) -> DeviceContract:
    return DeviceContract.from_entries(
        [
            DeviceContractEntry(
                workflow_id=str(workflow_id),
                source_name='monitor1',
                output_name='counts_total_cumulative',
                device_name='mon1_total',
            )
        ],
        registry,
    )


@pytest.fixture
def orchestrator(registry) -> JobOrchestrator:
    js = JobService()
    orch = JobOrchestrator(
        command_service=CommandService(sink=FakeMessageSink()),
        workflow_registry=registry,
        active_job_registry=ActiveJobRegistry(
            data_service=DataService(), job_service=js
        ),
        config_store=None,
    )
    js.on_status_updated = orch.on_job_status_updated
    return orch


def _body_html(overview: DerivedDevicesOverview) -> str:
    import panel as pn

    parts: list[str] = []
    stack = list(overview._body.objects)
    while stack:
        node = stack.pop()
        if isinstance(node, pn.pane.HTML) and isinstance(node.object, str):
            parts.append(node.object)
        if hasattr(node, 'objects'):
            stack.extend(node.objects)
    return ''.join(parts)


def test_empty_when_nothing_running(orchestrator, contract):
    overview = DerivedDevicesOverview(
        orchestrator=orchestrator, device_contract=contract
    )
    html = _body_html(overview)
    assert 'No devices are currently exposed' in html
    assert 'mon1_total' not in html


def test_lists_running_device(orchestrator, contract, workflow_id):
    orchestrator.clear_staged_configs(workflow_id)
    orchestrator.stage_config(
        workflow_id, source_name='monitor1', params={}, aux_source_names={}
    )
    orchestrator.commit_workflow(workflow_id)

    overview = DerivedDevicesOverview(
        orchestrator=orchestrator, device_contract=contract
    )
    html = _body_html(overview)
    assert 'mon1_total' in html
    assert 'Monitor' in html  # workflow title
    assert 'running' in html


def test_refresh_rebuilds_on_state_change(orchestrator, contract, workflow_id):
    overview = DerivedDevicesOverview(
        orchestrator=orchestrator, device_contract=contract
    )
    assert 'mon1_total' not in _body_html(overview)

    orchestrator.clear_staged_configs(workflow_id)
    orchestrator.stage_config(
        workflow_id, source_name='monitor1', params={}, aux_source_names={}
    )
    orchestrator.commit_workflow(workflow_id)

    overview._refresh()
    assert 'mon1_total' in _body_html(overview)
