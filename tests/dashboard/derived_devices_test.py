# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the dashboard-side derived-device derivation helper.

Pure logic over a real :class:`DeviceContract` and the live running-sources
snapshot from a real :class:`JobOrchestrator`.
"""

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
from ess.livedata.dashboard.derived_devices import (
    ContractDevice,
    affected_device_names,
    all_devices,
    declared_device_names,
    exposed_devices,
    view_device_names,
)
from ess.livedata.dashboard.job_orchestrator import JobOrchestrator
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.fakes import FakeMessageSink


class Params(pydantic.BaseModel):
    threshold: float = 1.0


def _scalar_cumulative() -> sc.DataArray:
    return sc.DataArray(sc.scalar(0.0, unit='counts'))


def _histogram() -> sc.DataArray:
    return sc.DataArray(
        sc.zeros(dims=['x'], shape=[0], unit='counts'),
        coords={'x': sc.arange('x', 0, unit='m')},
    )


class MonitorOutputs(WorkflowOutputsBase):
    """Outputs with a scalar-cumulative device output and a non-device view."""

    counts_total_cumulative: sc.DataArray = pydantic.Field(
        default_factory=_scalar_cumulative, title='Total'
    )
    histogram: sc.DataArray = pydantic.Field(
        default_factory=_histogram, title='Histogram'
    )

    output_views = (
        OutputView(
            name='total',
            title='Total',
            streams={'since_start': 'counts_total_cumulative'},
        ),
        OutputView(
            name='histogram',
            title='Histogram',
            streams={'since_start': 'histogram'},
        ),
    )


@pytest.fixture
def workflow_id() -> WorkflowId:
    return WorkflowId(instrument='dummy', name='monitor', version=1)


@pytest.fixture
def spec(workflow_id: WorkflowId) -> WorkflowSpec:
    return WorkflowSpec(
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


@pytest.fixture
def registry(workflow_id, spec):
    return {workflow_id: spec}


@pytest.fixture
def contract(workflow_id) -> DeviceContract:
    """Contract exposing monitor1's cumulative total as a device."""
    return DeviceContract(
        [
            DeviceContractEntry(
                workflow_id=str(workflow_id),
                source_name='monitor1',
                output_name='counts_total_cumulative',
                device_name='mon1_total',
            ),
            DeviceContractEntry(
                workflow_id=str(workflow_id),
                source_name='monitor2',
                output_name='counts_total_cumulative',
                device_name='mon2_total',
            ),
        ]
    )


class TestExposedDevices:
    def test_no_running_jobs_exposes_nothing(self, contract):
        assert exposed_devices({}, contract) == []

    def test_running_source_exposes_its_device(self, workflow_id, contract):
        running = {workflow_id: {'monitor1'}}
        assert exposed_devices(running, contract) == [
            ContractDevice(
                device_name='mon1_total',
                workflow_id=workflow_id,
                source_name='monitor1',
                output_name='counts_total_cumulative',
            )
        ]

    def test_intersection_only(self, workflow_id, contract):
        # monitor2 running but only contract entry for monitor1 + monitor2;
        # a non-contract source running exposes nothing extra.
        running = {workflow_id: {'monitor2', 'monitor_other'}}
        names = [d.device_name for d in exposed_devices(running, contract)]
        assert names == ['mon2_total']

    def test_both_running(self, workflow_id, contract):
        running = {workflow_id: {'monitor1', 'monitor2'}}
        names = [d.device_name for d in exposed_devices(running, contract)]
        assert names == ['mon1_total', 'mon2_total']

    def test_empty_contract(self, workflow_id):
        running = {workflow_id: {'monitor1'}}
        assert exposed_devices(running, DeviceContract(())) == []


class TestAllDevices:
    def test_lists_all_contract_devices_regardless_of_running(
        self, workflow_id, contract
    ):
        names = [d.device_name for d in all_devices(contract)]
        assert names == ['mon1_total', 'mon2_total']

    def test_is_running_in_reflects_snapshot(self, workflow_id, contract):
        running = {workflow_id: {'monitor1'}}
        states = {
            d.device_name: d.is_running_in(running) for d in all_devices(contract)
        }
        assert states == {'mon1_total': True, 'mon2_total': False}

    def test_empty_contract(self):
        assert all_devices(DeviceContract(())) == []


class TestDeclaredDeviceNames:
    def test_lists_workflow_devices_regardless_of_running(self, workflow_id, contract):
        assert declared_device_names(workflow_id, contract) == [
            'mon1_total',
            'mon2_total',
        ]

    def test_empty_for_other_workflow(self, contract):
        other = WorkflowId(instrument='dummy', name='other', version=1)
        assert declared_device_names(other, contract) == []


class TestAffectedDeviceNames:
    def test_lists_running_devices_in_order(self, workflow_id, contract):
        running = {workflow_id: {'monitor2', 'monitor1'}}
        assert affected_device_names(workflow_id, running, contract) == [
            'mon1_total',
            'mon2_total',
        ]

    def test_empty_when_not_device_bearing(self, workflow_id, contract):
        assert affected_device_names(workflow_id, {}, contract) == []


class TestViewDeviceNames:
    def test_device_view_maps_to_device(self, spec, workflow_id, contract):
        total_view = spec.get_output_view('total')
        names = view_device_names(
            workflow_id, {'monitor1', 'monitor2'}, total_view, contract
        )
        assert names == ['mon1_total', 'mon2_total']

    def test_non_device_view_maps_to_nothing(self, spec, workflow_id, contract):
        hist_view = spec.get_output_view('histogram')
        names = view_device_names(
            workflow_id, {'monitor1', 'monitor2'}, hist_view, contract
        )
        assert names == []

    def test_restricted_to_given_sources(self, spec, workflow_id, contract):
        total_view = spec.get_output_view('total')
        names = view_device_names(workflow_id, {'monitor1'}, total_view, contract)
        assert names == ['mon1_total']


class TestWithRealOrchestrator:
    """Derivation against a real orchestrator's running-sources snapshot."""

    @pytest.fixture
    def orchestrator(self, registry) -> JobOrchestrator:
        return JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            active_job_registry=ActiveJobRegistry(
                data_service=DataService(), job_service=JobService()
            ),
            config_store=None,
        )

    def test_no_running_jobs(self, orchestrator, contract):
        running = orchestrator.get_running_workflow_sources()
        assert running == {}
        assert exposed_devices(running, contract) == []

    def test_committed_job_becomes_device_bearing(
        self, orchestrator, workflow_id, contract
    ):
        orchestrator.clear_staged_configs(workflow_id)
        orchestrator.stage_config(
            workflow_id, source_name='monitor1', params={}, aux_source_names={}
        )
        orchestrator.commit_workflow(workflow_id)

        running = orchestrator.get_running_workflow_sources()
        assert running == {workflow_id: {'monitor1'}}
        assert affected_device_names(workflow_id, running, contract) == ['mon1_total']

    def test_stop_clears_device_bearing(self, orchestrator, workflow_id, contract):
        orchestrator.clear_staged_configs(workflow_id)
        orchestrator.stage_config(
            workflow_id, source_name='monitor1', params={}, aux_source_names={}
        )
        orchestrator.commit_workflow(workflow_id)
        orchestrator.stop_workflow(workflow_id)

        running = orchestrator.get_running_workflow_sources()
        assert affected_device_names(workflow_id, running, contract) == []
