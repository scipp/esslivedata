# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Device-bearing behavior of WorkflowStatusWidget.

Covers the confirmation gate (commit/stop/reset gated only when device-bearing,
staging never gated), the device badge, and the per-output device marker, using
a real JobOrchestrator and a real DeviceContract.
"""

import warnings
from contextlib import contextmanager

import pydantic
import pytest
import scipp as sc
from panel.util.warnings import PanelUserWarning

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
from ess.livedata.dashboard.widgets.workflow_status_widget import WorkflowStatusWidget
from ess.livedata.fakes import FakeMessageSink


class Params(pydantic.BaseModel):
    threshold: float = 1.0


class MonitorOutputs(WorkflowOutputsBase):
    counts_total_cumulative: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0.0, unit='counts')),
        title='Total',
    )
    histogram: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['x'], shape=[0], unit='counts'),
            coords={'x': sc.arange('x', 0, unit='m')},
        ),
        title='Histogram',
    )

    output_views = (
        OutputView(
            name='total',
            title='Total',
            streams={'since_start': 'counts_total_cumulative'},
        ),
        OutputView(
            name='histogram', title='Histogram', streams={'since_start': 'histogram'}
        ),
    )


@pytest.fixture
def workflow_id() -> WorkflowId:
    return WorkflowId(instrument='dummy', name='monitor', version=1)


@pytest.fixture
def spec(workflow_id) -> WorkflowSpec:
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
    return DeviceContract(
        [
            DeviceContractEntry(
                workflow_id=str(workflow_id),
                source_name='monitor1',
                output_name='counts_total_cumulative',
                device_name='mon1_total',
            )
        ]
    )


@pytest.fixture
def job_service() -> JobService:
    return JobService()


@pytest.fixture
def orchestrator(registry, job_service) -> JobOrchestrator:
    orch = JobOrchestrator(
        command_service=CommandService(sink=FakeMessageSink()),
        workflow_registry=registry,
        active_job_registry=ActiveJobRegistry(
            data_service=DataService(), job_service=job_service
        ),
        config_store=None,
    )
    job_service.on_status_updated = orch.on_job_status_updated
    return orch


@contextmanager
def _no_server():
    """Suppress Panel's "modal needs a server" warning when opening a modal.

    Opening a ``pn.Modal`` outside a Bokeh server emits a ``PanelUserWarning``;
    irrelevant to the gate's behavior, which we exercise headless.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PanelUserWarning)
        yield


def _run(orchestrator: JobOrchestrator, workflow_id: WorkflowId, source: str) -> None:
    orchestrator.clear_staged_configs(workflow_id)
    orchestrator.stage_config(
        workflow_id, source_name=source, params={}, aux_source_names={}
    )
    orchestrator.commit_workflow(workflow_id)


@pytest.fixture
def widget(workflow_id, spec, orchestrator, job_service, contract):
    return WorkflowStatusWidget(
        workflow_id=workflow_id,
        workflow_spec=spec,
        orchestrator=orchestrator,
        job_service=job_service,
        device_contract=contract,
    )


class TestGate:
    def test_stop_gated_when_device_bearing(
        self, widget, orchestrator, workflow_id, job_service
    ):
        _run(orchestrator, workflow_id, 'monitor1')
        assert orchestrator.get_active_job_number(workflow_id) is not None

        with _no_server():
            widget._on_stop_click()
        # Not stopped yet: a confirmation modal is open.
        assert orchestrator.get_active_job_number(workflow_id) is not None
        assert len(widget._modal_container) == 1

    def test_confirm_runs_action(self, widget, orchestrator, workflow_id):
        _run(orchestrator, workflow_id, 'monitor1')
        with _no_server():
            widget._on_stop_click()
        with _no_server():
            _click(widget, 'lt-confirm-proceed')
        assert orchestrator.get_active_job_number(workflow_id) is None

    def test_cancel_does_not_run_action(self, widget, orchestrator, workflow_id):
        _run(orchestrator, workflow_id, 'monitor1')
        with _no_server():
            widget._on_stop_click()
        with _no_server():
            _click(widget, 'lt-confirm-cancel')
        assert orchestrator.get_active_job_number(workflow_id) is not None

    def test_not_gated_when_running_source_not_in_contract(
        self, widget, orchestrator, workflow_id
    ):
        # monitor2 is not in the contract -> no gate, stop runs immediately.
        _run(orchestrator, workflow_id, 'monitor2')
        widget._on_stop_click()
        assert orchestrator.get_active_job_number(workflow_id) is None
        assert len(widget._modal_container) == 0

    def test_staging_never_gated(self, widget, orchestrator, workflow_id):
        # Device-bearing, but opening the gear (staging) must not gate.
        _run(orchestrator, workflow_id, 'monitor1')
        with _no_server():
            widget._on_gear_click(['monitor1'])
        # The gear opens a configuration modal, not a confirmation gate;
        # the orchestrator job is untouched.
        assert orchestrator.get_active_job_number(workflow_id) is not None


class TestBadge:
    def test_badge_present_when_contract_declares_device(self, widget):
        # Static: present regardless of run state, naming the declared device.
        html = widget._make_device_badge_html()
        assert 'NICOS' in html
        assert 'mon1_total' in html

    def test_badge_present_while_running(self, widget, orchestrator, workflow_id):
        _run(orchestrator, workflow_id, 'monitor1')
        html = widget._make_device_badge_html()
        assert 'NICOS' in html
        assert 'mon1_total' in html

    def test_badge_empty_without_contract(
        self, workflow_id, spec, orchestrator, job_service
    ):
        plain = WorkflowStatusWidget(
            workflow_id=workflow_id,
            workflow_spec=spec,
            orchestrator=orchestrator,
            job_service=job_service,
            device_contract=DeviceContract(()),
        )
        assert plain._make_device_badge_html() == ''


class TestMarker:
    def test_marker_on_device_output(self, widget):
        widget.set_expanded(True)
        section = widget._create_outputs_section()
        html = _collect_html(section)
        assert 'NICOS' in html
        assert 'mon1_total' in html

    def test_marker_absent_without_contract(
        self, workflow_id, spec, orchestrator, job_service
    ):
        plain = WorkflowStatusWidget(
            workflow_id=workflow_id,
            workflow_spec=spec,
            orchestrator=orchestrator,
            job_service=job_service,
            device_contract=DeviceContract(()),
        )
        section = plain._create_outputs_section()
        assert 'NICOS' not in _collect_html(section)


def _click(widget, css_class: str) -> None:
    """Trigger the button carrying ``css_class`` inside the modal container."""
    import panel as pn

    stack = list(widget._modal_container.objects)
    while stack:
        node = stack.pop()
        if isinstance(node, pn.widgets.Button) and css_class in (
            node.css_classes or []
        ):
            node.clicks += 1
            return
        if hasattr(node, 'objects'):
            stack.extend(node.objects)
    raise AssertionError(f'No button with css class {css_class!r} found')


def _collect_html(component) -> str:
    """Concatenate HTML pane strings within a Panel component tree."""
    import panel as pn

    parts: list[str] = []
    stack = [component]
    while stack:
        node = stack.pop()
        if isinstance(node, pn.pane.HTML) and isinstance(node.object, str):
            parts.append(node.object)
        if hasattr(node, 'objects'):
            stack.extend(node.objects)
    return ''.join(parts)
