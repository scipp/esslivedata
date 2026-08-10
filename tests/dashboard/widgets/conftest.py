# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Shared fixtures for dashboard widget tests."""

import pytest

from ess.livedata.dashboard.notification_queue import NotificationQueue
from ess.livedata.dashboard.plot_data_service import PlotDataService
from ess.livedata.dashboard.plot_orchestrator import PlotOrchestrator
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry
from ess.livedata.dashboard.session_updater import SessionUpdater
from ess.livedata.dashboard.stream_manager import StreamManager
from ess.livedata.dashboard.widgets.plot_grid_tabs import PlotGridTabs
from ess.livedata.dashboard.widgets.workflow_status_widget import (
    WorkflowStatusListWidget,
)


@pytest.fixture
def stream_manager(data_service):
    """Create a StreamManager for testing."""
    return StreamManager(data_service=data_service)


@pytest.fixture
def plotting_controller(stream_manager):
    """Create a PlottingController for testing."""
    return PlottingController(
        stream_manager=stream_manager,
    )


@pytest.fixture
def plot_data_service():
    """Create a PlotDataService for testing."""
    return PlotDataService()


@pytest.fixture
def plot_orchestrator(
    plotting_controller, job_orchestrator, data_service, plot_data_service
):
    """Create a PlotOrchestrator for testing."""
    return PlotOrchestrator(
        plotting_controller=plotting_controller,
        job_orchestrator=job_orchestrator,
        data_service=data_service,
        instrument='dummy',
        plot_data_service=plot_data_service,
    )


@pytest.fixture
def workflow_status_widget(job_orchestrator, job_service):
    """Create a WorkflowStatusListWidget for testing."""
    return WorkflowStatusListWidget(
        orchestrator=job_orchestrator,
        job_service=job_service,
    )


@pytest.fixture
def session_updater():
    """Create a SessionUpdater for testing."""
    registry = SessionRegistry()
    return SessionUpdater(
        session_id=SessionId('test-session'),
        session_registry=registry,
        notification_queue=NotificationQueue(),
    )


@pytest.fixture
def plot_grid_tabs(
    plot_orchestrator,
    workflow_registry,
    plotting_controller,
    workflow_status_widget,
    plot_data_service,
    session_updater,
):
    """Create a PlotGridTabs widget for testing."""
    return PlotGridTabs(
        plot_orchestrator=plot_orchestrator,
        workflow_registry=workflow_registry,
        plotting_controller=plotting_controller,
        workflow_status_widget=workflow_status_widget,
        plot_data_service=plot_data_service,
        session_updater=session_updater,
    )
