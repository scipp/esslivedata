# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
DESIGN HULL: JobOrchestrator - Public API only

We need to design and implement a JobOrchestrator. To explain its
purpose, here is some context:

- We have a series of available workflows.
- They can be selected and run by a user.
- A concrete "run" of a workflow is referred to as a *job*.
- Running jobs produce a dict of output streams (held by and managed
by DataService).
- Plots are created by selecting outputs.

Here is where JobOrchestrator comes in. Its purpose is to provide a
seemless interface that makes working with jobs easier for users,
since the mainly care about "workflows", not "jobs":

- We want to run only a single job per workflow at a time.
- If a user configures and starts a workflow, an existing job for that
 particular workflow should be stopped first.
- Plots that subscribe to DataService for updates for a particular job
 output should either be recreated (and replaced in the UI), or the
subscription should be redirected to the new job's output.

JobOrchestrator might be a component we place between
DataService/CommandService/WorkflowConfigService on the one side and
WorkflowController/PlottingController on the other. Precise
responsibilities are not clear.

Core idea: One active "run" per workflow, where a run is a set of jobs
(one per source) sharing the same job_number.

Latest idea:
We need to handle configuration independent of starting, e.g., to support different config for each source name.
This would also make this "symmetric" between loading config and creating new config.
Rough flow:
- Stage config for source1, source2
- Stage different config for source3 (but for some JobNumber).
- Commit (start workflow).

WorkflowState should keep track of:
- Config(s) of active jobs.
- Staging for next job.

It it still unclear of we need two JobSets (current and previous), or if we fully finish with the previous one before the next is created.
Does JobSet only hold jobs, or also staged configs?

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import pydantic

from ess.livedata.config.workflow_spec import JobId, JobNumber, WorkflowId
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.job_controller import JobController
from ess.livedata.dashboard.job_service import JobService


@dataclass
class JobSet:
    """A set of jobs sharing the same job_number."""

    job_number: JobNumber
    jobs: list[JobId]  # One per source_name


@dataclass
class WorkflowState:
    """State for an active workflow, including transitions."""

    current: JobSet
    previous: JobSet | None  # Present during handoff/transition
    plot_subscriptions: dict[
        JobId, dict[str | None, set[str]]
    ]  # job -> output -> plot_ids


@dataclass(frozen=True)
class PlotSubscription:
    plot_id: str
    job_id: JobId
    output_name: str | None


class PlotTransitionStrategy(Enum):
    RECREATE = "recreate"
    REDIRECT = "redirect"
    MANUAL = "manual"


class PlotTransitionHandler(Protocol):
    def transition_plots(
        self,
        old_job_ids: list[JobId],
        new_job_ids: list[JobId],
        strategy: PlotTransitionStrategy,
    ) -> None: ...


class JobOrchestrator:
    # Primary state: workflow-centric
    _workflows: dict[WorkflowId, WorkflowState]

    # Reverse lookups for efficiency
    _plot_job_map: dict[str, PlotSubscription]  # plot_id -> subscription

    # Dependencies
    _job_service: JobService
    _job_controller: JobController
    _command_service: CommandService
    _plot_handler: PlotTransitionHandler | None
    _default_transition_strategy: PlotTransitionStrategy

    def __init__(
        self,
        job_service: JobService,
        job_controller: JobController,
        command_service: CommandService,
        plot_handler: PlotTransitionHandler | None = None,
        default_transition_strategy: PlotTransitionStrategy = PlotTransitionStrategy.RECREATE,
    ): ...

    # Workflow lifecycle
    def start_workflow(
        self,
        workflow_id: WorkflowId,
        source_names: list[str],
        config: pydantic.BaseModel,
        aux_source_names: pydantic.BaseModel | None = None,
        transition_strategy: PlotTransitionStrategy | None = None,
    ) -> list[JobId]: ...

    def stop_workflow(self, workflow_id: WorkflowId) -> None: ...

    # Job queries
    def get_active_jobs(self, workflow_id: WorkflowId) -> list[JobId]: ...

    def get_active_job_number(self, workflow_id: WorkflowId) -> JobNumber | None: ...

    def is_workflow_running(self, workflow_id: WorkflowId) -> bool: ...

    def get_workflow_for_job(self, job_id: JobId) -> WorkflowId | None: ...

    def get_all_active_workflows(self) -> dict[WorkflowId, list[JobId]]: ...

    # Plot subscription tracking
    def register_plot(
        self,
        plot_id: str,
        job_id: JobId,
        output_name: str | None = None,
    ) -> None: ...

    def unregister_plot(self, plot_id: str) -> None: ...

    def get_plots_for_job(
        self, job_id: JobId, output_name: str | None = None
    ) -> set[str]: ...

    def get_subscription_for_plot(self, plot_id: str) -> PlotSubscription | None: ...
