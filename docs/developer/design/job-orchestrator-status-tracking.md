# Job Status Tracking in JobOrchestrator

## Problem Statement

The dashboard currently tracks job status at the **job-level** via `JobService`, but the UI is moving toward a **workflow-level** abstraction. We need to determine:

1. Should job status tracking move from `JobService` into `JobOrchestrator`?
2. Where in the orchestrator's data structures should status be stored?
3. How should `JobOrchestrator` receive status updates?
4. How can we transition cleanly without breaking existing job-level UI?

## Current Architecture

### Status Flow (Job-Level)

```
Backend → STATUS_STREAM_ID → Orchestrator.forward() → JobService.status_updated()
                                                           ↓
                                                    JobStatusWidget (UI)
```

- `JobService` maintains `_job_statuses: dict[JobId, JobStatus]`
- Tracks individual job state, errors, warnings, timing
- Provides subscriptions for UI widgets

### JobOrchestrator Structure

```python
@dataclass
class JobConfig:
    """Configuration for a single job within a JobSet."""
    params: dict
    aux_source_names: dict

@dataclass
class JobSet:
    """A set of jobs sharing the same job_number."""
    job_number: JobNumber = field(default_factory=uuid.uuid4)
    jobs: dict[SourceName, JobConfig] = field(default_factory=dict)

@dataclass
class WorkflowState:
    """State for an active workflow, including transitions."""
    current: JobSet | None = None
    previous: JobSet | None = None
    staged_jobs: dict[SourceName, JobConfig] = field(default_factory=dict)
```

- `JobOrchestrator` manages workflow lifecycle (staging, commit, transitions)
- Groups jobs into `JobSet` (jobs sharing same `job_number`)
- Already has stub `handle_response()` method (for command responses, not periodic status)

## Design Decisions

### 1. Where to Store Status?

**Decision: Add status to `JobConfig`**

#### Options Considered

**Option A: Separate dict in JobSet** ❌
```python
@dataclass
class JobSet:
    jobs: dict[SourceName, JobConfig]
    job_statuses: dict[SourceName, JobStatus]  # Parallel dict
```
- **Problem**: Maintaining two parallel dicts creates synchronization issues
- **Problem**: Keys can get out of sync, error-prone

**Option B: Status in JobConfig** ✅ **CHOSEN**
```python
@dataclass
class JobConfig:
    params: dict
    aux_source_names: dict
    status: JobStatus | None = None  # None = not started/staged
```
- **Benefit**: Single source of truth per job
- **Benefit**: No synchronization issues
- **Benefit**: Natural lifecycle (staged jobs have `status=None`, active jobs have `JobStatus`)
- **Benefit**: Persistence already selective (only extracts `params`/`aux_source_names`)

**Option C: Status in WorkflowState** ❌
```python
@dataclass
class WorkflowState:
    current: JobSet | None
    current_statuses: dict[SourceName, JobStatus]  # Duplicates current/previous pattern
    previous: JobSet | None
    previous_statuses: dict[SourceName, JobStatus]
```
- **Problem**: Duplicates the current/previous structure
- **Problem**: Less cohesive than keeping status with the JobSet it belongs to

#### Naming Consideration

`JobConfig` becomes less accurate when it includes runtime state. Options:
- **JobInfo** - neutral, covers both config and state
- **JobEntry** - generic container
- Keep **JobConfig** - accept broader meaning of "config + state for a job"

**Decision: TBD during implementation** (likely keep `JobConfig` for simplicity)

### 2. How JobOrchestrator Receives Status Updates

**Decision: Orchestrator routes status to both services independently**

#### Implementation

```python
class Orchestrator:
    def __init__(
        self,
        message_source: MessageSource,
        data_service: DataService,
        job_service: JobService,
        job_orchestrator: JobOrchestrator | None = None,  # NEW: Optional during transition
        workflow_config_service: WorkflowConfigService | None = None,
    ):
        self._job_orchestrator = job_orchestrator

    def forward(self, stream_id: StreamId, value: Any) -> None:
        if stream_id == STATUS_STREAM_ID:
            # Route to both services independently during transition
            self._job_service.status_updated(value)
            if self._job_orchestrator is not None:
                self._job_orchestrator.status_updated(value)
        # ... other routing ...
```

```python
class JobOrchestrator:
    def status_updated(self, job_status: JobStatus) -> None:
        """Update status for a job in our workflow state."""
        job_id = job_status.job_id

        # Find which workflow this job belongs to
        for workflow_id, state in self._workflows.items():
            if state.current is not None:
                if job_id.source_name in state.current.jobs:
                    # Update status in JobConfig
                    state.current.jobs[job_id.source_name].status = job_status
                    break
```

#### Rejected Alternatives

**Option A: JobOrchestrator subscribes to JobService** ❌
- Creates coupling between services
- Makes removal harder (JobOrchestrator depends on JobService)
- Not clean separation

**Option B: Only route to JobOrchestrator** ❌
- Breaks existing job-level UI
- No transition period
- All-or-nothing migration

### 3. Transition Strategy

**Decision: Parallel pipelines, then clean removal**

#### Phase 1: Parallel Operation (Both Services Active)

```
Backend → STATUS_STREAM_ID → Orchestrator → ┬→ JobService → JobStatusWidget (old UI)
                                             └→ JobOrchestrator → WorkflowWidget (new UI)
```

- ✅ Both services receive status independently
- ✅ Zero coupling between services
- ✅ UI can mix job-level and workflow-level widgets
- ✅ Both can evolve separately

#### Phase 2: Remove JobService

When ready to remove `JobService`:

```diff
  def forward(self, stream_id: StreamId, value: Any) -> None:
      if stream_id == STATUS_STREAM_ID:
-         self._job_service.status_updated(value)
-         if self._job_orchestrator is not None:
-             self._job_orchestrator.status_updated(value)
+         self._job_orchestrator.status_updated(value)
```

```diff
  def __init__(
      self,
      message_source: MessageSource,
      data_service: DataService,
-     job_service: JobService,
      job_orchestrator: JobOrchestrator,  # No longer optional
      workflow_config_service: WorkflowConfigService | None = None,
  ):
```

**Clean removal**: Just delete JobService references from Orchestrator. No refactoring of JobOrchestrator needed.

## Implementation Plan

### Step 1: Add Status to JobConfig

```python
@dataclass
class JobConfig:
    """Configuration and runtime state for a single job."""
    params: dict
    aux_source_names: dict
    status: JobStatus | None = None  # None = staged/not started
```

### Step 2: Add status_updated() to JobOrchestrator

```python
class JobOrchestrator:
    def status_updated(self, job_status: JobStatus) -> None:
        """Update status for a job in our workflow state.

        Parameters
        ----------
        job_status
            Job status update from backend service.
        """
        job_id = job_status.job_id

        # Find which workflow this job belongs to
        for workflow_id, state in self._workflows.items():
            if state.current is not None:
                source_name = job_id.source_name
                if source_name in state.current.jobs:
                    # Update status in JobConfig
                    state.current.jobs[source_name].status = job_status
                    self._logger.debug(
                        "Updated status for job %s in workflow %s: %s",
                        job_id,
                        workflow_id,
                        job_status.state,
                    )
                    return

        # Job not found in any current JobSet - may be from previous/stopped workflow
        self._logger.debug(
            "Received status update for job %s not in any active workflow", job_id
        )
```

### Step 3: Wire JobOrchestrator into Orchestrator

```python
# In dashboard app initialization
job_orchestrator = JobOrchestrator(...)

orchestrator = Orchestrator(
    message_source=message_source,
    data_service=data_service,
    job_service=job_service,
    job_orchestrator=job_orchestrator,  # NEW
    workflow_config_service=workflow_config_service,
)
```

### Step 4: Add Workflow-Level Status Aggregation (Optional)

```python
class JobSet:
    def get_aggregate_status(self) -> WorkflowAggregateStatus:
        """Compute workflow-level status from all job statuses.

        Aggregation rules:
        - Any job in error → workflow has error
        - All jobs active → workflow active
        - Any job with warning → workflow has warning
        - etc.
        """
        # TODO: Define aggregation logic based on UI requirements
        ...
```

### Step 5: Build Workflow-Level UI

Create new UI components that consume from `JobOrchestrator` instead of `JobService`.

### Step 6: Remove JobService (Future)

Once workflow-level UI is complete and job-level UI is deprecated:
1. Remove `JobService` parameter from `Orchestrator.__init__`
2. Remove `job_service.status_updated()` call from `Orchestrator.forward()`
3. Make `job_orchestrator` required (not optional)
4. Delete `JobService` class and related tests

## Benefits

1. **Workflow-level abstraction**: Status naturally grouped by workflow
2. **Clean transition**: Parallel operation without coupling
3. **Straightforward removal**: Delete JobService references when ready
4. **No duplication**: Single dict structure (no parallel dicts to sync)
5. **Automatic lifecycle**: Status moves with JobSet (current → previous)

## Related Issues

- #538 - Job orchestrator (main tracking issue)
- #496 - Dashboard data/job service controls
- #445 - Cleanup/fix reply mechanism to request from frontend

## Open Questions

1. **Naming**: Should `JobConfig` be renamed to `JobInfo` or similar?
2. **Aggregation logic**: How should workflow-level status be computed from job statuses?
3. **Previous JobSet status**: Should we preserve status when JobSet moves to `previous`?
4. **Stale status detection**: Does JobOrchestrator need equivalent of `JobService.is_job_status_stale()`?
