# Backend Service Architecture

## Overview

ESSlivedata backend services follow a consistent **message-driven architecture** with clear separation of concerns:

- **Service**: Manages lifecycle and processing loop
- **Processor**: Routes and batches messages (IdentityProcessor or OrchestratingProcessor)
- **Preprocessor/Accumulator**: Accumulates and transforms messages before workflow execution
- **JobManager**: Orchestrates job lifecycle and workflow execution
- **Workflow**: Implements scientific reduction logic using sciline workflows

All backend services follow this pattern, differing only in preprocessor implementations.

```mermaid
graph TD
    subgraph "Service Layer"
        Service[Service]
    end

    subgraph "Processing Layer"
        IP[IdentityProcessor]
        OP[OrchestratingProcessor]
    end

    subgraph "Preprocessing Layer"
        PF[PreprocessorFactory]
        MP[MessagePreprocessor]
        A1[Accumulator 1]
        A2[Accumulator 2]
        A3[Accumulator N]
    end

    subgraph "Job Management"
        JM[JobManager]
        WF[Workflows]
    end

    subgraph "Infrastructure"
        MS[MessageSource]
        MSink[MessageSink]
    end

    Service -->|process loop| IP
    Service -->|process loop| OP
    MS --> IP
    MS --> OP
    IP --> MSink
    OP --> MP
    PF -.creates.-> MP
    MP -.uses.-> A1
    PF -.creates.-> A2
    PF -.creates.-> A3
    A1 --> JM
    A2 --> JM
    A3 --> JM
    JM --> WF
    WF --> MSink
    OP --> MSink

    classDef service fill:#e3f2fd,stroke:#1976d2;
    classDef processor fill:#f3e5f5,stroke:#7b1fa2;
    classDef preproc fill:#fff3e0,stroke:#f57c00;
    classDef job fill:#fce4ec,stroke:#c2185b;
    classDef infra fill:#e8f5e9,stroke:#388e3c;

    class Service service;
    class IP,OP processor;
    class PF,MP,A1,A2,A3 preproc;
    class JM,WF job;
    class MS,MSink infra;
```

## Service-Processor-Workflow Pattern

This architecture provides clear separation with distinct responsibilities:

1. **Service**: Lifecycle management and processing loop
2. **Processor**: Message batching and pipeline coordination
3. **MessagePreprocessor**: Routes messages to appropriate accumulators
4. **Accumulator**: Accumulates and transforms messages per stream
5. **JobManager**: Job lifecycle and workflow execution coordination
6. **Workflow**: Scientific reduction logic on accumulated data

### Two Processor Types

**IdentityProcessor**: Simple pass-through for fake data producers (no preprocessing or job management).

**OrchestratingProcessor**: Full job-based processing for all backend services:

- Configuration message handling
- Time-based message batching
- Preprocessing via accumulators
- Job lifecycle management
- Workflow execution
- Periodic status reporting (every 2 seconds)

**Processing Flow**: Source → Split config/data → Route config to jobs → Batch data by time → Preprocess/accumulate → Push to JobManager → Compute workflows → Publish results

### Service Layer

Manages lifecycle, signal handling (SIGTERM/SIGINT), background threading, and resource cleanup via `ExitStack`. Supports step-by-step execution for testing.

### Preprocessor Layer

**PreprocessorFactory** creates stream-specific accumulators on-demand. Common preprocessors: `GroupByPixel`, `ToNXevent_data`, `ToNXlog`, `Cumulative`, `LatestValueHandler`. Implement `Accumulator` protocol with `add()`, `get()`, `clear()`.

## Job-Based Processing Architecture

All backend services (monitor_data, detector_data, data_reduction, timeseries) use the same job-based architecture:

```mermaid
sequenceDiagram
    participant S as Service
    participant OP as OrchestratingProcessor
    participant CP as ConfigProcessor
    participant MP as MessagePreprocessor
    participant JM as JobManager
    participant Sink as MessageSink

    S->>OP: process()
    OP->>OP: Get messages from source
    OP->>OP: Split config vs data messages
    OP->>CP: process_messages(config_msgs)
    CP->>JM: schedule_job() / job_command()
    OP->>JM: Report status
    OP->>OP: Batch data messages by time
    OP->>MP: preprocess_messages(batch)
    MP->>MP: Accumulate per stream
    MP->>OP: WorkflowData
    OP->>JM: push_data(workflow_data)
    OP->>JM: compute_results()
    JM->>OP: JobResults
    OP->>Sink: publish_messages(results + status)
```

**ConfigProcessor** (see `handlers/config_handler.py`) handles `workflow_config` and `job_command` messages, delegating to **JobManagerAdapter** (see `core/job_manager_adapter.py`), which bridges ConfigProcessor and JobManager. Other key features: time-based batching, preprocessing accumulation, job lifecycle management, periodic status reporting.

## Message Flow in Backend Services

For a detailed view of the message source chain, adapter composition, and message transformations, see [](message-flow-and-transformation.md).

## Job-Based Processing System

`OrchestratingProcessor` uses job management for multiple concurrent data reduction workflows.

### Job Lifecycle

```{note}
The error/warning states are not fully exclusive in the current implementation — a job can transition between them. This state machine is a simplification.
```

```mermaid
stateDiagram-v2
    [*] --> scheduled: schedule_job()
    scheduled --> active: Time reached
    active --> finishing: End time reached
    active --> stopped: stop_job()
    active --> paused: pause()
    paused --> active: resume()
    scheduled --> stopped: stop_job()
    finishing --> stopped: Finalization complete
    stopped --> [*]: remove_job()

    active --> error: Finalization error
    active --> warning: Processing error
    error --> active: Successful finalization
    warning --> active: Successful processing
```

**Job States**: `scheduled` (waiting for start time) → `active` (processing) → `finishing` (end time reached) → `stopped` (complete). `paused` is defined but pause/resume raise `NotImplementedError` (placeholder for future use). Error states: `error` (finalization failure), `warning` (processing error).

### JobManager

Orchestrates job operations: `schedule_job()`, `push_data()` (activates/processes jobs), `compute_results()` (only for jobs with primary data), `stop_job()`, `reset_job()`, `remove_job()`, `get_job_status()`.

**Key Features**: Time-based activation, primary data triggering, auxiliary data handling, error isolation, status tracking.

### Primary vs Auxiliary Data

**Primary Data** (triggers job activation and computation): Detector/monitor events specified in `WorkflowSpec.source_names`.

**Auxiliary Data** (non-triggering metadata): Sample environment, geometry, etc. specified in `WorkflowSpec.aux_source_names`.

Prevents unnecessary computations when only metadata updates; enables efficient slow/fast-changing data handling.

### Job Scheduling

Jobs process specific time ranges via `JobSchedule(start_time, end_time)`. All times in nanoseconds since Unix epoch (UTC), based on data timestamps not wall-clock time.

## Service Lifecycle Management

### Graceful Shutdown

Services handle SIGTERM/SIGINT gracefully: Signal received → Set `_running = False` → Exit processing loop → Join background threads → Cleanup resources (consumers, producers) → Exit.

### Resource Management

Services use `ExitStack` for automatic resource cleanup on service exit or errors.

### Error Handling

**Service Loop**: Processor exceptions logged, service stops and signals main thread (prevents running with broken processor).

**Preprocessor Errors**: Per-stream exceptions caught, logged, other streams continue. PreprocessorFactory may return None for unknown streams (skipped silently).

**Job Errors**: Processing errors → `warning` state; Finalization errors → `error` state. Job continues, errors in status messages, other jobs unaffected.

## Building Services with DataServiceBuilder

`DataServiceBuilder` constructs services consistently with `OrchestratingProcessor` by default. For command-line services, `DataServiceRunner` (see `service_factory.py`) wraps a builder and adds standard CLI arguments (`--instrument`, `--dev`, `--log-level`, `--sink-type`). Services can publish initialization messages on startup for workflow specifications or configuration values.

