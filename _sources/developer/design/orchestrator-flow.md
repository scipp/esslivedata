# Orchestrator Flow: Plot Creation Lifecycle

This document describes the interaction between PlotOrchestrator, JobOrchestrator, PlottingController, and DataService when a user requests a plot.

## Overview

The plot creation follows a **two-phase subscription model**:

1. **Phase 1**: Subscribe to JobOrchestrator → receive `job_number` when workflow is committed
2. **Phase 2**: Use `job_number` to set up data pipeline → wait for data → create plot

## Sequence Diagram

```mermaid
sequenceDiagram
    participant User as User/PlotGridTabs
    participant PO as PlotOrchestrator
    participant JO as JobOrchestrator
    participant PC as PlottingController
    participant Sub as DataSubscriber
    participant DS as DataService

    Note over User,DS: Setup Phase
    User->>PO: add_plot(grid_id, PlotCell)<br/>"Plot workflow A output X here"
    PO->>JO: subscribe_to_workflow(workflow_id, callback)

    alt Workflow not yet running
        JO-->>PO: (subscription_id, was_invoked=false)
        PO-->>User: on_cell_updated(cell, plot=None)<br/>"Waiting for workflow"

        Note over JO: Later: User commits workflow
        JO->>PO: callback(job_number=12345)
    else Workflow already running
        JO->>PO: callback(job_number=12345)
        JO-->>PO: (subscription_id, was_invoked=true)
    end

    Note over User,DS: Data Pipeline Setup
    PO->>PC: setup_data_pipeline(job_number,<br/>workflow_id, source_names,<br/>output_name, on_first_data)
    PC->>Sub: Create with extractors and callback
    Sub->>DS: Subscribe to ResultKey(s)

    opt Data not yet available
        PO-->>User: on_cell_updated(cell, plot=None)<br/>"Waiting for data"
    end

    Note over User,DS: Plot Creation (on first data)
    DS-->>Sub: Data for job 12345
    Sub->>PO: on_first_data(pipe)
    PO->>PC: create_plot_from_pipeline(plot_name,<br/>params, pipe)
    PC-->>PO: HoloViews DynamicMap
    PO-->>User: on_cell_updated(cell, plot=DynamicMap)

    Note over User,DS: Subsequent Data Updates
    loop Streaming updates
        DS-->>Sub: New data for job 12345
        Sub-->>User: Pipe triggers DynamicMap update
    end
```

## Key Components

| Component | Responsibility |
|-----------|----------------|
| **PlotGridTabs** | UI widget where user selects what to plot and where |
| **PlotOrchestrator** | Manages plot lifecycle, subscribes to job availability, coordinates plot creation |
| **JobOrchestrator** | Manages workflow jobs, generates job numbers, notifies subscribers on commit |
| **PlottingController** | Creates DataSubscribers and HoloViews plots |
| **DataSubscriber** | Subscribes to DataService, invokes `on_first_data` callback when data arrives |
| **DataService** | Holds job data, notifies subscribers when data arrives |

## Waiting States

The UI can receive intermediate states before a plot is ready:

1. **"Waiting for workflow"**: The workflow hasn't been committed yet. The plot cell is configured but no job exists.
2. **"Waiting for data"**: The workflow is running (job exists) but no data has arrived yet.
3. **Plot ready**: Data has arrived and the HoloViews DynamicMap is created.

## Lifecycle Callbacks

PlotGridTabs subscribes to PlotOrchestrator lifecycle events:

```python
plot_orchestrator.subscribe_to_lifecycle(
    on_grid_created=...,
    on_grid_removed=...,
    on_cell_updated=...,  # Receives plot or waiting state
    on_cell_removed=...,
)
```

The `on_cell_updated` callback is called:
- When a cell is first added (waiting for workflow)
- When job becomes available but no data yet (waiting for data)
- When plot is created (with the DynamicMap)
- When plot creation fails (with error message)
