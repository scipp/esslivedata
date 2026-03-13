# Orchestrator Flow: Plot Creation Lifecycle

This document describes the interaction between PlotOrchestrator, LayerSubscription, JobOrchestrator, PlottingController, and PlotDataService when a plot layer is created.

## Overview

The plot creation follows a **two-phase subscription model**:

1. **Phase 1**: LayerSubscription subscribes to JobOrchestrator for each data source role. When *all* workflows are running, it fires an `on_ready` callback with `SubscriptionReady` (containing `keys_by_role`).
2. **Phase 2**: PlotOrchestrator creates a plotter, sets up the data pipeline via PlottingController, and waits for data. When data arrives, `plotter.compute(data)` is called and PlotDataService transitions the layer to READY.

## Sequence Diagram

```mermaid
sequenceDiagram
    participant User as User/PlotGridTabs
    participant PO as PlotOrchestrator
    participant LS as LayerSubscription
    participant JO as JobOrchestrator
    participant PC as PlottingController
    participant PDS as PlotDataService

    Note over User,PDS: Phase 1 — Subscribe to workflows
    User->>PO: add_layer(cell_id, PlotConfig)
    PO->>PDS: layer_added(layer_id)
    PO->>LS: create(data_sources, on_ready, on_stopped)
    PO->>LS: start()
    loop For each role in data_sources
        LS->>JO: subscribe_to_workflow(workflow_id, callbacks)
    end

    alt Workflow not yet running
        PO-->>User: on_cell_updated(cell)<br/>[PDS state: WAITING_FOR_JOB]

        Note over JO: Later: user commits workflow
        JO->>LS: on_started(job_number)
    else Workflow already running
        JO->>LS: on_started(job_number) [synchronous]
    end

    Note over LS: When ALL roles have a job_number:
    LS->>PO: on_ready(SubscriptionReady{keys_by_role})

    Note over User,PDS: Phase 2 — Create plotter and data pipeline
    PO->>PC: create_plotter(plot_name, params)
    PC-->>PO: Plotter instance
    PO->>PDS: job_started(layer_id, plotter)<br/>[state → WAITING_FOR_DATA]

    PO->>PC: setup_pipeline(keys_by_role, plot_name,<br/>params, on_data callback)
    PC-->>PO: DataServiceSubscriber

    Note over User,PDS: Data arrival
    Note over PO: on_data(dict[ResultKey, Any]) fires
    PO->>PO: plotter.compute(data)
    PO->>PDS: data_arrived(layer_id)<br/>[state → READY]

    Note over User,PDS: Subsequent updates
    loop Streaming data
        Note over PO: on_data fires again
        PO->>PO: plotter.compute(data)
    end
```

## Key Components

| Component | Responsibility |
|-----------|----------------|
| **PlotGridTabs** | UI widget; subscribes to PlotOrchestrator lifecycle events, polls PlotDataService for layer state |
| **PlotOrchestrator** | Manages plot lifecycle: layer subscription, plotter creation, data pipeline setup. See `dashboard/plot_orchestrator.py` |
| **LayerSubscription** | Subscribes to one or more workflows for a layer. Fires `on_ready` when *all* workflows are running, `on_stopped` when *any* stops. See `dashboard/layer_subscription.py` |
| **JobOrchestrator** | Manages workflow jobs; notifies subscribers on start/stop via `WorkflowCallbacks`. See `dashboard/job_orchestrator.py` |
| **PlottingController** | Two-phase plot creation: `create_plotter()` and `setup_pipeline()`. See `dashboard/plotting_controller.py` |
| **PlotDataService** | Layer state machine (WAITING_FOR_JOB → WAITING_FOR_DATA → READY / STOPPED / ERROR). Version-based polling for UI. See `dashboard/plot_data_service.py` |
| **DataService** | Holds job result data; manages subscriber lifecycle |

## Layer States

Each layer transitions through explicit states managed by `PlotDataService`:

```mermaid
stateDiagram-v2
    [*] --> WAITING_FOR_JOB: layer_added()
    WAITING_FOR_JOB --> WAITING_FOR_DATA: job_started(plotter)
    WAITING_FOR_JOB --> ERROR: error_occurred()

    WAITING_FOR_DATA --> READY: data_arrived()
    WAITING_FOR_DATA --> STOPPED: job_stopped()
    WAITING_FOR_DATA --> ERROR: error_occurred()

    READY --> STOPPED: job_stopped()
    READY --> WAITING_FOR_DATA: job_started(plotter) [restart]
    READY --> ERROR: error_occurred()

    STOPPED --> WAITING_FOR_DATA: job_started(plotter)
    STOPPED --> ERROR: error_occurred()
```

Each transition increments a version counter. UI components poll for version changes to detect when rebuilds are needed.

## Multi-Source Layers (Correlation Plots)

`PlotConfig.data_sources` maps role names to `DataSourceConfig`:

- **"primary"**: Main data source (required for all layers)
- **"x_axis"**, **"y_axis"**: Correlation axes (optional, for correlation histograms)

`LayerSubscription` subscribes to *each* role's workflow independently. The `on_ready` callback only fires when all roles have running jobs. If any workflow stops, `on_stopped` fires immediately (a correlation plot cannot function with partial data).

The resulting `keys_by_role` dict flows through to `setup_pipeline()`, preserving role structure so plotters receive data grouped by role.

## Lifecycle Callbacks

PlotGridTabs subscribes to PlotOrchestrator lifecycle events (grid created/removed/updated, cell updated/removed). The `on_cell_updated` callback signals that a cell's layer configuration changed. Actual layer state (waiting, ready, error) is read from `PlotDataService` by the UI during periodic polling.
