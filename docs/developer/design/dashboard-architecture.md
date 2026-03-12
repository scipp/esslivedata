# Livedata Dashboard Architecture

## Overview

The Livedata dashboard is a real-time data visualization system built on Panel/HoloViews. It consumes processed detector data from Kafka, displays it as interactive plots, and publishes user commands back to backend services. The architecture supports multiple concurrent browser sessions with shared state and per-session rendering.

Data updates arrive at ~1 Hz. User controls (workflow start/stop, plot configuration) result in commands published to Kafka for backend consumption.

## System Context

```mermaid
flowchart TD
    ECDCTopic(["ECDC Topics"])
    DataTopic(["Livedata Data Topic"])
    StatusTopic(["Livedata Status Topic"])
    ResponsesTopic(["Livedata Responses Topic"])
    CommandsTopic(["Livedata Commands Topic"])

    subgraph BackendServices["Livedata Backend Services"]
        MonitorData["monitor_data"]
        DetectorData["detector_data"]
        DataReduction["data_reduction"]
    end

    DashboardApp["Dashboard"]

    ECDCTopic --> BackendServices
    MonitorData -- publishes --> DataTopic
    DetectorData -- publishes --> DataTopic
    DataReduction -- publishes --> DataTopic
    DataReduction -- publishes --> StatusTopic
    DataReduction -- publishes --> ResponsesTopic

    DataTopic --> DashboardApp
    StatusTopic --> DashboardApp
    ResponsesTopic --> DashboardApp
    DashboardApp -- publishes --> CommandsTopic
    CommandsTopic --> DataReduction

    classDef kafka fill:#fff3e0,stroke:#ef6c00,color:#e65100;
    classDef backend fill:#e3f2fd,stroke:#1976d2,color:#0d47a1;
    classDef dashboard fill:#ede7f6,stroke:#7b1fa2,color:#4a148c;
    class DataTopic,StatusTopic,ResponsesTopic,CommandsTopic,ECDCTopic kafka;
    class MonitorData,DetectorData,DataReduction backend;
    class DashboardApp dashboard;
```

The dashboard publishes job commands (start/stop workflows) to the Commands topic and receives acknowledgements via the Responses topic. Job and service status updates arrive via the Status topic.

## Layered Architecture

```mermaid
graph TD
    subgraph "Infrastructure Layer"
        T["Transport<br>(Kafka / Null)"]
        MS["MessageSource"]
        CS["CommandService"]
    end

    subgraph "Application Layer"
        O["Orchestrator"]
        DS["DataService"]
        JO["JobOrchestrator"]
        AJR["ActiveJobRegistry"]
        PO["PlotOrchestrator"]
        PC["PlottingController"]
        SM["StreamManager"]
        WC["WorkflowController"]
    end

    subgraph "Presentation Layer"
        PGT["PlotGridTabs"]
        WSW["WorkflowStatusWidget"]
        CW["ConfigurationWidget"]
        Plots["Plots<br>(HoloViews)"]
    end

    T --> MS
    MS --> O
    O --> DS
    O --> AJR
    CS --> T
    JO --> CS
    WC --> JO
    PO --> PC
    PC --> SM
    SM --> DS
    PO --> JO
    PGT --> PO
    WSW --> JO
    CW --> WC
    DS -.->|notifies| Plots

    classDef infra fill:#e3f2fd,stroke:#1976d2,color:#0d47a1;
    classDef app fill:#ede7f6,stroke:#7b1fa2,color:#4a148c;
    classDef presentation fill:#e8f5e9,stroke:#388e3c,color:#1b5e20;
    class T,MS,CS infra;
    class O,DS,JO,AJR,PO,PC,SM,WC app;
    class PGT,WSW,CW,Plots presentation;
```

- **Infrastructure**: Transport abstraction (Kafka or Null for testing), message sources and sinks
- **Application**: Data management, workflow lifecycle, plot orchestration
- **Presentation**: Panel widgets and HoloViews plots

## Service Composition

`DashboardServices` (see `dashboard/dashboard_services.py`) wires all components together. It is the central composition root, created once per dashboard process, and shared across all browser sessions.

```mermaid
flowchart TD
    DS[DashboardServices]
    DS --> Transport
    DS --> Orchestrator
    DS --> DataService
    DS --> JobOrchestrator
    DS --> PlotOrchestrator
    DS --> WorkflowController
    DS --> CommandService
    DS --> ActiveJobRegistry
    DS --> SessionRegistry
    DS --> PlotDataService
    DS --> NotificationQueue
```

`DashboardBase` (see `dashboard/dashboard.py`) is the entry point. It creates `DashboardServices`, starts the Panel server, and creates per-session layouts via `create_layout()`.

## Data Flow

```mermaid
sequenceDiagram
    participant K as Kafka
    participant O as Orchestrator
    participant AJR as ActiveJobRegistry
    participant DS as DataService
    participant Sub as Plot Subscribers
    participant UI as HoloViews Plots

    K->>O: Raw messages (polled in background thread)
    O->>AJR: ingestion_guard()
    Note over O,AJR: Serializes against job stop/cleanup
    O->>O: Filter by active job number
    O->>DS: transaction { store values }
    DS->>Sub: Notify subscribers (batched)
    Sub->>UI: Update plot data (via Pipe)
```

The `Orchestrator` (see `dashboard/orchestrator.py`) is the message pump. It consumes from the `MessageSource`, filters messages by active job numbers, and stores data in `DataService`. Status messages and command acknowledgements are routed to `JobOrchestrator`.

`DataService` (see `dashboard/data_service.py`) is a dict-like store keyed by `ResultKey`. Subscribers register interest in specific keys and receive batched notifications via a transaction mechanism.

## Workflow Lifecycle

`JobOrchestrator` (see `dashboard/job_orchestrator.py`) manages the full lifecycle of workflow jobs using a two-phase commit pattern:

```mermaid
stateDiagram-v2
    [*] --> Staging: stage_config()
    Staging --> Staging: stage_config() (more sources)
    Staging --> Committed: commit_workflow()
    Committed --> [*]: stop_workflow()
    Committed --> Committed: Receives acknowledgement

    note right of Staging
        Per-source configs staged
        in memory. No Kafka commands
        sent yet.
    end note

    note right of Committed
        Commands sent to backend.
        PendingCommandTracker awaits
        acknowledgements.
    end note
```

Key responsibilities:
- **Staging**: Collects per-source configurations before committing. Subscribers are notified of staging changes for UI preview.
- **Commit**: Sends `JobCommand` messages via `CommandService`, assigns job numbers, activates jobs in `ActiveJobRegistry`.
- **Stop**: Sends stop commands, deactivates jobs (which cleans up `DataService` buffers via `ActiveJobRegistry`).
- **Acknowledgement processing**: Tracks pending commands and processes backend responses.

`WorkflowController` (see `dashboard/workflow_controller.py`) is the interface between widgets and `JobOrchestrator`. It translates Pydantic model parameters into orchestrator calls and creates `WorkflowConfigurationAdapter` instances for the configuration UI.

## Plot Orchestration

`PlotOrchestrator` (see `dashboard/plot_orchestrator.py`) manages the plot grid lifecycle:

- Creates and removes plot grids (tab-level containers)
- Manages plot cells within grids (add, remove, configure)
- Subscribes to `JobOrchestrator` workflow events to create plots when jobs start
- Persists grid configurations via `ConfigStore`
- Loads grid templates for instrument-specific default layouts

`PlottingController` (see `dashboard/plotting_controller.py`) handles the mechanics of plot creation: finding compatible plotters for a given data shape, setting up data pipelines via `StreamManager`, and creating plotter instances.

`PlotDataService` holds per-plot shared state (Presenters with dirty flags) that is read by per-session `SessionPlotManager` instances during periodic updates.

## Threading Model

```mermaid
flowchart LR
    subgraph "Background Thread"
        UL["Update Loop<br>(DashboardServices)"]
        UL --> O["Orchestrator.update()"]
        UL --> SC["SessionRegistry.cleanup_stale()"]
    end

    subgraph "Per-Session Periodic Callback<br>(Tornado IOLoop)"
        SU["SessionUpdater.periodic_update()"]
        SU --> HB["Browser heartbeat check"]
        SU --> NQ["Poll NotificationQueue"]
        SU --> CH["Custom handlers<br>(plot pipe updates,<br>status widget refresh)"]
    end

    O -.->|writes| DS["DataService<br>(shared state)"]
    CH -.->|reads| DS
```

Two threading contexts exist:

1. **Background thread** (`orchestrator-update`): Runs `Orchestrator.update()` in a loop at ~5 Hz. Consumes Kafka messages and writes to `DataService`. Uses `ActiveJobRegistry.ingestion_guard()` to serialize against UI-thread job cleanup.

2. **Per-session Tornado callbacks**: Each browser session has a `SessionUpdater` that runs in the Tornado IOLoop at ~1 Hz. It batches all UI mutations inside `pn.io.hold()` + `doc.models.freeze()` to minimize Bokeh model recomputation.

## Session Management

```mermaid
flowchart TD
    Browser1["Browser Session A"] --> SU1["SessionUpdater A"]
    Browser2["Browser Session B"] --> SU2["SessionUpdater B"]
    SU1 --> SR["SessionRegistry"]
    SU2 --> SR
    SR --> Cleanup["Stale session cleanup<br>(heartbeat timeout)"]
```

Each browser session creates a `SessionUpdater` (see `dashboard/session_updater.py`) which:
- Registers with `SessionRegistry` for lifecycle tracking
- Embeds an invisible `HeartbeatWidget` that sends browser-side heartbeats via JavaScript
- Runs custom handlers (plot updates, status widgets) in the correct session/document context
- Batches all UI operations using `pn.io.hold()` + `doc.models.freeze()`

`SessionRegistry` (see `dashboard/session_registry.py`) tracks active sessions with heartbeat-based stale detection. Sessions are cleaned up when `pn.state.on_session_destroyed()` fires, or after a heartbeat timeout (defense-in-depth for browser crashes).

## Configuration Adapters

Configuration widgets use the `ConfigurationAdapter` pattern (see `dashboard/configuration_adapter.py`):

- `ConfigurationAdapter` is an abstract base providing: title, description, model class for parameters, available source names, aux source definitions, and a start action
- `WorkflowConfigurationAdapter` implements this for workflow start dialogs
- `PlotConfigurationAdapter` implements this for plot configuration modals
- `ConfigurationState` persists parameter choices across sessions via `ConfigStore`

The generic `ConfigurationWidget` (see `widgets/configuration_widget.py`) renders any adapter into a Panel form with source selection, parameter inputs, and a start button.

## Transport Abstraction

The `Transport` protocol (see `dashboard/transport.py`) abstracts message infrastructure:

- `DashboardKafkaTransport` (see `dashboard/kafka_transport.py`): Connects to Kafka, provides `MessageSource` (consumer) and `MessageSink` instances (for commands and ROI updates)
- `NullTransport`: No-op implementation for testing

Both return `DashboardResources` containing a `MessageSource`, a command `MessageSink`, and an ROI `MessageSink`.

## Key Widget Components

Widgets live in `dashboard/widgets/` and follow a pattern of receiving shared services in their constructor and registering periodic refresh handlers with `SessionUpdater`:

- `PlotGridTabs`: Tab container managing multiple plot grids, workflow configuration, and system status
- `WorkflowStatusListWidget`: Displays active workflow jobs and their status
- `SystemStatusWidget`: Shows session count, backend worker status, heartbeat info
- `ConfigurationWidget`: Generic form builder driven by `ConfigurationAdapter`
- `PlotConfigModal`: Modal dialog for configuring individual plot cells
