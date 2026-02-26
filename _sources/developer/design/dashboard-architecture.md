# Livedata Dashboard Architecture

## Table of Contents

1. [Overview](#overview)
2. [System Context: Dashboard and Kafka Integration](#system-context-dashboard-and-kafka-integration)
3. [High-Level Architecture](#high-level-architecture)
4. [Configuration Architecture](#configuration-architecture)
5. [Data Flow Architecture](#data-flow-architecture)
6. [The ConfigBackedParam Mechanism](#the-configbackedparam-mechanism)
7. [Background Threading Architecture](#background-threading-architecture)
8. [Extension Points](#extension-points)

## Overview

The Livedata dashboard is a real-time data visualization system that follows a layered architecture with clear separation of concerns between presentation, application logic, and infrastructure. The system is designed for live display of raw and processed detector data with configurable processing parameters, using dependency injection patterns for testability and maintainability.

A key architectural principle is the separation between **Pydantic models** (used for Kafka message validation and backend communication) and **Param models** (used for GUI widgets and user interaction).

The dashboard processes 1-D and 2-D data displayed using Holoviews with update rates on the order of 1Hz. Data updates are received via Kafka streams, and user controls result in configuration updates published to Kafka topics for backend consumption.

## System Context: Dashboard and Kafka Integration

The Livedata dashboard operates within a Kafka-based system, interacting with multiple backend services via Kafka topics for both data and configuration.

```mermaid
flowchart TD
    ConfigTopic(["Livedata Config Topic"])
    DataTopic(["Livedata Data Topic"])
    ECDCTopic(["ECDC Topics"])

    subgraph BackendServices["Livedata Backend Services"]
        MonitorData["monitor_data service"]
        DetectorData["detector_data service"]
        DataReduction["data_reduction service"]
    end

    DashboardApp["Livedata Dashboard"]

    %% Raw data from ECDC topics
    ECDCTopic --> BackendServices

    %% Data publishing
    MonitorData -- Publishes --> DataTopic
    DetectorData -- Publishes --> DataTopic
    DataReduction -- Publishes --> DataTopic

    %% Data consumption by dashboard
    DataTopic -- Feeds --> DashboardApp

    %% Config flows
    ConfigTopic -- Consumed by --> MonitorData
    ConfigTopic -- Consumed by --> DetectorData
    ConfigTopic -- Consumed by --> DataReduction
    ConfigTopic -- Consumed by --> DashboardApp
    DashboardApp -- Publishes config --> ConfigTopic
    DataReduction -- Publishes config (workflow specs) --> ConfigTopic

    classDef kafka fill:#fff3e0,stroke:#ef6c00,color:#e65100;
    classDef backend fill:#e3f2fd,stroke:#1976d2,color:#0d47a1;
    classDef dashboard fill:#ede7f6,stroke:#7b1fa2,color:#4a148c;
    class ConfigTopic,DataTopic,ECDCTopic kafka;
    class BackendServices,MonitorData,DetectorData,DataReduction backend;
    class DashboardApp dashboard;
```

**Key Points:**
- Backend Services publish data streams to a single Kafka topic
- The Dashboard consumes this data topic, feeding into internal `DataService` components
- The Dashboard both publishes to and consumes from the config topic
- All backend services consume the config topic for configuration updates
- The `data_reduction` service can publish configuration messages (workflow specs, status)

## High-Level Architecture

```mermaid
graph TD
    subgraph "Livedata Backend Services"
        K1[Kafka Data Streams]
        K2[Kafka Config Topic]
    end

    subgraph "Infrastructure Layer"
        KT[KafkaTransport]
        TMH["ThrottlingMessageHandler<br>(prevents flooding Kafka)"]
        MB["BackgroundMessageBridge<br>(prevents blocking UI)"]
        MS[MessageSource]
        DF[DataForwarder]
    end

    subgraph "Application Layer"
        CS[ConfigService]
        WCS["WorkflowConfigService<br>(adapts ConfigService)"]
        DS[["DataService(s)"]]
        DSU[["DataSubscriber(s)"]]
        PM[["ConfigBackedParam(s)<br>(param.Parameterized)"]]
        WC[WorkflowController]
        SM[["StreamManager(s)"]]
    end

    subgraph "Presentation Layer"
        W1[["Plot(s)<br>(Holoviews)"]]
        UI["Workflow Widgets<br>(Panel)"]
        W2[["Config Widgets<br>(Panel)"]]
    end

    DSU -- holoviews.streams.Pipe --> W1
    WCS <--> WC
    CS <--> WCS
    DS <-.-> WC
    WC <--> UI
    K1 --> MS
    K2 <--> KT
    MS --> DF
    DF --> DS
    DS --> DSU
    SM -.->|creates|DSU
    DSU -.->|subscribes|DS
    PM -.->|subscribes|CS
    WCS -.->|subscribes|CS
    MB <--> CS
    CS <-- Pydantic --> PM
    PM <--> W2
    KT <--> TMH
    TMH <--> MB

    classDef kafka fill:#fff3e0,stroke:#ef6c00,color:#e65100;
    classDef infra fill:#e3f2fd,stroke:#1976d2,color:#0d47a1;
    classDef app fill:#ede7f6,stroke:#7b1fa2,color:#4a148c;
    classDef prese fill:#ede7f6,stroke:#7b1fa2,color:#4a148c;
    class K1,K2 kafka;
    class KT,KT,TMH,MB,MS,DF infra;
    class CS,WCS,DS,DSU,PM,WC,SM app;
    class W1,UI,W2 prese;
```

### Component Overview

The architecture is structured in three main layers:

- **Infrastructure Layer**: Manages Kafka integration and external message sources
- **Application Layer**: Contains business logic, orchestration, and data management
- **Presentation Layer**: Handles GUI components and user interaction

## Configuration Architecture

The dashboard implements a configuration system that maintains a clear separation between frontend widgets (using Param for interactive controls) and backend validation/communication (using Pydantic models throughout). The `ConfigService` operates entirely with Pydantic models, ensuring type safety and validation consistency.

### Two-Way Configuration Flow

```mermaid
sequenceDiagram
    participant PW as Param Widget
    participant PM as Param Model
    participant CS as ConfigService
    participant SV as Schema Validator
    participant KB as KafkaBridge
    participant KT as Kafka Topic
    participant BE as Backend Services

    Note over PW,BE: User Changes Parameter
    PW->>PM: User input (num_edges=150)
    PM->>PM: Create Pydantic model from Param state
    PM->>CS: update_config(key, pydantic_model)
    CS->>CS: Validate isinstance(value, BaseModel)
    CS->>SV: validate(key, pydantic_model)
    CS->>KB: publish(key, model.model_dump(mode='json'))
    KB->>KT: JSON message
    KT->>BE: Backend consumes

    Note over PW,BE: Backend/Remote Updates Configuration
    BE->>KT: Updated config (JSON)
    KT->>KB: JSON message
    KB->>CS: incoming JSON message
    CS->>SV: deserialize(key, json_data) -> Pydantic model
    CS->>PM: callback(pydantic_model)
    PM->>PM: Extract dict from Pydantic model
    PM->>PW: param.update(**model.model_dump())
    PW->>PW: Widget reflects new state
```

### ConfigBackedParam Translation Mechanism

The `ConfigBackedParam` serves as a **translation layer** that:

1. **Outbound (Param → Pydantic)**: Creates Pydantic models from Param state using `self.schema.model_validate(kwargs)`
2. **Inbound (Pydantic → Param)**: Extracts data from Pydantic models using `model.model_dump()`
3. **Schema Binding**: Connects each Param model to its corresponding Pydantic schema
4. **Validation**: Ensures all data flowing through ConfigService is properly validated

## Data Flow Architecture

### Real-Time Data Flow

```mermaid
sequenceDiagram
    participant K as Kafka Stream
    participant MS as MessageSource
    participant O as Orchestrator
    participant DF as DataForwarder
    participant DS as DataService
    participant S as Subscribers
    participant UI as GUI Components

    K->>MS: Raw detector/monitor data
    MS->>O: Batch messages
    O->>DF: Forward with stream name
    Note over O,DF: Transaction batching
    DF->>DS: Store by DataKey
    DS->>S: Notify subscribers
    S->>UI: Update visualizations
```

## The ConfigBackedParam Mechanism

### Purpose and Role

`ConfigBackedParam` is a key architectural component for simple configuration widgets. It serves as a dedicated translation layer and per-widget controller, bridging the gap between:

- **Param models** (`param.Parameterized`): Used for GUI widgets and user interaction
- **Pydantic models**: Used for backend validation, serialization, and communication

This mechanism enables a clean, testable, and maintainable way to synchronize widget state with configuration state, without leaking infrastructure details into the presentation layer.

### How It Works

- Each simple configuration widget is backed by a `ConfigBackedParam` subclass
- The `ConfigBackedParam`:
  - Registers the relevant Pydantic schema with `ConfigService`
  - Subscribes to config updates for its key, updating the widget state when changes arrive
  - Propagates user changes from the widget to `ConfigService` by translating Param state to a Pydantic model

```mermaid
sequenceDiagram
    participant ConfigService
    participant ConfigBackedParam
    participant PanelWidget

    ConfigService->>ConfigBackedParam: Notify config update
    ConfigBackedParam->>PanelWidget: Update widget state
    PanelWidget->>ConfigBackedParam: User changes widget
    ConfigBackedParam->>ConfigService: Update config
```

### Architectural Implications

- **Localized Coupling**: `ConfigBackedParam` knows about both Param and Pydantic models, but this coupling is intentional and limited to the translation layer
- **No Architectural Problem**: This is not problematic coupling, but a necessary and well-encapsulated translation between two distinct model types
- **Testability**: The translation logic is isolated and can be tested independently
- **Extensibility**: More complex workflows can use a centralized controller, while simple controls benefit from this lightweight mechanism

## MVC Pattern Analysis

### Subscription and Notification Flow

```mermaid
sequenceDiagram
    participant ConfigService
    participant DataService
    participant WorkflowController
    participant ConfigBackedParam
    participant PanelWidgets

    ConfigService->>WorkflowController: Notify config update
    DataService->>WorkflowController: Notify data update
    WorkflowController->>PanelWidgets: Notify relevant update
    PanelWidgets->>WorkflowController: User actions (start/stop workflow)
    WorkflowController->>ConfigService: Update config
    WorkflowController->>DataService: Update data (e.g., cleanup)

    ConfigService->>ConfigBackedParam: Notify config update (simple controls)
    ConfigBackedParam->>PanelWidgets: Notify widget update
    PanelWidgets->>ConfigBackedParam: User changes widget (simple controls)
    ConfigBackedParam->>ConfigService: Update config (simple controls)
```

### Analysis

**Strengths:**
- Clear separation of concerns with well-defined responsibilities
- Testability through controller and widget isolation using fakes
- Maintainability with centralized business logic
- Extensibility for future requirements
- Efficient handling of simple controls via `ConfigBackedParam`

**Potential Pitfalls:**
- Controller bloat as it mediates more services
- State synchronization challenges between controller and services
- Subscription complexity with multiple layers

**Anti-Patterns Avoided:**
- No leaky abstractions between views and services
- No tight coupling between controllers and specific GUI frameworks
- No direct service access from views

## Background Threading Architecture

### KafkaBridge Threading Model

```mermaid
sequenceDiagram
    participant GUI as GUI Thread
    participant KB as KafkaBridge
    participant BT as Background Thread
    participant K as Kafka

    Note over GUI,K: Startup
    GUI->>KB: start()
    KB->>BT: spawn thread
    BT->>K: subscribe to topic

    Note over GUI,K: Publishing (Non-blocking)
    GUI->>KB: publish(key, value)
    KB->>KB: queue.put()
    BT->>KB: queue.get()
    BT->>K: produce message

    Note over GUI,K: Consuming (Batched)
    K->>BT: poll messages
    BT->>KB: incoming_queue.put()
    GUI->>KB: process_incoming_messages()
    KB->>GUI: pop_message() × N
```

### Message Processing Strategy

The KafkaBridge implements several optimizations:

1. **Batched Processing**: Consumes up to `max_batch_size` messages per poll
2. **Timed Polling**: Only checks for incoming messages at specified intervals
3. **Queue-based Communication**: Non-blocking queues between GUI and background threads
4. **Smart Idle Handling**: Minimal CPU usage when no messages are available

## Extension Points

### Adding New Configuration Parameters

1. **Create Pydantic Model** (Backend validation):
```python
class NewParam(BaseModel):
    value: float
    enabled: bool
```

2. **Create Param Model** (GUI widget):
```python  
class NewParamWidget(ConfigBackedParam):
    value = param.Number(default=1.0)
    enabled = param.Boolean(default=True)

    @property
    def schema(self) -> type[NewParam]:
        return NewParam
```

3. **Subscribe to ConfigService**:
```python
widget.subscribe(config_service)
```

### Adding New Data Types

1. Create new `DataKey` subclass
2. Implement corresponding `DataSubscriber`
3. Register subscriber with appropriate `DataService`

### Adding New Visualizations

1. Create new subscriber implementing `DataSubscriber`
2. Register with appropriate `DataService`  
3. Implement visualization using Holoviews/Panel
