# ESSlivedata Concept Inventory

A complete inventory of concepts and behaviors a developer needs to understand to reason about the system. Each entry is a topic that would warrant its own documentation section.

---

## A. System-Wide Concepts

### A1. Service Topology and Data Flow

The system is a pipeline of independent processes communicating via Kafka. Upstream (raw instrument data) flows through four processing services (monitor_data, detector_data, data_reduction, timeseries) which publish results to downstream topics. The dashboard consumes downstream data for visualization and publishes commands/ROI definitions back upstream. Each processing service is a separate OS process with its own OrchestratingProcessor.

### A2. Timestamps and the Time Model

All data timestamps are nanoseconds since epoch (UTC), carried in `Message.timestamp`. They originate from the instrument's event formation unit (EFU) or slow-control system, not from wall clocks. The `SimpleMessageBatcher` uses data timestamps as its clock: batches close only when a message with a future timestamp arrives. Wall clock is used only for heartbeats (2s), metrics logging (30s), and sleep durations. Time coordinates (`start_time`, `end_time`) are added to workflow outputs for lag calculation in the dashboard.

### A3. Stream Identity Model

Messages are identified by `StreamId(kind: StreamKind, name: str)`. `StreamKind` is an enum (10 members) classifying the data type. The `name` field distinguishes specific sources (e.g., different detector banks). On the Kafka side, streams are identified by `InputStreamKey(topic, source_name)`. A `StreamLUT` maps Kafka coordinates to internal stream names, decoupling the system from Kafka topic naming.

### A4. Message Envelope (`Message[T]`)

The universal message type: `(timestamp, stream: StreamId, value: T)`. Frozen, generic, comparable by timestamp. Everything in the system flows through this envelope, from raw Kafka bytes to processed scipp DataArrays.

### A5. Instrument Abstraction and Registry

An `Instrument` dataclass defines everything about a supported instrument: detector names, monitor names, log stream attributes, source metadata, detector pixel numbers, and registered workflow specs. Each instrument lives in its own package under `config/instruments/`. The `InstrumentRegistry` holds loaded instruments. A service typically loads a single instrument.

### A6. Two-Phase Workflow Registration

Workflow registration is split into lightweight spec registration (Phase 1: happens at import time, registers `WorkflowSpec` with metadata, params models, and output templates) and heavy factory attachment (Phase 2: happens at runtime via `load_factories()`, imports scientific libraries and attaches factory callables). This avoids slow imports of `essdiffraction`, `essspectroscopy`, etc. at startup.

### A7. Configuration System (YAML + Jinja2)

Configuration files use YAML with Jinja2 templating. They live in `config/defaults/` (shared) and `config/instruments/{name}/` (per-instrument). The `LIVEDATA_ENV` environment variable selects dev/staging/production. `load_config(namespace, env)` loads and merges configs. Kafka broker addresses, consumer group settings, and topic names are all configured this way.

### A8. DEV vs PROD Stream Mappings

Every instrument defines both `DEV` and `PROD` stream mappings. DEV uses simplified topic naming (e.g., `dummy_detector`) suitable for local testing with fake producers. PROD uses ESS-specific conventions (e.g., `dream_detector_mantle`, EPICS PV names as source names, CBM monitor naming). The `--dev` flag selects DEV mode.

### A9. FlatBuffer Schemas

The system uses five FlatBuffer schemas from the `streaming_data_types` library: ev44 (neutron events), da00 (N-D arrays), f144 (scalar log data), ad00 (area detector images), and x5f2 (status heartbeats). Each has dedicated adapter and serialization code. The da00 schema is also used for all ESSlivedata-internal data (results, ROI definitions).

### A10. Scipp as the Data Lingua Franca

After Kafka deserialization, all scientific data is represented as `sc.DataArray` or `sc.DataGroup` (from the scipp library). Scipp provides labeled N-D arrays with physical units, bin-edge coordinates, and uncertainty propagation. Workflows consume and produce scipp objects. The dashboard converts scipp to HoloViews for visualization.

### A11. Error Handling Philosophy

Data processing errors in `Job.add()` produce warnings (job may still finalize). Finalization errors in `Job.get()` produce errors (job retries next cycle). Command routing mismatches (wrong instrument, job not found) are silently ignored (expected in multi-worker setups). Unhandled processor exceptions cause the service to send SIGINT to its main thread. In the Kafka layer, adaptation errors log-and-skip by default; production errors log via delivery callback with no retry.

---

## B. Backend Processing Concepts

### B1. Service-Processor-Workflow Pattern

Each backend service is a `Service` wrapping a single `Processor` (always `OrchestratingProcessor` in production). The processor orchestrates the full pipeline: message separation (commands vs data), batching, preprocessing, job routing, result computation, and publishing. The `Service` manages the OS-level lifecycle (signal handling, threading, graceful shutdown).

### B2. Message Batching and the Data-Driven Clock

`SimpleMessageBatcher` groups messages into fixed-length time windows based on data timestamps. Batches close when a message with a timestamp beyond the current window arrives. This means the system's "clock" is the data itself: if data stops flowing, the last batch never closes. The `NaiveMessageBatcher` (used by timeseries) is the exception: it returns all messages immediately.

### B3. Accumulators and Preprocessing

Accumulators (`Accumulator[T, U]`) are per-stream stateful objects that transform and accumulate raw messages into workflow-consumable form. A `PreprocessorFactory` creates the right accumulator for each `StreamId`. Key accumulators: `Cumulative` (sums DataArrays), `CollectTOA` (concatenates event arrays), `ToNXevent_data` (builds NeXus event format), `ToNXlog` (builds NeXus log format with doubling-capacity buffer), `LatestValueHandler` (keeps latest value only).

### B4. Window vs Cumulative Accumulation Semantics

Some accumulators clear on `get()` (window semantics: you get the delta since last read). Others grow indefinitely (cumulative semantics: you get the full history). `Cumulative(clear_on_get=True)` and `CollectTOA` use window semantics. `ToNXlog` uses cumulative semantics. In detector view workflows, `NoCopyWindowAccumulator` vs `NoCopyAccumulator` embody this same distinction.

### B5. Jobs and the Job Lifecycle

A `Job` wraps a `Workflow` instance with routing metadata. Jobs are created by `JobManager` when a `WorkflowConfig` message arrives from the dashboard. Lifecycle: `scheduled -> active -> finishing -> stopped`, with `warning` and `error` side-states. Jobs are activated when their scheduled start time arrives (based on data timestamps). Jobs in `finishing` state still compute one final result before stopping.

### B6. Primary Data Gating

Results are only computed for jobs that received primary data in the current batch. Auxiliary-only updates accumulate silently without triggering computation. This prevents unnecessary workflow runs when only context data (e.g., ROI definitions, log values) changes.

### B7. Workflow Protocol (accumulate/finalize/clear)

The `Workflow` protocol has three methods: `accumulate(data_dict, start_time, end_time)` feeds preprocessed data, `finalize()` computes and returns results, `clear()` resets state. This matches `ess.reduce.streaming.StreamProcessor`. Key distinction: accumulators preprocess raw messages; workflows operate on the preprocessed output.

### B8. StreamProcessorWorkflow and the Sciline Bridge

`StreamProcessorWorkflow` adapts a Sciline-based `StreamProcessor` to the `Workflow` protocol. It manages the split between dynamic data (per-batch, triggers accumulation) and context data (infrequently changing, updates pipeline parameters). This is how the ROI system, log data, and other auxiliary inputs are handled differently from high-frequency detector/monitor data.

### B9. Namespace Scoping and Multi-Service Routing

All processing services receive the same command messages, but each only handles workflows matching its `active_namespace` (set by `DataServiceBuilder` to the service name, e.g., `detector_data`). `JobFactory.create()` raises `DifferentInstrument` for mismatches, which is silently caught. This enables a single commands topic to control all services.

### B10. Command-Acknowledge Pattern

The dashboard sends commands (`WorkflowConfig`, `JobCommand`) via the `livedata_commands` topic. Backend services process them and send `CommandAcknowledgement` (ACK/ERR) via the `livedata_responses` topic. The `message_id` field correlates requests with responses. The dashboard's `PendingCommandTracker` tracks in-flight commands and their ACK status.

### B11. Status Heartbeats

Backend services publish two types of status messages (x5f2 FlatBuffer) every 2 seconds: `ServiceStatus` (service-level: state, worker ID, active job count, messages processed) and `JobStatus` (per-job: state, workflow ID, timing, errors). The dashboard's `ServiceRegistry` and `JobService` consume these for monitoring.

### B12. Result Publishing and Unrolling

Workflow results are `sc.DataGroup` objects (multiple named outputs). The `UnrollingSinkAdapter` "unrolls" each DataGroup into individual Kafka messages, one per output. Each message's stream name encodes a `ResultKey(workflow_id, job_id, output_name)` as JSON.

---

## C. Kafka Layer Concepts

### C1. Message Adapter Pipeline

Raw Kafka messages pass through a chain of adapters: `RouteByTopicAdapter` dispatches to per-topic adapters, which may include `RouteBySchemaAdapter` (for topics carrying multiple schemas, like beam monitors with ev44 and da00). Each adapter chain deserializes FlatBuffers and transforms to domain types. The `RoutingAdapterBuilder` constructs the full adapter graph.

### C2. Background Message Consumption

`BackgroundMessageSource` polls Kafka in a daemon thread, buffering batches in a queue. The processing thread drains the queue on each cycle. Queue overflow drops the oldest batch (stale data is less valuable than fresh data). A circuit breaker stops after N consecutive errors. Health monitoring tracks thread liveness, time since last consume, and consumer lag.

### C3. Commands Topic and Message Keys

The `livedata_commands` topic uses `cleanup.policy=delete` with 48-hour retention. Messages are published with a `ConfigKey` as the Kafka message key (originally designed for log compaction, but compaction was disabled because workflow start commands on the same topic caused duplicate/zombie jobs on service restart). Backend consumers use `auto.offset.reset=latest`, so they only see commands sent while running. The dashboard overrides this to `earliest` for its own consumption (to restore previous workflow configurations within the retention window). `ConfigProcessor` performs manual deduplication (keeps latest per key/source pair) since Kafka no longer does this.

### C4. Consumer Group Model (Broadcast)

Each consumer instance gets a unique group ID (`{group}_{uuid4()}`), preventing consumer group rebalancing. This is intentional: every service instance sees all messages (broadcast/fan-out model), because each service handles a different namespace. Explicit partition assignment is used alongside subscribe for immediate assignment.

### C5. Serialization Formats

Outbound: results use da00, commands use JSON with `ConfigKey` as Kafka key, responses use JSON, status uses x5f2. The `scipp_da00_compat` module handles bidirectional scipp-to-da00 conversion, including datetime encoding and dtype mapping. The `ad00_to_scipp` conversion is one-way (input only).

---

## D. Dashboard Concepts

### D1. Session Management and Multi-Session

Each browser tab creates an independent session. `SessionRegistry` tracks active sessions with heartbeats. `SessionUpdater` runs a 500ms periodic callback per session that drives all UI updates. Sessions have their own `SessionLayer` per plot layer, holding per-session HoloViews `Pipe` objects for reactive data push. Stale sessions (no heartbeat for 30s) are cleaned up.

### D2. Two-Stage Plotter/Presenter Pattern

Plotters use a two-stage pattern: `Plotter.compute(data)` (stage 1) runs on the background thread, processing data into a cacheable form. `Presenter.render(state)` (stage 2) runs on the session thread, converting cached state to HoloViews elements and pushing to `Pipe`. This separates expensive computation from per-session rendering. A plotter has many presenters (one per session).

### D3. Dirty-Flag and Version-Based Polling

Cross-thread communication uses dirty flags and version counters instead of direct callbacks. The background thread sets dirty flags on plotters when data arrives. The periodic callback checks `is_dirty()` and calls `pipe.send()` only when needed. Status widgets compare `_last_state_version` with the current version from shared services, doing full rebuilds only on change.

### D4. Bokeh Model Freeze Batching

`doc.models.freeze()` batches Bokeh model graph recomputation. `pn.io.hold()` batches Panel document events. The `SessionUpdater._batched_update()` context wraps all per-session updates in both, collapsing N recomputes into 1. This is the primary performance optimization for the dashboard.

### D5. Data Flow: Backend to UI

Kafka messages arrive via `DashboardKafkaTransport` -> `Orchestrator.update()` (background thread) routes to `DataService` -> `StreamManager` callbacks fire -> `TemporalBufferManager` stores data -> `UpdateExtractor` extracts -> `Plotter.compute()` processes -> dirty flag set -> periodic callback -> `Presenter.render()` -> `Pipe.send()` -> HoloViews `DynamicMap` updates.

### D6. Transport Abstraction

The `Transport` protocol abstracts Kafka communication, with `DashboardKafkaTransport` (production) and `NullTransport` (testing). The transport handles bidirectional message flow: subscribing to downstream data/status/responses and publishing commands/ROI definitions upstream.

### D7. Temporal Buffers and Extractors

`TemporalBuffer` stores time-series data with efficient append (doubling-capacity `VariableBuffer`). `SingleValueBuffer` stores only the latest value. `UpdateExtractor` controls what data the plotter sees: `LatestValueExtractor` (single frame), `WindowAggregatingExtractor` (time-windowed aggregation with nansum/nanmean), `FullHistoryExtractor` (complete timeseries). The `TemporalBufferManager` switches buffer types based on extractor requirements.

### D8. Job Orchestrator and Workflow Lifecycle

`JobOrchestrator` manages the frontend view of workflow lifecycle. It tracks `WorkflowState` (staging, running, stopped) and `JobConfig` per workflow. Staging accumulates parameter changes without sending to backend. Committing sends `WorkflowConfig` messages via `CommandService`. Stopping sends `JobCommand(action=stop)`. The `PendingCommandTracker` tracks ACK/ERR responses.

### D9. Plot Orchestrator and Grid/Cell/Layer Management

`PlotOrchestrator` manages the plot grid layout. A grid contains cells, each cell contains layers. A layer binds a workflow output to a plotter. `PlotConfig` defines a cell's layers, each with a `DataSourceConfig` (workflow_id, output_name, source_names) and plotter params. `ConfigStore` persists grid configurations to disk (JSON). Grid templates (YAML) provide pre-configured layouts per instrument.

### D10. Plotter Registry and Spec-Based Selection

`PlotterRegistry` maps plotter names to `PlotterSpec` (capabilities, data requirements, params model). When a user adds a plot, the system filters available plotters by matching spec requirements against the workflow output's template (dimensionality, coords, shape). This enables automatic plotter recommendation before data arrives. Categories: DATA plotters (process data) and STATIC plotters (overlays from params only).

### D11. Configuration Widget System (Pydantic to Widgets)

Pydantic models are automatically rendered as Panel widgets via `ModelWidget` and `ParamWidget`. Nested models become collapsible cards. Field types map to widgets: `float`->`FloatInput`, `Enum`->`Select`, `bool`->`Checkbox`, `Color`->`ColorPicker`, etc. This provides type-safe configuration with validation for both workflow params and plot display params.

### D12. ROI System (End-to-End)

ROIs flow from dashboard to backend and back. Interactive tools (`BoxEdit`, `PolyDraw`) on HoloViews plots let users draw rectangles and polygons. The `ROIPublisher` serializes these as `sc.DataArray` and publishes to the `livedata_roi` topic. Backend detector workflows receive ROIs as auxiliary context data, precompute bounds/masks, extract spectra, and echo readback geometries in results. The dashboard renders both request (editable) and readback (confirmed) overlays. Each job gets its own ROI stream via `{source_name}/{job_number}/roi_{shape}` naming.

### D13. Overlay and Layer Composition

Plots support multiple overlaid layers. `Overlay1DPlotter` chains: e.g., `image_2d` (data) + `roi_readback` (confirmed shapes) + `roi_request` (editable shapes). Static overlays (rectangles, lines) can be layered on data plots. The `OVERLAY_PATTERNS` registry defines valid chain compositions.

### D14. Correlation Histograms

Multi-source correlation plots that histogram one data stream against another as axis values. `DataSubscriber` assembles multi-role data (PRIMARY, X_AXIS, Y_AXIS). `CorrelationHistogramPlotter` builds lookup tables to correlate primary data with axis data. Supports 1D and 2D correlation histograms with configurable binning and normalization (counts vs per-second).

### D15. Dashboard Threading Model

Three thread types: (a) Main thread (Panel/Tornado event loop) serves HTTP and runs periodic callbacks (all widget updates). (b) Background update thread runs `Orchestrator.update()` loop (~200ms), polls Kafka, routes messages, runs `Plotter.compute()`. (c) Kafka polling thread inside `BackgroundMessageSource`. Thread safety via locks on `SessionRegistry`, `PlotDataService`, `NotificationQueue`, `ConfigStore`, and version-counter-based change detection.

### D16. Plot Configuration Wizard

A 3-step wizard for adding plots: (a) Select namespace/workflow/output, (b) Select plotter type (filtered by spec compatibility) and optionally correlation axes, (c) Select source names and configure params. Edit mode starts at step 3 with pre-populated values. Static overlay support via synthetic `STATIC_OVERLAY_WORKFLOW`.

### D17. Autoscaling

`BidirectionalAutoscaler` manages dynamic axis ranges. Grows when data exceeds range (threshold: 10% outside). Shrinks when range is mostly empty (threshold: 50% unused). Returns `None` when no rescale needed. Used by plotters for both axis and color ranges.

---

## E. Detector View Concepts

### E1. Geometric vs Logical Projections

Two paradigms for mapping detector pixels to screen coordinates: Geometric projections use calibrated 3D positions from NeXus geometry files, projecting to 2D planes (`xy_plane` or `cylinder_mantle_z`). Logical projections use array operations (fold, flatten, slice) based on the detector's logical structure. Geometric projections support pixel noise replicas for smoother visualization.

### E2. Coordinate Modes (TOA/TOF/Wavelength)

Detector views can histogram by time-of-arrival (raw), time-of-flight (requires chopper info), or wavelength (requires full calibration). The coordinate mode determines which upstream provider converts raw events. Downstream histogram/image/ROI computation is generic over the coordinate type.

### E3. Detector Data Sources

Three strategies for providing empty detector geometry: `NeXusDetectorSource` (loads from NeXus file via Pooch), `DetectorNumberSource` (from explicit pixel ID array), `InstrumentDetectorSource` (from instrument config). NeXus geometry files are versioned by date and fetched/cached via Pooch.

## F. Instrument-Specific Concepts

### F1. Per-Instrument File Structure

Each instrument follows a 3-file pattern: `specs.py` (lightweight spec registration at import time), `streams.py` (Kafka topic mappings for DEV and PROD), `factories.py` (heavy initialization, imports scientific libraries). Some add `views.py` for logical view transforms.

### F2. Detector Bank Merging (Bifrost)

Bifrost has 45 detector banks (5 arcs x 9 channels x 3 tubes). The `merge_detectors` flag in `Ev44ToDetectorEventsAdapter` merges all banks into a single `unified_detector` stream. The factory then folds the flat pixel array into (arc, tube, channel, pixel) dimensions.

### F3. NeXus Geometry Versioning

Detector geometry files are versioned by date (e.g., `geometry-dream-2025-01-01.nxs`). `get_nexus_geometry_filename(instrument, date)` selects the appropriate file for a given date. Files are fetched via Pooch and cached locally. MD5 checksums ensure integrity.

### F4. Area Detector Workflows

Area detectors (ad00 schema) provide dense 2D images rather than event lists. `AreaDetectorView` accumulates frames, applies optional logical transforms (folding, downsampling), and computes cumulative and delta (current) images. Used by Dummy (area_panel) and TBL (Orca camera).

### F5. Reduction Workflows

Instrument-specific scientific reduction pipelines: DREAM powder diffraction (I(d), I(d,2theta)), Bifrost spectroscopy (Q-E maps, detector ratemeter), LOKI SANS (I(Q)). These use the `StreamProcessorWorkflow` bridge to Sciline. Auxiliary data (monitors, log values, rotation angles) is fed as context.

### F6. Grid Templates

Pre-configured dashboard layouts in YAML, per instrument. Define grid dimensions, cell geometry (row/col spans), and layer configurations (workflow, output, plotter, params). Support multi-layer cells (e.g., detector image + static overlay rectangles). The `PlotGridManager` widget allows applying, editing, and exporting templates.

---

## G. Behavioral/Cross-Cutting Concerns

### G1. Out-of-Order and Late Messages

The `SimpleMessageBatcher` does not drop late messages. Instead, they are included in the next batch (the batch's start_time may "lie" to accommodate them). Within a batch, messages are sorted by timestamp before preprocessing. The accumulator pattern naturally handles out-of-order data by treating each batch atomically.

### G2. Empty Batch Handling

When a batch period elapses with no messages (triggered by a future-timestamped message arriving), `SimpleMessageBatcher` returns an empty batch. This advances the time window and allows jobs to transition (activate, finish) even during data gaps.

### G3. Stale Data and Queue Overflow

`BackgroundMessageSource` drops the oldest batch if its queue fills. This is intentional for live data: stale data is less valuable than fresh data. The queue overflow counter is tracked in metrics.

### G4. Graceful Shutdown

On SIGTERM/SIGINT, `Service` notifies the processor (`shutdown()`), which publishes a `stopping` heartbeat. After the processing loop exits, `report_stopped()` publishes a `stopped` heartbeat. The dashboard distinguishes graceful shutdown (stopping->stopped) from stale disappearance (last heartbeat ages out).

### G5. Commands Topic as Configuration State

The `livedata_commands` topic was originally designed as a compacted topic (durable configuration store), but compaction is currently disabled (`cleanup.policy=delete`, 48h retention). Message keys (`ConfigKey`) are still set but Kafka does not deduplicate by key. Backend services consume from `latest` offset, so they do not replay historical configuration. The dashboard consumes from `earliest` to restore previous workflow configurations within the retention window. `ConfigProcessor` manually deduplicates by key (keeps latest per key/source pair) to handle the duplicates that compaction would otherwise remove.

### G6. Result Naming Convention (ResultKey)

Workflow outputs are identified by a JSON-serialized `ResultKey(workflow_id, job_id, output_name)` stored in the Kafka message's `StreamId.name`. The dashboard parses this to route results to the correct plot layer. The `UnrollingSinkAdapter` constructs these keys during DataGroup unrolling.

### G7. Multi-Output Workflows and Output Templates

Workflows declare their outputs via `WorkflowOutputsBase` subclasses with `sc.DataArray` default factories. These templates (empty DataArrays with correct dims, coords, units) enable the dashboard to select plotters and configure grids before any data arrives. Outputs can be 0D (scalar + time coord = timeseries), 1D (histogram), 2D (image), or 3D (slicer).

### G8. f144 Attribute Registry

The `f144_attribute_registry` on `Instrument` maps log stream source names to metadata (Pydantic `LogDataAttributes`: unit, dtype, optional `value_min`/`value_max`). Only log streams present in this registry are preprocessed; unknown sources are dropped. This controls which slow-control parameters are available in the system.
