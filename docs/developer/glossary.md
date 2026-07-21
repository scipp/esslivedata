# Glossary

Authoritative definitions of the terms used in ESSlivedata code and documentation.
When code, another document, and this glossary disagree, one of them has a bug:
fix it or file an issue.
References name modules (under `src/ess/livedata/`) rather than line numbers.

Terms are grouped into cross-cutting, backend, and dashboard sections.
The final section, [One word, several meanings](#one-word-several-meanings),
disambiguates overloaded words — consult it before introducing a new name.

## Cross-cutting

### Identity and keying

The naming stack, from the Kafka wire inwards (see ADR 0004 and
`design/stream-keying.md`):

- **Instrument** — an ESS beamline, and the central per-instrument configuration
  object (`config/instrument.py`) bundling detector/monitor names, streams,
  and the workflow factory. Registered in `config/instruments/`.
- **Topic** — a Kafka topic name (plain `str` alias in `kafka/stream_mapping.py`).
  The fixed per-instrument livedata topics (commands, data, responses, roi,
  status) are declared in `LivedataTopics`.
- **Source name** — the producer-declared name inside a FlatBuffers message
  (EV44/F144 field). `(topic, source_name)` (`InputStreamKey`) is the raw Kafka
  identity of a message and exists only at the Kafka boundary.
- **Stream name** (canonical stream name) — the instrument-facing routing handle,
  a NeXus-derived name resolved on ingress. All non-boundary code keys inputs by
  it (ADR 0004). In job context it is often called `source_name` — e.g.
  `JobId.source_name`, `WorkflowSpec.source_names` — which refers to the
  canonical stream name, *not* the raw Kafka source name.
- **StreamId** — internal stream key `(kind: StreamKind, name)` (`core/message.py`),
  isolating all code from Kafka topic names.
- **StreamKind** — enum of stream categories (`core/message.py`):
  `monitor_counts`, `monitor_events`, `detector_events`, `area_detector`, `log`,
  `device`, the `livedata_*` topics, `run_control`, `unknown`.
- **StreamMapping** — the boundary lookup
  `(topic, source_name) → stream name` (`kafka/stream_mapping.py`).
- **Synthesized stream** — a stream emitted in-process (`topic is None`), never
  read from Kafka, e.g. a chopper cascade or merged device stream (ADR 0001).

### Workflow and job identity

- **Workflow** — the scientific reduction logic: a protocol with
  `accumulate`/`finalize`/`clear` (`handlers/workflow_factory.py`), usually
  wrapping an `ess.reduce.streaming.StreamProcessor`. A workflow *runs as* a Job.
- **WorkflowId** — `(instrument, name, version)`; string form
  `instrument/name/version` (`config/workflow_spec.py`).
- **WorkflowSpec** — the declarative description of a workflow: title,
  source names, aux sources, parameter and output models
  (`config/workflow_spec.py`). Purely declarative; contains no runtime values.
- **WorkflowConfig** — runtime parameter *values* for a spec, sent as a command
  to start a job. Currently conflates "configure" and "start" (issue #445).
- **Job** — a running instance of a workflow, bound to a `JobId` and its input
  streams (`core/job.py`).
- **JobNumber** — a `uuid.UUID` minted per commit; the persistent identity
  component of a job and the generation marker (ADR 0007/0008).
- **JobId** — `(source_name, job_number)`; string form `source_name/job_number`.
- **Generation** — the epoch defined by one commit's `job_number`: results
  stamped with an older job number belong to a previous generation and are
  filtered out; a new generation clears buffers (ADR 0007/0008).
- **ResultKey** — wire key of one workflow output:
  `(workflow_id, job_id, output_name)`. Embeds the per-commit job number.
- **DataKey** — stable identity `(workflow_id, source_name, output_name)`,
  i.e. a ResultKey with the job number stripped. The dashboard data plane and
  NICOS derived devices (ADR 0006) key by DataKey, not ResultKey.
- **OutputView** — user-facing presentation of a workflow output, bundling
  backend output fields by window role (`since_start`/`per_update`)
  (`config/workflow_spec.py`).

### Data kinds and services

- **Backend services** — the four job-based workers: `monitor_data`,
  `detector_data`, `data_reduction`, `timeseries`
  (`services/`; run as `python -m ess.livedata.services.<name>`).
- **Monitor / detector / area detector / log / device data** — the domain data
  kinds, mirrored by `StreamKind`. *Logdata* (F144 log streams) is the input
  kind consumed by the *timeseries* service; the fake producer is accordingly
  named `fake_logdata`.
- **Fake services** — synthetic producers for Kafka-based local demos:
  `fake_monitors`, `fake_detectors`, `fake_logdata`.
- **Dev mode** (`--dev`) — simplified Kafka topic structure compatible with the
  fake producers.
- **Transport** — the dashboard's backend selector: `none` (UI only, workflows
  stay pending), `fake` (in-process fake backend, no Kafka), `kafka` (real
  backend). Also the abstraction name in `dashboard/transport.py`.
- **Run control** — `RunStart`/`RunStop` messages from the ESS filewriter topic
  (`core/message.py`), driving job schedule transitions.
- **Heartbeat** — periodic (~2 s) status publication from each backend worker:
  a `ServiceStatus` envelope carrying per-job `JobStatus` entries
  (`core/job.py`). The dashboard adopts running jobs from heartbeats (ADR 0008).

## Backend

### Service layer

- **Service** — top-level lifecycle manager (`core/service.py`): owns a worker
  thread that calls a Processor in a poll loop, handles signals and shutdown.
- **Processor** — the protocol a Service drives: `process()` + `finalize()`
  (`core/processor.py`). `IdentityProcessor` is the passthrough used by fake
  producers.
- **OrchestratingProcessor** — the Processor implementation for job-based
  services (`core/orchestrating_processor.py`): pulls messages, splits
  command/run-control/data, preprocesses batches, drives the JobManager, and
  publishes results and heartbeats.
- **ServiceState** — worker lifecycle enum: `starting`/`running`/`stopped`/`error`.
- **Service name** — string identity of a backend worker kind
  (`'data_reduction'`, `'monitor_data'`, `'detector_data'`, `'timeseries'`),
  matched against a workflow's registered service. The same literals also name
  `WorkflowGroup`s (display grouping in the UI); the coincidence is load-bearing:
  `register_spec` defaults a workflow's service to its group name.

### Messages and preprocessing

- **Message** — the universal internal envelope
  `(timestamp, stream: StreamId, value)` (`core/message.py`).
- **MessageSource / MessageSink** — protocols for consuming/publishing messages;
  Kafka implementations in `kafka/source.py` and `kafka/sink.py`.
- **MessageAdapter** — converts wire payloads to domain messages
  (`kafka/message_adapter.py`); `KafkaAdapter` maps `(topic, source_name)` to
  `StreamId`.
- **MessageBatch / MessageBatcher** — a time-windowed batch of messages and the
  strategies producing them (`core/message_batcher.py`).
- **Accumulator** — protocol accumulating data over time
  (`add`/`get`/`clear`, `core/handler.py`). *Batch* accumulators are consumed on
  `get()`; *context* accumulators (`is_context = True`) are idempotent and
  retain state.
- **Preprocessor** — an Accumulator in its pipeline role: bound to one StreamId,
  turning raw stream data into workflow input. Created by a
  **PreprocessorFactory** (`core/handler.py`); the concrete factories are the
  `*HandlerFactory` classes in `handlers/` (naming predates the
  preprocessor terminology). The similarly named `MessagePreprocessor` is
  internal OrchestratingProcessor wiring that owns the accumulators.

### Job management

- **JobManager** — owns all job records; schedules, activates, gates, and
  finishes jobs, fans data out, gathers results and statuses
  (`core/job_manager.py`).
- **Command** — wire type of the `livedata_commands` topic: discriminated union
  `WorkflowConfig | JobCommand` (`core/job_manager.py`). Dispatched by
  `ConfigProcessor` (`handlers/config_handler.py`).
- **JobCommand** — control message for a running job:
  `pause`/`resume`/`reset`/`stop` (pause/resume unimplemented).
- **JobSchedule** — optional start/end times (raw-data timestamps) governing
  activation and finish.
- **JobState** — wire-facing per-job status enum: `scheduled`, `active`,
  `finishing`, `pending_context`, `stopped`, `error`, `warning`. Not a flat
  state machine: it mixes lifecycle phase, the finishing overlay, and health
  into one enum, derived on demand from the internal record.
- **JobPhase** — internal lifecycle position only:
  `scheduled → pending_context → active` (`core/job_manager.py`). Orthogonal to
  health.
- **Primary / auxiliary / context data** — *primary* streams
  (the job's `source_names`) trigger computation; *auxiliary* streams
  (user-selected `AuxSources` plus framework context) accumulate and tolerate
  absence; *context* streams parametrize the workflow graph and, lacking a safe
  default, gate the job (ADR 0002/0003).
- **Gating** — holding a job in `pending_context` until all its gating (context)
  streams have a value (ADR 0002).
- **ContextBinding** — declaration mapping a context stream to a Sciline
  workflow key for given dependent sources (`config/stream.py`, ADR 0003).
- **Device** — a synthesized in-process stream merging EPICS substreams
  (RBV/VAL/DMOV → value/target/idle) into one consistent record
  (`config/stream.py`, ADR 0001/0006).

## Dashboard

### Data plane

- **DataService** — dict-like shared store keyed by **DataKey**, backed by
  temporal buffers; notifies subscribers with keys-only batched notifications
  (`dashboard/data_service.py`).
- **Subscriber** — a `DataServiceSubscriber`: registers interest in a key set
  with per-key extractors; on notification it *pulls* snapshots (it is never
  pushed data). `DataSubscriber` groups pulled data by data role.
- **UpdateExtractor** — pull-time transform fixing buffer type and retention per
  key: latest value, full history, or window aggregation
  (`dashboard/extractors.py`).
- **StreamManager** — factory registering a subscriber pipeline for a set of
  DataKeys; the result is informally called a *data stream*
  (`dashboard/stream_manager.py`).
- **Orchestrator** (unqualified, dashboard) — the message pump
  (`dashboard/orchestrator.py`): pulls from the transport's MessageSource,
  filters by active generation, writes into DataService, routes status and
  acknowledgements. Distinct from JobOrchestrator and PlotOrchestrator.

### Job and workflow lifecycle

- **JobOrchestrator** — owns the two-phase workflow lifecycle
  (stage → commit → stop): a commit mints one job number, starts the set of
  per-source jobs sharing it, publishes the commands, and tracks
  acknowledgements until confirmed or timed out
  (`dashboard/job_orchestrator.py`).
- **JobService** — read model of the latest `JobStatus` per JobId, with
  heartbeat staleness detection (`dashboard/job_service.py`).
- **ActiveJobRegistry** — thread-safe record of each workflow's current
  generation, gating ingestion against UI-thread commits
  (`dashboard/active_job_registry.py`).
- **WorkflowController** — thin interface between widgets and JobOrchestrator
  (`dashboard/workflow_controller.py`).
- **Adoption** — deriving the currently-running generation from live heartbeats
  instead of persisted job identity (ADR 0008).
- **ServiceRegistry** — backend worker health derived from heartbeats
  (`dashboard/service_registry.py`).

### Plot hierarchy

From coarse to fine: **grid → cell → layer → plotter → presenter → figure**.

- **Grid** — a titled `nrows × ncols` arrangement of cells (`PlotGridConfig`);
  shown as one dashboard tab. Grids are managed by **PlotOrchestrator**
  (`dashboard/plot_orchestrator.py`), which owns topology, persistence, and a
  version-bump polling contract (ADR 0007).
- **Cell** — one position in a grid (`PlotCell`: geometry + layers). Multiple
  layers in a cell are composed into one figure via `hv.Overlay`. Per-session
  view: `CellWidget` (`dashboard/widgets/cell.py`).
- **Layer** — the atomic plotted unit: a layer id plus `PlotConfig` (plotter
  name, params, data sources keyed by data role). When prose says "a plot",
  it almost always means a layer.
- **Plotter** — session-shared object producing HoloViews elements from
  subscribed data (`dashboard/plots.py`); subclasses per plot type
  (`LinePlotter`, `ImagePlotter`, `SlicerPlotter`, …). Registered in
  **PlotterRegistry** with a spec, factory, and data requirements.
- **Presenter** — per-session bridge from a plotter's cached state to a
  HoloViews `DynamicMap` via an `hv.streams.Pipe`; carries the dirty flag
  (`dashboard/plots.py`).
- **Figure** — the rendered HoloViews/Bokeh object placed in the document.
  Reserved for the rendered artifact; not a synonym for plotter or layer.
- **PlotDataService / LayerState** — per-layer shared state machine
  (`WAITING_FOR_DATA`/`READY`/`STOPPED`/`ERROR`) with version counters, read by
  per-session pollers (`dashboard/plot_data_service.py`).
- **Grid template / GridSpec** — declarative grid layout shipped with an
  instrument (`config/grid_template.py`).

### Sessions and updates

- **Session** — one browser connection (one Bokeh document). Tracked by
  **SessionRegistry** with heartbeat-based stale cleanup.
- **SessionUpdater** — per-session ~1 Hz driver on the session's IOLoop:
  polls its notification queue and runs update handlers inside a batched
  (`pn.io.hold` + `doc.models.freeze`) session context
  (`dashboard/session_updater.py`).
- **SessionLayer** — per-session render state for one layer: presenter, pipe,
  and DynamicMap (`dashboard/session_layer.py`).
- **Single-writer versioned pull** — the dashboard concurrency model: one writer
  mutates shared state and bumps a version; sessions poll the version and pull
  snapshots on their own IOLoop (ADR 0007).
- **Frame** (dashboard) / **FrameClock** — the frame-gated flush cycle batching
  plot updates per session (`dashboard/frame_clock.py`, ADR 0005). Unrelated to
  the neutron pulse-frame sense.
- **Reaper** — background teardown of dead browser sessions off their own IOLoop
  (`dashboard/session_updater.py`, ADR 0007).

### Configuration and transport

- **ConfigStore** — UI-state persistence (workflow and plot configs, YAML files
  under the user's config dir; `dashboard/config_store.py`). Not related to
  backend command handling.
- **ConfigurationAdapter** — bridges pydantic parameter models to generic form
  widgets (`dashboard/configuration_adapter.py`); implementations for workflow
  and plot configuration.
- **DashboardServices** — the per-process composition root shared across
  sessions (`dashboard/dashboard_services.py`).
- **`lt-*` hooks** — stable DOM classes for UI automation
  (`.claude/rules/dashboard-widgets.md`).

## One word, several meanings

- **role** — three senses: (1) *data role*: a plot data-source slot —
  `primary`/`x_axis`/`y_axis` (`dashboard/data_roles.py`); (2) *window role*
  (`StreamRole`): the time-window of an output — `since_start`/`per_update`
  (`config/workflow_spec.py`); (3) *aux-input role*: the logical name of an
  auxiliary workflow input a user selects a stream for (`AuxSources`).
  Qualify the word when ambiguity is possible.
- **stream** — (1) a Kafka/internal data stream (`StreamId`); (2) a dashboard
  subscriber pipeline (`StreamManager.make_stream`); (3) an
  `hv.streams.Pipe` per-session channel; (4) `OutputView.streams`, backend
  output field bundles. Sense (1) is the default; qualify the others.
- **source name** — raw Kafka FlatBuffers producer name at the boundary, but the
  canonical *stream name* everywhere else (see
  [Identity and keying](#identity-and-keying)).
- **view** — (1) `OutputView` / `view_name`: user-facing name of a workflow
  output; (2) the MVC sense: a widget rendering state (e.g. `CellWidget`).
- **plot** — informal umbrella over grid/cell/layer/plotter/figure; in precise
  writing name the level, usually *layer*.
- **frame** — neutron pulse frame (instrument context) vs. the dashboard
  update-flush cycle (ADR 0005).
- **orchestrator** — backend `OrchestratingProcessor`; dashboard `Orchestrator`
  (message pump), `JobOrchestrator` (workflow lifecycle), `PlotOrchestrator`
  (grid topology). Never write "the orchestrator" without qualification.
- **processor** — the Service-driven `Processor` protocol; *not* a workflow.
  (`ess.reduce.streaming.StreamProcessor` is an upstream class a Workflow may
  wrap.)
- **handler** — historical name surviving in `handlers/` module and
  `*HandlerFactory` class names; the concepts are preprocessor factories and
  accumulators. Avoid "handler" for new names.
- **config** — spans `WorkflowConfig` (runtime start command), instrument
  configuration (`Instrument`, YAML), dashboard `ConfigStore` (UI persistence),
  and `ConfigProcessor` (backend command dispatch). Always qualify.
- **state / status** — `JobState` (wire enum) vs `JobPhase` (internal lifecycle)
  vs `ServiceState` (worker lifecycle); `JobStatus`/`ServiceStatus` are the
  heartbeat payloads carrying them.
- **layout** — `hv.Layout` combine-mode subplots; the per-session page layout
  (`create_layout`); a grid's cell arrangement; Bokeh's layout pass. Qualify.
