# Glossary

Authoritative definitions of the terms used in ESSlivedata code and documentation.
When code, another document, and this glossary disagree, one of them has a bug:
fix it or file an issue.
References name modules (under `src/ess/livedata/`) rather than line numbers.

Terms are grouped into cross-cutting, backend, and dashboard sections;
dashboard terms live in `dashboard/glossary.md`, and the documentation build
renders all sections as one page.
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
  backend output fields by `Windowing` (`since_start`/`per_update`)
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

<!-- dashboard-glossary-insertion-point: the docs build inserts dashboard/glossary.md here -->

## One word, several meanings

- **role** — two senses: (1) *data role*: a plot layer's data-source key —
  `primary`/`x_axis`/`y_axis` (`dashboard/data_roles.py`); (2) *aux-input
  role*: the logical name of an auxiliary workflow input a user selects a
  stream for (`AuxSources`). Qualify the word when ambiguity is possible. The
  time-window flavor of an output field (`since_start`/`per_update`) is
  `Windowing`, not a role (`config/workflow_spec.py`).
- **stream** — (1) a Kafka/internal data stream (`StreamId`); (2) a dashboard
  subscriber pipeline (`StreamManager.make_stream`); (3) an
  `hv.streams.Pipe` per-session channel. Sense (1) is the default; qualify
  the others.
- **source name** — raw Kafka FlatBuffers producer name at the boundary, but the
  canonical *stream name* everywhere else (see
  [Identity and keying](#identity-and-keying)).
- **view** — (1) `OutputView` / `view_name`: user-facing name of a workflow
  output, held by `DataSourceConfig.view_name` and resolved to a backend
  pydantic field name (`ResolvedDataSource.output_name`, feeding
  `DataKey.output_name`) at layer-setup time — see `dashboard/glossary.md`;
  (2) the MVC sense: a widget rendering state (e.g. `CellWidget`).
- **plot** — informal umbrella over grid/cell/layer/plotter/figure; in precise
  writing name the level, usually *layer*.
- **frame** — neutron pulse frame (instrument context) vs. the dashboard
  update-flush cycle (ADR 0005).
- **orchestrator** — backend `OrchestratingProcessor`; dashboard
  `JobOrchestrator` (workflow lifecycle), `PlotOrchestrator` (grid topology).
  The dashboard's Kafka-to-`DataService` message pump is `MessagePump`, not
  an orchestrator.
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
