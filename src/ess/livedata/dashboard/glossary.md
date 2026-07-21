## Dashboard

Dashboard-specific terms. Cross-cutting terms (Job, DataKey, StreamId, ...) and
the disambiguation of overloaded words live in `src/ess/livedata/glossary.md`.

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

