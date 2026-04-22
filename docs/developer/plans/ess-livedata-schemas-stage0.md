# Stage 0: `WorkflowId` / `WorkflowSpec` semantic surgery

Preparatory stage for [`ess-livedata-schemas.md`](./ess-livedata-schemas.md). Performs the non-1:1 changes to `WorkflowId` and `WorkflowSpec` **in place**, before any files move to `ess.livedata.schemas`. After Stage 0, Stage 1's relocation is mechanical.

## Intent

The target design reduces `WorkflowId` identity from `(instrument, namespace, name, version)` to `(instrument, name, version)` and replaces `namespace` with a display-oriented `group`. The change is non-1:1:

- Two workflows that differ only in namespace collapse to the same identity.
- `namespace` carries two runtime responsibilities (service routing; worker identity) that must be relocated before the field can be removed.
- `ArraySpec` replaces `sc.DataArray`-based output templates — handled in Stage 1, not here.

Stage 0 cleans the semantics in place. One branch, independent commits. Packaging (single PR vs. split) decided when the branch is ready.

## Non-goals

- No files move. The schemas subpackage is created in Stage 1.
- No `ArraySpec`. Output templates stay `sc.DataArray`-based through Stage 0.
- No `InstrumentSpec` split. `Instrument` keeps its current shape.

## Commit structure

Five commits, dependency-ordered; each leaves the repo green.

### 0a — Route services without reading `WorkflowId.namespace`

`JobManager.create` currently rejects jobs where `workflow_id.namespace != instrument.active_namespace`. `active_namespace` is set by `ServiceFactory` at startup. Replace with an explicit service tag owned by the registering factory:

- `WorkflowFactory.register(...)` (or the decorator it exposes) gains a `service: str` argument.
- Factory keeps a `service` mapping `WorkflowId → str`.
- `JobManager` consults `factory.get_service(workflow_id)` and compares against the running service name (already known to the service; no need to read it from `Instrument`).
- `active_namespace` is **not** removed yet — removed in 0e once nothing else reads it. It remains unused by routing after this commit.

### 0b — `BackendStatus.service_name` replaces `.namespace`

`BackendStatus` publishes `namespace`; the dashboard uses it for the worker-identity key (`"{instrument}:{namespace}:{worker_id}"`) and for the status-widget label.

- Rename the wire field `namespace` → `service_name` on `BackendStatus`.
- Update `service_registry.py` key composition and `backend_status_widget.py` display.
- Wire break: producers and consumers land in the same commit.

### 0c — Resolve cross-namespace name collisions

Only three generic workflows collide once `namespace` leaves identity:

| Old `WorkflowId`              | New `name`                 |
| ----------------------------- | -------------------------- |
| `.../monitor_data/default/v`  | `monitor_data_default`     |
| `.../detector_data/default/v` | `detector_data_default`    |
| `.../timeseries/default/v`    | `timeseries_default`       |

**Convention:** `<old_namespace>_<old_name>`. Mechanical, greppable, no bikeshed.

Instrument-specific reduction specs (`data_reduction/...`) are not renamed — their names are already distinct within the reduced identity. If a later audit finds a collision with one of the new prefixed names, rename that one too; not expected.

### 0d — Add `WorkflowSpec.group` (pydantic model), rename `params` → `inputs`

Introduce a `WorkflowGroup` pydantic model providing display metadata alongside a locked-down identifier:

```python
class WorkflowGroup(BaseModel, frozen=True):
    name: Literal[
        "data_reduction", "monitor_data", "detector_data", "timeseries"
    ]
    title: str
    description: str = ""
```

Canonical instances as module-level constants; specs reference them by identity:

```python
REDUCTION = WorkflowGroup(
    name="data_reduction", title="Reduction",
    description="...",
)
MONITORS = WorkflowGroup(name="monitor_data", title="Monitors", description="...")
DETECTORS = WorkflowGroup(name="detector_data", title="Detectors", description="...")
TIMESERIES = WorkflowGroup(name="timeseries", title="Timeseries", description="...")
```

Spec usage: `group=MONITORS` — single source of truth for display metadata. The `Literal` on `WorkflowGroup.name` is belt-and-suspenders against ad-hoc construction sneaking in new categories.

`WorkflowSpec` changes in this commit:

- Add `group: WorkflowGroup` (required, no default). Set on every existing spec, mirroring the current namespace (`data_reduction`→`REDUCTION`, etc.).
- Rename `params` → `inputs` (field, and in `WorkflowConfig.params` → `WorkflowConfig.inputs` if we're consistent; confirm scope when writing the commit).

Relaxing to free-form `name` and picking nicer display wording are follow-ups the dashboard can drive.

### 0e — Drop `namespace` from identity and types

- Remove `namespace` field from `WorkflowId`, `WorkflowSpec`, and every spec registration site.
- `WorkflowId.__str__` → `"{instrument}/{name}/{version}"`; `from_string` parses 3 parts.
- Remove `Instrument.active_namespace` and the setter in `ServiceFactory` (already unused after 0a).
- No wire-format compatibility. Durable breakage is acceptable per discussion.

## Open items

- **Scope of `params`→`inputs` rename.** Confirm whether `WorkflowConfig.params` follows suit. Default: yes, for consistency with the spec. Decide at implementation time.
- **Group ordering in the UI.** Display order of groups is not yet specified. Declaration order of the module-level constants is the natural answer; confirm the dashboard honors it (not alphabetical sort).
- **Stage 0 PR packaging.** One branch, five commits. Single PR most likely; split only if review bandwidth demands it.
