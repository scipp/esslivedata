# Declarative session reconciler — migration plan

Companion to the
[declarative session reconciler proposal](declarative-session-reconciler.md).
Four phases, each shippable and useful on its own; phase 2 doubles as the
go/no-go spike for the rest.

```mermaid
flowchart LR
    P0["Phase 0<br/>characterization<br/>tests"]
    P1["Phase 1<br/>occupancy from<br/>topology"]
    P2["Phase 2<br/>desired() + differ<br/>(the spike)"]
    P3["Phase 3<br/>fold in flush +<br/>freshness"]
    P4["Phase 4<br/>stable widget shell<br/>(speculative)"]

    P0 --> P2
    P1 --> P2
    P2 -- "criteria met" --> P3 --> P4
    P2 -- "criteria missed" --> Stop(["stop, keep 0+1,<br/>write down why"])

    classDef ship fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef spec fill:#fff3e0,stroke:#ef6c00,color:#e65100;
    class P0,P1 ship;
    class P2,P3 spec;
    class P4 spec;
```

## Phase 0 — characterization tests

Lock in what the current pass does before changing how it does it. The tests
from #1220 (hidden cell not built, reveal builds once, watcher preserves
pre-warm, deferred rebuild) are the seed; add:

- Tab reconcile: focus preserved across rebuilds, local-creation focus,
  disabled-grid tab omission, mid-removal identity lookups.
- Orphan sweep: cell removed from topology → widget removed and disposed;
  disabled grid's cells survive for re-enable.
- Flush gating: one flush per frame generation, tab-switch flush without a
  generation change does not consume the next frame's flush.
- Freshness: pill updates on flush/rebuild, stall aging at the 2 s cadence,
  no double-update beat.
- Predicate/pass agreement: for each gate, "pass would do work ⇒ predicate
  fires" (this test is only writable per-gate today; under phase 2 it becomes
  a single generic property).

Also enumerate the framework-trap invariants that tests cannot easily cover
as a checklist in the module docstring (they are currently scattered across
`IMPORTANT:` comments): modal container has one stable parent at top level;
wake registration deferred until browser load; batching around multi-widget
updates; disposal breaks toolbar reference cycles; markup-pane reveal
workaround.

Value: independent of this proposal — it is the safety net any future change
to this module needs.

## Phase 1 — occupancy from topology (#1219, structural half)

`PlotGrid` stops deriving free positions from inserted widgets. Instead the
reconciler hands it the occupied geometry set from the topology snapshot on
each pass (or `PlotGrid` receives a callable). `_occupied_cells` shrinks to
"which widget sits at which geometry" (needed for removal/replacement), and
`_is_region_available` consults the topology-derived set.

Small, local, already agreed as the right fix in #1219. Kills the
offered-then-refused wizard path entirely — including the cross-session race
the `ValueError` handler now merely reports politely.

## Phase 2 — `desired()` + differ for structure and materialization

The spike. Scope: concerns 1–4 of the current pass (tabs, cell composition,
layer lifecycle, tokens) move to the new shape; flush (5) and freshness (6)
stay as they are, driven after the apply step exactly as today.

Sketch of the types (final naming via the glossary before implementation):

```python
@dataclass(frozen=True)
class LayerBuildInput:
    layer_id: LayerId
    plotter: object | None      # identity comparison only
    has_plot: bool              # plotter holds a computed frame

@dataclass(frozen=True)
class CellPlan:
    grid_id: GridId
    geometry: CellGeometry
    materialize: bool
    build_inputs: tuple[...]    # composition + per-layer LayerBuildInput

def desired(
    topology: Mapping[GridId, PlotGridConfig],   # orchestrator snapshot
    layer_states: Mapping[LayerId, LayerSnapshot],
    view: SessionView,                            # active grid, modal open
    watched: Callable[[LayerId], bool],           # other sessions' tokens
) -> dict[CellId, CellPlan]: ...
```

The differ compares `CellPlan`s against the applied record (which stores each
widget's `build_inputs`) and emits build/insert/dispose actions; token
acquire/release is derived from the same plans (materialized-and-visible ⇒
hold tokens). The wake predicate becomes: input version stamp ≠ stamp at last
apply, plus the freshness time term.

Deleted on success: `_cell_signatures`, `SessionLayer.last_seen_version`
(as a rebuild trigger), `_last_layer_version` and its before/after
choreography, the rebuild half of `_has_pending_work`, and the
deliberately-stale-records device from #1220. `_cell_grid` folds into the
applied record. `_tab_composition` and the tab-focus logic can stay initially
— tabs are a small, self-contained concern.

Order-of-operations invariants to carry over explicitly (they do not
disappear, they become documented steps of the apply stage):

- Token 0→1 computes synchronously; sample time bounds and flush *after*
  activation so a reveal renders on its own pass.
- Rebuild+insert runs inside the same batched update as removal, in one pass.
- Records update only after a successful build (exception ⇒ retry next tick) —
  under the differ this is automatic: a failed build leaves the applied record
  unchanged.

### Acceptance criteria (agree before starting, judge after ≤ 1 week)

1. ≥ half of the bookkeeping fields deleted (measured against the
   [inventory](declarative-session-reconciler-current-state.md)).
2. The hand-mirrored rebuild predicate deleted, replaced by the generic
   stamp comparison.
3. All phase-0 characterization tests and the browser smoke tests pass
   unchanged (behavioral equivalence, not "tests updated to match").
4. No new framework workaround was needed (a rewrite that trips a new
   Panel/Bokeh trap has negative value).
5. `desired()` has direct unit tests with plain data and no Panel imports.

Any criterion missed → stop, keep phases 0–1, record the reason in this
document.

## Phase 3 — fold in flush and freshness (optional)

Model the data flush as part of the desired state ("grid G should display
frame generation N") and freshness as a derived output. Only worth it if
phase 2 leaves flush/freshness as the odd ones out complicating the tick
entry points; they are edge-triggered by nature and may legitimately stay a
small imperative tail. Decide on the phase-2 outcome, same criteria style.

## Phase 4 — stable widget shell (speculative spike)

Today a plotter swap rebuilds the whole `CellWidget` (titlebar, toolbars,
panes, figure). If the shell were stable and only the figure pane swapped,
rebuild cost would drop enough that much gating precision becomes
unnecessary — the differ's job shrinks further. Risk: Panel pane replacement
has its own churn and known traps (reveal latch, reparenting); history in
this codebase says prototype first. Strictly optional; nothing in phases 0–3
depends on it.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Silently dropping a framework workaround | Phase 0 checklist + tests; criterion 4 |
| Behavioral drift hidden by rewritten tests | Criterion 3 forbids test rewrites during the spike |
| `desired()` grows imperative side effects over time | Enforce purity: no Panel imports in its module; unit tests construct inputs as plain data |
| Differ becomes the new dumping ground | Differ rules are fixed and generic; policy changes go to `desired()` only — review rule to state in `.claude/rules/dashboard-widgets.md` |
| Spike overruns | Hard time box; abandoning keeps phases 0–1 |
