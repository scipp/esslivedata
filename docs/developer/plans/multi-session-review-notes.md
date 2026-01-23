# Multi-Session Architecture Review Notes

Review of changes since `adc9f382`, focusing only on modified (not new) files.

## Overall Architecture

The change introduces a **two-stage plotter architecture**:
1. **Compute stage** (shared): `plotter.compute(data)` runs once, stores result in `PlotDataService`
2. **Present stage** (per-session): `Presenter.present(pipe)` creates session-bound DynamicMaps

Polling-based updates via `SessionUpdater` replace direct push to plots.

## Things That Feel Off or Unfinished

### 1. plotting_controller.py still has both patterns

Lines 168-243 have `create_plot_from_pipeline()` that does the old pattern:
- `plotter.initialize_from_data(pipe.data)`
- Direct `hv.DynamicMap(plotter, streams=[pipe], kdims=plotter.kdims)`

Meanwhile `setup_pipeline()` (lines 122-166) uses the new callback-based approach. Either the old method is legacy that should be removed, or there's a reason ROI plotters can't use two-stage—but that's not documented.

### 2. SlicerPlotter.compute() breaks the type contract

`Plotter.compute()` signature says it returns `hv.Overlay | hv.Layout | hv.Element`, but `SlicerPlotter.compute()` (plots.py:716) returns `SlicerState`. This works at runtime because of duck typing, but it's a type-safety smell. The base class and custom plotters have different contracts.

### 3. Split between LayerState and PlotDataService

`PlotOrchestrator._layer_state` stores `LayerState(plot=..., error=..., stopped=...)` but actual plot data is in `PlotDataService`. For static plots, `LayerState.plot` is used directly; for streaming plots, `PlotDataService` is used. This split seems accidental rather than designed.

### 4. Widget notification uses two mechanisms (resolved)

`JobOrchestrator` uses two different mechanisms:
- `NotificationQueue` → for command success/error toasts (multi-session)
- `WidgetLifecycleCallbacks` → for widget rebuilds (direct callbacks)

**Resolution**: This split is intentional and correct:
- **Toasts** (NotificationQueue): Must be triggered from the current session's context because Panel's notification system is session-bound. The background thread cannot show toasts directly—each session must poll the queue and display its own toasts.
- **Widget rebuilds** (WidgetLifecycleCallbacks): Work correctly with direct callbacks because widgets subscribe individually and Panel handles the session context correctly when the callback updates widget state.

The original architecture plan included `WidgetStateStore` for polling-based widget updates, but this was never wired up. Since the callback mechanism works correctly, we removed `WidgetStateStore` and decided to keep the simpler callback approach for widgets.

### 5. Polling efficiency concern

`_poll_for_plot_updates()` in plot_grid_tabs.py iterates through all grids → all cells → all layers every 500ms, even when nothing changed. The `PlotDataService.update()` sets a version, but every session still does this loop.

## Things That Look Clean

- `SessionUpdater` as per-session coordinator receiving it in widget constructors
- `staging_transaction` context manager in JobOrchestrator for batching
- `WidgetLifecycleCallbacks` following the pattern from `.claude/rules/dashboard-widgets.md`
- `DataSubscriber` with role-based assembly is cleaner than before
- `pn.io.hold()` usage for batched widget updates

## Cleanup Performed

### Removed WidgetStateStore

The original architecture plan (`multi-session-architecture.md`) included `WidgetStateStore` for polling-based widget state synchronization. However:
- The class was implemented but never wired up
- No production code called `widget_state_store.update()` to write state
- No widget called `register_widget_handler()` to listen for changes
- Only tests exercised these methods

Since the existing `WidgetLifecycleCallbacks` mechanism works correctly for widget synchronization, we removed `WidgetStateStore` and all related code:
- Removed `WidgetStateStore`, `StateKey`, and `VersionedState` from `state_stores.py`
- Removed `widget_state_store` creation from `DashboardServices`
- Removed widget state polling from `SessionUpdater`
- Removed related tests

## Questions for Discussion

1. Is `create_plot_from_pipeline()` in plotting_controller still needed, or legacy?
2. Should `SlicerPlotter` have a separate protocol/base class since it returns different types?
3. Would version-based polling be more efficient? (Only rebuild if version changed for specific layer)
