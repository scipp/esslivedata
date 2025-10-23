# PlotGrid Dashboard Integration Plan

## Status: Steps 1-4 Complete ✅

Implementation complete. The PlotGrid has been successfully integrated into the dashboard.

## Implementation Summary

### Completed Components

1. **PlotGrid enhancements** ([plot_grid.py](../../src/ess/livedata/dashboard/widgets/plot_grid.py))
   - In-flight state tracking to prevent concurrent workflows
   - Deferred insertion methods: `insert_plot_deferred()` and `cancel_pending_selection()`
   - Async callback pattern (no return value from callback)

2. **JobPlotterSelectionModal** ([job_plotter_selection_modal.py](../../src/ess/livedata/dashboard/widgets/job_plotter_selection_modal.py))
   - Two-step wizard: job/output selection → plotter type selection
   - Success and cancel callbacks
   - Modal close detection for proper cleanup

3. **PlotGridTab** ([plot_grid_tab.py](../../src/ess/livedata/dashboard/widgets/plot_grid_tab.py))
   - Orchestrates PlotGrid + two-modal workflow
   - Handles JobPlotterSelectionModal → ConfigurationModal chain
   - Proper error handling and state cleanup

4. **PlotConfigurationAdapter** ([plot_configuration_adapter.py](../../src/ess/livedata/dashboard/widgets/plot_configuration_adapter.py))
   - Extracted to separate file to avoid circular imports
   - Reusable adapter for plot configuration modal

5. **PlotCreationWidget integration** ([plot_creation_widget.py](../../src/ess/livedata/dashboard/widgets/plot_creation_widget.py))
   - Added "Plot Grid" tab between "Create Plot" and "Plots"
   - Registered with job service updates

## Next Steps

### Testing and Refinement
- [ ] Manual testing with live dashboard
- [ ] Test modal workflows (success and cancellation paths)
- [ ] Test grid cell selection and plot insertion
- [ ] Verify proper cleanup on modal close
- [ ] Test with multiple plotters and job types

### Potential Enhancements
- [ ] Add keyboard shortcuts (ESC to cancel selection)
- [ ] Make grid size configurable (currently fixed 3x3)
- [ ] Add plot resize/move functionality
- [ ] Extract shared code between PlotCreationWidget and JobPlotterSelectionModal
- [ ] Add plot titles or labels in grid cells
- [ ] Persist grid layout across sessions

## Architecture Overview

### Component Structure

```
PlotCreationWidget
├── Jobs Tab
├── Create Plot Tab
├── Plot Grid Tab ← NEW
│   └── PlotGridTab
│       ├── PlotGrid (3x3 fixed)
│       ├── JobPlotterSelectionModal
│       └── ConfigurationModal (reused)
└── Plots Tab
```

### Modal Workflow

1. User selects region in grid → `PlotGrid._on_cell_click()`
2. PlotGrid calls `plot_request_callback()` (no return value)
3. PlotGridTab shows JobPlotterSelectionModal
4. User selects job/output and plotter
5. PlotGridTab shows ConfigurationModal
6. User configures parameters and sources
7. PlotGridTab creates plot via PlottingController
8. PlotGridTab calls `PlotGrid.insert_plot_deferred(plot)`

### Key Design Decisions

- **Fixed 3x3 grid**: Simplifies initial implementation
- **In-flight state tracking**: Prevents concurrent workflows
- **Deferred insertion**: Enables async modal workflow
- **Reuse existing modals**: ConfigurationModal works unchanged
- **Separate PlotConfigurationAdapter**: Avoids circular imports