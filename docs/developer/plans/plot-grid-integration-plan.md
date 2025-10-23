# PlotGrid Dashboard Integration Plan

## Goal

Integrate PlotGrid into the dashboard as a new tab alongside the existing PlotCreationWidget. The new tab provides a 3x3 grid where users can place plots of varying sizes by selecting rectangular regions.

## Architecture Overview

### Component Structure

```
PlotCreationWidget (modified)
├── Jobs Tab (existing)
├── Create Plot Tab (existing)
├── Plot Grid Tab (NEW)
│   └── PlotGridTab widget
│       ├── PlotGrid (3x3 fixed)
│       ├── JobPlotterSelectionModal (NEW)
│       └── ConfigurationModal (existing, reused)
└── Plots Tab (existing)
```

### Modal Workflow

**Two-modal chain** triggered when user completes cell selection in grid:

1. **JobPlotterSelectionModal**: Two-step wizard in single modal
   - Step 1: Job/output selection (Tabulator table, same as PlotCreationWidget)
   - Step 2: Plotter type selection (filtered by compatibility)
   - Success → opens ConfigurationModal

2. **ConfigurationModal**: Existing modal, reused as-is
   - Source selection + parameter configuration
   - Success → creates plot and inserts into grid

### Key Design Decisions

**1. Keep existing PlotCreationWidget functional**
- Add PlotGridTab as new tab, not replacement
- Both mechanisms coexist during transition
- No changes to existing plot creation flow

**2. Fixed 3x3 grid size**
- Simplifies initial implementation
- Can add configurability later if needed

**3. Prevent concurrent plot creation workflows**
- PlotGrid tracks `_plot_creation_in_flight` boolean state
- Blocks new cell selections while modal workflow is active
- Cleared on modal success or cancellation

**4. Deferred plot insertion**
- PlotGrid cannot return plot synchronously (modal workflow is async)
- Add `insert_plot_deferred(plot)` method to complete insertion after workflow
- Add `cancel_pending_selection()` to abort and reset state

**5. Minimize code duplication initially**
- JobPlotterSelectionModal duplicates logic from PlotCreationWidget
- Refactor to extract shared components only if duplication becomes problematic
- Premature abstraction adds complexity

## Implementation Steps

### Step 1: Enhance PlotGrid (plot_grid.py)

Add in-flight tracking and deferred insertion:

**New state:**
- `_plot_creation_in_flight: bool` - Tracks active modal workflow

**New methods:**
- `insert_plot_deferred(plot)` - Complete plot insertion after async workflow
- `cancel_pending_selection()` - Abort workflow and reset state

**Modified behavior:**
- `_on_cell_click()`: Check in-flight state, reject if busy
- After second click: Set in-flight flag, call callback (no return value needed)
- `_refresh_all_cells()`: Show disabled state when in-flight

### Step 2: Create JobPlotterSelectionModal (new file)

**File:** `src/ess/livedata/dashboard/widgets/job_plotter_selection_modal.py`

**Purpose:** Two-step wizard modal for selecting job/output and plotter type

**Dependencies:**
- JobService (for job data)
- PlottingController (for available plotters)

**Flow:**
1. Show job/output Tabulator (extracted pattern from PlotCreationWidget)
2. On selection → enable "Next" button
3. Next → show plotter selector (RadioButtonGroup or Select)
4. On plotter selection → enable "Configure Plot" button
5. Configure Plot → close modal, call success_callback(job_number, output_name, plot_name)

**Callbacks:**
- `success_callback(JobNumber, str | None, str)` - Called with selected parameters
- `cancel_callback()` - Called on modal close/cancel

**UI Elements:**
- Job/output table (reuse PlotCreationWidget._create_job_output_table pattern)
- Plotter selector (reuse PlotCreationWidget._create_plot_selector pattern)
- Navigation: Next button (step 1) → Configure Plot button (step 2)
- Cancel button (always available)

### Step 3: Create PlotGridTab (new file)

**File:** `src/ess/livedata/dashboard/widgets/plot_grid_tab.py`

**Purpose:** Orchestrate PlotGrid + modal workflow

**Dependencies:**
- JobService, JobController, PlottingController
- PlotGrid
- JobPlotterSelectionModal
- ConfigurationModal + PlotConfigurationAdapter

**Structure:**
- PlotGrid instance with callback to `_on_plot_requested()`
- Modal container (pn.Column) for modal lifecycle management
- State: References to active modals for cleanup

**Workflow:**
1. User selects region → `_on_plot_requested()` called
2. Create and show JobPlotterSelectionModal
3. On success → `_on_job_plotter_selected(job, output, plotter)`
4. Create PlotConfigurationAdapter
5. Create and show ConfigurationModal
6. On success → `_on_plot_created(plot)`
7. Call `grid.insert_plot_deferred(plot)`

**Error Handling:**
- Modal cancellation → `_on_modal_cancelled()` → `grid.cancel_pending_selection()`
- Modal close (X button) → Watch modal.param 'open' → call cancel callback
- Configuration errors → handled by ConfigurationModal (existing behavior)

### Step 4: Integrate into PlotCreationWidget

**File:** `src/ess/livedata/dashboard/widgets/plot_creation_widget.py`

**Changes:**
- Import PlotGridTab
- Instantiate in `__init__`: `self._plot_grid_tab = PlotGridTab(...)`
- Add to main tabs: `("Plot Grid", self._plot_grid_tab.widget)`
- Register with job service updates: `job_service.register_job_update_subscriber(self._plot_grid_tab.refresh)`

**Tab order:**
1. Jobs
2. Create Plot (existing mechanism)
3. Plot Grid (NEW)
4. Plots

### Step 5: PlotConfigurationAdapter Integration

**Already implemented, reuse as-is:**
- Takes job_number, output_name, plot_spec, available_sources
- Works with ConfigurationModal
- Handles source selection + parameter configuration
- Success callback receives (plot, selected_sources)

**Adapter instantiation in PlotGridTab:**
```python
config = PlotConfigurationAdapter(
    job_number=job_number,
    output_name=output_name,
    plot_spec=plotting_controller.get_spec(plot_name),
    available_sources=list(job_service.job_data[job_number].keys()),
    plotting_controller=plotting_controller,
    success_callback=lambda plot, sources: self._on_plot_created(plot),
)
```

## Code Organization

### New Files
1. `src/ess/livedata/dashboard/widgets/job_plotter_selection_modal.py`
2. `src/ess/livedata/dashboard/widgets/plot_grid_tab.py`

### Modified Files
1. `src/ess/livedata/dashboard/widgets/plot_grid.py`
   - Add in-flight tracking
   - Add deferred insertion methods

2. `src/ess/livedata/dashboard/widgets/plot_creation_widget.py`
   - Instantiate and integrate PlotGridTab

### Unchanged (Reused)
- `ConfigurationModal` - Works as-is with new workflow
- `PlotConfigurationAdapter` - Works as-is with new workflow
- `PlottingController` - API already supports all needed operations
- `JobService` - Existing data access patterns

## Technical Details

### In-Flight State Management

**Purpose:** Prevent multiple concurrent plot creation workflows

**Implementation in PlotGrid:**
- Flag set after region selection, before callback invocation
- All cell clicks rejected while flag is true
- Visual feedback: Show notification "Plot creation in progress"
- Cleared on successful insertion or cancellation

### Modal Lifecycle

**Problem:** Modals must trigger cleanup when closed via X button or ESC

**Solution:** Watch modal.param 'open' event
```python
modal.param.watch(self._on_modal_closed, 'open')

def _on_modal_closed(self, event):
    if not event.new:  # Modal was closed
        self._cancel_callback()
```

ConfigurationModal already implements this pattern; apply to JobPlotterSelectionModal.

### Job/Output Table Data

**Source:** JobService.job_data and JobService.job_info

**Format:** Same as PlotCreationWidget
- One row per (job_number, output_name) pair
- Grouped by workflow_name and job_number
- Columns: job_number, workflow_name, output_name, source_names

**Refresh:** Subscribe to JobService updates via `register_job_update_subscriber()`