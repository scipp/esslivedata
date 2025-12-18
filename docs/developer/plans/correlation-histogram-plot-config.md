# Correlation Histogram Support in PlotConfigModal

This document describes the design and implementation status for supporting correlation histogram plotters in the PlotConfigModal wizard.

## Problem Context

Issue #611 identified that the existing correlation histogram workflow mechanism doesn't fit well with the PlotOrchestrator-based plot configuration:

1. **Current correlation histogram workflows** require auxiliary source names (`aux-source-names`) to specify which timeseries to correlate against
2. **The PlotConfigModal** was designed for single-data-source plots and couldn't handle the multi-source nature of correlation histograms
3. **User mental model mismatch**: Users think of "plot detector counts against temperature" rather than "configure a correlation workflow with aux sources"

The solution treats correlation histograms as **special plotters** that consume timeseries data and need additional timeseries for their correlation axes, rather than as separate workflows.

## Agreed Solution Design

### Conceptual Model

Correlation histograms are presented as plotter types (like Timeseries, Lines, Image) rather than as separate workflows. When a user selects a timeseries-compatible output:

1. **Step 1**: Select workflow and output (unchanged) - e.g., "Timeseries: detector_counts"
2. **Step 2**: Select plotter type - includes "Correlation Histogram 1D" and "Correlation Histogram 2D"
   - When correlation histogram is selected, additional dropdown(s) appear for axis selection
   - For 1D: one dropdown for X-axis timeseries
   - For 2D: two dropdowns for X-axis and Y-axis timeseries
3. **Step 3**: Configure plotter parameters (bin edges, normalization, etc.)

### Data Flow

The `PlotConfig.data_sources` list stores:
- `data_sources[0]`: Primary data (the output selected in Step 1, e.g., detector counts)
- `data_sources[1]`: X-axis timeseries to correlate against
- `data_sources[2]`: Y-axis timeseries (for 2D only)

### Finding Available Timeseries

Instead of querying the DataService at runtime (which requires data to exist), we use `find_timeseries_outputs()` to scan the workflow registry for outputs that:
- Have `ndim == 0` (scalar)
- Have a `'time'` coordinate

This requires output fields to have `default_factory` templates that define the expected structure.

## Implementation Status

### Completed

#### 1. `find_timeseries_outputs()` Helper (`workflow_spec.py`)

```python
def find_timeseries_outputs(
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
) -> list[tuple[WorkflowId, str, str]]:
    """Find all timeseries outputs in the workflow registry."""
```

Returns tuples of `(workflow_id, source_name, output_name)` for all timeseries-compatible outputs.

#### 2. Output Spec Templates with `default_factory`

Added `default_factory` to timeseries output fields so `find_timeseries_outputs()` can identify them:

| File | Field | Template |
|------|-------|----------|
| `timeseries_workflow_specs.py` | `TimeseriesOutputs.delta` | 0-D scalar with time coord |
| `detector_view_specs.py` | `DetectorViewOutputs.counts_total` | 0-D counts with time coord |
| `detector_view_specs.py` | `DetectorViewOutputs.counts_in_toa_range` | 0-D counts with time coord |
| `instruments/bifrost/specs.py` | `DetectorRatemeterOutputs.detector_region_counts` | 0-D counts with time coord |
| `instruments/dummy/specs.py` | `TotalCountsOutputs.total_counts` | 0-D counts with time coord |

#### 3. Correlation Histogram Plotter Registration (`plotting.py`)

Registered two new plotters with the same `DataRequirements` as the timeseries plotter:

```python
plotter_registry.register_plotter(
    name='correlation_histogram_1d',
    title='Correlation Histogram 1D',
    data_requirements=DataRequirements(
        min_dims=0,
        max_dims=0,
        multiple_datasets=True,
        required_extractor=FullHistoryExtractor,
    ),
    factory=_correlation_histogram_1d_factory,  # Dummy - handled by PlotOrchestrator
)

plotter_registry.register_plotter(
    name='correlation_histogram_2d',
    title='Correlation Histogram 2D',
    # ... same requirements
)
```

#### 4. Simplified Parameter Models (`correlation_histogram.py`)

Added pragmatic parameter models for the PlotConfigModal wizard that use simple bin counts instead of full edge specifications. The plotter auto-determines bin ranges from the data at runtime.

The bin parameters are wrapped in nested models (required by `ModelWidget`):

```python
class Bin1dParams(pydantic.BaseModel):
    x_bins: int = Field(default=50, ge=1, le=1000, title="X Bins")

class Bin2dParams(pydantic.BaseModel):
    x_bins: int = Field(default=50, ge=1, le=1000, title="X Bins")
    y_bins: int = Field(default=50, ge=1, le=1000, title="Y Bins")

class SimplifiedCorrelationHistogram1dParams(CorrelationHistogramParams):
    bins: Bin1dParams = Field(default_factory=Bin1dParams)

class SimplifiedCorrelationHistogram2dParams(CorrelationHistogramParams):
    bins: Bin2dParams = Field(default_factory=Bin2dParams)
```

A mapping `SIMPLIFIED_CORRELATION_PARAMS` maps plotter names to their param classes for use in the wizard.

#### 5. PlotConfigModal Wizard Extensions (`plot_config_modal.py`)

**Extended `PlotterSelection` dataclass:**
```python
@dataclass
class PlotterSelection:
    workflow_id: WorkflowId
    output_name: str
    plot_name: str
    correlation_axes: list[DataSourceConfig] | None = None  # NEW
```

**Extended `PlotterSelectionStep`:**
- Added state for correlation axis selection
- `is_valid()` requires axis selection for correlation histogram plotters
- `_update_axis_selection()` creates dropdown(s) when correlation histogram selected
- `_get_available_timeseries()` uses `find_timeseries_outputs()` (cached)
- `commit()` includes selected axes in `PlotterSelection.correlation_axes`

**Extended `SpecBasedConfigurationStep`:**
- `_on_config_collected()` includes correlation axes in `PlotConfig.data_sources`

#### 6. Tests

Added `TestFindTimeseriesOutputs` test class in `tests/config/workflow_spec_test.py`:
- `test_finds_timeseries_output_with_time_coord`
- `test_ignores_multidimensional_outputs`
- `test_ignores_outputs_without_time_coord`
- `test_ignores_outputs_without_default_factory`
- `test_empty_registry_returns_empty_list`
- `test_multiple_workflows_combined`

#### 7. PlotOrchestrator Multi-Source Support (`plot_orchestrator.py`)

Extended PlotOrchestrator to handle layers with multiple data sources (generic, not correlation-histogram-specific):

- `_subscribe_multi_source_layer()`: Subscribes to all unique workflows from data_sources
- `_on_multi_source_workflow_available()`: Tracks which workflows are ready
- `_setup_multi_source_pipeline()`: Sets up data pipeline when all workflows available
- Updated `_unsubscribe_and_cleanup_layer()` to handle multi-source subscriptions

The multi-source flow:
1. Extract unique workflow IDs from all data sources
2. Subscribe to all workflows, tracking readiness
3. When ALL workflows report available, build ResultKeys for all data sources
4. Call `setup_data_pipeline_from_keys()` to create merged data stream

#### 8. PlottingController Extensions (`plotting_controller.py`)

Added `setup_data_pipeline_from_keys()` - a lower-level API that accepts pre-built ResultKeys:

```python
def setup_data_pipeline_from_keys(
    self,
    keys: list[ResultKey],
    plot_name: str,
    params: dict | pydantic.BaseModel,
    on_first_data: Callable[[Any], None],
) -> None:
```

For correlation histograms, uses `OrderedCorrelationAssembler` instead of the default `MergingStreamAssembler` to preserve key order (important: first key = primary data, subsequent = correlation axes).

#### 9. Actual Correlation Histogram Plotters (`correlation_histogram.py`)

Implemented actual plotter classes that compute histograms from multiple data sources:

**`OrderedCorrelationAssembler`**: Stream assembler that preserves key order (unlike MergingStreamAssembler which sorts). Returns data as a list where [0]=primary, [1]=x-axis, [2]=y-axis.

**`CorrelationHistogram1dPlotter`**:
- Receives ordered list of DataArrays from assembler
- Computes bin edges from x-axis data range and `x_bins` param
- Uses `CorrelationHistogrammer` to compute histogram
- Returns holoviews Curve via `to_holoviews()`

**`CorrelationHistogram2dPlotter`**: Same pattern for 2D with x and y axes.

Helper function `_compute_edges_from_data()` computes evenly-spaced bin edges from data range.

#### 10. Edit Mode Support (`plot_config_modal.py`)

Extended `_update_axis_selection()` to pre-populate axis dropdowns when editing existing correlation histogram plots:
- Extracts correlation axes from `PlotConfig.data_sources[1:]`
- Matches against available timeseries options
- Pre-selects matching values in dropdowns

### Future Considerations

#### Integration with Existing Correlation Histogram Workflows

The existing correlation histogram workflow approach (via `WorkflowController.create_correlation_adapter()`) will be deprecated in favor of the plotter-based approach. The old approach can be removed cleanly without legacy support.

## File Reference

| File | Purpose |
|------|---------|
| `src/ess/livedata/config/workflow_spec.py` | `find_timeseries_outputs()` helper |
| `src/ess/livedata/dashboard/plotting.py` | Plotter registration with actual factories |
| `src/ess/livedata/dashboard/plotting_controller.py` | `setup_data_pipeline_from_keys()` for multi-source |
| `src/ess/livedata/dashboard/widgets/plot_config_modal.py` | Wizard UI with axis selection + edit mode |
| `src/ess/livedata/dashboard/plot_orchestrator.py` | Multi-source workflow coordination |
| `src/ess/livedata/dashboard/correlation_histogram.py` | Plotter classes, assembler, histogram computation |
| `tests/config/workflow_spec_test.py` | Tests for `find_timeseries_outputs()` |

## UI Mockup

When "Correlation Histogram 1D" is selected in Step 2:

```
┌────────────────────────────────────────────────┐
│ Step 2: Select Plotter Type                    │
├────────────────────────────────────────────────┤
│ ○ Timeseries                                   │
│ ○ Bars                                         │
│ ● Correlation Histogram 1D                     │
│ ○ Correlation Histogram 2D                     │
├────────────────────────────────────────────────┤
│ Correlation Axis (correlate against):          │
│ ┌──────────────────────────────────────────┐   │
│ │ ▼ Select...                              │   │
│ │   Timeseries data: motion1               │   │
│ │   Timeseries data: proton_charge         │   │
│ │   Detector View: panel_0 (counts_total)  │   │
│ └──────────────────────────────────────────┘   │
└────────────────────────────────────────────────┘
```

For 2D, two dropdowns appear: "X-Axis (correlate against)" and "Y-Axis (correlate against)".
