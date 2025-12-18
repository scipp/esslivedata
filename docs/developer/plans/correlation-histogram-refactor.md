# Correlation Histogram Refactor Plan

## Problem

The current correlation histogram implementation has two issues:

1. **Bug**: Only `data[0]` is histogrammed - if user selects monitor1 AND monitor2, monitor2 is ignored
2. **Missing display options**: Correlation histograms don't support layout/scale/aspect options that Lines and Image plotters have

## Root Cause

The interface conflates data sources (to histogram) with axis sources (to correlate against):
- `OrderedCorrelationAssembler` returns `list[sc.DataArray]` with implicit ordering
- Plotter assumes `data[0]` = single primary, `data[1]` = x-axis
- No way to express "multiple data sources" vs "axis sources"

## Solution Overview

1. **Schema change**: Separate `data_source` (singular) from `axis_sources` in PlotConfig
2. **New assembler**: `CorrelationHistogramAssembler` with explicit data/axis separation
3. **Smarter triggering**: Wait for all axes + at least one data source
4. **Display params**: Add layout/scale/ticks to correlation histogram params
5. **Delegation**: Correlation plotters delegate rendering to LinePlotter/ImagePlotter

## Implementation Steps

### Step 1: PlotConfig Schema Change

**File**: `plot_orchestrator.py`

Change from:
```python
data_sources: list[DataSourceConfig]
```

To:
```python
data_source: DataSourceConfig      # Primary data (multiple source_names OK)
axis_sources: list[DataSourceConfig]  # Empty for regular plots, 1-2 for correlation
```

- Update `PlotConfig` dataclass
- Update convenience properties
- Update `is_correlation_histogram()` helper
- Update serialization/deserialization
- Handle backward compatibility in `_parse_raw_layer()`

### Step 2: Display Params Hierarchy

**File**: `plot_params.py`

Create intermediate classes without `window`:
```python
class PlotDisplayParams1d(PlotParamsBase):
    plot_scale: PlotScaleParams
    ticks: TickParams

class PlotDisplayParams2d(PlotParamsBase):
    plot_scale: PlotScaleParams2d
    ticks: TickParams

# Existing classes now inherit:
class PlotParams1d(PlotDisplayParams1d):
    window: WindowParams
```

### Step 3: Update Correlation Histogram Params

**File**: `correlation_histogram.py`

Add display params to simplified param models:
```python
class SimplifiedCorrelationHistogram1dParams(CorrelationHistogramParams):
    bins: Bin1dParams
    display: PlotDisplayParams1d = Field(default_factory=PlotDisplayParams1d)

class SimplifiedCorrelationHistogram2dParams(CorrelationHistogramParams):
    bins: Bin2dParams
    display: PlotDisplayParams2d = Field(default_factory=PlotDisplayParams2d)
```

### Step 4: New Assembler with Structured Output

**File**: `correlation_histogram.py`

```python
@dataclass
class CorrelationHistogramData:
    data_sources: dict[ResultKey, sc.DataArray]
    axis_data: dict[str, sc.DataArray]  # 'x' and optionally 'y'

class CorrelationHistogramAssembler(StreamAssembler[ResultKey]):
    def __init__(self, data_keys: list[ResultKey], axis_keys: dict[str, ResultKey]):
        ...

    def can_trigger(self, available_keys: set[ResultKey]) -> bool:
        # All axes required + at least one data source
        ...

    def assemble(self, data: dict[ResultKey, Any]) -> CorrelationHistogramData:
        ...
```

Remove `OrderedCorrelationAssembler` (replaced).

### Step 5: Update Plotters to Handle Multiple Data Sources

**File**: `correlation_histogram.py`

```python
class CorrelationHistogram1dPlotter:
    def __init__(self, params: SimplifiedCorrelationHistogram1dParams):
        self._renderer = LinePlotter(
            scale_opts=params.display.plot_scale,
            tick_params=params.display.ticks,
            layout_params=params.display.layout,
            aspect_params=params.display.plot_aspect,
        )

    def __call__(self, data: CorrelationHistogramData) -> hv.Element:
        histograms: dict[ResultKey, sc.DataArray] = {}
        for key, source in data.data_sources.items():
            histograms[key] = self._histogrammer(source, coords=data.axis_data)
        return self._renderer(histograms)
```

### Step 6: Update PlottingController

**File**: `plotting_controller.py`

Update `setup_data_pipeline_from_keys()`:
- Accept separate `data_keys` and `axis_keys` parameters
- Use `CorrelationHistogramAssembler` instead of `OrderedCorrelationAssembler`

### Step 7: Update PlotOrchestrator

**File**: `plot_orchestrator.py`

Update multi-source handling:
- `_subscribe_multi_source_layer()`: Track axis vs data workflows separately
- `_on_multi_source_workflow_available()`: Check "all axes ready AND any data ready"
- `_setup_multi_source_pipeline()`: Pass structured keys to controller

### Step 8: Update PlotConfigModal

**File**: `plot_config_modal.py`

- Update `PlotterSelection` to use new schema
- Update `_on_config_collected()` to populate `data_source` + `axis_sources`

### Step 9: Update DataSubscriber

**File**: `data_subscriber.py`

Change `requires_all_keys: bool` to `can_trigger(available_keys)` method for flexible triggering logic.

## Testing

- Update existing correlation histogram tests
- Add tests for multiple data sources
- Add tests for progressive data arrival (axes ready, then data sources arrive one by one)

## Migration

Existing serialized configs use `data_sources: list[...]`. Handle in `_parse_raw_layer()`:
- If `axis_sources` missing and plotter is correlation histogram, extract from end of `data_sources`
- Otherwise, treat first `data_sources` entry as `data_source`
