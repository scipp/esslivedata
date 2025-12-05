# Plot Composition System

## Motivation

### The ROI Detector Problem

The current `roi_detector` plotter requires special-case handling in `PlottingController`:
- Custom per-detector pipe creation in `setup_data_pipeline`
- Custom plot composition in `create_plot_from_pipeline`
- A dummy factory registration that raises `NotImplementedError`

This exists because ROI editing needs:
- Interactive streams (BoxEdit, PolyDraw) that persist across data updates
- Kafka subscriptions for ROI readback (bidirectional sync)
- Kafka publishing when user edits ROIs

These requirements don't fit the `Plotter` abstraction, which is designed for pure data→visualization transformation.

### The Key Insight

ROI editing is not a different way to plot data. It's **additional visual elements composed with a regular image plot**. The underlying visualization is just `ImagePlotter`—we're adding interactive shapes and Kafka sync alongside it.

### Generalization: Composition

Users have requested other composition features:
- **Peak markers**: Vertical lines at known peak positions on a spectrum
- **Reference curves**: Theoretical values overlaid on measured data
- **Multiple spectra**: Compare two data streams on one axis
- **Masks**: Show masked regions on 2D images

These all follow the same pattern: **multiple visual elements composed on a shared canvas**, each with its own data source and configuration.

## Core Insight: Everything is a Layer

Instead of distinguishing "plotters" from "overlays," we recognize that all visual elements are **layers** that get composed. Layers differ only in:

| Attribute | Examples |
|-----------|----------|
| Element type | image, curve, rectangles, vlines |
| Data source | pipeline, kafka stream, file, user interaction |
| Interaction tool | box_edit, poly_draw, or none |
| Update behavior | reactive (streams) or static |

A spectrum plot with peak markers and ROI editing on a detector image are the same mechanism—just different layer configurations.

## Architecture

### Layer Specification

```python
@dataclass(frozen=True)
class PlotLayer:
    """Specification for one visual layer in a composed plot."""

    name: str                       # Unique identifier within composition
    element: str                    # Element type: "image", "curve", "rectangles", ...
    source: DataSource              # Where data comes from
    params: dict = field(default_factory=dict)  # Element styling/options
    interaction: InteractionSpec | None = None  # Optional editing tool


@dataclass(frozen=True)
class InteractionSpec:
    """Specification for interactive editing capability on a layer."""

    tool: str                       # "box_edit" or "poly_draw"
    publish_to: str | None = None   # Result key pattern for publishing edits
    max_objects: int = 10


@dataclass(frozen=True)
class PlotComposition:
    """Complete specification for a composed plot."""

    layers: list[PlotLayer]
    shared_axes: bool = True
```

### Data Sources

Data sources abstract where layer data comes from:

```python
class DataSource(Protocol):
    """Protocol for layer data sources."""

    def get_result_keys(self) -> list[ResultKey]:
        """Return ResultKeys this source will provide data for."""
        ...


@dataclass(frozen=True)
class PipelineSource(DataSource):
    """Data from reduction pipeline via StreamManager subscription."""

    result_keys: list[ResultKey]
    window: int | None = None  # Rolling window size


@dataclass(frozen=True)
class KafkaSource(DataSource):
    """Direct Kafka subscription (e.g., ROI readback)."""

    result_keys: list[ResultKey]


@dataclass(frozen=True)
class FileSource(DataSource):
    """Static data loaded from file."""

    path: str
    format: str = "csv"


@dataclass(frozen=True)
class StaticSource(DataSource):
    """Fixed values provided directly."""

    data: Any  # e.g., [1.5, 2.3, 4.1] for VLine positions


@dataclass(frozen=True)
class InteractiveSource(DataSource):
    """Data originates from user interaction (BoxEdit, PolyDraw)."""

    initial_data: Any = None
    coordinate_reference: str | None = None  # Layer name to get coordinate info from
```

### Element Factories

Element factories transform data into HoloViews elements. Two levels:

- **Simple factories**: Single data item → Element (e.g., rectangles, vlines)
- **Multi-key factories**: Pipeline data with multiple ResultKeys → Element (existing Plotter classes)

Today's `Plotter` classes become `MultiKeyElementFactory` implementations with minimal changes.

### Plot Composer

The `PlotComposer` is responsible for:

1. Creating a `Pipe` and `DynamicMap` for each layer
2. Setting up data subscriptions based on source type
3. Attaching interactive streams (BoxEdit, PolyDraw) to appropriate layers
4. Composing all layer DynamicMaps via the `*` operator

The composer maintains `LayerState` for each layer, tracking:
- The layer specification
- Runtime artifacts: pipe, dmap, interaction stream
- Data source/generator references

## Key Implementation Insights

These insights emerged from prototyping and are critical for correct implementation:

### DynamicMap Composition Pattern

Interactive streams (BoxEdit, PolyDraw) **must attach to the DynamicMap, not the element**:

| Pattern | Works? | Issue |
|---------|--------|-------|
| `DynamicMap(Image * Rectangles)` | No | BoxEdit doesn't respond to clicks |
| `DynamicMap(Image) * Rectangles` | No | Programmatic updates don't work |
| `DynamicMap(Image) * DynamicMap(Rectangles)` | **Yes** | Both work correctly |

The layer model naturally produces the working pattern: each layer creates its own DynamicMap, and layers compose via the `*` operator.

### Tool Registration Varies by Stream Type

- **BoxEdit/PolyDraw**: Automatically add their toolbar tool
- **BoundsX/BoundsXY**: Require explicit tool registration via `.opts(tools=['xbox_select'])`

This is a HoloViews quirk that must be handled per-stream-type.

### Panel Session Closure Capture

When creating callbacks for Panel applications, variables must be explicitly captured to avoid closure issues across sessions:

```python
# Correct: capture variables as default arguments
def make_element(data, _factory=factory, _params=params):
    return _factory(data, **_params)

# Incorrect: relies on closure (breaks across Panel sessions)
def make_element(data):
    return factory(data, **params)
```

### DynamicMap Truthiness

HoloViews `DynamicMap` evaluates to `False` when empty. Use explicit `is not None` checks:

```python
# Correct
dmaps = [s.dmap for s in layers.values() if s.dmap is not None]

# Incorrect (filters out empty but valid DynamicMaps)
dmaps = [s.dmap for s in layers.values() if s.dmap]
```

### Single Update Callback with Batching

Use one shared periodic callback for all dynamic layers, with `pn.io.hold()` for batched updates:

```python
def update_all_layers():
    with pn.io.hold():
        for layer in dynamic_layers:
            layer.pipe.send(new_data)
```

This is more efficient than per-layer callbacks and reduces visual flicker.

### Full Composition Rebuild is Acceptable

Layer add/remove operations are rare (user-initiated). Rebuilding the entire composition on each change is simpler than incremental updates and performs well in practice.

## Integration with Existing Configuration

The layer model integrates with the existing `PlotConfig` / `GridSpec` configuration system.

### Current PlotConfig

```python
@dataclass
class PlotConfig:
    workflow_id: WorkflowId
    source_names: list[str]
    plot_name: str  # "image", "lines", "roi_detector"
    params: pydantic.BaseModel
    output_name: str = 'result'
```

This is a **single-layer specification**: one pipeline data source, one element type.

### Evolution: Symmetric Layer Model

`PlotCell` evolves to hold a list of peer layers:

```python
@dataclass
class PlotCell:
    geometry: CellGeometry
    layers: list[LayerConfig]  # All layers are peers
```

Where `LayerConfig` generalizes `PlotConfig` using the `DataSource` abstraction:

```python
@dataclass
class LayerConfig:
    name: str
    element: str  # "image", "curve", "rectangles", "vlines"
    source: DataSource  # PipelineSource, KafkaSource, StaticSource, InteractiveSource
    params: dict
    interaction: InteractionSpec | None = None
```

The current `PlotConfig` maps directly to a `LayerConfig` with a `PipelineSource`:

```python
# PlotConfig → LayerConfig conversion
LayerConfig(
    name=config.plot_name,
    element=config.plot_name,  # "image", "lines", etc.
    source=PipelineSource(
        workflow_id=config.workflow_id,
        source_names=config.source_names,
        output_name=config.output_name,
    ),
    params=config.params,
)
```

For backward compatibility, existing single-`PlotConfig` YAML converts to a single-element `layers` list. The exact migration strategy should be determined during implementation

## Reusing Existing Code

Most existing code can be reused:

| Existing | Becomes |
|----------|---------|
| `ImagePlotter` | `ImageElementFactory` (multi-key) |
| `LinesPlotter` | `CurveElementFactory` (multi-key) |
| `ROIPlotState` | Reused in interaction handling |
| `GeometryHandler`, converters | Reused for ROI serialization |
| `parse_readback_by_type` | Reused for Kafka parsing |

The main change is organizational: ROI logic moves from a "special plotter" to interaction handling within the composition model.

## Comparison: Overlay Model vs Layer Model

| Aspect | Overlay Model | Layer Model |
|--------|---------------|-------------|
| Core abstraction | Plotter + Overlay (two concepts) | Layer (one concept) |
| Hierarchy | Overlay "decorates" base plot | Layers are peers |
| ROI editing | "ROI Overlay" on image | Image layer + ROI layers |
| Peak markers | "Peak Overlay" on spectrum | Spectrum layer + VLines layer |
| Multi-spectrum | Not addressed | Multiple curve layers |
| Special cases | Less than before, but "overlay" is still special | None—uniform handling |
| Composition | `base * overlay` | `layer * layer * layer` |

## Migration Path

The migration is split into two phases. Phase 1 validates the composition model with simpler use cases before tackling ROI complexity.

### Phase 1: Multi-layer composition (no interaction)

This phase delivers useful features (multi-workflow comparison, peak markers) while validating the core model.

1. **Extend PlotCell**: Change from single `config: PlotConfig` to `layers: list[LayerConfig]`
2. **Implement simple data sources**: `PipelineSource` (wraps existing workflow subscription) and `StaticSource` (fixed data like peak positions)
3. **Create PlotComposer**: Handles DynamicMap composition for multi-layer cells
4. **Update PlottingController**: Minimal changes to use composer for multi-layer cells; existing plotters remain as-is
5. **Add UI for composition**: Multi-curve overlay (select workflows to compare) and peak marker layer (select from templates)

At this point, existing `roi_detector` plotter continues to work unchanged.

### Phase 2: Interactive layers and ROI

Once Phase 1 is validated in production, extend to interactive layers.

6. **Implement remaining data sources**: `KafkaSource` (ROI readback) and `InteractiveSource` (user-drawn shapes)
7. **Add interaction handling**: Extract BoxEdit/PolyDraw logic from `ROIDetectorPlotFactory`
8. **Deprecate roi_detector**: Map to equivalent composition (image layer + ROI layers)
9. **Update UI**: Full layer management in plot settings

## Benefits

1. **Unified model**: No artificial Plotter/Overlay distinction
2. **Removes all special cases** from PlottingController
3. **Composable**: Any layers can be combined
4. **Extensible**: New element types and data sources without core changes
5. **Clear data flow**: Each layer has explicit data source
6. **Enables new features**: Multi-spectrum, reference curves, annotations
7. **Correct by construction**: Layer model produces working DynamicMap composition

## Implementation Status

### Phase 1 Complete

Phase 1 has been fully implemented with UI integration.

#### Core Infrastructure (`src/ess/livedata/dashboard/plot_composer.py`)

**Data Sources:**
- `DataSource` protocol defining the interface
- `PipelineSource` - wraps workflow subscription (workflow_id, job_number, source_names)
- `StaticSource` - fixed data provided directly (e.g., peak positions)

**Layer Configuration:**
- `LayerConfig` - frozen dataclass specifying name, element type, source, and params

**Element Factories:**
- `ElementFactory` base class for creating HoloViews elements
- `VLinesFactory` - vertical lines for peak markers
- `HLinesFactory` - horizontal lines for thresholds

**PlotComposer:**
- `add_pipeline_layer()` - adds layers with pipeline data sources
- `add_static_layer()` - adds layers with static data
- `remove_layer()` - removes layers by name
- `update_static_layer()` - updates static layer data
- `get_composition()` - returns composed DynamicMap via `*` operator
- Proper handling of DynamicMap truthiness (explicit `is not None` checks)
- Closure capture pattern for Panel session safety

#### PlotCell Extension (`src/ess/livedata/dashboard/plot_orchestrator.py`)

- `PlotCell` now has `additional_layers: list[PlotConfig]` field
- Primary layer remains in `config` for backward compatibility
- New methods: `add_layer()`, `remove_layer()`, `get_layer_count()`, `get_all_layers()`
- Serialization/deserialization updated to persist additional layers
- `_on_job_available()` routes to single-layer or multi-layer pipeline setup

#### PlottingController Integration (`src/ess/livedata/dashboard/plotting_controller.py`)

- `setup_multi_layer_pipeline()` - sets up pipelines for all layers, waits for all to be ready
- `_compose_layers()` - composes multiple DynamicMaps via `*` operator
- Each layer creates its own plotter and DynamicMap
- Composed result wrapped in `hv.Layout` with `shared_axes=False`

#### UI Integration (`src/ess/livedata/dashboard/widgets/`)

**plot_widgets.py:**
- `create_add_layer_button()` - green "+" button for adding layers

**plot_grid_tabs.py:**
- Add layer button shown on all plot cells (status and plot widgets)
- `_on_add_layer()` handler shows config modal for new layer
- Reuses existing `PlotConfigModal` for layer configuration

#### Tests

- `tests/dashboard/plot_composer_test.py` - 27 tests for core composition
- All 671 existing dashboard tests continue to pass

The existing `roi_detector` plotter continues to work unchanged until Phase 2.

## Open Questions

1. **Layer ordering in UI**: Should users be able to reorder layers? Does visual order matter?
2. **Shared vs independent axes**: When overlaying curves with different ranges, union or separate?
3. **Layer visibility toggles**: Should layers be individually hideable?
4. **Composition templates**: Pre-defined compositions for common patterns (roi_detector, comparison, etc.)?
