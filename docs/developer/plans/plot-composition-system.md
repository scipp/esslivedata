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

## Configuration Format

Plot compositions can be specified in YAML:

```yaml
# Simple spectrum plot
layers:
  - name: spectrum
    element: curve
    source:
      type: pipeline
      result_keys:
        - workflow_id: monitor_reduction
          output_name: normalized_counts

# Spectrum with peak markers
layers:
  - name: spectrum
    element: curve
    source:
      type: pipeline
      result_keys:
        - workflow_id: monitor_reduction
          output_name: normalized_counts
  - name: peaks
    element: vlines
    source:
      type: file
      path: config/known_peaks.csv
    params:
      color: red
      line_dash: dashed

# Two spectra comparison
layers:
  - name: measured
    element: curve
    source:
      type: pipeline
      result_keys:
        - workflow_id: reduction
          output_name: spectrum
  - name: reference
    element: curve
    source:
      type: pipeline
      result_keys:
        - workflow_id: reduction
          output_name: reference_spectrum
    params:
      color: gray
      alpha: 0.5

# ROI detector (replaces special roi_detector plotter)
layers:
  - name: detector
    element: image
    source:
      type: pipeline
      result_keys:
        - workflow_id: detector_reduction
          output_name: detector_image
  - name: rect_readback
    element: rectangles
    source:
      type: kafka
      result_keys:
        - workflow_id: detector_reduction
          output_name: roi_rectangle
    params:
      line_color: blue
      fill_alpha: 0.1
  - name: rect_request
    element: rectangles
    source:
      type: interactive
      coordinate_reference: detector
    params:
      line_color: blue
      line_dash: dashed
    interaction:
      tool: box_edit
      publish_to: roi_rectangle
      max_objects: 4
```

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

1. **Create layer infrastructure**: `PlotLayer`, `DataSource`, `PlotComposer`
2. **Adapt existing plotters**: Rename to `ElementFactory`, minimal changes
3. **Implement data sources**: `PipelineSource`, `KafkaSource`, `InteractiveSource`
4. **Implement interaction handling**: Extract from `ROIDetectorPlotFactory`
5. **Update PlottingController**: Use `PlotComposer`, remove special cases
6. **Add configuration parsing**: YAML → `PlotComposition`
7. **Deprecate roi_detector**: Map to equivalent composition
8. **Update UI**: Composition builder instead of plotter selector

## Benefits

1. **Unified model**: No artificial Plotter/Overlay distinction
2. **Removes all special cases** from PlottingController
3. **Composable**: Any layers can be combined
4. **Extensible**: New element types and data sources without core changes
5. **Clear data flow**: Each layer has explicit data source
6. **Enables new features**: Multi-spectrum, reference curves, annotations
7. **Correct by construction**: Layer model produces working DynamicMap composition

## Open Questions

1. **Layer ordering in UI**: Should users be able to reorder layers? Does visual order matter?
2. **Shared vs independent axes**: When overlaying curves with different ranges, union or separate?
3. **Layer visibility toggles**: Should layers be individually hideable?
4. **Composition templates**: Pre-defined compositions for common patterns (roi_detector, comparison, etc.)?
