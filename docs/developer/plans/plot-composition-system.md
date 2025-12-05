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
    # Future: axis configuration, layout options
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

    def get_result_keys(self) -> list[ResultKey]:
        return self.result_keys


@dataclass(frozen=True)
class KafkaSource(DataSource):
    """Direct Kafka subscription (e.g., ROI readback)."""

    result_keys: list[ResultKey]  # Keys for the readback data

    def get_result_keys(self) -> list[ResultKey]:
        return self.result_keys


@dataclass(frozen=True)
class FileSource(DataSource):
    """Static data loaded from file."""

    path: str
    format: str = "csv"  # csv, json, etc.

    def get_result_keys(self) -> list[ResultKey]:
        return []  # No ResultKeys, data is static


@dataclass(frozen=True)
class StaticSource(DataSource):
    """Fixed values provided directly."""

    data: Any  # e.g., [1.5, 2.3, 4.1] for VLine positions

    def get_result_keys(self) -> list[ResultKey]:
        return []


@dataclass(frozen=True)
class InteractiveSource(DataSource):
    """Data originates from user interaction (BoxEdit, PolyDraw)."""

    initial_data: Any = None
    coordinate_reference: str | None = None  # Layer name to get coordinate info from

    def get_result_keys(self) -> list[ResultKey]:
        return []
```

### Element Factories

Element factories transform data into HoloViews elements. Two levels:

```python
class ElementFactory(Protocol):
    """Simple factory: single data item → Element."""

    def __call__(self, data: Any, **params) -> hv.Element:
        ...


class MultiKeyElementFactory(Protocol):
    """Factory for pipeline data with multiple ResultKeys."""

    def initialize(self, data: dict[ResultKey, sc.DataArray]) -> None:
        """One-time setup from first data (dimensions, ranges, etc.)."""
        ...

    def __call__(self, data: dict[ResultKey, sc.DataArray]) -> hv.Element:
        """Transform keyed data to element(s)."""
        ...

    @property
    def kdims(self) -> list[hv.Dimension]:
        """Key dimensions for widget integration."""
        ...
```

Today's `Plotter` classes become `MultiKeyElementFactory` implementations. Simple elements (rectangles, vlines) use `ElementFactory`.

```python
# Simple element factories
def rectangles_element(data: list[dict], **params) -> hv.Rectangles:
    """Create Rectangles from list of {x0, y0, x1, y1} dicts."""
    if not data:
        return hv.Rectangles([])
    df = pd.DataFrame(data)
    return hv.Rectangles(df).opts(**params)


def vlines_element(data: list[float], **params) -> hv.Overlay:
    """Create vertical lines at given x positions."""
    return hv.Overlay([hv.VLine(x).opts(**params) for x in data])


# Multi-key factories (existing plotters, renamed)
class ImageElementFactory:
    """Creates Image elements from detector data."""
    # ... existing ImagePlotter logic ...


class CurveElementFactory:
    """Creates Curve elements from 1D data."""
    # ... existing LinesPlotter logic ...
```

### Element Registry

```python
class ElementRegistry:
    """Registry of available element factories."""

    def __init__(self):
        self._simple: dict[str, ElementFactory] = {}
        self._multi_key: dict[str, type[MultiKeyElementFactory]] = {}

    def register_simple(self, name: str, factory: ElementFactory) -> None:
        self._simple[name] = factory

    def register_multi_key(
        self, name: str, factory_class: type[MultiKeyElementFactory]
    ) -> None:
        self._multi_key[name] = factory_class

    def get_simple(self, name: str) -> ElementFactory:
        return self._simple[name]

    def create_multi_key(
        self, name: str, params: dict
    ) -> MultiKeyElementFactory:
        return self._multi_key[name](**params)

    def is_multi_key(self, name: str) -> bool:
        return name in self._multi_key


# Default registry
element_registry = ElementRegistry()
element_registry.register_simple("rectangles", rectangles_element)
element_registry.register_simple("polygons", polygons_element)
element_registry.register_simple("vlines", vlines_element)
element_registry.register_multi_key("image", ImageElementFactory)
element_registry.register_multi_key("curve", CurveElementFactory)
```

### Plot Composer

The composer creates composed plots from layer specifications:

```python
class PlotComposer:
    """Creates composed HoloViews plots from layer specifications."""

    def __init__(
        self,
        stream_manager: StreamManager,
        element_registry: ElementRegistry,
        roi_publisher: ROIPublisher | None = None,
        logger: logging.Logger | None = None,
    ):
        self._stream_manager = stream_manager
        self._element_registry = element_registry
        self._roi_publisher = roi_publisher
        self._logger = logger or logging.getLogger(__name__)

        # Track created resources for cleanup
        self._layer_state: dict[str, LayerState] = {}

    def compose(
        self,
        composition: PlotComposition,
        on_first_data: Callable | None = None,
    ) -> hv.DynamicMap:
        """Create composed plot from specification.

        Parameters
        ----------
        composition:
            The plot composition specification.
        on_first_data:
            Callback when first data arrives (for UI initialization).

        Returns
        -------
        :
            Composed DynamicMap ready for display.
        """
        # Build coordinate info from data-bearing layers (for interactive layers)
        coord_info = self._build_coordinate_info(composition)

        # Create DynamicMap for each layer
        dmaps: list[hv.DynamicMap] = []
        for layer in composition.layers:
            dmap = self._create_layer(layer, coord_info, on_first_data)
            dmaps.append(dmap)
            # Only first layer triggers on_first_data
            on_first_data = None

        # Compose all layers
        # Order matters: first layer is "bottom", subsequent layers on top
        result = dmaps[0]
        for dmap in dmaps[1:]:
            result = result * dmap

        return result

    def _create_layer(
        self,
        layer: PlotLayer,
        coord_info: CoordinateInfo,
        on_first_data: Callable | None,
    ) -> hv.DynamicMap:
        """Create DynamicMap for a single layer."""

        # Create data pipe for this layer
        pipe = self._create_pipe_for_source(layer.source, on_first_data)

        # Get element factory
        if self._element_registry.is_multi_key(layer.element):
            factory = self._element_registry.create_multi_key(
                layer.element, layer.params
            )
            element_fn = self._wrap_multi_key_factory(factory, pipe)
            kdims = factory.kdims
        else:
            simple_factory = self._element_registry.get_simple(layer.element)
            element_fn = lambda data, f=simple_factory, p=layer.params: f(data, **p)
            kdims = []

        # Create DynamicMap
        dmap = hv.DynamicMap(element_fn, streams=[pipe], kdims=kdims)

        # Attach interaction if specified
        if layer.interaction:
            self._attach_interaction(layer, dmap, pipe, coord_info)

        # Track state for cleanup
        self._layer_state[layer.name] = LayerState(pipe=pipe, dmap=dmap)

        return dmap

    def _create_pipe_for_source(
        self,
        source: DataSource,
        on_first_data: Callable | None,
    ) -> hv.streams.Pipe:
        """Create appropriate Pipe for data source type."""

        if isinstance(source, PipelineSource):
            # Subscribe to pipeline via stream_manager
            extractors = self._create_extractors(source)
            return self._stream_manager.make_merging_stream(
                extractors, on_first_data=on_first_data
            )

        elif isinstance(source, KafkaSource):
            # Direct Kafka subscription for readback
            return self._create_kafka_subscription(source)

        elif isinstance(source, FileSource):
            # Load file data once, create static pipe
            data = self._load_file(source)
            pipe = hv.streams.Pipe(data=data)
            return pipe

        elif isinstance(source, StaticSource):
            # Static data, no updates
            return hv.streams.Pipe(data=source.data)

        elif isinstance(source, InteractiveSource):
            # Empty pipe, will be populated by interaction stream
            return hv.streams.Pipe(data=source.initial_data or [])

        else:
            raise ValueError(f"Unknown source type: {type(source)}")

    def _attach_interaction(
        self,
        layer: PlotLayer,
        dmap: hv.DynamicMap,
        pipe: hv.streams.Pipe,
        coord_info: CoordinateInfo,
    ) -> None:
        """Attach interactive editing stream to layer.

        Critical: The interaction stream (BoxEdit, PolyDraw) must use the
        DynamicMap as its source. This is required for both:
        1. Click detection (HoloViews routes clicks to the source element)
        2. Programmatic updates (Pipe updates flow through DynamicMap)
        """
        interaction = layer.interaction

        if interaction.tool == "box_edit":
            stream = hv.streams.BoxEdit(
                source=dmap,
                num_objects=interaction.max_objects,
            )
        elif interaction.tool == "poly_draw":
            stream = hv.streams.PolyDraw(
                source=dmap,
                num_objects=interaction.max_objects,
            )
        else:
            raise ValueError(f"Unknown interaction tool: {interaction.tool}")

        # Subscribe to edits
        def on_edit(**data):
            # Update pipe to re-render
            normalized = self._normalize_stream_data(data, interaction.tool)
            pipe.send(normalized)

            # Publish if configured
            if interaction.publish_to and self._roi_publisher:
                self._publish_edit(interaction, normalized, coord_info)

        stream.add_subscriber(on_edit)

        # Store stream reference
        self._layer_state[layer.name].interaction_stream = stream
```

### Coordinate Information

For ROI publishing, we need coordinate units from the data:

```python
@dataclass
class CoordinateInfo:
    """Coordinate system information for serialization."""

    x_unit: str | None = None
    y_unit: str | None = None
    x_dim: str | None = None
    y_dim: str | None = None


class PlotComposer:
    def _build_coordinate_info(
        self, composition: PlotComposition
    ) -> CoordinateInfo:
        """Extract coordinate info from data-bearing layers.

        Interactive layers may reference a specific layer for coordinates,
        or we use the first multi-key layer (typically the image/base data).
        """
        # Find layers that interactive sources reference
        references = {
            layer.source.coordinate_reference
            for layer in composition.layers
            if isinstance(layer.source, InteractiveSource)
            and layer.source.coordinate_reference
        }

        # Find first data-bearing layer (fallback)
        for layer in composition.layers:
            if isinstance(layer.source, (PipelineSource, KafkaSource)):
                # This layer has data we can extract coords from
                # Actual extraction happens when data arrives
                return CoordinateInfo()  # Populated lazily

        return CoordinateInfo()
```

### Multi-Detector Handling

For multi-detector cases, layers can be expanded per ResultKey:

```python
@dataclass(frozen=True)
class PlotLayer:
    # ... existing fields ...
    per_result_key: bool = False  # If True, create separate layer per key


class PlotComposer:
    def compose(self, composition: PlotComposition, ...) -> hv.DynamicMap:
        # Expand layers marked as per_result_key
        expanded_layers = []
        for layer in composition.layers:
            if layer.per_result_key and isinstance(layer.source, PipelineSource):
                for key in layer.source.result_keys:
                    expanded = PlotLayer(
                        name=f"{layer.name}_{key.job_id.source_name}",
                        element=layer.element,
                        source=PipelineSource(result_keys=[key]),
                        params=layer.params,
                        interaction=layer.interaction,
                    )
                    expanded_layers.append(expanded)
            else:
                expanded_layers.append(layer)

        # Continue with expanded layers...
```

Alternatively, for ROI editing across multiple detectors, each detector gets its own set of ROI layers:

```python
def create_roi_detector_composition(
    detector_keys: list[ResultKey],
    workflow_id: str,
) -> PlotComposition:
    """Create composition for ROI editing on multiple detectors."""
    layers = []

    # Image layer (handles multiple keys internally)
    layers.append(PlotLayer(
        name="detector_image",
        element="image",
        source=PipelineSource(result_keys=detector_keys),
        params={"colorbar": True},
    ))

    # Per-detector ROI layers
    for key in detector_keys:
        source_name = key.job_id.source_name

        # Rectangle readback (from backend)
        layers.append(PlotLayer(
            name=f"rect_readback_{source_name}",
            element="rectangles",
            source=KafkaSource(result_keys=[
                key.model_copy(update={"output_name": "roi_rectangle"})
            ]),
            params={"line_color": "blue", "fill_alpha": 0.1},
        ))

        # Rectangle request (user interaction)
        layers.append(PlotLayer(
            name=f"rect_request_{source_name}",
            element="rectangles",
            source=InteractiveSource(coordinate_reference="detector_image"),
            params={"line_color": "blue", "line_dash": "dashed", "fill_alpha": 0.0},
            interaction=InteractionSpec(
                tool="box_edit",
                publish_to=f"{source_name}/roi_rectangle",
                max_objects=4,
            ),
        ))

        # Polygon readback
        layers.append(PlotLayer(
            name=f"poly_readback_{source_name}",
            element="polygons",
            source=KafkaSource(result_keys=[
                key.model_copy(update={"output_name": "roi_polygon"})
            ]),
            params={"line_color": "green", "fill_alpha": 0.1},
        ))

        # Polygon request (user interaction)
        layers.append(PlotLayer(
            name=f"poly_request_{source_name}",
            element="polygons",
            source=InteractiveSource(coordinate_reference="detector_image"),
            params={"line_color": "green", "line_dash": "dashed", "fill_alpha": 0.0},
            interaction=InteractionSpec(
                tool="poly_draw",
                publish_to=f"{source_name}/roi_polygon",
                max_objects=4,
            ),
        ))

    return PlotComposition(layers=layers)
```

## DynamicMap Composition: Why It Works

HoloViews has specific requirements for interactive streams to work correctly:

| Pattern | Works? | Issue |
|---------|--------|-------|
| `DynamicMap(Image * Rectangles)` | No | BoxEdit doesn't respond to clicks |
| `DynamicMap(Image) * Rectangles` | No | Programmatic updates don't work |
| `DynamicMap(Image) * DynamicMap(Rectangles)` | **Yes** | Both work correctly |

The layer model naturally produces the working pattern:
- Each layer creates its own `DynamicMap`
- Interactive streams attach to their layer's `DynamicMap` as `source`
- Layers compose via `*` operator: `dmap1 * dmap2 * dmap3`

This is the correct pattern, achieved without special-casing.

## PlottingController Simplification

The current `PlottingController` has special cases for `roi_detector`. With the layer model:

```python
class PlottingController:
    def __init__(
        self,
        job_service: JobService,
        stream_manager: StreamManager,
        plot_composer: PlotComposer,  # New
        logger: logging.Logger | None = None,
    ):
        self._job_service = job_service
        self._stream_manager = stream_manager
        self._composer = plot_composer
        self._logger = logger or logging.getLogger(__name__)

    def create_plot(
        self,
        composition: PlotComposition,
        on_first_data: Callable | None = None,
    ) -> hv.DynamicMap:
        """Create plot from composition specification.

        No special cases—all plot types use the same path.
        """
        return self._composer.compose(composition, on_first_data)
```

The `setup_data_pipeline` / `create_plot_from_pipeline` split can also be unified since the composer handles both subscription setup and plot creation.

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
  # ... polygon layers ...
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
