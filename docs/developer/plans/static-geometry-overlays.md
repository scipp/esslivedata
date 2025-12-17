# Static Geometry Overlays Implementation Plan

**Issue**: #626
**Related**: #611 (correlation histograms - multiple workflow subscriptions), #619 (detector separators)

## Overview

Add support for plot layers that draw static geometric elements (rectangles, vertical lines, etc.) without subscribing to any workflow. Design the data model to also support future N-workflow subscriptions (#611).

## Schema Changes

### New `DataSourceConfig` Dataclass

```python
@dataclass
class DataSourceConfig:
    workflow_id: WorkflowId
    source_names: list[str]
    output_name: str = 'result'
```

### Modified `PlotConfig`

Replace single `workflow_id` with a list:

```python
@dataclass
class PlotConfig:
    data_sources: list[DataSourceConfig]  # len=0 static, len=1 normal, len=N correlation
    plot_name: str
    params: pydantic.BaseModel
```

- `len(data_sources) == 0`: Static overlay (no subscriptions)
- `len(data_sources) == 1`: Current behavior (single workflow)
- `len(data_sources) > 1`: Future correlation histograms (#611)

### Serialization

Update `serialize_grid()` and `_parse_single_spec()` to handle the new structure. Empty `data_sources` serializes as `[]`.

## Plotter Registry Changes

### New `PlotterCategory` Enum

```python
class PlotterCategory(enum.Enum):
    DATA = "data"      # Requires workflow data (current plotters)
    STATIC = "static"  # No data required (geometric overlays)
```

### Extended `PlotterSpec`

Add `category: PlotterCategory = PlotterCategory.DATA` field.

### Filtering Logic

- `get_available_plotters_from_spec()`: Return only `DATA` category plotters
- New `get_static_plotters()`: Return only `STATIC` category plotters

This ensures static plotters never appear for regular workflows and vice versa.

## New Static Plotters

Register plotters for geometric elements with `category=PlotterCategory.STATIC`:

- `rectangles`: Wraps `hv.Rectangles`
- `vlines`: Wraps `hv.VLines`
- `hlines`: Wraps `hv.HLines`

Each plotter's `params` Pydantic model defines the geometry data directly (no external data source).

## Static Geometry Params Models

### Widget System Constraint

`ParamWidget` supports: `float`, `int`, `bool`, `str`, `Path`, `Enum`, `Literal`. Complex types fall back to `TextInput`. `ModelWidget` renders each nested model field as a separate card with full width.

### Model Structure Guidelines

**Nested models for layout control**: `ModelWidget` renders each nested model as a collapsible card. `ParamWidget` renders fields within a model side-by-side in a `Row`. For coordinate input fields that need more space, wrap them in a dedicated nested model containing only that field — this gives the text input full card width instead of sharing a row.

**User-facing titles and descriptions**: Add `title` and `description` to the top-level fields in the params model (e.g., `geometry`, `style`). These appear as card headers and are immediately visible. Descriptions on inner fields (inside nested models) only show on hover, so rely on top-level descriptions to explain what the user should enter.

**Avoid technical jargon**: Don't use terms like "JSON" in user-facing text. Describe the expected format in plain terms (e.g., "List of [x0, y0, x1, y1] corner coordinates").

**Validation for early feedback**: String-based coordinate input is error-prone. Implement Pydantic `field_validator` on coordinate fields that validates structure (correct nesting, correct number of elements per item) and provides clear error messages. The widget system calls validation on change, so users see errors before submitting. Validation should catch: malformed syntax, wrong number of coordinates per element, non-numeric values.

### Example

```python
class CoordinatesInput(pydantic.BaseModel):
    """Wrapper for coordinate input to get full-width card."""

    coordinates: str = pydantic.Field(
        default="[]",
        title="Coordinates",
        description='e.g., [[0,0,1,1],[2,2,3,3]]'
    )

    @pydantic.field_validator('coordinates')
    @classmethod
    def validate_coordinates(cls, v: str) -> str:
        import json
        coords = json.loads(v)
        # validation logic...
        return v


class StyleParams(pydantic.BaseModel):
    """Style options grouped together."""

    color: str = pydantic.Field(default="red", title="Color")
    alpha: float = pydantic.Field(default=0.3, ge=0.0, le=1.0, title="Opacity")


class RectanglesParams(pydantic.BaseModel):
    """Top-level params with nested models for card layout."""

    geometry: CoordinatesInput = pydantic.Field(
        default_factory=CoordinatesInput,
        title="Rectangle Coordinates",
        description="List of [x0, y0, x1, y1] corner coordinates for each rectangle.",
    )
    style: StyleParams = pydantic.Field(
        default_factory=StyleParams,
        title="Appearance",
        description="Visual styling options.",
    )
```

The plotter's `create_static_plot()` parses `geometry.coordinates` into data for `hv.Rectangles`.

### Future Enhancement

If manual coordinate entry becomes common, extend `ParamWidget` to recognize `list[tuple[float, ...]]` and render a dynamic table editor. Out of scope for initial implementation.

## PlotOrchestrator Changes

### `_subscribe_layer()`

Branch on `len(config.data_sources)`:

- **0 (static)**: Skip subscription, call new `_create_static_layer_plot()` directly
- **1 (normal)**: Current behavior via `subscribe_to_workflow()`
- **>1**: Future work for #611

### New `_create_static_layer_plot()`

Creates plot immediately from params without data pipeline:

1. Call `plotter_registry.create_plotter(plot_name, params)`
2. Invoke plotter's `create_static_plot()` method (new protocol method)
3. Store in `_layer_state[layer_id]`
4. Notify cell updated

### Plotter Protocol Extension

Add optional `create_static_plot(self) -> hv.Element` method for static plotters. Data plotters don't implement this.

## Wizard Changes

### Step 1: WorkflowAndOutputSelectionStep

Add synthetic entries to namespace/workflow structure:

- Namespace: "Static Overlay" (display) / `static_overlay` (internal)
- Workflow: "Geometric" (display) / `WorkflowId(namespace="static_overlay", name="geometric")` (internal)
- Output: Single dummy option or skip output selection entirely

The synthetic workflow ID is only used for wizard navigation — it maps to `data_sources=[]` in the final `PlotConfig`.

### Step 2: PlotterSelectionStep

Modify `_get_available_plotters()`:

- If selected workflow is the synthetic static overlay → call `get_static_plotters()`
- Otherwise → call `get_available_plotters_from_spec()` (current behavior)

### Step 3: SpecBasedConfigurationStep

When `data_sources` will be empty:

- Hide source selection UI entirely
- Show only the params configuration panel

### PlotConfig Construction

In wizard completion, translate selections to `PlotConfig`:

- Synthetic static workflow → `data_sources=[]`
- Real workflow → `data_sources=[DataSourceConfig(...)]`

## Migration

Existing persisted/template configs use old schema. In `_parse_single_spec()`:

- If layer has `workflow_id` (old format): Convert to `data_sources=[DataSourceConfig(...)]`
- If layer has `data_sources` (new format): Use directly

This provides backward compatibility without a migration script.

## Testing Strategy

1. **Unit tests for schema**: Serialization round-trip with 0, 1, N data sources
2. **PlotOrchestrator tests**: Static layer creation without subscription, immediate plot availability
3. **Wizard tests**: Static overlay namespace appears, correct plotters shown per category
4. **Integration**: Add static rectangles layer to existing plot cell, verify overlay composition works

## Out of Scope

- Implementing correlation histogram support (#611) — only the schema accommodates it
- Interactive geometry editing — params are static after configuration
- Dynamic geometry from workflow data — that's a data plotter, not static overlay
