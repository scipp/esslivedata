# Multi-View Detector Workflow Implementation Plan

## Goal

Extend the `detector_view` subpackage (in `handlers/detector_view/`) to support multiple simultaneous views of the same detector data. For example: one geometric projection (xy_plane) and two logical views (panel fold, tube fold).

## Current State

The workflow currently supports **one projection type at a time**:
- Either geometric (`xy_plane`, `cylinder_mantle_z`) via `add_geometric_projection()`
- Or logical (transform + reduction) via `add_logical_projection()`

Key limitation: Both paths produce `ScreenBinnedEvents`, so only one can be active.

## Design Approach

**Move polymorphism from providers to params.**

Instead of having separate provider functions for geometric vs logical projection, we:
1. Define a `Projector` protocol that both projection types implement
2. Set concrete projector instances as workflow params: `workflow[Projector[ViewType]] = ...`
3. Keep all providers generic over `ViewType` - they delegate to the projector

This avoids provider conflicts and keeps the Sciline graph clean.

## Type System Changes

### New View Type Markers

```python
# View type markers (empty classes for type discrimination)
class XYPlaneView:
    """Marker for xy_plane geometric projection."""

class CylinderMantleView:
    """Marker for cylinder_mantle_z geometric projection."""

# Logical views are instrument-specific, defined by factory users
# Example: class PanelView, class TubeView
```

### ViewType TypeVar

```python
ViewType = TypeVar('ViewType')
"""TypeVar for view discrimination. Constrained at pipeline level."""
```

### Parameterized Types

The following types become generic over `ViewType`:

**Projection layer:**
- `Projector[ViewType]` - the projector instance (set as param)
- `ScreenBinnedEvents[ViewType]` - projection output
- `ScreenCoordInfo[ViewType]` - coordinate metadata

**Histogram layer:**
- `DetectorHistogram3D[ViewType]`
- `CumulativeHistogram[ViewType]`
- `WindowHistogram[ViewType]`

**Output layer:**
- `CumulativeDetectorImage[ViewType]`
- `CurrentDetectorImage[ViewType]`
- `CountsTotal[ViewType]`
- `CountsInTOARange[ViewType]`

**ROI layer:**
- `ROIRectangleRequest[ViewType]`
- `ROIPolygonRequest[ViewType]`
- `ROIRectangleBounds[ViewType]`
- `ROIPolygonMasks[ViewType]`
- `CumulativeROISpectra[ViewType]`
- `CurrentROISpectra[ViewType]`
- `ROIRectangleReadback[ViewType]`
- `ROIPolygonReadback[ViewType]`

**Shared (not parameterized):**
- `TOFBins` - same for all views
- `TOFSlice` - same for all views (could be per-view if needed later)
- `RawDetector[SampleRun]` - input, consumed by all projectors

### Projector Protocol

Note: The param type is `Projector[ViewType]` to avoid confusion with the existing
`EventProjector` class (which becomes `GeometricProjector`).

```python
from typing import Protocol

class ProjectorProtocol(Protocol):
    """Protocol for event projection strategies."""

    def project_events(self, events: sc.DataArray) -> sc.DataArray:
        """Project events from detector pixels to screen coordinates."""
        ...

    @property
    def y_dim(self) -> str:
        """Name of the y (vertical) screen dimension."""
        ...

    @property
    def x_dim(self) -> str:
        """Name of the x (horizontal) screen dimension."""
        ...

    @property
    def y_edges(self) -> sc.Variable | None:
        """Bin edges for y dimension. None if logical projection has no coords."""
        ...

    @property
    def x_edges(self) -> sc.Variable | None:
        """Bin edges for x dimension. None if logical projection has no coords."""
        ...
```

### Concrete Projector Implementations

**GeometricProjector** (refactor from existing `EventProjector` class):
```python
class GeometricProjector:
    """Projects events using geometric coordinate transformation."""

    def __init__(
        self,
        coords: sc.DataGroup,  # CalibratedPositionWithNoisyReplicas
        edges: sc.DataGroup,
    ) -> None: ...

    def project_events(self, events: sc.DataArray) -> sc.DataArray: ...

    # Properties for coord info
    @property
    def y_dim(self) -> str: ...
    @property
    def x_dim(self) -> str: ...
    @property
    def y_edges(self) -> sc.Variable: ...
    @property
    def x_edges(self) -> sc.Variable: ...
```

**LogicalProjector** (new class):
```python
class LogicalProjector:
    """Projects events using logical reshape and optional reduction."""

    def __init__(
        self,
        transform: Callable[[sc.DataArray], sc.DataArray] | None,
        reduction_dim: str | list[str] | None,
        # Coord info from applying transform to EmptyDetector
        y_dim: str,
        x_dim: str,
        y_edges: sc.Variable | None,
        x_edges: sc.Variable | None,
    ) -> None: ...

    def project_events(self, events: sc.DataArray) -> sc.DataArray: ...

    # Properties for coord info
    @property
    def y_dim(self) -> str: ...
    # ... etc
```

## Provider Changes

### Projection Provider (unified)

Replace `project_events_geometric` and `project_events_logical` with single generic provider:

```python
def project_events(
    raw_detector: RawDetector[SampleRun],
    projector: Projector[ViewType],
) -> ScreenBinnedEvents[ViewType]:
    """Project events to screen coordinates using the configured projector."""
    raw_detector = sc.values(raw_detector)
    return ScreenBinnedEvents[ViewType](projector.project_events(raw_detector))
```

### ScreenCoordInfo Provider (unified)

Replace `screen_coord_info_geometric` and `screen_coord_info_logical`:

```python
def screen_coord_info(
    projector: Projector[ViewType],
) -> ScreenCoordInfo[ViewType]:
    """Extract screen coordinate info from projector."""
    return ScreenCoordInfo[ViewType](
        y_dim=projector.y_dim,
        x_dim=projector.x_dim,
        y_edges=projector.y_edges,
        x_edges=projector.x_edges,
    )
```

### Downstream Providers

All downstream providers become generic over `ViewType`. Example:

```python
def compute_detector_histogram_3d(
    screen_binned_events: ScreenBinnedEvents[ViewType],
    tof_bins: TOFBins,
) -> DetectorHistogram3D[ViewType]:
    # ... same logic, just parameterized
```

### Converting NewTypes to Generic Types

**For `sc.DataArray` wrappers** (e.g., `ScreenBinnedEvents`, `CumulativeHistogram`), use `sciline.Scope`:

```python
# Before
ScreenBinnedEvents = NewType('ScreenBinnedEvents', sc.DataArray)

# After
class ScreenBinnedEvents(sciline.Scope[ViewType, sc.DataArray], sc.DataArray):
    """Events binned by screen coordinates."""
```

This follows the essreduce pattern (e.g., `RawDetector`, `Filename`).

**For structured types** like `ScreenCoordInfo` (dataclass with fields), use a generic dataclass:

```python
@dataclass
class ScreenCoordInfo(Generic[ViewType]):
    """Screen coordinate info parameterized by view type."""
    y_dim: str
    x_dim: str
    y_edges: sc.Variable | None
    x_edges: sc.Variable | None
```

Generic dataclasses work correctly as Sciline type keys (`ScreenCoordInfo[XYPlaneView]`
creates a distinct type from `ScreenCoordInfo[PanelView]`).

## Factory Changes

### DetectorViewScilineFactory

The factory signature changes to accept multiple view configurations:

```python
@dataclass
class GeometricViewConfig:
    """Configuration for a geometric projection view."""
    view_type: type  # e.g., XYPlaneView
    name: str  # Human-readable name for output keys, e.g., 'xy_plane'
    projection_type: Literal['xy_plane', 'cylinder_mantle_z']
    resolution: dict[str, int]
    pixel_noise: Literal['cylindrical'] | sc.Variable | None = None
    roi_support: bool = True  # Geometric views typically support ROI

@dataclass
class LogicalViewConfig:
    """Configuration for a logical projection view."""
    view_type: type  # e.g., PanelView
    name: str  # Human-readable name for output keys, e.g., 'panel'
    # Transform signature includes source_name for multi-panel instruments
    transform: Callable[[sc.DataArray, str], sc.DataArray] | None
    reduction_dim: str | list[str] | None = None
    roi_support: bool = True  # Set False for 1D views (after reduction)

class DetectorViewScilineFactory:
    def __init__(
        self,
        *,
        instrument: Any,
        tof_bins: sc.Variable,
        nexus_filename: pathlib.Path,
        views: list[GeometricViewConfig | LogicalViewConfig],  # Multiple views
    ) -> None: ...
```

### Workflow Construction

The `make_workflow` method:

1. Creates base workflow with generic providers
2. Sets pipeline constraints: `ViewType` bound to all configured view types
3. For each view config, creates and sets the concrete projector:
   ```python
   for config in self._views:
       if isinstance(config, GeometricViewConfig):
           projector = self._create_geometric_projector(config, source_name)
       else:
           # Bind source_name to the transform at projector creation time
           projector = self._create_logical_projector(config, source_name)
       workflow[Projector[config.view_type]] = projector
   ```
4. Creates per-view accumulators
5. Sets up per-view target keys and context keys

**source_name binding**: The `LogicalProjector` is created with a bound transform:

```python
def _create_logical_projector(
    self, config: LogicalViewConfig, source_name: str
) -> LogicalProjector:
    if config.transform is not None:
        bound_transform = lambda da: config.transform(da, source_name)
    else:
        bound_transform = None
    # Compute coord info from empty detector + transform
    coord_info = self._compute_logical_coord_info(bound_transform)
    return LogicalProjector(
        transform=bound_transform,
        reduction_dim=config.reduction_dim,
        **coord_info,
    )
```

### Target Key Mapping

Target keys become view-prefixed (see ROI Streams section for full details):

```python
target_keys = {}
for config in self._views:
    view_name = config.name  # e.g., 'xy_plane', 'panel', or '' for legacy
    target_keys.update({
        _prefixed_key('cumulative', view_name): CumulativeDetectorImage[config.view_type],
        _prefixed_key('current', view_name): CurrentDetectorImage[config.view_type],
        # ... etc
    })
```

### Accumulator Configuration

Each view needs its own accumulators:

```python
def create_accumulators(view_types: list[type]) -> dict[type, Any]:
    accumulators = {}
    for view_type in view_types:
        accumulators[CumulativeHistogram[view_type]] = EternalAccumulator()
        accumulators[WindowHistogram[view_type]] = WindowAccumulator()
    return accumulators
```

## Implementation Steps

### Phase 0: Split File into Modules

The current file is ~1450 lines. Split before refactoring for cleaner diffs and
easier code review. Suggested structure:

```
handlers/
├── detector_view/
│   ├── __init__.py                  # Re-export public API
│   ├── types.py                     # All type definitions, ViewType TypeVar
│   ├── projectors.py                # GeometricProjector, LogicalProjector, protocol
│   ├── providers.py                 # All Sciline providers
│   ├── roi.py                       # ROI-related providers and helpers
│   ├── workflow.py                  # create_base_workflow, add_*_projection
│   ├── data_source.py               # DetectorDataSource hierarchy
│   └── factory.py                   # DetectorViewScilineFactory, config classes
```

### Phase 1: Type System Foundation

1. Define `ViewType` TypeVar in `types.py`
2. Define built-in view markers (`XYPlaneView`, `CylinderMantleView`)
3. Define `ProjectorProtocol` in `projectors.py`
4. Update `ScreenCoordInfo` to be a generic dataclass

### Phase 2: Projector Implementations

1. Rename existing `EventProjector` class to `GeometricProjector`
2. Add protocol properties (`y_dim`, `x_dim`, `y_edges`, `x_edges`)
3. Create `LogicalProjector` class implementing the protocol
4. Add factory functions to create projectors from config

### Phase 3: Parameterize Types

1. Update all intermediate/output NewTypes to be generic over `ViewType`
2. Use `sciline.Scope` for `sc.DataArray` wrappers
3. Use generic dataclass for structured types like `ScreenCoordInfo`

### Phase 4: Update Providers

1. Replace dual projection providers with single generic `project_events`
2. Replace dual coord info providers with single generic `screen_coord_info`
3. Update all downstream providers to be generic over `ViewType`
4. Update ROI providers to be generic over `ViewType`

### Phase 5: Update Factory

1. Define `GeometricViewConfig` and `LogicalViewConfig` dataclasses
2. Update `DetectorViewScilineFactory.__init__` to accept `views` list
3. Update `make_workflow` to:
   - Set pipeline constraints
   - Create and set projector instances for each view
   - Create per-view accumulators
   - Map target keys with view prefixes

### Phase 6: Update Tests

1. Split test file to match module structure:
   ```
   tests/handlers/detector_view/
   ├── projectors_test.py
   ├── providers_test.py
   ├── roi_test.py
   ├── factory_test.py
   └── integration_test.py
   ```
2. Update existing tests to use new API (single view configs)
3. Add tests for multi-view workflows
4. Test that views are independent (ROI on one doesn't affect another)

## ROI Streams for Multi-View

Each view has its own coordinate system, so ROIs are view-specific. Each view gets
independent ROI request/readback streams.

### Legacy Compatibility

For single-view workflows, the view `name` can be empty (`''`). When `name=''`:
- Stream/key prefixes are omitted
- Output keys match the current format: `cumulative`, `roi_rectangle`, etc.
- Allows new backend to work with existing frontend

This requires validation: only one view may have `name=''`.

### Stream Naming

Keeps existing separate rectangle/polygon streams, with view prefix (omitted if `name=''`):

```
{job_id}/{view_name}/roi_rectangle
{job_id}/{view_name}/roi_polygon
```

Example for a job with views named 'xy_plane' and 'panel':
```
detector_panel/abc-123/xy_plane/roi_rectangle
detector_panel/abc-123/xy_plane/roi_polygon
detector_panel/abc-123/panel/roi_rectangle
detector_panel/abc-123/panel/roi_polygon
```

### Key/Stream Name Helpers

```python
def _prefixed_key(base: str, view_name: str) -> str:
    """Generate key with optional view prefix."""
    return base if view_name == '' else f'{view_name}_{base}'

def _prefixed_stream(job_id: str, base: str, view_name: str) -> str:
    """Generate stream path with optional view prefix."""
    return f"{job_id}/{base}" if view_name == '' else f"{job_id}/{view_name}/{base}"
```

### AuxSources Factory

Use a factory function to create the AuxSources class with view configs captured in closure
(avoids adding public fields that would generate UI widgets):

```python
def make_detector_view_roi_aux_sources(
    views: list[GeometricViewConfig | LogicalViewConfig],
) -> type[AuxSourcesBase]:
    """Create an AuxSources class for multi-view detector ROI streams."""

    # Only include views that support ROI
    roi_views = [v for v in views if v.roi_support]

    class DetectorViewROIAuxSources(AuxSourcesBase):
        def render(self, job_id: JobId) -> dict[str, str]:
            result = {}
            for config in roi_views:  # Captured in closure
                view_name = config.name
                result[_prefixed_key('roi_rectangle', view_name)] = \
                    _prefixed_stream(job_id, 'roi_rectangle', view_name)
                result[_prefixed_key('roi_polygon', view_name)] = \
                    _prefixed_stream(job_id, 'roi_polygon', view_name)
            return result

    return DetectorViewROIAuxSources
```

### Workflow Context Keys

```python
context_keys = {}
for config in self._views:
    if config.roi_support:
        view_name = config.name
        context_keys[_prefixed_key('roi_rectangle', view_name)] = \
            ROIRectangleRequest[config.view_type]
        context_keys[_prefixed_key('roi_polygon', view_name)] = \
            ROIPolygonRequest[config.view_type]
```

### Output Keys

```python
target_keys = {}
for config in self._views:
    view_name = config.name
    target_keys.update({
        _prefixed_key('cumulative', view_name): CumulativeDetectorImage[config.view_type],
        _prefixed_key('current', view_name): CurrentDetectorImage[config.view_type],
        _prefixed_key('counts_total', view_name): CountsTotal[config.view_type],
        _prefixed_key('counts_in_toa_range', view_name): CountsInTOARange[config.view_type],
    })
    if config.roi_support:
        target_keys.update({
            _prefixed_key('roi_spectra_cumulative', view_name): CumulativeROISpectra[config.view_type],
            _prefixed_key('roi_spectra_current', view_name): CurrentROISpectra[config.view_type],
            _prefixed_key('roi_rectangle', view_name): ROIRectangleReadback[config.view_type],
            _prefixed_key('roi_polygon', view_name): ROIPolygonReadback[config.view_type],
        })
```

### DetectorViewOutputs

The outputs spec needs per-view fields. Extend `make_detector_view_outputs()` to
generate view-specific output fields:

```python
def make_detector_view_outputs(
    views: list[GeometricViewConfig | LogicalViewConfig],
) -> type[WorkflowOutputsBase]:
    """Create DetectorViewOutputs with per-view fields."""

    fields = {}
    for config in views:
        view_name = config.name
        # Per-view image outputs (prefix omitted if view_name='')
        fields[_prefixed_key('cumulative', view_name)] = (sc.DataArray, ...)
        fields[_prefixed_key('current', view_name)] = (sc.DataArray, ...)
        fields[_prefixed_key('counts_total', view_name)] = (sc.DataArray, ...)
        fields[_prefixed_key('counts_in_toa_range', view_name)] = (sc.DataArray, ...)

        if config.roi_support:
            fields[_prefixed_key('roi_spectra_cumulative', view_name)] = (sc.DataArray, ...)
            fields[_prefixed_key('roi_spectra_current', view_name)] = (sc.DataArray, ...)
            fields[_prefixed_key('roi_rectangle', view_name)] = (sc.DataArray, ...)
            fields[_prefixed_key('roi_polygon', view_name)] = (sc.DataArray, ...)

    return pydantic.create_model('DetectorViewOutputs', __base__=WorkflowOutputsBase, **fields)
```

### Frontend Changes

- When user draws ROI on a view, frontend knows the view name
- Publish to view-specific stream: `{job_id}/{view_name}/roi_rectangle` or `roi_polygon`
- Subscribe to view-specific readback streams

## Example Usage

### Legacy-Compatible Single View

```python
# Single view with name='' for backwards compatibility with existing frontend
factory = DetectorViewScilineFactory(
    instrument=instrument,
    tof_bins=sc.linspace('tof', 0, 1e8, 100, unit='ns'),
    nexus_filename=nexus_file,
    views=[
        GeometricViewConfig(
            view_type=XYPlaneView,
            name='',  # Empty name = no prefix = legacy-compatible
            projection_type='xy_plane',
            resolution={'screen_x': 200, 'screen_y': 200},
        ),
    ],
)

workflow = factory.make_workflow('detector_panel')
# Target keys (no prefix): 'cumulative', 'current', 'roi_rectangle', 'roi_polygon', ...
# Works with existing frontend without changes
```

### Multi-View Configuration

```python
# Define custom logical view types for instrument
class PanelView:
    """Logical view showing detector panels."""

class TubeView:
    """Logical view showing detector tubes."""

# Create factory with multiple views
factory = DetectorViewScilineFactory(
    instrument=instrument,
    tof_bins=sc.linspace('tof', 0, 1e8, 100, unit='ns'),
    nexus_filename=nexus_file,
    views=[
        GeometricViewConfig(
            view_type=XYPlaneView,
            name='xy_plane',
            projection_type='xy_plane',
            resolution={'screen_x': 200, 'screen_y': 200},
            pixel_noise='cylindrical',
        ),
        LogicalViewConfig(
            view_type=PanelView,
            name='panel',
            # Transform receives source_name for multi-panel instruments
            transform=lambda da, src: da.fold('detector_number', {'panel': 10, 'pixel': 1000}),
        ),
        LogicalViewConfig(
            view_type=TubeView,
            name='tube',
            transform=lambda da, src: da.fold('detector_number', {'tube': 100, 'pixel': 100}),
            reduction_dim='pixel',
            roi_support=False,  # 1D view after reduction
        ),
    ],
)

# Workflow produces outputs for all views
workflow = factory.make_workflow('detector_panel')
# Target keys: 'xy_plane_cumulative', 'panel_cumulative', 'tube_cumulative', ...
# ROI keys only for views with roi_support=True:
#   'xy_plane_roi_rectangle', 'xy_plane_roi_polygon',
#   'panel_roi_rectangle', 'panel_roi_polygon'
```
