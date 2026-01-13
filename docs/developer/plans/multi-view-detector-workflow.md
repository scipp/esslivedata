# Multi-View Detector Workflow Implementation Plan

## Goal

Extend the `detector_view` subpackage (in `handlers/detector_view/`) to support multiple simultaneous views of the same detector data. For example: one geometric projection (xy_plane) and two logical views (panel fold, tube fold).

## Current State

The workflow supports **one projection type at a time** via a unified projector abstraction:
- `Projector` protocol defining the interface
- `GeometricProjector` for xy_plane and cylinder_mantle_z projections
- `LogicalProjector` for fold/slice transforms with optional dimension reduction
- Single `project_events` provider that works with any projector type

### Type System

**Projector Protocol:**

```python
class Projector(Protocol):
    """Protocol for event projection strategies."""

    def project_events(self, events: sc.DataArray) -> sc.DataArray:
        """Project events from detector pixels to screen coordinates."""
        ...

    @property
    def screen_coords(self) -> dict[str, sc.Variable | None]:
        """Screen coordinate bin edges, keyed by dimension name.
```

**Projector Implementations:**

- `GeometricProjector`: Projects events using calibrated positions and noise replicas
- `LogicalProjector`: Reshapes detector data using fold/slice transforms

**Projector Factories (Sciline providers):**

```python
def make_geometric_projector(
    coords: CalibratedPositionWithNoisyReplicas,
    projection_type: ProjectionType,
    resolution: DetectorViewResolution,
) -> Projector: ...

def make_logical_projector(
    empty_detector: EmptyDetector[SampleRun],
    transform: LogicalTransform,
    reduction_dim: ReductionDim,
) -> Projector: ...
```

### Provider Structure

Single unified projection provider:

```python
def project_events(
    raw_detector: RawDetector[SampleRun],
    projector: Projector,
) -> ScreenBinnedEvents:
    """Project events to screen coordinates using the configured projector."""
    raw_detector = sc.values(raw_detector)
    return ScreenBinnedEvents(projector.project_events(raw_detector))
```

ROI precomputation providers take `Projector` directly:

```python
def precompute_roi_rectangle_bounds(
    projector: Projector,
    rectangle_request: ROIRectangleRequest,
) -> ROIRectangleBounds: ...

def precompute_roi_polygon_masks(
    projector: Projector,
    polygon_request: ROIPolygonRequest,
) -> ROIPolygonMasks: ...
```

## Design Approach for Multi-View

**Move polymorphism from providers to params.**

To support multiple views, parameterize types by `ViewType`:
1. Define view type markers (e.g., `XYPlaneView`, `PanelView`)
2. Set concrete projector instances as workflow params: `workflow[Projector[ViewType]] = ...`
3. Keep all providers generic over `ViewType` - they delegate to the projector

This avoids provider conflicts and keeps the Sciline graph clean.

## Parameterized Types for Multi-View

The following types would become generic over `ViewType`:

**Projection layer:**
- `Projector[ViewType]` - the projector instance (set as param)
- `ScreenBinnedEvents[ViewType]` - projection output

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

## Implementation Steps

### Phase 1: Type System Foundation (Future Work)

1. Define `ViewType` TypeVar
2. Define built-in view markers (`XYPlaneView`, `CylinderMantleView`)
3. Update types to be generic over `ViewType` using `sciline.Scope`

### Phase 2: Update Providers (Future Work)

1. Update all providers to be generic over `ViewType`
2. Example:
   ```python
   def project_events(
       raw_detector: RawDetector[SampleRun],
       projector: Projector[ViewType],
   ) -> ScreenBinnedEvents[ViewType]:
       ...
   ```

### Phase 3: Update Factory (Future Work)

1. Define `GeometricViewConfig` and `LogicalViewConfig` dataclasses
2. Update factory to accept multiple view configurations
3. Create per-view accumulators and target key mappings

## Module Structure

```
handlers/
├── detector_view/
│   ├── __init__.py           # Re-export public API
│   ├── types.py              # Type definitions
│   ├── projectors.py         # GeometricProjector, LogicalProjector, protocol
│   ├── providers.py          # Sciline providers
│   ├── roi.py                # ROI-related providers
│   ├── workflow.py           # create_base_workflow, add_*_projection
│   ├── data_source.py        # DetectorDataSource hierarchy
│   └── factory.py            # DetectorViewScilineFactory
```

## Example Usage

### Current Single-View

```python
# Create factory with geometric projection
factory = DetectorViewScilineFactory(
    data_source=NeXusDetectorSource(nexus_file),
    tof_bins=sc.linspace('tof', 0, 1e8, 100, unit='ns'),
    projection_type='xy_plane',
    resolution={'screen_x': 200, 'screen_y': 200},
    pixel_noise='cylindrical',
)

workflow = factory.make_workflow('detector_panel')
```

### Future Multi-View

```python
# Define custom logical view types
class PanelView:
    """Logical view showing detector panels."""

class TubeView:
    """Logical view showing detector tubes."""

# Create factory with multiple views
factory = DetectorViewScilineFactory(
    data_source=NeXusDetectorSource(nexus_file),
    tof_bins=sc.linspace('tof', 0, 1e8, 100, unit='ns'),
    views=[
        GeometricViewConfig(
            view_type=XYPlaneView,
            name='xy_plane',
            projection_type='xy_plane',
            resolution={'screen_x': 200, 'screen_y': 200},
        ),
        LogicalViewConfig(
            view_type=PanelView,
            name='panel',
            transform=lambda da, src: da.fold('detector_number', {'panel': 10, 'pixel': 1000}),
        ),
    ],
)

# Workflow produces outputs for all views
workflow = factory.make_workflow('detector_panel')
# Target keys: 'xy_plane_cumulative', 'panel_cumulative', ...
```
