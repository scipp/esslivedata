# Multi-View Detector Workflow Implementation Plan

## Goal

Extend the `detector_view` subpackage (in `handlers/detector_view/`) to support multiple simultaneous views of the same detector data. For example: one geometric projection (xy_plane) and two logical views (panel fold, tube fold).

## Current State

The workflow supports **one view per source** with per-detector configuration via `ViewConfig` classes:

### View Configuration

```python
@dataclass(frozen=True, slots=True)
class GeometricViewConfig:
    projection_type: Literal['xy_plane', 'cylinder_mantle_z']
    resolution: dict[str, int]
    pixel_noise: Literal['cylindrical'] | sc.Variable | None = None

@dataclass(frozen=True, slots=True)
class LogicalViewConfig:
    transform: Callable[[sc.DataArray, str], sc.DataArray] | None = None
    reduction_dim: str | list[str] | None = None

ViewConfig = GeometricViewConfig | LogicalViewConfig
```

### Factory API

The factory accepts either a single config (applied to all sources) or a dict for per-detector settings:

```python
# Single config for all sources
factory = DetectorViewScilineFactory(
    data_source=NeXusDetectorSource(nexus_file),
    tof_bins=sc.linspace('tof', 0, 1e8, 100, unit='ns'),
    view_config=GeometricViewConfig(
        projection_type='xy_plane',
        resolution={'y': 200, 'x': 200},
        pixel_noise=sc.scalar(4.0, unit='mm'),
    ),
)

# Per-detector configs (e.g., DREAM with different detector types)
factory = DetectorViewScilineFactory(
    data_source=NeXusDetectorSource(nexus_file),
    tof_bins=sc.linspace('tof', 0, 1e8, 100, unit='ns'),
    view_config={
        'mantle_detector': GeometricViewConfig(
            projection_type='cylinder_mantle_z',
            resolution={'arc_length': 80, 'z': 320},
            pixel_noise=sc.scalar(4.0, unit='mm'),
        ),
        'endcap_detector': GeometricViewConfig(
            projection_type='xy_plane',
            resolution={'y': 240, 'x': 160},
            pixel_noise=sc.scalar(4.0, unit='mm'),
        ),
    },
)

workflow = factory.make_workflow('mantle_detector')
```

### Projector Abstraction

```python
class Projector(Protocol):
    """Protocol for event projection strategies."""

    def project_events(self, events: sc.DataArray) -> sc.DataArray:
        """Project events from detector pixels to screen coordinates."""
        ...

    @property
    def screen_coords(self) -> dict[str, sc.Variable | None]:
        """Screen coordinate bin edges, keyed by dimension name."""
        ...
```

**Implementations:**
- `GeometricProjector`: Projects using calibrated positions and noise replicas
- `LogicalProjector`: Reshapes data using fold/slice transforms

## Design Approach for Multi-View

**Move polymorphism from providers to params.**

To support multiple simultaneous views of the same data:
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

### Phase 1: Type System Foundation

1. Define `ViewType` TypeVar
2. Define built-in view markers (`XYPlaneView`, `CylinderMantleView`)
3. Update types to be generic over `ViewType` using `sciline.Scope`

### Phase 2: Update Providers

1. Update all providers to be generic over `ViewType`
2. Example:
   ```python
   def project_events(
       raw_detector: RawDetector[SampleRun],
       projector: Projector[ViewType],
   ) -> ScreenBinnedEvents[ViewType]:
       ...
   ```

### Phase 3: Update Factory

1. Extend `ViewConfig` classes with `view_type` field for multi-view scenarios
2. Update factory to accept list of view configurations
3. Create per-view accumulators and target key mappings

## Module Structure

```
handlers/
├── detector_view/
│   ├── __init__.py           # Re-export public API
│   ├── types.py              # Type definitions incl. ViewConfig classes
│   ├── projectors.py         # GeometricProjector, LogicalProjector, protocol
│   ├── providers.py          # Sciline providers
│   ├── roi.py                # ROI-related providers
│   ├── workflow.py           # create_base_workflow, add_*_projection
│   ├── data_source.py        # DetectorDataSource hierarchy
│   └── factory.py            # DetectorViewScilineFactory
```

## Example: Future Multi-View Usage

```python
# Define custom logical view types
class PanelView:
    """Logical view showing detector panels."""

class TubeView:
    """Logical view showing detector tubes."""

# Create factory with multiple views of the same data
factory = DetectorViewScilineFactory(
    data_source=NeXusDetectorSource(nexus_file),
    tof_bins=sc.linspace('tof', 0, 1e8, 100, unit='ns'),
    views=[
        GeometricViewConfig(
            view_type=XYPlaneView,
            projection_type='xy_plane',
            resolution={'x': 200, 'y': 200},
        ),
        LogicalViewConfig(
            view_type=PanelView,
            transform=lambda da, src: da.fold('detector_number', {'panel': 10, 'pixel': 1000}),
        ),
    ],
)

# Workflow produces outputs for all views
workflow = factory.make_workflow('detector_panel')
# Target keys: CumulativeDetectorImage[XYPlaneView], CumulativeDetectorImage[PanelView], ...
```
