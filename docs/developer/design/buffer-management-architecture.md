# Buffer Management Architecture

## Overview

This document describes the architecture for managing data buffers in the dashboard, focusing on the separation between temporal requirements (user/UI concerns) and spatial constraints (implementation concerns).

## The Temporal/Spatial Duality

Buffer management involves two distinct domains:

**Temporal Domain** (User/UI concerns):
- "Show me the latest value"
- "Aggregate the last 5 seconds"
- "Plot all history"

**Spatial Domain** (Implementation concerns):
- Memory allocation in frames
- Performance constraints
- Storage capacity limits

**Frame rate** is the runtime-observed conversion factor between domains:
```
spatial_size = temporal_duration × frame_rate
```

Frame rate cannot be known at design time—it's a property of the actual data stream that must be observed during operation.

## Component Responsibilities

### Extractor

**Purpose**: Define what data view is needed for a specific use case (e.g., plotting, display widget).

**Responsibilities**:
- Declare temporal coverage requirements
- Extract specific views from buffers
- Aggregate or transform data as needed for presentation

**Does NOT**:
- Know or care about frame rates
- Make buffer sizing decisions
- Manage memory constraints

**Interface**:
```
get_temporal_requirement() -> TemporalRequirement
    Returns the temporal coverage needed for this extractor.
    Examples:
    - LatestFrame: "I need the most recent single frame"
    - TimeWindow(5.0): "I need 5 seconds of temporal coverage"
    - CompleteHistory: "I need all available history"

extract(buffer: Buffer) -> Any
    Extract and transform data from the buffer.
    Uses buffer's temporal query methods (get_latest, get_window_by_duration).
```

### DataService

**Purpose**: Coordinate data distribution to subscribers and manage buffer lifecycle.

**Responsibilities**:
- Register subscribers and track their dependencies
- Route incoming data to appropriate buffers
- Trigger subscriber notifications on updates
- Create and delegate buffer management to BufferManager

**Does NOT**:
- Make retention policy decisions
- Translate temporal requirements to spatial sizes
- Manage buffer resizing or compaction

**Interface**:
```
register_subscriber(subscriber: Subscriber) -> None
    Register a subscriber with its temporal requirements.
    Delegates buffer management to BufferManager.

update(key: K, data: V) -> None
    Update buffer with new data.
    Delegates to BufferManager, then notifies subscribers.

__getitem__(key: K) -> Buffer
    Access buffer for a given key.
```

### BufferManager

**Purpose**: Translate temporal requirements into spatial sizing decisions and manage buffer retention policies.

**Responsibilities**:
- Create buffers with appropriate initial sizes
- Observe buffer metrics (frame rate, temporal coverage)
- Validate that buffers meet temporal requirements
- Resize or compact buffers to satisfy requirements under constraints
- Apply retention policies (simple sizing, compaction, downsampling)

**Does NOT**:
- Store data (delegates to Buffer)
- Know about extractors or subscribers
- Handle data routing

**Interface**:
```
create_buffer(key: K, requirements: list[TemporalRequirement]) -> Buffer
    Create a buffer sized to satisfy the given temporal requirements.
    Starts with conservative default, refines based on observations.

update_buffer(buffer: Buffer, data: V) -> None
    Update buffer with new data and apply retention policy.
    Observes metrics and resizes if needed to meet requirements.

validate_coverage(buffer: Buffer, requirements: list[TemporalRequirement]) -> bool
    Check if buffer currently provides sufficient coverage.
    Returns False if resize/compaction is needed.

add_requirement(buffer: Buffer, requirement: TemporalRequirement) -> None
    Register additional temporal requirement for an existing buffer.
    May trigger immediate resize if needed.
```

**Policy Strategies** (future extensibility):
- `SimpleRetentionPolicy`: Size buffer based on frame rate × duration
- `CompactingRetentionPolicy`: Downsample old data (keep every Nth frame)
- `MultiResolutionPolicy`: Recent high-res, older low-res
- `MemoryPressurePolicy`: Adaptive based on available memory

### Buffer

**Purpose**: Store time-series data and provide temporal query interface.

**Responsibilities**:
- Allocate and manage storage (via BufferInterface)
- Append incoming data
- Provide temporal query methods (get_latest, get_window_by_duration)
- Report observable metrics (frame rate, coverage duration, frame count)
- Support dynamic resizing (grow, never shrink)

**Does NOT**:
- Interpret temporal requirements
- Make sizing decisions
- Apply retention policies

**Interface**:
```
append(data: T) -> None
    Add new data to the buffer.

get_latest() -> T | None
    Get the most recent single frame (temporal query).

get_window_by_duration(duration_seconds: float) -> T | None
    Get data covering specified time duration (temporal query).
    Uses actual time coordinates from data.

get_all() -> T | None
    Get all buffered data.

# Observable metrics
get_observed_frame_rate() -> float | None
    Report the observed frame rate (Hz) based on received data.
    Returns None if insufficient data to estimate.

get_temporal_coverage() -> float | None
    Report the time span (seconds) currently covered by buffer.
    Returns None if buffer is empty or has no time coordinate.

get_frame_count() -> int
    Report the number of frames currently stored.

# Sizing
set_max_size(new_max_size: int) -> None
    Resize buffer capacity (can only grow, never shrink).
```

### BufferInterface

**Purpose**: Provide type-specific storage implementation (DataArray, Variable, list).

**Responsibilities**:
- Allocate storage with specific capacity
- Write data to storage in-place
- Provide views/slices of stored data
- Extract temporal windows using time coordinates
- Report storage metrics

**Does NOT**:
- Make sizing decisions
- Track frame rates
- Manage buffer lifecycle

**Interface** (unchanged from current implementation):
```
allocate(template: T, capacity: int) -> T
write_slice(buffer: T, start: int, data: T) -> None
shift(buffer: T, src_start: int, src_end: int, dst_start: int) -> None
get_view(buffer: T, start: int, end: int) -> T
get_size(data: T) -> int
get_window_by_duration(buffer: T, end: int, duration_seconds: float) -> T
extract_latest_frame(data: T) -> T
unwrap_window(view: T) -> T
```

## Interaction Flow

### Subscriber Registration

1. Subscriber registers with DataService
2. DataService extracts temporal requirements from subscriber's extractors
3. DataService delegates to BufferManager: "Create/configure buffer for key X with requirements [5 seconds, latest]"
4. BufferManager creates buffer with conservative default size (e.g., 100 frames)
5. DataService triggers subscriber with existing data

### Data Update

1. New data arrives at DataService
2. DataService delegates to BufferManager: "Update buffer for key X"
3. BufferManager:
   - Appends data to buffer via `buffer.append(data)`
   - Observes metrics: `buffer.get_observed_frame_rate()`
   - Validates coverage: "Does current coverage meet requirements?"
   - If insufficient: computes new size using observed frame rate
   - Resizes buffer: `buffer.set_max_size(new_size)`
4. DataService notifies subscribers
5. Extractors query buffer using temporal methods: `buffer.get_window_by_duration(5.0)`

### Adding New Subscriber to Existing Buffer

1. New subscriber registers with different temporal requirement (e.g., needs 10 seconds vs existing 5 seconds)
2. DataService delegates to BufferManager: "Add requirement to existing buffer"
3. BufferManager:
   - Recalculates required size using observed frame rate
   - Resizes buffer if needed
4. DataService triggers new subscriber

## Temporal Requirement Types

```
TemporalRequirement (base protocol)
    Describes what temporal coverage is needed.

LatestFrame
    Requires only the most recent single data point.

TimeWindow(duration_seconds: float)
    Requires temporal coverage of specified duration.
    Example: TimeWindow(5.0) = "last 5 seconds of data"

CompleteHistory
    Requires all available history.
    May have practical upper limit for memory constraints.
```

## Benefits of This Architecture

### Separation of Concerns
- Extractors work in temporal domain (natural for users/UI)
- Buffers work in spatial domain (natural for implementation)
- BufferManager mediates between domains

### Eliminates Guessing
- No hard-coded frame rate assumptions
- Sizing decisions based on observed metrics
- Adaptive to actual data characteristics

### Extensibility
- New temporal requirement types don't affect buffers
- New retention policies don't affect extractors
- Policy strategies can be swapped without changing interfaces

### Testability
- Components have clear responsibilities
- Temporal requirements are declarative
- Observable metrics are factual

## Future Extensions

### Advanced Retention Policies

**Compaction Policy**:
- When buffer grows too large, downsample old data
- Keep every Nth frame for data older than threshold
- Maintains temporal coverage at reduced resolution

**Multi-Resolution Policy**:
- Recent data: full resolution
- Medium age: reduced resolution (every 2nd frame)
- Old data: sparse sampling (every 10th frame)
- Still provides requested temporal coverage

**Memory-Pressure Policy**:
- Monitor system memory usage
- Adaptively reduce buffer sizes when under pressure
- Prioritize critical buffers over less-important ones

### Instrument-Specific Strategies

Different instruments may have different characteristics:
- High-rate detectors: aggressive compaction needed
- Low-rate monitors: simple sizing sufficient
- Bursty sources: over-provision for spikes

BufferManager can select appropriate policy based on instrument configuration.

## Migration Path

1. Add `get_temporal_requirement()` to extractor interface alongside existing `get_required_size()`
2. Implement BufferManager with simple policy (replicates current behavior)
3. Add observable metrics to Buffer (`get_observed_frame_rate()`, `get_temporal_coverage()`)
4. Update DataService to delegate buffer management to BufferManager
5. Migrate extractors to use temporal requirements
6. Remove `get_required_size()` from extractor interface
7. Implement advanced retention policies as needed
