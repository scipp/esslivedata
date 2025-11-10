# DataService Buffering Integration

## Problem Statement

The current architecture has a fundamental issue with the transaction mechanism "swallowing" intermediate data updates:

1. **Orchestrator** batches Kafka messages in a transaction (`orchestrator.py:57`)
2. **Transaction coalescing**: Multiple updates to the same key → only final value notifies subscribers
3. **Example**: Key updated with values 20, 30, 40 → subscribers only see 40

This is:
- ✅ **Perfect** for regular plotters (they only want the latest image/data)
- ❌ **Fatal** for time-series buffers that need every update (SlidingWindowPlotter)

Current workaround (HistoryBufferService) subscribes to DataService but gets coalesced data, missing intermediate updates.

## Architectural Decision

**Integrate buffering directly into DataService** - "latest value" is just a buffer of size 1.

### Key Insight
All plotters subscribe to the same service and specify what they need via an `UpdateExtractor`:
- **Regular plotters**: `LatestValueExtractor` (extracts last element)
- **SlidingWindowPlotter**: `WindowExtractor(size=100)`
- **Future use cases**: `FullHistoryExtractor`, custom extractors

## Design Details

### Shared Buffers Per Key

```
Key → Buffer (size = max requirement of all subscribers)
       ↓
       ├→ Subscriber A extracts via LatestValueExtractor
       ├→ Subscriber B extracts via WindowExtractor(size=100)
       └→ Subscriber C extracts via WindowExtractor(size=50)
```

**One buffer per key** (not per-subscriber-per-key like current HistoryBufferService).

### Buffer Sizing Logic

- **Default**: size 1 (latest value only)
- **On subscriber registration**: `buffer_size = max(current_size, all_subscribers_requirements)`
- **Extractor requirements**:
  - `LatestValueExtractor` → size 1
  - `WindowExtractor(n)` → size n
  - `FullHistoryExtractor` → size 10000 (or DEFAULT_MAX_SIZE)

### Buffer Lifecycle Examples

1. **No subscribers**: Buffer has `max_size=1` (latest value only)
2. **First subscriber** (`WindowExtractor(100)`): Buffer grows to `max_size=100`
3. **Second subscriber** (`WindowExtractor(50)`): Buffer stays at `max_size=100` (sufficient)
4. **Third subscriber** (`FullHistoryExtractor`): Buffer grows to `max_size=10000`

### Buffer Shrinking

**Decision**: Don't implement initially. Once grown, buffers stay grown.
- Simpler implementation
- Avoids data loss if subscriber re-registers
- Can add later if memory becomes a concern

## Implementation Approach

### 1. Buffer Size Calculation

Generate dynamically from subscribers (no separate requirements store):

```python
def _get_required_buffer_size(self, key: K) -> int:
    """Calculate required buffer size for a key based on all subscribers."""
    max_size = 1  # Default: latest value only
    for subscriber in self._subscribers:
        if key in subscriber.keys:
            extractor = subscriber.extractors[key]
            if isinstance(extractor, WindowExtractor):
                max_size = max(max_size, extractor.window_size)
            elif isinstance(extractor, FullHistoryExtractor):
                max_size = max(max_size, DEFAULT_MAX_SIZE)
            # LatestValueExtractor -> size 1 (no change)
    return max_size
```

### 2. Subscriber Registration

```python
def register_subscriber(self, subscriber: SubscriberProtocol):
    self._subscribers.append(subscriber)

    # Update buffer sizes for affected keys
    for key in subscriber.keys:
        if key in self._buffers:
            required_size = self._get_required_buffer_size(key)
            self._buffers[key].set_max_size(required_size)
        # If no buffer yet, created on first data arrival with correct size
```

### 3. Data Updates

```python
def __setitem__(self, key: K, value: V):
    # Create buffer lazily if needed
    if key not in self._buffers:
        required_size = self._get_required_buffer_size(key)
        self._buffers[key] = self._buffer_factory.create_buffer(value, required_size)

    # Always append to buffer (even during transaction)
    self._buffers[key].append(value)

    # Mark for notification
    self._pending_updates.add(key)
    self._notify_if_not_in_transaction()
```

### 4. Notification

```python
def _notify_subscribers(self, updated_keys: set[K]) -> None:
    for subscriber in self._subscribers:
        if hasattr(subscriber, 'keys') and hasattr(subscriber, 'trigger'):
            if updated_keys & subscriber.keys:
                # Extract data per key using subscriber's extractors
                extracted_data = {}
                for key in (updated_keys & subscriber.keys):
                    if key in self._buffers:
                        extractor = subscriber.extractors[key]
                        data = extractor.extract(self._buffers[key])
                        if data is not None:
                            extracted_data[key] = data

                if extracted_data:
                    subscriber.trigger(extracted_data)
        else:
            # Plain callable - gets key names only (legacy support)
            subscriber(updated_keys)
```

## Required Changes

### 1. Buffer Class Enhancement

Add dynamic resizing to `Buffer`:

```python
class Buffer:
    def set_max_size(self, new_max_size: int):
        """Grow max_size (never shrink)."""
        if new_max_size > self._max_size:
            self._max_size = new_max_size
            self._max_capacity = int(new_max_size * self._overallocation_factor)
```

### 2. UpdateExtractor Types

Already exist in `history_buffer_service.py`:
- `UpdateExtractor` (ABC)
- `FullHistoryExtractor`
- `WindowExtractor`

Need to add:
- `LatestValueExtractor` (for backward compatibility with existing plotters)

### 3. ListBuffer Implementation

Add simple list-based buffer for testing and non-scipp types:

```python
class ListBuffer(BufferInterface[list]):
    """Simple list-based buffer for non-scipp types."""

    def allocate(self, template: Any, capacity: int) -> list:
        """Allocate empty list."""
        return []

    def write_slice(self, buffer: list, start: int, end: int, data: Any) -> None:
        """Append data to list."""
        # For ListBuffer, we just append (ignore indices)
        if isinstance(data, list):
            buffer.extend(data)
        else:
            buffer.append(data)

    def shift(self, buffer: list, src_start: int, src_end: int, dst_start: int) -> None:
        """Shift list elements."""
        buffer[dst_start:dst_start + (src_end - src_start)] = buffer[src_start:src_end]

    def get_view(self, buffer: list, start: int, end: int) -> list:
        """Get slice of list."""
        return buffer[start:end]

    def get_size(self, data: Any) -> int:
        """Get size of data."""
        if isinstance(data, list):
            return len(data)
        return 1
```

### 4. SubscriberProtocol Update

```python
class SubscriberProtocol(Protocol[K]):
    @property
    def keys(self) -> set[K]:
        """Return the set of data keys this subscriber depends on."""

    @property
    def extractors(self) -> dict[K, UpdateExtractor]:
        """Return extractors for obtaining data views."""

    def trigger(self, store: dict[K, Any]) -> None:
        """Trigger the subscriber with extracted data."""
```

### 4. BufferFactory - Separation of Concerns

DataService should not know about buffer implementation details (concat_dim, DataArrayBuffer, etc.).
A unified factory handles type-based dispatch:

```python
class BufferFactory:
    """
    Factory that creates appropriate buffers based on data type.

    Maintains a registry of type → BufferInterface mappings.
    """

    def __init__(self,
                 concat_dim: str = "time",
                 initial_capacity: int = 100,
                 overallocation_factor: float = 2.5) -> None:
        self._concat_dim = concat_dim
        self._initial_capacity = initial_capacity
        self._overallocation_factor = overallocation_factor

        # Default type registry
        self._buffer_impls: dict[type, Callable[[], BufferInterface]] = {
            sc.DataArray: lambda: DataArrayBuffer(concat_dim=self._concat_dim),
            sc.Variable: lambda: VariableBuffer(concat_dim=self._concat_dim),
            # ListBuffer as fallback for simple types (int, str, etc.)
        }

    def create_buffer(self, template: T, max_size: int) -> Buffer[T]:
        """Create buffer appropriate for the data type."""
        data_type = type(template)

        # Find matching buffer implementation
        if data_type in self._buffer_impls:
            buffer_impl = self._buffer_impls[data_type]()
        else:
            # Default fallback for unknown types
            buffer_impl = ListBuffer()

        return Buffer(
            max_size=max_size,
            buffer_impl=buffer_impl,
            initial_capacity=self._initial_capacity,
            overallocation_factor=self._overallocation_factor,
            concat_dim=self._concat_dim,
        )

    def register_buffer_impl(
        self, data_type: type, impl_factory: Callable[[], BufferInterface]
    ) -> None:
        """Register custom buffer implementation for a type."""
        self._buffer_impls[data_type] = impl_factory
```

**Usage:**
```python
# Production - one factory for all types
factory = BufferFactory(concat_dim="time")
data_service = DataService(buffer_factory=factory)

# Tests - same factory, uses ListBuffer for simple types automatically
factory = BufferFactory()
data_service = DataService(buffer_factory=factory)
data_service["key"] = 42  # Automatically uses ListBuffer
```

### 5. DataService Updates - Buffers as Primary Storage

**Key change**: DataService inherits from `MutableMapping` instead of `UserDict`. Buffers ARE the storage.

```python
from collections.abc import MutableMapping

class DataService(MutableMapping[K, V]):
    """
    Service for managing data with buffering and subscriber notifications.

    Buffers serve as the primary storage. __getitem__ returns the latest value
    from the buffer.
    """

    def __init__(self, buffer_factory: BufferFactory[V]) -> None:
        self._buffer_factory = buffer_factory
        self._buffers: dict[K, Buffer[V]] = {}
        self._subscribers: list[SubscriberProtocol[K] | Callable[[set[K]], None]] = []
        # ... transaction fields (unchanged)
```

**Benefits:**
- ✅ No data duplication (UserDict storage vs buffers)
- ✅ Single source of truth
- ✅ Cleaner mental model
- ✅ DataService knows nothing about buffer implementation details

## Migration Strategy

### Backward Compatibility

Existing subscribers without `extractors` property:
- Default to "latest value only" behavior
- Use `LatestValueExtractor` as default when `extractors` property is missing
- Legacy callable subscribers continue to work (receive key names only)

### Phased Approach

1. **Phase 1**: Add buffering infrastructure to DataService (with backward compatibility)
2. **Phase 2**: Update existing plotters to use extractors (optional, for consistency)
3. **Phase 3**: Remove HistoryBufferService (once no longer needed)

## Benefits

1. ✅ **Solves transaction problem**: Buffer captures all updates, extractor chooses what to return
2. ✅ **Single source of truth**: No dual DataService/HistoryBufferService
3. ✅ **Unified subscription interface**: All plotters use same mechanism
4. ✅ **Memory efficient**: Size-1 buffers for keys that only need latest value
5. ✅ **Transaction batching preserved**: Notify once, but with access to full update history

## Open Questions

These can be resolved during implementation or postpone till later:

1. Should `Buffer` initialization require a template, or can we defer until first data?
2. How to handle type checking with `extractors` property (Protocol vs ABC)?
3. Should we add buffer size metrics/monitoring?
4. What's the cleanup strategy for buffers when all subscribers for a key unregister?
