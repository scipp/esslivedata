# FastAPI Transport Alternative

## Overview

Add FastAPI-based HTTP transport as an alternative to Kafka for service-to-service communication, primarily for local development and to enable longer-term reusability of services without Kafka dependency.

## Architecture

### Topology
- **Star topology**: Each service exposes its own FastAPI endpoint
- **Coexistence**: Kafka and HTTP transports can be mixed (e.g., raw data from Kafka, results via HTTP)
- Consumers poll directly from producer services via HTTP

### Design Decisions

1. **Queue Strategy**: Bounded queue with configurable max size (Option A/D)
   - Oldest messages dropped when full
   - Potential for time-based expiry in future iterations

2. **HTTP API**: Stateless `GET /messages` endpoint
   - Returns all currently available messages
   - No offset tracking needed
   - Consumer gets "whatever is available now"

3. **Serialization**: Pluggable layer via `MessageSerializer[T]` protocol
   - Start with JSON for ease of debugging
   - Can swap to binary formats (da00, f144) or other schemes later
   - Avoids unnecessary serialization/deserialization where possible

4. **Service Discovery**: Configuration file approach
   - Explicitly map streams to transport type (kafka vs http)
   - Central, visible configuration for what uses which transport
   - Supports incremental migration stream-by-stream

### Key Components

**New Abstractions:**
- `QueueBasedMessageSink[T]` - implements `MessageSink[T]`, wraps bounded queue
- `HTTPMessageSource[T]` - implements `MessageSource[T]`, polls HTTP endpoint
- `MessageSerializer[T]` - protocol for pluggable serialization
- FastAPI app factory - exposes sink's queue via HTTP

**Leverage Existing:**
- `MessageSource[T]` and `MessageSink[T]` protocols already in place
- `DataServiceBuilder` accepts any source/sink - just inject different implementations
- No changes to core processing logic needed

## Implementation Phases

### Phase 1: Minimum Viable (Output Flow)
**Goal:** Prove HTTP transport for service results

**Test case:** `fake_monitors` → `monitor_data` → CLI helper

- `fake_monitors` publishes to Kafka (unchanged)
- `monitor_data` consumes from Kafka, exposes results via FastAPI
- Simple CLI helper polls HTTP endpoint and displays results
- Pre-configured workflow (no config channel yet)

**Deliverables:**
- `QueueBasedMessageSink` implementation
- `HTTPMessageSource` implementation
- FastAPI app factory
- CLI helper tool
- Unit tests

### Phase 2: Bidirectional (Config Flow)
**Goal:** Prove HTTP can replace config channel

**Additions:**
- `POST /config` endpoint on services
- CLI helper can trigger workflow changes via HTTP
- Services can receive config updates without Kafka

**Deliverables:**
- HTTP config endpoint implementation
- Bidirectional HTTP communication

### Phase 3: Dashboard Integration
**Goal:** Full end-to-end with GUI

**Scope:**
- Wire dashboard to consume via HTTP instead of Kafka
- Dashboard config updates via HTTP
- Full production-ready alternative to Kafka

### Phase 4: Configuration System
**Goal:** Enable mixing transports per deployment

**Scope:**
- Configuration file schema for transport selection
- Per-stream transport configuration
- Runtime selection of Kafka vs HTTP

## Benefits

- **Local development**: No Kafka container required
- **Incremental adoption**: Migrate stream-by-stream via configuration
- **Debugging**: JSON serialization makes inspection trivial
- **Reusability**: Services can run standalone with HTTP API
- **Minimal invasiveness**: Leverages existing protocol abstractions

## Non-Goals

- Replace Kafka in production deployments
- Implement Kafka-level features (persistence, replay, partitioning)
- Performance optimization (initially)
