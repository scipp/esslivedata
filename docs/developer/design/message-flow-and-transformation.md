# Message Flow and Transformation

## Overview

ESSlivedata processes neutron detector data flowing through Kafka topics. Messages transform from raw Kafka messages to typed domain objects ready for processing.

**Key Principles**: Isolation from Kafka details, composable transformations, type safety, performance optimization.

## End-to-End Message Journey

```mermaid
flowchart TB
    subgraph Kafka["Kafka Topics"]
        DetTopic["Detector Topic<br/>(EV44)"]
        MonTopic["Monitor Topic<br/>(EV44)"]
        LogTopic["Log Topic<br/>(F144)"]
        DataTopic["Data Topic<br/>(DA00)"]
    end

    subgraph Source["Message Source Chain"]
        KC[Kafka Consumer]
        KMS[KafkaMessageSource]
        BMS[BackgroundMessageSource]
        AMS[AdaptingMessageSource]
    end

    subgraph Adaptation["Adapter Chain"]
        RTA[RouteByTopicAdapter]
        EV44A[KafkaToEv44Adapter]
        DetA[Ev44ToDetectorEventsAdapter]
        MonA[KafkaToMonitorEventsAdapter]
        F144A[KafkaToF144Adapter]
        LogA[F144ToLogDataAdapter]
        DA00A[KafkaToDa00Adapter]
        ScippA[Da00ToScippAdapter]
    end

    subgraph Domain["Domain Messages"]
        DetMsg["Message&lt;DetectorEvents&gt;"]
        MonMsg["Message&lt;MonitorEvents&gt;"]
        LogMsg["Message&lt;LogData&gt;"]
        DataMsg["Message&lt;DataArray&gt;"]
    end

    subgraph Processing["Processor"]
        P[StreamProcessor /<br/>OrchestratingProcessor]
    end

    DetTopic --> KC
    MonTopic --> KC
    LogTopic --> KC
    DataTopic --> KC

    KC --> KMS
    KMS --> BMS
    BMS --> AMS

    AMS --> RTA
    RTA --> EV44A
    RTA --> MonA
    RTA --> F144A
    RTA --> DA00A

    EV44A --> DetA
    DetA --> DetMsg
    MonA --> MonMsg
    F144A --> LogA
    LogA --> LogMsg
    DA00A --> ScippA
    ScippA --> DataMsg

    DetMsg --> P
    MonMsg --> P
    LogMsg --> P
    DataMsg --> P

    classDef kafka fill:#fff3e0,stroke:#ef6c00;
    classDef source fill:#e3f2fd,stroke:#1976d2;
    classDef adapter fill:#f3e5f5,stroke:#7b1fa2;
    classDef domain fill:#e8f5e9,stroke:#388e3c;
    classDef processor fill:#fff9c4,stroke:#fbc02d;

    class DetTopic,MonTopic,LogTopic,DataTopic kafka;
    class KC,KMS,BMS,AMS source;
    class RTA,EV44A,DetA,MonA,F144A,LogA,DA00A,ScippA adapter;
    class DetMsg,MonMsg,LogMsg,DataMsg domain;
    class P processor;
```

**Stages**: Kafka Topics (FlatBuffers: EV44, F144, DA00) → Message Source Chain (consumption, polling, adaptation) → Adapter Chain (deserialization to domain types) → Domain Messages (typed) → Processor (routes by StreamId).

## Message Abstraction

### The Message Type

`Message[T]` is an immutable (frozen) dataclass with `timestamp` (nanoseconds since Unix epoch UTC), `stream` (StreamId), and `value` (generic type T). Generic over value type, timestamp-based ordering, stream identification.

### StreamId: Identifying Message Streams

`StreamId` contains `kind` (StreamKind enum: detector, monitor, log, config, data, status) and `name` (specific identifier). Isolates internal code from Kafka topic names.

## Stream Mapping

Isolates ESSlivedata from Kafka topic structure: Kafka uses `(topic, source_name)` tuples, ESSlivedata uses `StreamId(kind, name)`. Enables topic renaming, split/merged topics, simplified testing.

### StreamMapping and StreamLUT

`InputStreamKey` = `(topic, source_name)`. `StreamLUT` = dict mapping InputStreamKey to internal name. `StreamMapping` contains detector_lut, monitor_lut, log_topics, config/data/status topics. Adapters use stream_lut to convert Kafka identifiers to StreamId.

## Message Adapters

### Adapter Pattern

`MessageAdapter` protocol: `adapt(message: T) -> Sequence[U]`. Each adapter returns a sequence, enabling 1:N message expansion (e.g., splitting multi-pulse ev44 messages into one message per pulse). Most adapters return a single-element tuple. Benefits: composable, type-safe, testable, reusable.

### Core Adapters

**KafkaToEv44Adapter**: Deserializes EV44 event data (uses stream_lut, extracts timestamp, returns `Message[EventData]`).

**KafkaToMonitorEventsAdapter**: Optimized—directly extracts time-of-flight arrays from FlatBuffers, skips unused fields, reduces memory allocations.

**Ev44ToDetectorEventsAdapter**: Converts `Message[EventData]` → `Message[DetectorEvents]`. Option: `merge_detectors=True` merges all detector banks.

**KafkaToDa00Adapter** and **Da00ToScippAdapter**: Process DataArray messages, handle DA00/Scipp compatibility.

**KafkaToF144Adapter** and **F144ToLogDataAdapter**: Process log data, decouple from upstream schema changes.

### Adapter Composition

**ChainedAdapter**: Chains two adapters with flatmap semantics — for each intermediate result from the first adapter, all results from the second adapter are collected into a flat sequence.

**RouteByTopicAdapter**: Routes by Kafka topic to different adapters. Provides `.topics` list for subscription.

**RouteBySchemaAdapter**: Routes by FlatBuffers schema identifier. Use case: multiple schemas on same topic.

### Error Handling

`AdaptingMessageSource` with `raise_on_error=False` (default) logs errors and skips messages. `raise_on_error=True` re-raises exceptions. Unknown schema exceptions logged as warnings.

## Message Batching

Groups messages into time-aligned windows for time-aligned processing, job scheduling, efficient accumulation, predictable latency.

### MessageBatch Structure

`MessageBatch` = `(start_time, end_time, messages)`. Times in nanoseconds, `[start_time, end_time)` (inclusive, exclusive).

### SimpleMessageBatcher

Time-aligned batching used by `OrchestratingProcessor`. Initial batch contains all messages; subsequent batches aligned to `batch_length_s` intervals. Late messages included in next batch. Empty batches returned when next batch starts. Batch completion triggered by first message for next batch.

**Important**: Relies on message timestamps (not wall-clock time). If messages stop arriving, current batch may never complete.

### NaiveMessageBatcher

Simple batcher for testing—all messages in one batch. Use for testing without time-based logic or services not needing time alignment.

## Message Source Chain

### MessageSource Protocol

Protocol: `get_messages() -> Sequence[T]`. Called by processor each iteration, returns available messages (may be empty).

### KafkaMessageSource

Basic Kafka consumer wrapper. Blocking: polls Kafka for up to `timeout` seconds, processor blocked during poll.

### BackgroundMessageSource

Non-blocking Kafka consumer. Background thread polls continuously, messages queued in memory, `get_messages()` drains queue without blocking, automatic overflow handling (drops oldest batches). Benefits: processor never blocked on Kafka I/O, handles bursts, reduces risk of falling behind. Context manager: start/stop background thread.

### AdaptingMessageSource

Wraps source to apply adapters. Returns `list[Message[DomainType]]`. Error handling: catches exceptions during adaptation, logs/skips messages (if `raise_on_error=False`).

### MultiConsumer

Combines multiple Kafka consumers. Use case: different offset strategies for config topics (earliest) vs data topics (latest).

## Serialization and Schema

### FlatBuffers Schemas

ESSlivedata uses FlatBuffers schemas from `streaming_data_types`:

**EV44 (Event Data)**: Fields: `source_name`, `reference_time` (pulse times), `time_of_flight`, `detector_id`.

**F144 (Log Data)**: Fields: `source_name`, `value` (scalar/array), `timestamp_unix_ns`.

**DA00 (DataArray)**: Compatibility layer via `scipp_da00_compat`: `scipp_to_da00`, `da00_to_scipp`.

### Kafka Sink Serialization

`KafkaSink` publishes messages using built-in serializers: `serialize_dataarray_to_da00` (default), `serialize_dataarray_to_f144` (logs). Stream-specific: config (JSON), status (X5f2), data (DA00).

### UnrollingSinkAdapter

Handles DataGroup results: splits `Message[DataGroup]` with multiple outputs into separate DA00 messages per output. Stream name includes output: `workflow_id/job_id/output_name`.

## Summary

Message transformation pipeline: Kafka Messages (FlatBuffers) → Source Chain (consumption, polling, adaptation) → Adapters (deserialization, domain conversion) → Stream Mapping (Kafka isolation) → Batching (time-aligned windows) → Domain Messages (type-safe).

**Key Abstractions**: `Message[T]`, `StreamId`, `MessageAdapter`, `StreamMapping`, `MessageBatcher`, `BackgroundMessageSource`.

Enables high-rate data stream processing with type safety, testability, and Kafka isolation.
