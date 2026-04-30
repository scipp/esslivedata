# Figure of Merit — Phase 1 Implementation Plan

Companion to `figure-of-merit-premise.md`, `figure-of-merit-concept.md`, and
`figure-of-merit-plan.md`. This document is the concrete coding plan for
Phase 1 (the core mechanism, backend-only). Phases 2 (dedicated topic) and 3
(scheduled reset/start), and all dashboard/NICOS-side work, are out of scope.

## Scope

Phase 1 delivers:

- A generic **stream-alias binding** mechanism (no FOM-specific schema).
- Mirror emission of bound `(job_id, output_name)` results on a new
  `StreamKind.LIVEDATA_FOM`, in **parallel** with the existing
  `LIVEDATA_DATA` emission (copy semantics).
- Backend tests, including an end-to-end test using the existing
  `LivedataApp` helper.

Out of scope: dashboard widget, NICOS interface, dedicated FOM topic,
scheduled reset, persistence, the `fom-N` naming convention as schema
(it remains a documentation-level agreement).

## Decisions agreed in design conversation

| # | Decision | Rationale |
|---|---|---|
| 1 | Mirror emitted by extending `UnrollingSinkAdapter` (option (a)). | Decision is local — `(job_id, output_name)` is already in hand inside the unrolling loop. No `ResultKey` re-parse. Switching to reroute-and-swallow later is one branch in the same loop. |
| 2 | Initial semantics: **copy** (parallel mirror, original `LIVEDATA_DATA` unchanged). Designed for an easy switch to reroute-and-swallow later. | Keeps existing dashboard subscriptions working. FOM workflow may have non-FOM outputs the operator wants to see. Reroute is a cleaner steady state once the dashboard FOM widget exists; switching is a body-of-the-loop change. |
| 3 | Mirror message value is a wrapper type (no extra fields on `Message`). | `Message` stays neutral. Kafka-encoding state lives in the value, decoded by a kind-specific serializer. |
| 4 | Two pydantic command models: `BindStreamAlias` / `UnbindStreamAlias`. Routed via existing `ConfigProcessor` actions. New `StreamAliasAdapter` parallel to `JobManagerAdapter`. | Reuses the commands-topic / ACK / responses-topic plumbing. No new schema for NICOS to care about (NICOS doesn't see Bind/Unbind in any case). |
| 5 | No explicit flush-on-Unbind. Rely on the per-tick `flush(timeout=3)` already in `KafkaSink.publish_messages`. | Adapter doesn't need a sink reference. Residual race during reconfigure is acknowledged in the plan as "brief outage" class. Can add later if observed. |
| 6 | Keep `StreamKind.LIVEDATA_FOM` (per the plan), comment the FOM-coupling. | Phase 2 wants a FOM-specific topic anyway. The generic mechanism is preserved; only the *output kind* picked by the mirror is FOM-flavoured. |

## File-level changes

### New files

#### `src/ess/livedata/core/stream_alias.py`

Generic stream-alias binding primitives. No FOM-specific code.

```python
class BindStreamAlias(pydantic.BaseModel):
    """Bind a stable stream alias to a specific (job, output) pair."""
    key: ClassVar[str] = "bind_stream_alias"
    message_id: str | None = None
    alias: str
    job_id: JobId
    output_name: str

class UnbindStreamAlias(pydantic.BaseModel):
    """Release a previously bound stream alias."""
    key: ClassVar[str] = "unbind_stream_alias"
    message_id: str | None = None
    alias: str

class StreamAliasRegistry:
    """In-memory registry: alias -> (JobId, output_name)."""
    def bind(self, alias: str, job_id: JobId, output_name: str) -> None: ...
    def unbind(self, alias: str) -> None: ...
    def lookup(self, job_id: JobId, output_name: str) -> str | None: ...
    def has(self, alias: str) -> bool: ...

@dataclass(frozen=True, slots=True, kw_only=True)
class AliasedResult(Generic[T]):
    """Wrapper carrying the underlying data plus the bound alias for serialization."""
    data: T
    alias: str
```

`bind()` raises `ValueError` if the alias is already present (no-replace,
D1). `unbind()` is a no-op for unknown aliases (the adapter checks
`has()` to decide whether to ACK).

#### `src/ess/livedata/core/stream_alias_adapter.py`

Parallel to `job_manager_adapter.py`. Produces `CommandAcknowledgement`s
following the same actor / silent-non-actor pattern.

```python
class StreamAliasAdapter:
    def __init__(
        self,
        *,
        registry: StreamAliasRegistry,
        job_manager: JobManager,
    ) -> None: ...

    def bind(self, source_name: str, value: dict) -> CommandAcknowledgement | None:
        # 1. parse value -> BindStreamAlias
        # 2. if not job_manager.has_job(cmd.job_id): return None  (silent)
        # 3. if registry.has(cmd.alias): ACK error (no-replace)
        # 4. registry.bind(...); ACK success

    def unbind(self, source_name: str, value: dict) -> CommandAcknowledgement | None:
        # 1. parse value -> UnbindStreamAlias
        # 2. if not registry.has(cmd.alias): return None  (silent)
        # 3. registry.unbind(...); ACK success
```

`source_name` is unused (the alias is global per-service, not source-keyed),
mirroring how `JobManagerAdapter.job_command` ignores it.

### Modified files

#### `src/ess/livedata/core/message.py`

Add one line:

```python
class StreamKind(StrEnum):
    ...
    LIVEDATA_FOM = "livedata_fom"  # Mirror of bound stream-alias outputs (FOM use)
```

#### `src/ess/livedata/config/streams.py`

Map the new kind to the existing data topic for Phase 1, with a comment
flagging the planned Phase 2 split:

```python
case StreamKind.LIVEDATA_FOM:
    # Phase 1: shares LIVEDATA_DATA topic; NICOS filters by Kafka key.
    # Phase 2: switch to f'{instrument}_livedata_fom'.
    return f'{instrument}_livedata_data'
```

#### `src/ess/livedata/core/job_manager.py`

Add `has_job(job_id) -> bool`:

```python
def has_job(self, job_id: JobId) -> bool:
    return job_id in self._active_jobs or job_id in self._scheduled_jobs
```

#### `src/ess/livedata/kafka/sink.py`

Extend `UnrollingSinkAdapter`:

```python
class UnrollingSinkAdapter(MessageSink[T | sc.DataGroup[T]]):
    def __init__(
        self,
        sink: MessageSink[T | AliasedResult[T]],
        *,
        alias_registry: StreamAliasRegistry | None = None,
    ) -> None:
        self._sink = sink
        self._alias_registry = alias_registry

    def publish_messages(self, messages: list[Message[T | sc.DataGroup[T]]]) -> None:
        unrolled: list[Message[T | AliasedResult[T]]] = []
        for msg in messages:
            if isinstance(msg.value, sc.DataGroup):
                result_key = ResultKey.model_validate_json(msg.stream.name)
                for name, value in msg.value.items():
                    key = ResultKey(
                        workflow_id=result_key.workflow_id,
                        job_id=result_key.job_id,
                        output_name=name,
                    )
                    stream = replace(msg.stream, name=key.model_dump_json())
                    unrolled.append(replace(msg, stream=stream, value=value))
                    if self._alias_registry is not None:
                        alias = self._alias_registry.lookup(
                            result_key.job_id, name
                        )
                        if alias is not None:
                            mirror_stream = replace(
                                stream, kind=StreamKind.LIVEDATA_FOM
                            )
                            unrolled.append(
                                replace(
                                    msg,
                                    stream=mirror_stream,
                                    value=AliasedResult(data=value, alias=alias),
                                )
                            )
            else:
                unrolled.append(msg)
        self._sink.publish_messages(unrolled)
```

When `alias_registry is None` (any test that doesn't care about FOM, plus
all unrolling-only tests), behaviour is identical to today.

#### `src/ess/livedata/kafka/sink_serializers.py`

Add `FomDa00Serializer` and register it. The serializer composes — it
encodes the same Da00 payload as `Da00Serializer` would, then sets the
Kafka key to `alias.encode('utf-8')`.

```python
class FomDa00Serializer(MessageSerializer[AliasedResult[sc.DataArray]]):
    def __init__(self, *, instrument: str) -> None:
        self._inner = Da00Serializer(instrument=instrument)

    def serialize(
        self, message: Message[AliasedResult[sc.DataArray]]
    ) -> SerializedMessage:
        # Build a Message[sc.DataArray] for the inner serializer; stream.name
        # stays the ResultKey JSON so Da00 source_name is unchanged.
        inner_msg = replace(message, value=message.value.data)
        serialized = self._inner.serialize(inner_msg)
        return replace(serialized, key=message.value.alias.encode('utf-8'))
```

Registered:

```python
return RouteByStreamKindSerializer(
    {
        ...,
        StreamKind.LIVEDATA_DATA: data_serializer,
        StreamKind.LIVEDATA_FOM: FomDa00Serializer(instrument=instrument),
        ...,
    }
)
```

#### `src/ess/livedata/handlers/config_handler.py`

Register two more actions:

```python
self._actions = {
    'workflow_config': self._job_manager_adapter.set_workflow_with_config,
    JobCommand.key: self._job_manager_adapter.job_command,
    BindStreamAlias.key: self._stream_alias_adapter.bind,
    UnbindStreamAlias.key: self._stream_alias_adapter.unbind,
}
```

`ConfigProcessor.__init__` gains a `stream_alias_adapter` parameter.

#### `src/ess/livedata/core/orchestrating_processor.py`

`OrchestratingProcessor` constructs the adapter and registers it in
`ConfigProcessor`. The registry itself is **passed in** from the builder
(see below) — the processor doesn't own it, because the same instance
must be visible to the sink wrapper.

```python
def __init__(
    self,
    *,
    source: ...,
    sink: MessageSink[Tout],
    preprocessor_factory: ...,
    stream_alias_registry: StreamAliasRegistry,
    ...,
) -> None:
    ...
    self._stream_alias_adapter = StreamAliasAdapter(
        registry=stream_alias_registry,
        job_manager=self._job_manager,
    )
    self._config_processor = ConfigProcessor(
        job_manager_adapter=self._job_manager_adapter,
        stream_alias_adapter=self._stream_alias_adapter,
    )
```

#### `src/ess/livedata/service_factory.py`

`DataServiceBuilder` creates the registry, exposes it, and passes it to
`OrchestratingProcessor` via `from_source`. Both the production runner
(`DataServiceRunner.run`) and `LivedataApp.from_service_builder` use the
exposed registry to wrap the sink with `UnrollingSinkAdapter`.

```python
class DataServiceBuilder:
    def __init__(self, ...) -> None:
        ...
        self._stream_alias_registry = StreamAliasRegistry()

    @property
    def stream_alias_registry(self) -> StreamAliasRegistry:
        return self._stream_alias_registry

    def from_source(self, ...):
        ...
        processor = self._processor_cls(
            ...,
            stream_alias_registry=self._stream_alias_registry,
        )
```

Production runner change (`service_factory.py:333`):

```python
sink = UnrollingSinkAdapter(sink, alias_registry=builder.stream_alias_registry)
```

#### `tests/helpers/livedata_app.py`

`LivedataApp.from_service_builder` mirrors the production wiring:

```python
service = builder.from_consumer(
    consumer=consumer,
    sink=UnrollingSinkAdapter(
        sink, alias_registry=builder.stream_alias_registry
    ),
    ...,
)
```

## Tests

### Unit tests

| File | Coverage |
|---|---|
| `tests/core/stream_alias_test.py` | `StreamAliasRegistry`: bind/unbind/lookup, no-replace raises, unbind unknown is no-op. |
| `tests/core/stream_alias_adapter_test.py` | `StreamAliasAdapter`: actor ACK success, non-actor returns `None`, no-replace ACK error, unbind unknown returns `None`. Uses real `JobManager` with a scheduled job and an alternate empty `JobManager`. |
| `tests/kafka/sink_test.py` (extend existing) | `UnrollingSinkAdapter` mirror: bound output emits a parallel `LIVEDATA_FOM` `Message[AliasedResult]`; unbound output emits no mirror; `alias_registry=None` matches current behaviour byte-for-byte. |
| `tests/kafka/sink_serializers_test.py` (extend existing) | `FomDa00Serializer`: Kafka key equals alias bytes; payload bytes equal `Da00Serializer` output for the same data; topic resolves correctly. |

### End-to-end via `LivedataApp`

`tests/services/stream_alias_test.py` (new file) — modelled on
`tests/services/detector_data_test.py`, using the `dummy` instrument
detector service (workflow with deterministic scalar outputs).

Three tests:

1. **`test_bind_emits_fom_mirror`** — schedule a workflow, bind an alias
   to a known output, push events, step, assert sink contains both
   `LIVEDATA_DATA` (per-output) and exactly one `LIVEDATA_FOM`
   `Message[AliasedResult]` for the bound output, with the right alias.

2. **`test_unbind_stops_mirror`** — extends (1): publish `UnbindStreamAlias`,
   step, push more events, step, assert no further `LIVEDATA_FOM` messages
   appear; `LIVEDATA_DATA` continues unchanged.

3. **`test_bind_unknown_job_is_silent`** — publish `BindStreamAlias` for a
   `job_id` that no service holds, step, assert no `CommandAcknowledgement`
   on responses topic.

Multi-service rebind is **not** tested here. (1)–(3) cover the
single-service ACK semantics; cross-service correctness reduces to
"the non-actor stays silent", which is exercised in the adapter unit
test via an empty `JobManager`.

## Implementation order

1. `StreamAliasRegistry` + `BindStreamAlias` / `UnbindStreamAlias` +
   `AliasedResult` → unit tests.
2. `StreamAliasAdapter` + `JobManager.has_job` → unit tests.
3. `StreamKind.LIVEDATA_FOM` + `streams.py` mapping.
4. Extend `UnrollingSinkAdapter` → unit tests, including the
   `alias_registry=None` no-op path.
5. `FomDa00Serializer` + register in `make_default_sink_serializer` →
   unit tests.
6. Wire `StreamAliasRegistry` through `DataServiceBuilder`,
   `OrchestratingProcessor`, `ConfigProcessor`, production runner, and
   `LivedataApp`.
7. End-to-end test in `tests/services/stream_alias_test.py`.

Each step is independently testable; (4) and (5) can be parallelised.

## What this plan does *not* change

- `Message` shape (no new fields).
- `JobCommand` / `WorkflowConfig` schemas.
- `ResultKey` (still propagated as-is into Da00 `source_name` for both
  the data and FOM mirror messages).
- The unrolling rule for non-`DataGroup` values (passed through
  unchanged).
- `Da00Serializer` (`FomDa00Serializer` composes it; the existing
  serializer is untouched).
