# Drop `ConfigKey` envelope on `livedata_commands` topic

Tracking: #853

## Motivation

The `livedata_commands` topic uses an asymmetric wire format inherited from a
compacted-topic design that no longer applies (topic now has
`cleanup.policy=delete`, see `scripts/setup-kafka-topics.sh:78-86`):

- Kafka **key** = encoded `ConfigKey` (`source_name/service_name/key`)
- Kafka **value** = payload JSON only

Investigation surfaced that most of `ConfigKey` is dead or redundant:

- `service_name` is **fully unused**. Both registered specs
  (`config/keys.py:21,33`) set it to `None`; nothing reads it. Service affinity
  moved into `WorkflowId.instrument` + `WorkflowFactory.get_service` after
  b6bb1e83 dropped `WorkflowId.namespace`.
- `source_name` for `JobCommand` is redundant — `JobCommand.job_id` already
  carries it. `JobManagerAdapter.job_command` explicitly discards it
  (`job_manager_adapter.py:32`: `_ = source_name  # Legacy, not used.`).
- `source_name` for `WorkflowConfig` is the only meaningful use: the consumer
  needs it to construct a `JobId` at `job_manager.py:243`. It is smuggled via
  the Kafka key because `WorkflowConfig` does not carry it itself.
- The dedup wildcard branch (`config_handler.py:101-105`,
  `source_name=None overrides all previous source-specific updates`) is dead
  with the current dashboard.
- `SchemaRegistry` runtime methods (`get_model`, `list_specs`,
  `get_produced_keys`, `FakeSchemaRegistry`) have no callers today. After
  this refactor `keys.py` (the only `schema_registry` importer) and
  `WORKFLOW_CONFIG.create_key` both go away, so the entire module becomes
  unused.

## End state

- Wire format is symmetric: Kafka value carries a JSON-serialized
  `Command = WorkflowConfig | JobCommand` (Pydantic discriminated union).
  No Kafka key.
- `WorkflowConfig` carries `job_id: JobId | None` instead of
  `job_number: JobNumber | None`. `JobId` packages the `(source_name,
  job_number)` pair that `JobManager.schedule_job` already needs.
- `ConfigKey`, `ConfigUpdate`, `RawConfigItem`, `CommandsAdapter` key parsing
  and the dedup wildcard logic are all removed.
- `ConfigProcessor` dispatches by `type(command)` (or a discriminator field).

## Deployment

Wire-format break. Producer (dashboard) and consumer (backend services) must
deploy together. Topic has 2-day retention and no persistent state, so a
coordinated restart is sufficient. No staged rollout needed.

## Plan

### 1. Verify `JobId` works embedded in Pydantic models

`JobId` is a frozen `@dataclass`. It is already embedded in
`JobCommand.job_id: JobId` (`job_manager.py:76`) and
`ServiceId.job_id: JobId` (`x5f2_compat.py:38`), so Pydantic v2 handles it
transparently in `model_dump_json` / `model_validate_json`. No conversion to
`BaseModel` should be needed; verify with a small round-trip test before
touching `JobId`.

### 2. Replace `WorkflowConfig.job_number` with `job_id: JobId`

`workflow_spec.py:419+`:

- Remove `job_number` field, add `job_id: JobId` (**required**, no `None`).
  The producer already supplies both `source_name` and `job_number` in every
  call (`JobSet.job_number = Field(default_factory=uuid.uuid4)` at
  `job_orchestrator.py:69`), so the historical `None`/uuid-fallback was a
  leak from the envelope design and should not survive.
- Update `from_params(...)` signature: drop `job_number`, add
  `job_id: JobId`. (Drop `from_params` entirely if call sites are simpler
  with the direct constructor.)
- Update consumer `JobManager.schedule_job` (`job_manager.py:238-253`) to
  take `WorkflowConfig` directly and read `config.job_id`. The
  `source_name` arg disappears from `schedule_job` and so does the
  `or uuid.uuid4()` fallback at `job_manager.py:243`.
- Update producer `job_orchestrator.py:414-430` to construct
  `JobId(source_name=source_name, job_number=job_set.job_number)` and pass
  it via the config.

### 3. Define the wire `Command` type

New `Command` in (probably) `core/job_manager.py` or `config/commands.py`:

```python
Command = Annotated[
    WorkflowConfig | JobCommand,
    Field(discriminator="kind"),
]
```

Add a literal `kind` field to each model (`"workflow_config"` /
`"job_command"`) — this replaces the `key` field on `ConfigKey` and matches
what `JobCommand.key: ClassVar[str]` already documents.

### 4. Replace `CommandSerializer` and `CommandsAdapter`

`kafka/sink_serializers.py:160-172`:

```python
class CommandSerializer(_TopicResolvingSerializer[Command]):
    def _encode(self, message: Message[Command]) -> tuple[None, bytes]:
        return None, message.value.model_dump_json().encode("utf-8")
```

`kafka/message_adapter.py:411-425`:

```python
class CommandsAdapter(MessageAdapter[KafkaMessage, Message[Command]]):
    def adapt(self, message: KafkaMessage) -> Message[Command]:
        return Message(
            stream=COMMANDS_STREAM_ID,
            timestamp=Timestamp.from_ms(message.timestamp()[1]),
            value=TypeAdapter(Command).validate_json(message.value()),
        )
```

Delete `RawConfigItem`.

### 5. Simplify `ConfigProcessor`

`handlers/config_handler.py`:

- Delete `ConfigUpdate` and `from_raw`.
- `process_messages` takes `list[Message[Command]]`.
- Replace `_actions: dict[str, Callable]` with type-based dispatch:

```python
match command:
    case WorkflowConfig(): result = self._adapter.set_workflow_with_config(command)
    case JobCommand():     result = self._adapter.job_command(command)
```

- Delete the dedup wildcard branch (lines 101-105) and the
  `latest_updates` two-level dict — keep only the simplest "process each
  message in order" loop unless a real dedup need is identified.
- `JobManagerAdapter.job_command` and `set_workflow_with_config` lose their
  `source_name: str` parameter and take the typed command directly.

### 6. Update producer call sites

`dashboard/job_orchestrator.py:397, 429, 437, 874-880, 889`:

- Stop building `ConfigKey`. Send the command directly:
  `self._command_service.send_batch([config1, command1, ...])`.
- `CommandService.send` / `send_batch` (`dashboard/command_service.py:34-71`)
  signature changes from `(ConfigKey, Any)` / `list[tuple[ConfigKey, Any]]`
  to `Command` / `list[Command]`.

### 7. Delete `ConfigKey`, `keys.py`, and `schema_registry.py`

- `WORKFLOW_CONFIG.create_key(source_name=...)` at `job_orchestrator.py:429`
  goes away with step 6. That removes the only call into `keys.py`.
- `keys.py` is the only importer of `schema_registry.py`. With `keys.py`
  deleted, `schema_registry.py` (`SchemaRegistry`, `ConfigItemSpec`,
  `FakeSchemaRegistry`, `get_schema_registry`, `SchemaRegistryBase`) has no
  remaining callers. Delete the whole module.
- Delete `ConfigKey` from `config/models.py` and its test file
  `tests/config/models_test.py`.

### 8. Tests

- Delete `tests/handlers/config_handler_test.py::TestConfigUpdate` (gone).
- Delete `tests/config/models_test.py` (`ConfigKey` gone).
- Update `tests/kafka/sink_serializers_test.py`,
  `tests/kafka/message_adapter_test.py`, `tests/kafka/kafka_sink_test.py`,
  `tests/dashboard/command_service_test.py`,
  `tests/dashboard/job_orchestrator_test.py`,
  `tests/dashboard/workflow_controller_test.py`,
  `tests/handlers/config_handler_test.py` to use the new `Command` shape.
- Service tests under `tests/services/` that build `ConfigKey` directly need
  the same migration.

