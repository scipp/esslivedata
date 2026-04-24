# Deployment pins

Sub-document of `ess-livedata-schemas.md`. Resolves how deployment-pinned fields interact with param models, reduced-schema emission, and sciline-side pipeline configuration. Concrete enough to be implementable; narrow enough not to re-open composition-root scope.

## What a pin is (and is not)

A **deployment pin** is a field whose value is determined by the *deployment* — the particular service instance, its config, the files on its disk — rather than by the user at job launch or by a pydantic field default. Concretely: normalisation file paths, calibration IDs, detector-mask files, NeXus geometry filenames, lookup-table files, "this instrument lacks a direct-beam source" flags.

Contrast:

- **Pydantic field default.** Compile-time constant, same across every deployment. Part of the schema itself.
- **Deployment pin.** Resolved at service startup from deployment context. Same across every job on that service instance, different between instances.
- **Job-start routing.** Resolved when a job is launched from user selection. Different per job. Example: "which Kafka source is the mantle detector for this run".
- **User input.** The knobs the UI renders.

Defaults and job-start routing are orthogonal to pins. This doc is about pins only.

## Relationship to the multi-level-defaults rule

From the parent plan: *if a field exists in the param model, the param model is authoritative and the translation layer always applies it to the pipeline.* A value set via `pipeline[X] = ...` in the factory is permitted only for structural constants no user, ops config, or future param will touch.

Under that rule, every deployment pin lives as a field in the param model. The factory never does `pipeline[Filename[SampleRun]] = '...'` directly. Instead the param model carries a `nexus_geometry_filename: Path` (or equivalent) field, the pin mechanism fills it at job start, and the existing translation layer applies it through the usual path.

This is the structural fix for the streaming-vs-batch divergence class of bug: there is exactly one authority (the param model) for every parametric input, whether the user sets it or the deployment does.

## What currently pins vs. what needs to move

Looking at today's `dream/factories.py` and `loki/factories.py`:

| Current site                                                     | Category                | Target                                   |
|------------------------------------------------------------------|-------------------------|------------------------------------------|
| `wf[Filename[SampleRun]] = get_nexus_geometry_filename(...)`      | Deployment pin          | Param-model field, pin-resolved          |
| `wf[LookupTableFilename] = _resolve_lookup_table_filename()`      | Deployment pin          | Param-model field, pin-resolved          |
| `wf[DetectorMasks] = DetectorMasks({})`                           | Deployment pin (empty)  | Param-model field, pin-resolved          |
| `wf[DirectBeam] = None`                                           | Deployment pin          | Param-model field, pin-resolved          |
| `wf[UncertaintyBroadcastMode] = ...drop`                          | Structural constant     | Stays sciline-side; never a user param   |
| `wf[KeepEvents[SampleRun]] = False`                               | Structural constant     | Stays sciline-side                       |
| `wf[CalibrationData] = None`                                      | Structural constant     | Stays sciline-side                       |
| `wf[sans_types.TransmissionFraction[SampleRun]] = sc.scalar(1.0)` | **Graph reshape**       | Push upstream (see parent plan §Direction)|
| `wf[CorrectedMonitor[EmptyBeamRun, ...]] = sc.scalar(1.0)`        | **Graph reshape**       | Push upstream                            |

The category boundary that matters: a pin is a value that *could* sensibly be user-controlled in some other deployment or in batch mode. A structural constant is something no user ever touches (e.g. the uncertainty-broadcast mode is an algorithm-global policy; making it a user param would be nonsense). Graph reshapes don't belong here at all — parent plan's "no graph reshapes" rule excludes them.

## Declaration shape

Pins are declared at workflow registration, alongside the `WorkflowSpec` and the `apply_params` / `dynamic_keys` functions. Sketch:

```python
@dataclass(frozen=True)
class DeploymentPin:
    """Fills a single param-model field at job start from deployment context."""
    field: str                                    # param-model field name
    resolve: Callable[[DeploymentContext], Any]   # returns the pinned value

factory.register(
    workflow_id=...,
    spec=DreamPowderWorkflowSpec,
    params_model=DreamPowderWorkflowParams,
    apply_params=apply_params,
    dynamic_keys=dynamic_keys,
    pipeline=build_pipeline,
    pins=[
        DeploymentPin(
            field='nexus_geometry_filename',
            resolve=lambda ctx: ctx.geometry_file('dream-no-shape'),
        ),
        DeploymentPin(
            field='lookup_table_filename',
            resolve=lambda ctx: ctx.lookup_table('dream'),
        ),
        DeploymentPin(
            field='detector_masks',
            resolve=lambda ctx: DetectorMasks({}),
        ),
    ],
)
```

The `DeploymentContext` is a small object the service constructs at startup, carrying the ops-config it was started with (paths, host, instrument, whatever). Pin resolvers are ordinary Python callables — free to read files, compute "latest in directory", or return a constant.

**Why callable-over-a-context rather than a static value:** pins that look static today (`'dream-no-shape'`) tomorrow want to be "latest geometry file" or "the one matching this run's calibration version". Threading context through the resolver from day one keeps the declaration shape unchanged when the resolution gets smarter.

**Why registration-time and not a pydantic annotation:** different deployments may pin different fields. The same DREAM powder workflow could run with a fixed vanadium file on one deployment and with a user-selectable one on another (e.g. commissioning). Pydantic-annotating the field as "pinned" bakes that into the schema, which is wrong. Registration-time declaration is a deployment concern living in deployment code.

## Value source (initial rollout)

For the near-term plan, `DeploymentContext` carries only what is already implicitly threaded through `ServiceFactory` / `setup_factories()`: instrument name, data directories, whatever else the services already read at startup. Pin resolvers are hand-written Python alongside the registration.

Structured ops-config files (YAML per-deployment with an explicit schema for what pins are set where) are a follow-up. Not required to land the rest.

## Application at job start

When a job launches:

1. The UI / API submits `user_inputs: dict[str, Any]` — only fields the user sees.
2. The composition-root-lite layer constructs the full param instance:
   ```python
   pin_values = {pin.field: pin.resolve(ctx) for pin in workflow.pins}
   params = params_model(**user_inputs, **pin_values)
   ```
   Collision — a user-supplied field matching a pinned field — is a registration bug; the pin list defines which fields the UI must not have sent. Fail loudly.
3. `apply_params(pipeline, params)` runs as usual. No special-casing of pinned fields in the translation layer; they are just fields.
4. `dynamic_keys(params)` runs as usual.

This is deliberately not a "composition root" in the full section-6 sense (defaults / pins / routing / user input with explicit declared partitioning). It is the minimum that makes pins work: one declared partition (pinned vs. user-visible), with routing and defaults handled by existing code paths. The full composition-root refactor is Tier A follow-up.

## Reduced-schema emission

When a service emits the reduced JSON Schema for a workflow on the status topic:

1. Start from `params_model.model_json_schema()`.
2. For each `pin.field`, delete the corresponding `properties[<field>]` entry and remove it from `required`.
3. Publish the result.

Job-start-routed fields (source-name fields) are handled by the same mechanism — the registration declares which fields are resolved at job start, reduced-schema stripping removes them too. The parent plan already mentions source-name fields as job-start routing; same code path as pins, different declaration list. (Bookkeeping: reduce to one list or two? Two — `pins` and `routed` — keeps the semantic distinction legible; merging into one "filled by the system" list loses it. The emission code handles both identically.)

## Open items

- **Ops-config schema.** Once pin resolvers start reading from a structured config rather than hand-written callables, what shape does that config take? Likely a per-instrument YAML under `config/instruments/<inst>/deployment.yaml` with `pins:` and `routing:` sections. Not required for initial rollout.
- **Collision detection at startup.** When a service starts, every registered workflow's pin list should be checked against its params_model fields: every pin must name an existing field; no pin-field overlap between pin and routed lists. Enforce at registration time, not at job-launch time.
- **Computed pins with side effects.** `"latest file in /data/norm"` is a real use case (instrument scientists don't want to edit config every time a new normalisation is generated). The resolver is a callable, so it can do the filesystem read — but when? On service startup only (cached), or per-job (re-resolved each time)? Startup-only keeps pins genuinely deployment-scoped; per-job blurs the pin/routing boundary. Default: startup-only, cached. Per-job dynamism is job-start routing by definition.
- **Pin vs. structural-constant boundary in practice.** The table above is a first pass. Walking every factory during Stage 2 will surface cases that genuinely belong in one category but are today written in the other. Expect to re-classify a handful.
- **Does the dashboard need the pin values?** Not for rendering — the reduced schema is what it needs. But for telemetry ("the backend currently uses calibration v7"), a separate status channel is cleaner than conflating pins with specs. Parent plan already says this; flagging here so it is not forgotten when the emission format is designed.
