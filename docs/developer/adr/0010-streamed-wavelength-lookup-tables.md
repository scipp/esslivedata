# ADR 0010: Wavelength lookup tables as streamed context inputs

- Status: proposed
- Deciders: Simon
- Date: 2026-08-11

## Context

`wavelength_lut_workflow.py` computes a wavelength lookup table from the live chopper
cascade: it consumes each chopper's rotation-speed and delay setpoints as context
(ADR 0003 spec-scope bindings), rebuilds the cascade whenever `ChopperSynthesizer`
emits a `chopper_cascade` tick, and publishes the table as an ordinary workflow result.
Nothing consumes that result. It exists to be looked at.

Meanwhile every backend workflow that converts time-of-arrival to wavelength loads a
lookup table from a file fixed at import time. `LookupTableFilename` is assigned
bare-generic (`wf[LookupTableFilename] = ...`) in each instrument's `factories.py`, so a
single McStas-derived table serves every component in the job. The files describe a
*nominal* chopper configuration. Run in a different configuration and the reduction is
silently wrong -- the whole motivation for computing the table live.

Three properties of the current file-based tables shape the decision.

**The shipped ranges are already wrong for monitors.** A table is indexed by `distance`;
a lookup outside its range yields `NaN` (`interpolator_numba.py:63-65`,
`interpolator_scipy.py:40-46`, `fill_value=np.nan`), those events fall outside every
histogram bin, and the component renders empty with no error anywhere. Measured against
the registered geometry artifacts:

| instrument | shipped LUT range | component outside it |
|---|---|---|
| DREAM | 59.85-80.15 m | `monitor_bunker` at 6.62 m |
| LOKI | 8.8-35.1 m | `m0` at 6.80 m |
| BIFROST | 161.24-163.44 m | monitors at 6.79, 27.90, 78.06 m |

**One table per job wastes almost all of its rows.** Per-component `Ltotal` spreads are
tiny -- DREAM `mantle_detector` 0.945 m, `high_resolution_detector` 0.374 m, LOKI banks
0.11-0.41 m, ESTIA 0.131 m -- and monitors are scalars. A union table spanning detectors
*and* monitors is ~750 rows at the default 0.1 m resolution where 4-12 are used. That is
not a bandwidth problem (`config/defaults/kafka_dev.yaml` allows ~96 MB messages); it is
a recompute problem, since the polygon rasterization runs over every distance row on
every chopper change.

**The range is currently a user parameter.** `LtotalRange` defaults to 5-30 m
(`wavelength_lut_workflow_specs.py:113-121`), which covers none of BIFROST and only part
of LOKI. An operator starting the LUT workflow with defaults would silently blank
detectors facility-wide.

The mechanisms this ADR builds on already exist: context bindings and JobManager gating
(ADR 0002, ADR 0003), uniform stream-name keying (ADR 0004), and the NICOS derived-device
mirror (ADR 0006), which republishes selected workflow outputs onto a dedicated topic
under names that deliberately exclude job identity.

What does not exist, and is the new concept here: **a workflow output republished as an
input stream for other workflows.** The NICOS mirror is publish-only -- nothing in the
system consumes `*_livedata_nicos_data`. No backend service subscribes to
`livedata_data`; only the dashboard does. This is the first cross-service feedback edge.

## Decision

### One LUT job, one output per component

The LUT workflow keeps its single `chopper_cascade` source and gains one output per
component, each covering exactly that component's `Ltotal` range. `LtotalRange` ceases to
be a user parameter.

One job rather than one job per component. A job's identity is
`(workflow_id, source_name)` where `source_name` *is the stream it consumes*; the LUT job
consumes the chopper cascade and nothing per-component. Per-component jobs would require
inventing N synthetic trigger streams so `ChopperSynthesizer` could fan one tick into N --
manufacturing input identity to express output identity -- and would recompute the same
cascade N times, take N times the chopper context bindings, and allow N ways to drift on
`source_offset` or `pulse_stride`. Consistency across components is the point of the
feature.

Per-component messages come for free: `UnrollingSinkAdapter` (`kafka/sink.py:176-195`)
already splits a multi-output result into one message per output.

`WavelengthLutOutputs` is built per instrument (`pydantic.create_model`), since DREAM's
components are not LOKI's. This mirrors the existing per-instrument `params` override on
`register_wavelength_lut_workflow_spec` and stays dashboard-safe: no science imports.

Per-component tables are ordinary outputs. Nothing auto-plots them; a user who wants one
adds a plot. During commissioning that visibility is the point -- "why is this detector
empty" is answered by looking at its table, which today is unreadable because the
relevant rows are buried in tens of metres of empty distance.

### Ranges are derived, with two orthogonal declarations

The range must be computed with the same `Ltotal` definition the consumer uses at lookup
time, and those differ:

| rule | distance | applies to |
|---|---|---|
| scatter | \|sample-source\| + \|pixel-sample\| | DREAM, LOKI, ESTIA detectors |
| straight-line | \|component-source\| | all monitors |
| source-to-sample | \|sample-source\| | BIFROST QCut |

BIFROST is indirect geometry: `DetectorLtotal(sample_data.coords['L1'])`
(`ess/spectroscopy/indirect/time_of_flight.py:66`, `ki.py:48-53`), so its lookup distance
is ~162.0 m and the analyser and detector legs are excluded. For LUT purposes its
detector behaves like a monitor at the sample. A generic walk to each pixel would produce
164-166 m and every event would `NaN` -- reintroducing the exact failure this ADR
eliminates. The rule is therefore *declared* per component, not inferred, so the physics
distinction stays visible rather than hiding inside a hand-written constant.

Motion is a second, independent declaration. Which components ride a moving axis is
already known from `Instrument.chain_patch_bindings` (`dependent_sources` plus the
resolved `transform_path`). Only the travel envelope is underivable, so instruments
declare one number per moving axis and the affected component set is derived. A bank
hung off the carriage later inherits the padding instead of silently getting a
nominal-only range.

Only LOKI needs an envelope today: `detector_carriage` is a translation along `[0,0,1]`
that moves `Ltotal` 1:1 (carriage 0 -> bank-0 at 28.605-28.786 m; carriage +3 m ->
31.605-31.754 m), and `beam_monitor_m4` rides it. LOKI's shipped 8.8-35.1 m range is
this headroom, hand-rolled. BIFROST's tank rotation is direction-only and its lookup uses
L1; ESTIA's arm rotation is bit-identical at 0, 0.6 and 2.0 degrees.

The range is static, covering the full motion envelope. The LUT workflow does *not*
consume motion context: a live range would gate the LUT job on motion and create a
cross-job race between the carriage value the LUT job saw and the one the consumer
patched into its geometry.

A guard test in the shape of `tests/config/motion_binding_test.py` recomputes each
component's range from the registered artifact and fails if a declared range no longer
contains it -- catching artifact regeneration as a loud failure rather than a silent one.

### A generic mirror: workflow outputs republished as input streams

Modelled on ADR 0006 but named for the seam rather than for the LUT, because the shape --
an output fed back as another workflow's input -- recurs (a fitted beam centre, a live
calibration).

- `context_outputs: dict[str, str]` on `WorkflowSpec`, mapping output field name to a
  stream-name template, validated against real output fields exactly as `device_outputs`
  is.
- A new `StreamKind` and topic `{instrument}_livedata_context`
  (`core/message.py`, `config/streams.py`).
- Sink routing to the existing da00 serializer (`kafka/sink_serializers.py`); the wire
  `source_name` is the rendered stream name.
- An **ingest** route, which ADR 0006 has no analogue for: a `KafkaToDa00Adapter` with no
  `StreamLUT`, so the internal stream name *is* the da00 source name -- the same shape as
  the `livedata_roi` route.
- A preprocessor case returning `LatestValueAccumulator`, already `is_context = True`, so
  `MessagePreprocessor.get_context` and the JobManager gate work unchanged.
- The route added to `detector_data`, `monitor_data` and `data_reduction`.

A dedicated topic rather than reusing `livedata_data`: backend services must not
subscribe to every detector image in the facility to receive a lookup table.

Stream names are `wavelength_lut/{component}`. The slash prefix keeps the namespace
greppable and collision-free against f144 device names. Job identity is excluded, per
ADR 0006 -- a `ContextBinding` is declared at import time and cannot know a job number,
and excluding it is what lets a relaunched LUT job transparently resume feeding its
consumers.

### Consumers bind the LUT as gated context

Spec-scope `ContextBinding` (ADR 0003), keyed by the consumer's own source:

```python
handle.add_context_binding(
    stream_name=f'wavelength_lut/{source_name}',
    workflow_key=WavelengthLutArray[snx.NXdetector],
    dependent_sources={source_name},
)
```

The wire value is a `sc.DataArray` carrying the four provenance scalars as 0-D coords
(`_attach_provenance`, `wavelength_lut_workflow.py:177-196`) -- deliberately the legacy
on-disk layout that `load_lookup_table_from_file` already rehydrates
(`ess/reduce/unwrap/lut.py:864-878`). A provider `WavelengthLutArray[Component] ->
LookupTable[RunType, Component]` inverts it in ~6 lines. `WavelengthLutArray` is generic
over `Component` so a reduction job can bind a detector table and a monitor table as
distinct sciline keys. da00 round-trips variances, which `mask_large_uncertainty_in_lut`
requires unconditionally even at threshold `inf`.

Monitors are treated as statically known. The user-selectable aux-monitor mechanism on
LOKI I(Q) and DREAM powder is not designed around; a check at job creation raises when
the chosen aux monitor differs from the bound one, rather than silently supplying a table
for the wrong distance.

Detectors need no new type parameter: `Component` is hard-wired to `snx.NXdetector`
(`to_wavelength.py:486-488`) and `DetectorLtotal[RunType]` has no `Component` parameter
at all (`types.py:87`), but livedata already runs one bank per job, so per-component
delivery is achieved by `dependent_sources`, not by parametrising `Component`.

### Wavelength specs are split from TOA specs

Gating is resolved per `(workflow_id, source_name)` and never per params:
`Instrument.resolve_context_keys` takes no params, and `Job.gating_streams` is its key set
(`core/job_manager.py:175-197`). `coordinate_mode` is a param. A spec offering a
TOA/wavelength dropdown would therefore gate TOA-mode jobs on a LUT they never read.

ADR 0003 accepted exactly this over-gate for motion and prescribed the remedy for when it
becomes real: split the spec. It has become real, and the cost is far higher than for
motion. Motion streams are f144 PVs pushed continuously by the control system and arrive
within seconds of any service start. The LUT is published by an operator-started livedata
job that re-emits only on chopper change. Over-gating TOA on the LUT means TOA -- the mode
you fall back to when everything else is broken -- depends on the most fragile link in the
chain.

New wavelength-variant specs are therefore added *additively*, carrying the LUT binding;
existing specs are untouched. TOA keeps working unconditionally, no version bumps, no
saved-config migration, and the file-based wavelength mode remains available until the
streamed path is trusted. The new path carries no file default at all, so "silently
reducing with a simulation table for the wrong chopper configuration" is gone by
construction rather than by warning text.

This lands as a preparatory PR ahead of the main work.

### Consumers clear on a changed LUT

A new LUT means the chopper phasing changed, so the wavelength for a given time-of-arrival
changed: data accumulated before and after are not the same measurement. Downstream
workflows clear.

Consumers do not compare tables. They clear when the LUT's **identity coord** changes: a
0-D fingerprint the producer stamps alongside the four provenance scalars, derived from
the inputs that determine the table -- per-chopper speed and delay setpoints, pulse period
and stride, the resolutions, the source offset and the component's range -- each rounded
to a declared precision.

Clearing on *any* received LUT would be simpler, and correctly locates the decision at the
producer rather than having consumers second-guess it. It fails on one case that matters:
restarting the LUT job re-emits an unchanged table, and restarting the LUT job is the v0
recovery action for a lost stream. Recovery must not destroy every consumer's statistics.
The same objection rules it out for the liveness heartbeat below, where every beat would
clear the facility.

ADR 0006's `start_time` generation marker is the other obvious candidate and is wrong for
the same reason: it changes on a restart with identical configuration.

Comparing table *contents* at the consumer also works and is worse on every axis. It is N
comparisons instead of one stamp; `sc.identical` cannot be used at all, since it returns
`False` for bit-identical arrays containing `NaN` and a LUT is full of `NaN` rows wherever
the cascade blocks the beam (`ess/reduce/unwrap/lut.py:609-612`); and the tolerance needed
to work around that puts the noise-rejection knob at the consumer, far from the setpoint
jitter it is filtering.

Deriving the fingerprint from inputs rather than from the table's bytes keeps it stable
across producer restarts -- the setpoints replay from retained context, so the same cascade
yields the same fingerprint in a fresh process -- and avoids a float hash that would not
survive a library or platform change. The rounding precision is the noise knob, sitting at
the source. It backs up the plateau filtering `ChopperSynthesizer` already applies
(`_StabilityDetector`, `chopper_synthesizer.py`).

Clearing is opt-in per binding -- `clear_on_change: bool = False` on `ContextBinding`, set
only for LUT bindings. Universal clearing is arguably more correct (a carriage move does
invalidate an accumulated histogram, since the pixels moved) but would change motion
behaviour on LOKI, the v0 acceptance instrument. Whether motion should clear is a separate
question to decide on its own merits.

No new plumbing: `Job.clear()` (`core/job.py:471`) is already driven by an external event
via `reset_on_run_transition` (`core/job_manager.py:508`); this is a second trigger on
that path.

A LUT emitted because the operator reconfigured the LUT job is indistinguishable
downstream from one emitted because the choppers moved, and should be: both mean the same
thing.

### Scope

v0 covers **DREAM and LOKI**. LOKI is included specifically so that carriage motion
composing correctly with the derived ranges is an acceptance criterion rather than an
assumption.

BIFROST is deferred: it needs the source-to-sample rule, and only its QCut reduction
consumes a LUT (its detector views, ratemeter and monitors are TOA-only). ESTIA is out
entirely -- it has LUT consumers but declares no choppers, so
`register_wavelength_lut_workflow_spec` never fires and there is no producer; it keeps its
file. TBL is producer-only: it has choppers, so it gets the LUT job, but nothing there
consumes a table.

## Alternatives considered

| Option | Notes |
|---|---|
| **Per-component tables, one job, N outputs (chosen)** | Removes the range parameter, cuts recompute ~100x, makes each table readable, and maps 1:1 onto the per-output mirror. |
| One instrument-wide table, range = union over all components | Also removes the range parameter, with far less plumbing. But ~750 rows where 4-12 are used, and DREAM's union spans 6.6-79.4 m -- a factor 12 in distance, almost all of it empty. Rejected. |
| One job per component | Requires N synthetic trigger streams to give each job a `source_name`, recomputes the cascade N times, and multiplies the ways params can drift between components. Rejected. |
| Derive every range generically from the geometry artifact | Silently wrong for BIFROST (164-166 m against a 162.0 m lookup). The failure mode is exactly the one this ADR exists to remove. Rejected. |
| Hand-declare every range, no derivation | ~14 numbers per instrument that nobody re-checks when an artifact is regenerated. Rejected. |
| LUT workflow consumes motion context for a live range | Gates the LUT job on motion, races the consumer's own geometry patch, and couples motion to LUT-driven clearing so every carriage nudge would discard statistics. Static envelope is simpler and sufficient. Rejected. |
| Route the LUT as ungated aux, file as default (ROI precedent) | Survives a cold start, but leaves "reducing with a stale or nominal table" as a silent mode -- the thing the feature exists to prevent. Rejected in favour of the gate plus an explicit v0 limitation. |
| Param-dependent gating on `coordinate_mode` | Explicitly ruled out by ADR 0003. Rejected. |
| Reuse `livedata_data` instead of a dedicated topic | Backend services would subscribe to every detector image in the facility. Rejected. |
| Generation marker (`start_time`) to detect LUT change | Fires on a LUT-job restart with identical config, wiping consumer statistics mid-run. Rejected in favour of content comparison. |
| Reset on any received LUT, with no identity at all | Simplest, and correctly puts the decision at the producer. But a LUT-job restart re-emits an unchanged table, and that restart is the v0 recovery action; the follow-up liveness heartbeat would also clear the facility on every beat. Rejected in favour of a producer-stamped fingerprint. |
| Content comparison at the consumer | Works with `sc.allclose(..., equal_nan=True)`, but is N comparisons rather than one stamp and puts the noise tolerance at the consumer instead of at the setpoints it filters. `sc.identical` is not usable at all -- `False` for bit-identical `NaN`-bearing arrays. Rejected. |
| Fingerprint hashed from the table's bytes rather than its inputs | Not stable across a library or platform change, and says nothing about *why* the table changed. Rejected. |
| **LUT workflow consumes component motion streams** | Tables would always describe where the component actually is, and the static travel envelope -- the one number this design needs from the instrument team -- would disappear. Does *not* remove motion from consumers: they still need pixel positions for scattering geometry and for the per-pixel `Ltotal` that indexes the table. Costs: motion joins the LUT job's gating set, so a dead motion PV stops all wavelength reduction; every sample during a move re-emits and clears; and the LUT job's motion value can lag the one the consumer patched into its geometry, so padding is still needed. The strongest alternative here; recorded to revisit, not taken for v0. |
| `sc.identical` for that comparison | Returns `False` for bit-identical arrays containing `NaN`, which every LUT has. Would clear on every republish. Rejected in favour of `sc.allclose(..., equal_nan=True)`. |
| Mirror named for the LUT rather than for the seam | Same code either way; the generic name is the honest one. Rejected. |

## Consequences

- `WorkflowSpec` gains `context_outputs`; `ContextBinding` gains `clear_on_change`.
- The LUT carries an identity coord alongside its four provenance scalars.
- A new `StreamKind`, topic, sink route, ingest route and preprocessor case; the ingest
  half has no precedent in ADR 0006 and is the genuinely new mechanism.
- `gather_source_names` will include the LUT stream names, and `resolve_stream_names`
  will drop them because they appear in no `StreamLUT`. Harmless -- the context route is
  added unconditionally rather than derived from the mapping -- but it needs a test
  pinning that, since it looks like a bug.
- New wavelength-variant specs on DREAM and LOKI; existing TOA and file-based wavelength
  specs unchanged.
- `LtotalRange` disappears from the LUT workflow's user parameters.

### Known limitations, accepted for v0

- **Nothing guarantees the LUT job is running.** Under gating it is a hard prerequisite
  for all wavelength reduction on the instrument. If nobody starts it after a deployment,
  or someone stops it, every wavelength workflow goes `pending_context`. Recovery in v0 is
  manual: restart the LUT job.
- **A backend restart loses the LUT.** Consumers pin to the high watermark
  (`kafka/consumer.py:83-127`) -- no replay, no compaction, no seed (ADR 0002). The
  `chopper_cascade` tick fires only on `any_input_changed and all_locked`
  (`chopper_synthesizer.py:210-221`), and setpoints do not change during a steady run, so
  a restart mid-run blocks every wavelength job until the LUT job is restarted by hand.
  Acceptable while commissioning; backend restarts are rare.
- **The gate protects startup, not steady state.** Gate graduation is one-way and
  monotonic, and context is sticky (`stream_processor_workflow.py:60-75`). Stopping the
  LUT job does not re-block consumers; they keep reducing with the last table
  indefinitely, silently.
- **Two concurrent LUT jobs would flap**, both publishing the same stream names.
  `DeviceContract` catches duplicate names at construction (`device_contract.py:116-126`)
  but that is static across specs and cannot see two live jobs of one spec.

### Intended follow-up

The three limitations above are one problem -- the LUT stream has no liveness -- and have
one shape of answer, borrowed from how a physical device reports to NICOS: an **alarm
status**.

Content-compare clearing is what makes it affordable. Because a re-emitted identical table
causes no clear, the LUT job can heartbeat: "re-emitted within the last N seconds" becomes
liveness, "silent for more than k*N" becomes an alarm surfaced on every dependent job, and
the heartbeat doubles as recovery after a backend restart. A compacted topic consumed from
earliest is the more principled recovery mechanism and remains the long-term answer, but it
needs a per-topic exception to the unconditional high-watermark pin and message keys the
sink does not currently set.

Alongside it: a dashboard guard warning before a running LUT job is reconfigured or
stopped, since the blast radius is every wavelength workflow on the instrument; and a
decision on whether motion changes should clear consumers too.
