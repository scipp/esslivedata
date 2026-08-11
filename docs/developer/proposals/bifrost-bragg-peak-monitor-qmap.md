# BIFROST Bragg peak monitor Q-map — handoff

Status as of 2026-08-11. Supersedes PR #555 (open since 2025-11-26, branch
`bifrost-bragg-peak-monitor-qmap`, 2411 commits behind main and not worth rebasing).

## What the scientists asked for

> Using the full wavelength band, and during a whole sample rotation scan, the Bragg
> peak monitor collects a 2D intensity map of (Qx, Qy) in the laboratory reference
> system. This would normally yield a Bragg peak map to be plotted alongside the
> expected reciprocal lattice, and thus it is easy to find irregularities, spot
> magnetic peaks, do slices and see 1D plots, etc. This would be useful for the user,
> but for experts to setup experiments it would not be needed for hot commissioning.

Note the last sentence: this is not commissioning-critical.

## Identity: the Bragg peak monitor is `elastic_monitor` (cbm5)

Established by evidence, not naming:

- It is the only single-pixel event-mode monitor — 17.8M events with `event_id`
  uniformly 1, matching PR #555's `detector_fakes[...] = (1, 1)`.
- Upstream renamed the type to `ElasticMonitor`.
- `591151ebf` (2026-06-18) renamed cbm5 `bragg_peak_monitor` -> `elastic_monitor` and
  deleted `grid_templates/bragg_peak_monitor.yaml`.

For contrast, `normalization_monitor` (cbm4) is 1024-pixel and position-sensitive.

The spec is titled **"Elastic Q map (Bragg peak monitor)"**. The bare "Elastic Q map"
that #993 asked for is already taken by the `elastic_qmap` unified-detector workflow,
and workflow titles are not checked for uniqueness — two identical entries would appear
in the UI.

## Why the map is accumulated in the backend

The alternative was to emit a scalar timeseries and let the dashboard's
`correlation_histogram_2d` build the map. That is what #425 asked for ("aggregated
intensity in that interval as a function of instrument parameters") and it was closed
via #454 when the correlation histogram landed. `monitor_histogram`'s `total_in_range`
view already emits exactly that scalar.

That route is right for **alignment scans** and wrong for this. Two reasons:

- The requirement is the *full wavelength band*, so each update carries events spanning
  many Q. That is a distribution per update, not a scalar, so it must be histogrammed.
- BIFROST runs for days. A histogram costs memory proportional to bin count; a
  timeseries costs memory proportional to run length. `FullHistoryExtractor` returns
  `inf` from `get_required_timespan`, which makes `_trim_to_timespan` return without
  trimming, so the buffer falls back to ring-buffer eviction against
  `DEFAULT_MAX_MEMORY = 20 MB` — it silently discards the start of the run.

A "Q-stream" workflow (emit Q per update, histogram in the frontend) is the worst of
both: it inherits the frontend ceiling, and at a fixed wavelength Q is a deterministic
function of (a3, a4), so it is a relabeling of the angle streams rather than new
information.

## What is done

### Upstream — `scipp/ess`, branch `bragg-peak-monitor-nxmonitor` (pushed, no PR)

`0cabb8dc8` and `475730b00`, in
`packages/essspectroscopy/src/ess/bifrost/single_crystal/`. 18/18 essspectroscopy tests
pass, ruff clean.

The workflow could not run against *any* real BIFROST file:
`get_calibrated_bragg_peak_detector` required `NeXusComponent[snx.NXdetector]`, but
`elastic_monitor` is an `NXmonitor` in the CODA file, both geometry artifacts, and the
McStas simulation file alike. Only the user guide worked, by standing a detector triplet
in for the monitor (it carries an explicit warning saying so).

- `detector.py`: `get_calibrated_bragg_peak_monitor` takes the component, transformation
  and offset from `ElasticMonitor`. All those nodes already exist because the workflow
  declares `monitor_types=(ElasticMonitor, NormalizationMonitor)` — this is a
  re-pointing, not new machinery. Position is the transformed origin (a monitor has no
  pixel offsets); a `detector_number` is synthesised for the single pixel. The
  `Analyzer` dependency is dropped: `get_base_calibrated_detector_bifrost` never reads
  it, and a monitor in the direct beam has no analyzer.
- `detector.py`: `assemble_bragg_peak_monitor_data` assigns geometry onto events rather
  than calling `assemble_detector_data`, which groups by `event_id` — streamed monitor
  events carry none. This was only found by running the live path; the file-based path
  worked because CODA's `cbm5_events` *does* have `event_id`.
- `time_of_flight.py`: fixes a latent defect. A time-dependent position makes `ltotal`
  carry `time`, but `group_by_rotation` has already renamed that dimension to `a4`, so
  the broadcast rejected it. **This bites whenever the tank rotates and is not specific
  to the monitor.** It stayed hidden because the user guide's stand-in position is
  static, and because the inelastic/Q-cut path uses a different `detector_wavelength_data`
  that takes no `ltotal` at all.
- The detector stand-in moved to `simulation_providers`, keeping the user guide working
  (verified: its `EmptyDetector` still computes).

### esslivedata — branch `555-elastic-qmap` (pushed)

Spec `bifrost/bragg_peak_qmap/1` on `source_names=['elastic_monitor']`,
`BraggPeakQMapParams` (Q∥/Q⊥ edges), 2D `q_map` output, and a factory accumulating
`IntensityQparQperp`. Geometry comes from the geometry artifact; the detector tank angle
is an instrument-scope chain-patch binding (`DetectorTankAngleLog`) that also feeds the
a4 grouping coordinate, and the sample rotation stays a spec-scope direct bind.

This is the first reduction spec whose primary source is a monitor. That path works:
monitor events reach `data_reduction`, the job goes `pending_context` -> holds events
without crashing -> `job_activated` once both rotation devices publish their
RBV/VAL/DMOV substreams.

Three strict markers record the single remaining blocker and clear themselves when the
artifact is fixed: the service test's `xfail`, `_KNOWN_UNRESOLVABLE_CHAINS` in
`tests/config/motion_binding_test.py`, and `_BLOCKED_ON_GEOMETRY_ARTIFACT` in
`tests/config/registered_workflow_factories_test.py`.

## The chain-patch binding is done; one producer-side blocker remains

The monitor is mounted on the detector tank, so its transformation chain runs through
`detector_tank_angle`, whose value is live. That is now handled by a chain-patch context
binding (ADR 0003, `config/value_log.py`): `DetectorTankAngleLog` is a `ValueLog`
subclass, so the routing layer substitutes the live f144 NXlog into the NeXus chain at
the path derived from the stream name (`Instrument.chain_patch_path`).

Three things had to be settled to make that work.

**Geometry had to move off the McStas file onto the geometry artifact.** The patch path
is the f144 stream's `nexus_path`,
`/entry/instrument/detector_tank_angle/transformations/detector_tank_angle_r0/value`.
The artifact writes the chain entry at exactly that path (a length-0 NXlog placeholder);
the McStas file keys the same transform one level up, at
`.../detector_tank_angle_r0`, and stores the 720-sample rotation scan in it. So on the
McStas file the patch cannot land at all:

```
KeyError: "Transformation entry '.../detector_tank_angle_r0/value' not found in chain."
```

`_init_bragg_peak_workflow` therefore uses `instrument.nexus_file`. Nothing else was
holding it to the simulation file: against the registered artifact the monitor chain is
the *only* thing that fails (unlike `BifrostQCutWorkflow`, which also needs analyzer
geometry the artifacts lack).

**The tank angle is needed twice, but a stream carries one context key per spec.**
`resolve_context_keys` returns `{stream_name: workflow_key}` and `validate` rejects the
same stream at both scopes, so `detector_tank_angle_r0` cannot be bound to both
`InstrumentAngle[SampleRun]` (what `group_by_rotation` bins on) and the `ValueLog`. The
chain patch is the binding, and `_instrument_angle_from_tank_log` derives the coordinate
from the same log — the payload is identical, direct-bind just passes it unwrapped.

**The plain monitor histogram had to opt out.** Chain-patch bindings must be declared at
instrument scope, and `dependent_sources={'elastic_monitor'}` also catches
`monitor_histogram` on that monitor, which would have gated a counts-over-TOA workflow on
the motor readback. Hence `specs.monitor_handle.skip_instrument_contexts()`.

### Verified end-to-end

With the monitor's transformations restored in the artifact (see below), the service test
**XPASSes**: streamed monitor events plus the two rotation devices yield a populated
(100, 100) `(Q_perpendicular, Q_parallel)` map — 1952 of 2000 events land in it. The live
a4 drives both the monitor position and the grouping coordinate. This is the real path,
not the earlier static-position stand-in.

Reproduce by rebuilding the fixture — copy `geometry-bifrost-2026-08-11.nxs`, then copy
`elastic_monitor/transformations` (`r0` -> `t0_z` -> `t0_x`) in from
`coda_bifrost_999999_00006061.hdf`, rewriting the final `depends_on` to
`/entry/instrument/detector_tank_angle/transformations/detector_tank_angle_r0/value` —
and pointing `instrument.nexus_file` at it.

## Geometry artifacts

`geometry-bifrost-2026-08-11.nxs` was regenerated from
`coda_bifrost_999999_00016610.hdf` (md5 `6291ecb5b1c0627dc7759e31f126c679`, not
uploaded, registry untouched).

Relevant to the #962 pin: all 45 detector chains resolve and there are **no stale
`117_` groups**, so both defects the pin comment names are gone. But only structural
resolution was checked, not numerical correctness of the positions.

**The writer also changed how it references the tank angle, and that is what makes
chain-patching possible.** In the new file the chains end at
`.../detector_tank_angle_r0/value` — the NXlog itself, terminating with `depends_on='.'`
— which is both the LOKI convention and exactly the path `chain_patch_path` derives from
the f144 stream. Older files reference the enclosing `detector_tank_angle_r0`
NXpositioner *group* instead, whose own `depends_on` points at
`/entry/instrument/117_detector_tank_angle/...`, a group that exists in neither the
artifact nor its CODA source. So in the currently registered
`geometry-bifrost-2026-06-08.nxs` **no BIFROST chain resolves at all** — not the
monitor's and not the detectors' — and its `elastic_monitor` transformations
(`r0` 59 deg -> `t0_z` 0.412 m -> `t0_x` 0.686 m -> `detector_tank_angle_r0`) are
structurally present but unusable. Only the values are still good, which is why they can
be lifted into the fixture above.

**Producer bug, blocking regeneration.** In `coda_bifrost_999999_00016610.hdf` the two
event-mode monitors (`elastic_monitor`, `normalization_monitor`) have a dangling
`depends_on` pointing at a `transformations` group that does not exist; they have no
such group at all. The three histogram-mode monitors are fine. A report is drafted.
This is now the *only* thing between the branch and a working Q map: everything else —
the binding, the artifact's tank-angle placeholder, the workflow — is verified.

## Dead ends — do not repeat

- **"Artifact chains must stop before the dynamic node."** Wrong. `depends_on` runs
  toward the root, so truncating discards static transforms beyond the dynamic one. It
  only appeared to work because nothing follows `detector_tank_angle_r0` in this file.
  `make_geometry_nexus.py` needs no truncation change; its NXlog trimming to length 0 is
  correct, because the value is patched in at runtime.
- **"The position must be derived from live a4 inside upstream code."** Wrong for the
  same reason — the mechanism already exists as chain-patching.
- **Forcing a static position to make the chain compute.** It does produce a (100, 100)
  map, but it discards the a4 dependence and the result is not physically meaningful.
- **Rebasing PR #555.** Its `exclude_from_merge` plumbing in `kafka/routes.py` and
  `message_adapter.py` exists only because it assumed the monitor arrived on the
  *detector* topic. cbm5 is an NXmonitor on `bifrost_beam_monitor`; delete, do not port.
- **Keeping the McStas file as this workflow's geometry source.** Its tank-angle
  transform is keyed one level above the f144 path, so the chain patch can never land;
  the workflow would be permanently inconsistent with its own binding.
- **Binding the tank angle twice (direct + chain-patch) for the same spec.** One context
  key per stream: `resolve_context_keys` is keyed by stream name and cross-scope
  duplicates are rejected outright. Derive the second use with a provider instead.

## Open questions

- Can the *detector* Q-map workflows move off the McStas file? The monitor one has
  (it must, to chain-patch), but pointing `BifrostQCutWorkflow` at the artifact still
  fails with `No NXcrystal found in the inputs of 135_channel_1_4_triplet` — the
  artifacts carry no analyzer geometry. So BIFROST currently reads geometry from two
  different files, which is worth resolving.
- The scientists say "(Qx, Qy) in the **laboratory** reference system", but upstream's
  `project_momentum_transfer` projects `sample_table_momentum_transfer` — the sample-table
  frame, which is what you would want to compare against an expected reciprocal lattice.
  Worth settling with them before the upstream PR goes up.
- Upstream test data contains no Bragg peak monitor with event data (the simulation
  file's `elastic_monitor` is histogram-mode), so there is no way to add a regression
  test for the monitor path upstream yet.
- `single_crystal/detector.py` imports `_assign_detector_position` from the sibling
  `..detector`; intra-package but private. Reviewers may want it promoted.

## Next steps

1. Get the producer to restore the event-mode monitors' transformations, then regenerate
   and register the artifact. All three strict markers flip together when it lands.
2. Open the upstream PR, ideally once the frame question above is settled.
3. Decide on the #962 pin separately, on the strength of a numerical position check.
   Note the registered artifact's chains do not resolve at all, so whatever the pin
   says, BIFROST detector geometry is not currently coming from a resolvable chain.
