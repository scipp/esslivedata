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
`BraggPeakQMapParams` (Q∥/Q⊥ edges), 2D `q_map` output, spec-scope context bindings, and
a factory accumulating `IntensityQparQperp`. 1212 tests pass, 1 strict xfail.

This is the first reduction spec whose primary source is a monitor. That path works:
monitor events reach `data_reduction`, the job goes `pending_context` -> holds events
without crashing -> `job_activated` once both rotation devices publish their
RBV/VAL/DMOV substreams.

## The one remaining blocker

The monitor is mounted on the detector tank, so its transformation chain runs through
`detector_tank_angle`, whose value is live. Geometry currently comes from the McStas
simulation file, whose tank-angle NXlog is a 720-sample rotation scan, so the position
arrives time-dependent on *that file's* time base and cannot be assigned onto streamed
events:

```
DimensionError: Expected dimension to be in [event_time_zero:1, ], got time.
```

**The fix is a chain-patch context binding**, which the framework already supports.
A `ContextBinding` whose `workflow_key` is a `ValueLog` subclass is routed via the fused
per-component patched-chain provider, substituting the live f144 NXlog into the NeXus
chain at a path derived from the stream name (`Instrument.chain_patch_path`). See ADR
0003 and `config/value_log.py`.

The existing BIFROST a3/a4 bindings are explicitly *direct-bind*: they feed
`group_by_rotation` a coordinate, not geometry. This workflow needs a4 **both** ways.
Note `ValueLog` keys must be declared at instrument scope, so the spec-scope bindings
currently in `factories.py` will need rethinking.

### Verified that the workflow itself is correct

Given a static tank-frame position, the end-to-end service test **XPASSes** and produces
a 2D map from streamed monitor events. Reproduce with
`/workspace/esslivedata/geometry-bifrost-2026-08-11-tankframe.nxs` as
`Filename[SampleRun]` in `_init_bragg_peak_workflow`. Note that file was made by
truncating the chain before the tank rotation, which is **not** a correct general
approach — see the dead ends below. It only demonstrates that everything downstream of
the position works.

## Geometry artifacts

`geometry-bifrost-2026-08-11.nxs` was regenerated from
`coda_bifrost_999999_00016610.hdf` (md5 `6291ecb5b1c0627dc7759e31f126c679`, not
uploaded, registry untouched).

Relevant to the #962 pin: all 45 detector chains resolve and there are **no stale
`117_` groups**, so both defects the pin comment names are gone. But only structural
resolution was checked, not numerical correctness of the positions.

Also worth knowing: in the pinned `geometry-bifrost-2025-01-01.nxs` the detector chains
are three links long and **never reach `detector_tank_angle`**, so the geometry we ship
today has detector positions that do not depend on the tank angle at all.

**Producer bug, blocking regeneration.** In `coda_bifrost_999999_00016610.hdf` the two
event-mode monitors (`elastic_monitor`, `normalization_monitor`) have a dangling
`depends_on` pointing at a `transformations` group that does not exist; they have no
such group at all. The three histogram-mode monitors are fine. This is a regression —
`geometry-bifrost-2026-06-08.nxs` has a complete chain for `elastic_monitor`
(`r0` 59 deg -> `t0_z` 0.412 m -> `t0_x` 0.686 m -> `detector_tank_angle_r0`). A report
is drafted; without those transformations there is no chain to patch into.

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

## Open questions

- Can the Q-map workflows move off the McStas file at all? Pointing `BifrostQCutWorkflow`
  at the pinned artifact fails with `No NXcrystal found in the inputs of
  135_channel_1_4_triplet` — the artifacts carry no analyzer geometry, which looks like
  why the McStas file is used. Affects the instruction to use the same artifact
  everywhere.
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

1. Declare the tank angle as a chain-patch binding for this spec and re-run the test;
   the strict xfail flips loudly when it works.
2. Get the producer to restore the monitor transformations, then regenerate the artifact.
3. Open the upstream PR, ideally once the frame question above is settled.
4. Decide on the #962 pin separately, on the strength of a numerical position check.
